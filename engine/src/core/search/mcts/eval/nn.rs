use burn::tensor::{DType, Shape, Tensor, TensorData, backend::Backend};
use itertools::Itertools;
use std::rc::Rc;

use crate::core::search::mcts::{
    nn::{
        self, BOARD_INPUT_HISTORY, BoardInputFloats, Model, StateInputFloats, VALUE_OUTPUTS,
        board_input, state_input,
    },
    search::{BatchItem, Selection},
};

use super::*;

#[derive(Debug, PartialEq)]
pub struct InputFloats {
    pub board: BoardInputFloats,
    pub state: StateInputFloats,
}

impl InputFloats {
    pub fn new(pos: &Position) -> Self {
        InputFloats {
            board: board_input(pos),
            state: state_input(pos),
        }
    }
}

#[derive(Debug)]
pub struct TraceInfo {
    /// Input floats for the eval model.
    pub inputs: InputFloats,
}

impl TraceInfo {
    pub fn new(pos: &Position) -> Self {
        Self { inputs: InputFloats::new(pos) }
    }
}

/// X: batch size
#[derive(Debug)]
pub struct NNEvaluator<B: Backend> {
    /// NN Model
    model: Rc<Model<B>>,

    // Device on which the nn will run.
    device: Rc<B::Device>,
}

/// Returns the history of the given selected leaf in following order:
/// - the oldest board state is the first index
/// - the youngest board state is the last index
pub fn get_node_history<const X: usize>(
    selection: &Selection<X, TraceInfo>,
    leaf: &BatchItem<TraceInfo>,
) -> Vec<BoardInputFloats> {
    let mut vec: Vec<BoardInputFloats> = vec![];

    // 1. Insert the leaf's own board input first
    vec.insert(0, leaf.data.inputs.board);

    // 2. Traverse up the tree to gather parent board inputs
    for node in selection.iter_path_up(leaf.parent) {
        if vec.len() == BOARD_INPUT_HISTORY {
            break;
        }

        let board_input = node.data.inputs.board;
        vec.insert(0, board_input);
    }

    vec
}

impl<B: Backend> NNEvaluator<B> {
    pub fn new(model: Rc<Model<B>>, device: Rc<B::Device>) -> Self {
        Self { model, device }
    }

    fn device(&self) -> &B::Device {
        &self.device
    }

    /// batch: The iterator of the selected leaf nodes that should be evaluated
    /// in this batch.
    fn build_board_batch<'c, const X: usize>(
        &self,
        selection: &Selection<X, TraceInfo>,
        batch: impl Iterator<Item = &'c BatchItem<TraceInfo>>,
    ) -> Tensor<B, 4> {
        // concatenate the board inputs along the batch dimension.
        Tensor::cat(
            batch
                .map(|leaf| get_node_history(selection, leaf))
                // concatenate the board inputs along the channel dimension.
                .map(|history| nn::board_history_input(&history, self.device()))
                .collect_vec(),
            0,
        )
    }

    fn build_state_batch<'c>(
        &self,
        batch: impl Iterator<Item = &'c BatchItem<TraceInfo>>,
    ) -> Tensor<B, 2> {
        // concatenate the state inputs along the batch dimension.
        Tensor::cat(
            batch
                .map(|leaf| {
                    let state_input = leaf.data.inputs.state;
                    Tensor::from_floats([state_input], self.device())
                })
                .collect_vec(),
            0,
        )
    }
}

impl<B: Backend> Evaluator for NNEvaluator<B> {
    type TraceData = TraceInfo;

    fn trace<S: HasBranches>(
        &self,
        _node: NodeId<S>,
        _tree: &Tree,
        pos: &mut Position,
    ) -> Self::TraceData {
        TraceInfo::new(pos)
    }

    fn eval_batch<const X: usize>(
        &mut self,
        tree: &Tree,
        selection: &Selection<X, Self::TraceData>,
        leafs: &[&BatchItem<Self::TraceData>],
    ) -> impl Iterator<Item = Evaluation> {
        let batch_size = leafs.len();

        let values_shape = Shape::new([batch_size, VALUE_OUTPUTS]);
        let policy_shape = Shape::new([batch_size, POLICY_OUTPUTS]);

        let (values, raw_logits) = if batch_size != 0 {
            let board_batch = self.build_board_batch(selection, leafs.iter().copied());
            let state_batch = self.build_state_batch(leafs.iter().copied());
            let (values, raw_policies) = self.model.forward(board_batch, state_batch);

            assert_eq!(values.shape(), values_shape);
            assert_eq!(raw_policies.shape(), policy_shape);

            let values = values.into_data();
            let raw_logits = raw_policies.into_data();

            (values, raw_logits)
        }
        else {
            (
                TensorData::from_bytes_vec(vec![], values_shape, DType::F32),
                TensorData::from_bytes_vec(vec![], policy_shape, DType::F32),
            )
        };

        let values = values
            .as_slice::<f32>()
            .expect("Qualities could not be converted to vec.");

        let raw_logits = raw_logits
            .as_slice::<f32>()
            .expect("Policy could not be converted to vec.");

        let values = values.chunks(VALUE_OUTPUTS);

        let raw_logits = raw_logits
            .chunks(POLICY_OUTPUTS)
            .map(|raw_logits| RawLogits(raw_logits.try_into().unwrap()));

        leafs
            .iter()
            .zip(values)
            .zip(raw_logits)
            .map(|((&leaf, value), raw_logits)| {
                let turn = leaf.turn;
                let moves = tree.move_indices(leaf.node);

                Evaluation::Guess(Box::new(Guess {
                    relative_to: turn,
                    quality: Quality::new(value[0]),
                    policy: Policy::from_raw_logits(&raw_logits, moves, 1.).expect("a policy"),
                }))
            })
            .collect_vec()
            .into_iter()
    }
}
