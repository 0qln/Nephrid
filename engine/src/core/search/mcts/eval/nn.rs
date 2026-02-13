use burn::tensor::{DType, TensorData};

use crate::core::search::mcts::search::SelectionNode;

use super::*;

#[derive(Debug, PartialEq)]
pub struct InputFloats {
    board: BoardInputFloats,
    state: StateInputFloats,
}

impl InputFloats {
    pub fn new(pos: &Position) -> Self {
        InputFloats {
            board: board_input(pos),
            state: state_input(pos),
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct TraceInfo {
    /// Input floats for the eval model.
    inputs: InputFloats,

    /// The node that this eval info is for.
    node: Rc<RefCell<Node>>,
}

impl TraceInfo {
    pub fn new(node: Rc<RefCell<Node>>, pos: &Position) -> Self {
        Self {
            inputs: InputFloats::new(pos),
            node,
        }
    }
}

/// X: batch size
pub struct NNEvaluator<'a, 'b, B: Backend, const X: usize> {
    /// NN Model
    model: &'a Model<B>,

    // Device on which the nn will run.
    device: &'b B::Device,
}

impl<'a, 'b, B: Backend, const X: usize> NNEvaluator<'a, 'b, B, X> {
    pub fn new(model: &'a Model<B>, device: &'b B::Device) -> Self {
        Self { model, device }
    }

    fn device(&self) -> &B::Device {
        self.device
    }

    /// batch: The iterator of the selected leaf nodes that should be evaluated
    /// in this batch.
    fn build_board_batch(
        &self,
        batch: impl Iterator<Item = SelectionNodeRef<<Self as Evaluator>::TraceData>>,
    ) -> Tensor<B, 4> {
        // concatenate the board inputs along the batch dimension.
        Tensor::cat(
            batch
                .map(|leaf| Self::get_node_history(leaf))
                // concatenate the board inputs along the channel dimension.
                .map(|history| nn::board_history_input(&history, self.device()))
                .collect_vec(),
            0,
        )
    }

    /// Returns the history of the given selected leaf in following order:
    /// - the oldest board state is the first index
    /// - the youngest board state is the last index
    fn get_node_history(
        leaf: SelectionNodeRef<<Self as Evaluator>::TraceData>,
    ) -> Vec<BoardInputFloats> {
        let mut vec: Vec<BoardInputFloats> = vec![];

        _ = SelectionNode::try_fold_up_mut(leaf.clone(), (), |_, leaf| {
            if vec.len() == BOARD_INPUT_HISTORY {
                return ControlFlow::Break(());
            }

            let board_input = leaf.borrow().data().trace_data.inputs.board;
            vec.insert(0, board_input);

            ControlFlow::Continue::<(), ()>(())
        });

        vec
    }

    fn build_state_batch(
        &self,
        batch: impl Iterator<Item = SelectionNodeRef<<Self as Evaluator>::TraceData>>,
    ) -> Tensor<B, 2> {
        // concatenate the state inputs along the batch dimension.
        Tensor::cat(
            batch
                .map(|leaf| {
                    let state_input = leaf.borrow().data().trace_data.inputs.state;
                    Tensor::from_floats([state_input], self.device())
                })
                .collect_vec(),
            0,
        )
    }
}

impl<'a, 'b, B: Backend, const X: usize> Evaluator for NNEvaluator<'a, 'b, B, X> {
    type TraceData = TraceInfo;

    fn trace(&self, node: Rc<RefCell<Node>>, pos: &Position) -> Self::TraceData {
        TraceInfo::new(node, pos)
    }

    fn eval_batch(
        &mut self,
        leafs: &[SelectionNodeRef<Self::TraceData>],
    ) -> impl Iterator<Item = Evaluation> {
        let batch_size = leafs.len();

        let values_shape = Shape::new([batch_size, VALUE_OUTPUTS]);
        let policy_shape = Shape::new([batch_size, POLICY_OUTPUTS]);

        let (values, raw_policies) = if batch_size != 0 {
            let board_batch = self.build_board_batch(leafs.iter().cloned());
            let state_batch = self.build_state_batch(leafs.iter().cloned());
            let (values, raw_policies) = self.model.forward(board_batch, state_batch);

            assert_eq!(values.shape(), values_shape);
            assert_eq!(raw_policies.shape(), policy_shape);

            let values = values.into_data();
            let raw_policies = raw_policies.into_data();

            (values, raw_policies)
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

        let raw_policies = raw_policies
            .as_slice::<f32>()
            .expect("Policy could not be converted to vec.");

        let values = values.chunks(VALUE_OUTPUTS);

        let raw_policies = raw_policies
            .chunks(POLICY_OUTPUTS)
            .map(|raw_policy| RawPolicy(raw_policy.try_into().unwrap()));

        leafs
            .iter()
            .zip(values)
            .zip(raw_policies)
            .map(|((b, c), d)| (b, c, d))
            .map(|(leaf, value, raw_policy)| {
                let leaf = leaf.borrow();
                let data = leaf.data();

                Evaluation::Guess(Box::new(Guess {
                    relative_to: data.turn,
                    quality: value[0],
                    policy: raw_policy,
                }))
            })
            .collect_vec()
            .into_iter()
    }
}
