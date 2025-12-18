use super::utils::*;
use crate::core::Position;
use crate::core::color::Color;
use crate::core::coordinates::files;
use crate::core::coordinates::ranks;
use crate::core::position::CheckState;
use crate::core::search::mcts::nn::BOARD_INPUT_CHANNELS;
use crate::core::search::mcts::nn::BOARD_INPUT_HISTORY;
use crate::core::search::mcts::nn::BoardInputFloats;
use crate::core::search::mcts::nn::BoardInputTensor;
use crate::core::search::mcts::nn::Model;
use crate::core::search::mcts::nn::POLICY_OUTPUTS;
use crate::core::search::mcts::nn::StateInputFloats;
use crate::core::search::mcts::nn::VALUE_OUTPUTS;
use crate::core::search::mcts::nn::board_input;
use crate::core::search::mcts::nn::state_input;
use crate::core::search::mcts::node::Node;
use crate::core::turn::Turn;
use burn::Tensor;
use burn::tensor::Shape;
use burn::tensor::backend::Backend;
use itertools::Itertools;
use std::cell::RefCell;
use std::ops::ControlFlow;
use std::rc::Rc;

#[cfg(test)]
pub mod test;

pub trait Evaluator<const X: usize> {
    // Prepare an eval_info_node with the required info for this evaluator.
    fn register_info(
        &mut self,
        eval_node: Rc<RefCell<EvalInfoNode>>,
        node: Rc<RefCell<Node>>,
        pos: &Position,
    );

    /// Evluate all the nodes in the batch.
    /// (Which is, all the nodes that are eval `None`)
    fn eval_guesses(&mut self);

    /// Evaluate a node's terminal state. If the node is terminal, return the evaluation, else
    /// return None.
    fn eval_terminal(node: &Node, pos: &Position) -> Option<Evaluation> {
        // First check if the position is a normal game ending.
        if !node.has_branches() {
            Some(if pos.get_check_state() != CheckState::None {
                // If in check and no moves, it's a loss for the current player
                Evaluation::Terminal(GameResult::Win { relative_to: !pos.get_turn() })
            } else {
                // Stalemate
                Evaluation::Terminal(GameResult::Draw)
            })
        }
        // Then check if the position has reached some of the extra-rule endings.
        else if pos.has_threefold_repetition()
            || pos.fifty_move_rule()
            || pos.is_insufficient_material()
        {
            Some(Evaluation::Terminal(GameResult::Draw))
        }
        // Otherwise not a terminal evaluation.
        else {
            None
        }
    }

    /// Set the evaluation at a specific index.
    fn set_eval(&mut self, index: usize, eval: Evaluation);

    /// Mark the node at `index` to be evaluated when doing `eval_guesses`.
    fn batch_eval(&mut self, index: usize, eval_node: Rc<RefCell<EvalInfoNode>>);

    /// Get the evaluation at a specific index.
    fn get_eval(&self, index: usize) -> Option<&Evaluation>;

    /// Initializes and returns a reference to the root eval info node.
    fn init(&mut self) -> Rc<RefCell<EvalInfoNode>>;

    fn iter(&self) -> impl Iterator<Item = &EvalState>;
}

#[derive(Debug, PartialEq)]
pub struct InputFloats {
    board: BoardInputFloats,
    state: StateInputFloats,
}

#[derive(PartialEq, Debug)]
pub struct EvalInfo {
    /// Input floats for the eval model.
    inputs: InputFloats,

    /// The node that this eval info is for.
    node: Rc<RefCell<Node>>,

    /// Turn of the current player.
    turn: Turn,
}

#[derive(Clone, Debug)]
pub enum EvalState {
    OnBatch(Rc<RefCell<EvalInfoNode>>),
    Evaluated(Evaluation),
    None,
}

pub type EvalInfoNode = DoubleLinkedNode<Option<EvalInfo>>;

pub struct EvaluationInfos<const X: usize> {
    root: Option<Rc<RefCell<EvalInfoNode>>>,
    evals: [EvalState; X],
}

/// X: batch size
pub struct NNEvaluator<'a, 'b, B: Backend, const X: usize> {
    /// Eval infos
    eval_infos: EvaluationInfos<X>,

    /// NN Model
    model: &'a Model<B>,

    // Device on which the nn will run.
    device: &'b B::Device,
}

impl<'a, 'b, B: Backend, const X: usize> NNEvaluator<'a, 'b, B, X> {
    pub fn new(model: &'a Model<B>, device: &'b B::Device) -> Self {
        Self {
            model,
            device,
            eval_infos: EvaluationInfos {
                evals: [const { EvalState::None }; X],
                root: None,
            },
        }
    }

    fn device(&self) -> &B::Device {
        self.device
    }

    fn build_board_batch(&self) -> Tensor<B, 4> {
        // concatenate the board inputs along the batch dimension.
        Tensor::cat(
            self.iter_batch()
                .map(|eval_node| Self::get_node_history(eval_node))
                // concatenate the board inputs along the channel dimension.
                .map(|history| {
                    // pad missing history info with zeroes.
                    let padding_len = BOARD_INPUT_HISTORY - history.len();
                    let padding_tensor = BoardInputTensor::<B>::zeros(
                        [
                            1,
                            padding_len * BOARD_INPUT_CHANNELS,
                            ranks::N_VARIANTS,
                            files::N_VARIANTS,
                        ],
                        self.device(),
                    );

                    // convert input floats to tensors
                    let history_tensor = Tensor::cat(
                        history
                            .into_iter()
                            .map(|b| Tensor::from_floats([b], self.device()))
                            .collect_vec(),
                        1,
                    );

                    // concat padding with history
                    Tensor::cat(vec![padding_tensor, history_tensor], 1)
                })
                .collect_vec(),
            0,
        )
    }

    /// return the history where:
    /// the oldest board state is the first index
    /// the youngest board state is the last index
    fn get_node_history(eval_info: Rc<RefCell<EvalInfoNode>>) -> Vec<BoardInputFloats> {
        let mut vec: Vec<BoardInputFloats> = vec![];

        _ = EvalInfoNode::try_fold_up_mut(eval_info.clone(), (), |_, eval_info| {
            let board_input = eval_info
                .borrow()
                .data()
                .as_ref()
                .expect("Eval info is missing")
                .inputs
                .board;

            vec.insert(0, board_input);

            ControlFlow::Continue::<(), ()>(())
        });

        vec
    }

    fn build_state_batch(&self) -> Tensor<B, 2> {
        // concatenate the state inputs along the batch dimension.
        Tensor::cat(
            self.iter_batch()
                .map(|eval_info| {
                    let state_input = eval_info
                        .borrow()
                        .data()
                        .as_ref()
                        .expect("Eval info is missing")
                        .inputs
                        .state;

                    Tensor::from_floats([state_input], self.device())
                })
                .collect_vec(),
            0,
        )
    }

    fn iter_batch(&self) -> impl Iterator<Item = Rc<RefCell<EvalInfoNode>>> {
        self.eval_infos.evals.iter().filter_map(|x| {
            if let EvalState::OnBatch(eval_info) = x {
                Some(eval_info.clone())
            } else {
                None
            }
        })
    }
}

impl<'a, 'b, B: Backend, const X: usize> Evaluator<X> for NNEvaluator<'a, 'b, B, X> {
    fn init(&mut self) -> Rc<RefCell<EvalInfoNode>> {
        let root = Rc::new(RefCell::new(EvalInfoNode::new_root(None)));
        self.eval_infos.root = Some(root.clone());
        root
    }

    fn register_info(
        &mut self,
        eval_node: Rc<RefCell<EvalInfoNode>>,
        node: Rc<RefCell<Node>>,
        pos: &Position,
    ) {
        let board = board_input(pos);
        let state = state_input(pos);
        let input = InputFloats { board, state };
        let data = EvalInfo {
            inputs: input,
            node,
            turn: pos.get_turn(),
        };
        eval_node.borrow_mut().set_data(Some(data));
    }

    fn get_eval(&self, index: usize) -> Option<&Evaluation> {
        if let Some(x) = self.eval_infos.evals.get(index)
            && let EvalState::Evaluated(eval) = x
        {
            Some(eval)
        } else {
            None
        }
    }

    fn set_eval(&mut self, index: usize, eval: Evaluation) {
        if let Some(x) = self.eval_infos.evals.get_mut(index) {
            *x = EvalState::Evaluated(eval);
        }
    }

    fn batch_eval(&mut self, index: usize, eval_node: Rc<RefCell<EvalInfoNode>>) {
        if let Some(x) = self.eval_infos.evals.get_mut(index) {
            println!("prepare batch index: {index}");
            *x = EvalState::OnBatch(eval_node);
        } else {
            panic!("Out of range");
        }
    }

    fn eval_guesses(&mut self) {
        let batch_size = self.iter_batch().count();
        println!("batchsize: {batch_size}");
        if batch_size == 0 {
            return;
        }

        let board_batch = self.build_board_batch();
        let state_batch = self.build_state_batch();
        let (values, raw_policies) = self.model.forward(board_batch, state_batch);

        assert_eq!(values.shape(), Shape::new([batch_size, VALUE_OUTPUTS]));
        assert_eq!(
            raw_policies.shape(),
            Shape::new([batch_size, POLICY_OUTPUTS])
        );

        let values = values.into_data();
        let values = values
            .as_slice::<f32>()
            .expect("Qualities could not be converted to vec.");
        let values = values.chunks(VALUE_OUTPUTS);

        let raw_policies = raw_policies.into_data();
        let raw_policies = raw_policies
            .as_slice::<f32>()
            .expect("Policy could not be converted to vec.");
        let raw_policies = raw_policies
            .chunks(POLICY_OUTPUTS)
            .map(|raw_policy| RawPolicy(raw_policy.try_into().unwrap()));

        // Enumerate all eval_infos, which have not yet been assigned and assign them a their guess.
        for (index, eval_info, value, raw_policy) in self
            .eval_infos
            .evals
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                if let EvalState::OnBatch(eval_info) = x {
                    Some((i, eval_info.clone()))
                } else {
                    None
                }
            })
            // ---
            // todo: we allocate here bc if we borrow above, we cannot borrow mutably in the
            // for loop to set the eval_infos...
            // find a better solution
            .collect_vec()
            .into_iter()
            // ---
            .zip(values)
            .zip(raw_policies)
            .map(|(((a, b), c), d)| (a, b, c, d))
        {
            let eval_info = eval_info.borrow();
            let eval_info = eval_info
                .data()
                .as_ref()
                .expect("This should be a leaf and leafes should have data.");

            let eval = Evaluation::Guess(Box::new(Guess {
                relative_to: eval_info.turn,
                quality: value[0],
                policy: raw_policy,
            }));
            self.eval_infos.evals[index] = EvalState::Evaluated(eval);
        }
    }

    fn iter(&self) -> impl Iterator<Item = &EvalState> {
        self.eval_infos.evals.iter()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct RawPolicy([f32; POLICY_OUTPUTS]);

impl RawPolicy {
    pub fn get(&self, i: usize) -> Option<f32> {
        self.0.get(i).cloned()
    }

    pub fn new(p: [f32; POLICY_OUTPUTS]) -> Self {
        Self(p)
    }

    pub fn sum(&self) -> f32 {
        self.0.iter().sum::<f32>()
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(POLICY_OUTPUTS, self.0.len());
        POLICY_OUTPUTS
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> {
        self.0.iter().cloned()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Policy(Vec<f32>);

impl Policy {
    pub fn get(&self, i: usize) -> Option<f32> {
        self.0.get(i).cloned()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn from_raw<I>(raw_policy: &RawPolicy, indeces_of_interest: I) -> Option<Self>
    where
        I: Iterator<Item = usize>,
    {
        let mut policy = Vec::<f32>::new();
        for index in indeces_of_interest {
            policy.push(raw_policy.get(index)?);
        }

        Some(Self::new(policy))
    }

    pub fn new(mut policy: Vec<f32>) -> Self {
        // Renormalize the policy to a sum of 1, since not all of the probabilities
        // were assigned to moves that are actually playable in this position:
        let policy_sum = {
            let sum = policy.iter().sum();
            if sum == 0.0 {
                // Fallback to uniform distribution
                policy.len() as f32
            } else {
                sum
            }
        };
        for policy in &mut policy {
            *policy /= policy_sum;
        }

        // Evaluator should return a probability distribution.
        let f32_eq = |a: f32, b: f32, e: f32| f32::abs(a - b) < e;
        debug_assert!(f32_eq(policy.iter().sum::<f32>(), 1., 0.001));

        Self(policy)
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GameResult {
    Win { relative_to: Color },
    Draw,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Guess {
    pub relative_to: Color,
    pub quality: f32,
    pub policy: RawPolicy,
}

impl Guess {
    pub fn policy(&self) -> &RawPolicy {
        &self.policy
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Evaluation {
    /// we will go further and have a guess about this game.
    Guess(Box<Guess>),
    /// we cannot go any further.
    Terminal(GameResult),
    /// we don't feel like going any further.
    Nope,
}

impl GameResult {
    /// Returns a number between 0 and 1, where 0 is a loss and 1 is a win.
    const fn to_value(self, turn: Color) -> f32 {
        match self {
            Self::Win { relative_to } => {
                if relative_to.v() == turn.v() {
                    Self::win_value()
                } else {
                    Self::loss_value()
                }
            }
            Self::Draw => Self::draw_value(),
        }
    }

    // these are functions, because maybe later we want to have different values for e.g. a
    // win that is close to the root node or further.

    pub const fn draw_value() -> f32 {
        0.5
    }

    pub const fn win_value() -> f32 {
        1.0
    }

    pub const fn loss_value() -> f32 {
        0.0
    }
}

impl Evaluation {
    /// Returns a number between 0 and 1, where 0 is a loss and 1 is a win.
    /// `turn`: Relative to which player should the value of the evaoluation be?
    pub fn to_value(&self, turn: Color) -> f32 {
        match self {
            Self::Terminal(result) => result.to_value(turn),
            Self::Nope => GameResult::draw_value(),
            Self::Guess(guess) => {
                // The quality is between -1 and 1, so we have to convert it to a 0 to 1 range.
                let quality = (guess.quality + 1.0) / 2.0;
                if guess.relative_to == turn {
                    quality
                } else {
                    1.0 - quality
                }
            }
        }
    }

    pub fn guess(&self) -> Option<&Guess> {
        match self {
            Self::Guess(guess) => Some(guess),
            _ => None,
        }
    }

    pub fn terminal(&self) -> Option<&GameResult> {
        match self {
            Self::Terminal(x) => Some(x),
            _ => None,
        }
    }
}
