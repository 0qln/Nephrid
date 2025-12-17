use super::utils::*;
use crate::core::Position;
use crate::core::color::Color;
use crate::core::coordinates::files;
use crate::core::coordinates::ranks;
use crate::core::position::CheckState;
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
use std::rc::Rc;

pub struct EvalItem {
    node: Rc<RefCell<EvalInfoNode>>,
}

pub trait Evaluator<const X: usize> {
    // Prepare an eval_info_node with the required info for this evaluator.
    fn prepare_node(
        &mut self,
        index: usize,
        eval_node: Rc<RefCell<EvalInfoNode>>,
        node: Rc<RefCell<Node>>,
        pos: &Position,
    ) -> ();

    /// Evluate all the nodes in the batch.
    /// (Which is, all the nodes that are eval `None`)
    fn eval_guesses(&mut self) -> ();

    /// Evaluate a node's terminal state. If the node is terminal, return the evaluation, else
    /// return None.
    fn eval_terminal(node: &Node, pos: &Position) -> Option<Evaluation>;

    /// Set the evaluation at a specific index.
    fn set_eval(&mut self, index: usize, eval: Evaluation) -> ();

    /// Clear the evaluation at a specific index.
    /// (Mark the node at `index` to be evaluated by `eval_guesses`.)
    fn clear_eval(&mut self, index: usize);

    /// Get the evaluation at a specific index.
    fn get_eval(&self, index: usize) -> Option<&Evaluation>;
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

#[derive(Clone)]
pub enum EvalState {
    OnBatch(Rc<RefCell<EvalInfoNode>>),
    Evaluated(Evaluation),
    None,
}

pub type EvalInfoNode = DoubleLinkedNode<Option<EvalInfo>>;

/// X: batch size
pub struct NNEvaluator<B: Backend, const X: usize> {
    /// Eval info root
    // eval_info_root: EvalInfoNode,

    // todo: i think if we don't hold onto the root, the parent's will be dropped since the
    // children just have weak references to them...

    /// Eval infos
    eval_infos: [EvalState; X],

    /// NN Model
    model: Rc<Model<B>>,

    // Device on which the nn will run.
    device: Rc<B::Device>,
}

impl<B: Backend, const X: usize> NNEvaluator<B, X> {
    pub fn new(model: Rc<Model<B>>, device: Rc<B::Device>) -> Self {
        Self {
            model,
            device,
            eval_infos: [const { EvalState::None }; X],
        }
    }

    fn device(&self) -> &B::Device {
        &self.device
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
                        [X, padding_len, ranks::N_VARIANTS, files::N_VARIANTS],
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
    fn get_node_history(mut eval_info: Rc<RefCell<EvalInfoNode>>) -> Vec<BoardInputFloats> {
        let mut vec: Vec<BoardInputFloats> = vec![];

        let board_input = eval_info
            .borrow()
            .data()
            .as_ref()
            .expect("Eval info is missing")
            .inputs
            .board;
        vec.insert(0, board_input);

        while let Some(parent) = eval_info.clone().borrow().parent() {
            eval_info = parent.upgrade().expect("can't get a Rc from a Weak");

            let board_input = eval_info
                .borrow()
                .data()
                .as_ref()
                .expect("Eval info is missing")
                .inputs
                .board;
            vec.insert(0, board_input);
        }

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

                    let state_tensor = Tensor::from_floats([state_input], self.device());

                    state_tensor
                })
                .collect_vec(),
            0,
        )
    }

    fn iter_batch(&self) -> impl Iterator<Item = Rc<RefCell<EvalInfoNode>>> {
        self.eval_infos.iter().filter_map(|x| {
            if let EvalState::OnBatch(eval_info) = x {
                Some(eval_info.clone())
            } else {
                None
            }
        })
    }
}

impl<B: Backend, const X: usize> Evaluator<X> for NNEvaluator<B, X> {
    /// Push the eval info as to be guessed.
    fn prepare_node(
        &mut self,
        index: usize,
        eval_node: Rc<RefCell<EvalInfoNode>>,
        node: Rc<RefCell<Node>>,
        pos: &Position,
    ) -> () {
        let board = board_input(pos);
        let state = state_input(pos);
        let input = InputFloats { board, state };
        let data = EvalInfo {
            inputs: input,
            node,
            turn: pos.get_turn(),
        };
        eval_node.borrow_mut().set_data(Some(data));

        if let Some(x) = self.eval_infos.get_mut(index) {
            *x = EvalState::OnBatch(eval_node);
        } else {
            panic!("Out of range");
        }
    }

    fn get_eval(&self, index: usize) -> Option<&Evaluation> {
        if let Some(x) = self.eval_infos.get(index) {
            if let EvalState::Evaluated(eval) = x {
                return Some(eval);
            }
        }
        None
    }

    fn set_eval(&mut self, index: usize, eval: Evaluation) {
        if let Some(x) = self.eval_infos.get_mut(index) {
            *x = EvalState::Evaluated(eval);
        }
    }

    fn clear_eval(&mut self, index: usize) {
        if let Some(x) = self.eval_infos.get_mut(index) {
            *x = EvalState::None;
        }
    }

    fn eval_terminal(node: &Node, pos: &Position) -> Option<Evaluation> {
        // First check if the position is a normal game ending.
        if node.has_branches() {
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

    fn eval_guesses(&mut self) {
        let board_batch = self.build_board_batch();
        let state_batch = self.build_state_batch();
        let (values, raw_policies) = self.model.forward(board_batch, state_batch);

        assert_eq!(values.shape(), Shape::new([X, VALUE_OUTPUTS]));
        assert_eq!(raw_policies.shape(), Shape::new([X, POLICY_OUTPUTS]));

        let values = values.into_data();
        let values = values
            .as_slice::<f32>()
            .expect("Qualities could not be converted to vec.");

        let raw_policies = raw_policies.into_data();
        let raw_policies = raw_policies
            .as_slice::<f32>()
            .expect("Policy could not be converted to vec.");

        // Enumerate all eval_infos, which have not yet been assigned and assign them a their guess.
        for (index, eval_info, value, raw_policy) in self
            .eval_infos
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
            .zip(values.chunks(VALUE_OUTPUTS))
            .zip(raw_policies.chunks(POLICY_OUTPUTS))
            .map(|(((a, b), c), d)| (a, b, c, d))
        {
            let eval_info = eval_info.borrow();
            let eval_info = eval_info
                .data()
                .as_ref()
                .expect("This should be a leaf and leafes should have data.");

            let mut policies = Vec::<f32>::new();
            for branch in eval_info.node.borrow().iter_branches() {
                policies.push(raw_policy[usize::from(branch.mov())]);
            }

            // Renormalize the policy to a sum of 1, since not all of the probabilities
            // were assigned to moves that are actually playable in this position:

            let policy_sum = {
                let sum = policies.iter().sum();
                if sum == 0.0 {
                    // Fallback to uniform distribution
                    policies.len() as f32
                } else {
                    sum
                }
            };
            for policy in &mut policies {
                *policy /= policy_sum;
            }

            // Evaluator should return a probability distribution.
            let f32_eq = |a: f32, b: f32, e: f32| f32::abs(a - b) < e;
            debug_assert!(f32_eq(policies.iter().sum::<f32>(), 1., 0.001));

            let eval = Evaluation::Guess(Guess {
                relative_to: eval_info.turn,
                quality: value[0],
                policies,
            });
            self.eval_infos[index] = EvalState::Evaluated(eval);
        }
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
    pub policies: Vec<f32>,
}

impl Guess {
    pub fn policies(&self) -> &[f32] {
        &self.policies
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Evaluation {
    /// we will go further and have a guess about this game.
    Guess(Guess),
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
            Self::Guess(Guess {
                quality,
                relative_to,
                policies: _policies,
            }) => {
                // The quality is between -1 and 1, so we have to convert it to a 0 to 1 range.
                let quality = (quality + 1.0) / 2.0;
                if *relative_to == turn { quality } else { 1.0 - quality }
            }
        }
    }
}
