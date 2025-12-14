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
use crate::core::search::mcts::nn::board_input;
use crate::core::search::mcts::nn::state_input;
use crate::core::search::mcts::node::Node;
use burn::Tensor;
use burn::tensor::backend::Backend;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::rc::Weak;

pub struct EvalItem {
    node: Rc<RefCell<EvalInfoNode>>,
}

pub trait Evaluator<const X: usize> {
    /// prepare the newest position that needs to be evaluated.
    // fn push(&mut self, pos: &Position, x: usize) -> ();

    /// pop the latest position that was prepared.
    // fn pop(&mut self, x: usize) -> ();

    /// clear all prepaered
    // fn clear(&mut self) -> ();

    /// Returns: (quality [-1;1], policy [over ALL moves])
    // fn evaluate(&self) -> [(f32, [f32; POLICY_OUTPUTS]); X];

    fn push(&mut self, parent: Rc<RefCell<EvalInfoNode>>, pos: &Position) -> ();

    fn push_item(&mut self, item: Rc<RefCell<EvalInfoNode>>) -> ();

    fn eval_guess(&self) -> Vec<Evaluation>;

    fn eval_terminal(node: &Node, pos: &Position) -> Option<Evaluation>;
}

pub struct InputFloats {
    board: BoardInputFloats,
    state: StateInputFloats,
}

pub struct EvalInfo {
    /// input floats for the eval model
    inputs: InputFloats,

    /// The node that this eval info is for.
    node: Weak<RefCell<Node>>,
}

pub type EvalInfoNode = DoubleLinkedNode<Option<EvalInfo>>;

// type BoardHistoryBuffer = StaticRb<BoardInputFloats, BOARD_INPUT_HISTORY>;

/// X: batch size
pub struct NNEvaluator<B: Backend, const X: usize> {
    /// Eval info root
    eval_info_root: EvalInfoNode,

    /// Eval infos
    eval_infos: Vec<Rc<RefCell<EvalInfoNode>>>,

    /// NN Model
    model: Rc<Model<B>>,

    /// shut up the compiler, i will probably use X later on...
    _x: [PhantomData<()>; X],
}

impl<B: Backend, const X: usize> NNEvaluator<B, X> {
    pub fn new(model: Rc<Model<B>>) -> Self {
        Self { model, ..Default::default() }
    }

    fn device(&self) -> B::Device {
        self.model.device
    }

    fn build_board_batch(&self) -> Tensor<B, 4> {
        // concatenate the board inputs along the batch dimension.
        let board_batch = Tensor::cat(
            self.eval_nodes
                .iter()
                .cloned()
                .map(|eval_node| Self::get_node_history(eval_node))
                // concatenate the board inputs along the channel dimension.
                .map(|history| {
                    // convert input floats to tensors
                    let history_tensor = Tensor::cat(
                        history
                            .into_iter()
                            .map(|b| Tensor::from_floats([b], self.device()))
                            .collect_vec(),
                        1,
                    );

                    // pad missing history info with zeroes.
                    let padding_len = BOARD_INPUT_HISTORY - history.len();
                    let padding_tensor = BoardInputTensor::<B>::zeros(
                        [
                            X,
                            BOARD_INPUT_CHANNELS,
                            ranks::N_VARIANTS,
                            files::N_VARIANTS,
                        ],
                        self.device(),
                    );

                    // concat padding with history
                    Tensor::cat(vec![padding_tensor, history], 1)
                })
                .collect_vec(),
            0,
        );
    }

    /// return the history where:
    /// the oldest board state is the first index
    /// the youngest board state is the last index
    fn get_node_history(mut eval_info: Rc<RefCell<EvalInfoNode>>) -> Vec<BoardInputFloats> {
        let mut vec: Vec<BoardInputFloats> = vec![];

        let board_input = eval_info
            .borrow()
            .data()
            .expect("Eval info is missing")
            .board;
        vec.prepend(board_input);

        while let Some(parent) = eval_info.borrow().parent() {
            eval_info = parent.upgrade().expect("can't get a Rc from a Weak");

            let board_input = eval_info
                .borrow()
                .data()
                .expect("Eval info is missing")
                .board;
            vec.prepend(board_input);
        }

        vec
    }

    fn build_state_batch(&self) -> Tensor<B, 2> {
        // concatenate the state inputs along the batch dimension.
        let state_batch = Tensor::cat(
            self.eval_infos
                .iter()
                .cloned()
                .map(|eval_info| {
                    let state_input = eval_info
                        .borrow()
                        .data()
                        .expect("Eval info is missing")
                        .state;
                    [state_input]
                })
                .collect_vec(),
            0,
        );
    }
}

impl<B: Backend, const X: usize> Evaluator<X> for NNEvaluator<B, X> {
    fn push(&mut self, parent: Rc<RefCell<EvalInfoNode>>, pos: &Position) -> () {
        debug_assert_eq!(parent.borrow().data(), None);

        let board = board_input(pos);
        let state = state_input(pos);
        let input = InputFloats { board, state };
        parent.borrow_mut().set_data(input);
    }

    fn push_item(&mut self, item: Rc<RefCell<EvalInfoNode>>) -> () {
        self.eval_infos.push(item);
    }

    // fn pop(&mut self, x: usize) {
    //     assert!(
    //         self.board_history[x].len() > BOARD_INPUT_HISTORY,
    //         "Cannot pop the padding."
    //     );
    //     self.board_history[x].pop();
    // }

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

    // todo: the below should be used.
    // fn eval_guess(&self) -> [Evaluation; X] {
    //
    fn eval_guess(&self) -> Vec<Evaluation> {
        let board_batch = self.build_board_batch();
        let state_batch = self.build_state_batch();
        let (qualities, raw_policies) = self.model.forward(board_batch, state_batch);

        let qualities = qualities
            .to_data()
            .to_vec::<f32>()
            .expect("Qualities could not be converted to vec.");

        let raw_policies = TryInto::<Vec<[f32; POLICY_OUTPUTS]>>::try_into(
            raw_policies
                .to_data()
                .to_vec::<[f32; POLICY_OUTPUTS]>()
                .expect("Policy could not be converted to vec.")
                .into_boxed_slice(),
        );

        let evals = self
            .eval_infos
            .iter()
            .map(|x| x.node.clone())
            .zip(qualities)
            .zip(raw_policies)
            .map(|(node, quality, raw_policy)| {
                let mut policies = Vec::<f32>::new();
                for branch in &node.branches {
                    policies.push(raw_policy[usize::from(branch.mov)]);
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

                Evaluation::Guess(Guess {
                    relative_to: node.turn(),
                    quality,
                    policies,
                })
            });

        evals.collect_vec();
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GameResult {
    Win { relative_to: Color },
    Draw,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Guess {
    relative_to: Color,
    quality: f32,
    policies: Vec<f32>,
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
    fn to_value(&self, turn: Color) -> f32 {
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
