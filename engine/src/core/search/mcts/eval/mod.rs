use crate::core::{
    Position,
    color::Color,
    search::mcts::{
        nn::{
            BOARD_INPUT_HISTORY, BoardInputFloats, Model, POLICY_OUTPUTS, StateInputFloats,
            VALUE_OUTPUTS, board_history_input, board_input, state_input,
        },
        node::{Node, NodeRef, NodeState},
        search::SelectionNodeRef,
    },
    turn::Turn,
};
use burn::{
    Tensor,
    tensor::{Shape, backend::Backend},
};
use core::fmt;
use itertools::Itertools;
use std::{assert_matches::assert_matches, cell::RefCell, ops::ControlFlow, rc::Rc};

#[cfg(test)]
pub mod test;

pub mod nn;
pub mod playout;
pub mod r#static;

pub trait Evaluator {
    type TraceData;

    fn trace(&self, node: NodeRef, pos: &Position) -> Self::TraceData;

    /// Evaluate a node's terminal state. If the node is terminal, return the
    /// evaluation, else return None.
    fn eval_terminal(node: &Node, pos: &Position) -> Option<Evaluation> {
        assert_matches!(
            node.state(),
            NodeState::Terminal | NodeState::Expanded,
            "We need a terminal or expanded node for proper evaluation."
        );

        let has_moves = node.has_branches();
        let game_result = pos.game_result_with(has_moves);
        game_result.map(Evaluation::Terminal)
    }

    fn eval_batch(
        &mut self,
        batch: &[SelectionNodeRef<Self::TraceData>],
    ) -> impl Iterator<Item = Evaluation>;
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GameResult {
    Win { relative_to: Color },
    Draw,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Guess {
    pub relative_to: Color,
    // value in the range from -1 to 1
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

impl fmt::Display for Evaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Guess(_) => write!(f, "Evaluation::Guess(...)"),
            Self::Terminal(result) => write!(f, "Evaluation::Terminal({result:?})"),
            Self::Nope => write!(f, "Evaluation::Nope"),
        }
    }
}

impl GameResult {
    /// Returns a number between 0 and 1, where 0 is a loss and 1 is a win.
    const fn to_value(self, turn: Color) -> f32 {
        match self {
            Self::Win { relative_to } => {
                if relative_to.v() == turn.v() {
                    Self::win_value()
                }
                else {
                    Self::loss_value()
                }
            }
            Self::Draw => Self::draw_value(),
        }
    }

    // these are functions, because maybe later we want to have different values for
    // e.g. a win that is close to the root node or further.

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
                }
                else {
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

#[derive(Debug, PartialEq, Clone)]
pub struct RawPolicy([f32; POLICY_OUTPUTS]);

impl Default for RawPolicy {
    fn default() -> Self {
        Self::null()
    }
}

impl RawPolicy {
    pub fn get(&self, i: usize) -> Option<f32> {
        self.0.get(i).cloned()
    }

    pub fn set(&mut self, i: usize, val: f32) {
        self.0[i] = val;
    }

    pub fn null() -> Self {
        Self::new([0.0; POLICY_OUTPUTS])
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

    pub fn normalize(&mut self) {
        let policy_sum = {
            let sum = self.iter().sum();
            if sum == 0.0 {
                // Fallback to uniform distribution
                self.len() as f32
            }
            else {
                sum
            }
        };
        for policy in &mut self.0 {
            *policy /= policy_sum;
        }

        // Evaluator should return a probability distribution.
        let f32_eq = |a: f32, b: f32, e: f32| f32::abs(a - b) < e;
        debug_assert!(f32_eq(self.iter().sum::<f32>(), 1., 0.001));
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

    pub fn iter(&self) -> impl Iterator<Item = f32> {
        self.0.iter().cloned()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.0.iter_mut()
    }

    pub fn normalize(&mut self) {
        let policy_sum = {
            let sum = self.iter().sum();
            if sum == 0.0 {
                // Fallback to uniform distribution
                self.len() as f32
            }
            else {
                sum
            }
        };
        for policy in &mut self.0 {
            *policy /= policy_sum;
        }

        // Evaluator should return a probability distribution.
        let f32_eq = |a: f32, b: f32, e: f32| f32::abs(a - b) < e;
        debug_assert!(f32_eq(self.iter().sum::<f32>(), 1., 0.001));
    }

    pub fn new(policy: Vec<f32>) -> Self {
        debug_assert!(!policy.is_empty(), "Should have atleast one policy item");
        let mut result = Self(policy);
        result.normalize();
        result
    }
}
