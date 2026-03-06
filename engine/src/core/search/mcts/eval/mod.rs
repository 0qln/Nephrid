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
    pub quality: Quality,
    pub policy: RawPolicy,
}

impl Guess {
    pub fn policy(&self) -> &RawPolicy {
        &self.policy
    }

    /// Returns the value of the guess relative to `turn`.
    pub fn to_value(&self, turn: Color) -> Value {
        let value = Value::from(self.quality);

        if self.relative_to == turn {
            value
        }
        else {
            value.inverse()
        }
    }

    pub fn relative_to(&self) -> Color {
        self.relative_to
    }

    pub fn quality(&self) -> Quality {
        self.quality
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
        // todo: this is the same as debug impl?
        match self {
            Self::Guess(x) => write!(f, "Evaluation::Guess({x:?})"),
            Self::Terminal(x) => write!(f, "Evaluation::Terminal({x:?})"),
            Self::Nope => write!(f, "Evaluation::Nope"),
        }
    }
}

impl GameResult {
    /// Returns the value of the game result relative to `turn`.
    fn to_value(self, turn: Color) -> Value {
        match self {
            Self::Win { relative_to } if relative_to == turn => Value::win(),
            Self::Win { .. } => Value::loss(),
            Self::Draw => Value::draw(),
        }
    }
}

impl Evaluation {
    /// Returns the value of the evaluation relative to `turn`.
    pub fn to_value(&self, turn: Color) -> Value {
        match self {
            Self::Terminal(result) => result.to_value(turn),
            Self::Nope => Value::draw(),
            Self::Guess(guess) => guess.to_value(turn),
        }
    }

    /// Get the guess, if this is a guess.
    pub fn guess(&self) -> Option<&Guess> {
        match self {
            Self::Guess(x) => Some(x),
            _ => None,
        }
    }

    /// Get the terminal eval, if this is a GameResult.
    pub fn terminal(&self) -> Option<&GameResult> {
        match self {
            Self::Terminal(x) => Some(x),
            _ => None,
        }
    }
}

#[derive(PartialEq, Clone)]
pub struct RawPolicy([f32; POLICY_OUTPUTS]);

impl std::fmt::Debug for RawPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RawPolicy").field(&"...").finish()
    }
}

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

    pub fn new(policy: Vec<f32>) -> Self {
        debug_assert!(!policy.is_empty(), "Should have atleast one policy item");
        let mut result = Self(policy);
        normalize(&mut result.0);
        result
    }
}

pub fn normalize(xs: &mut [f32]) {
    let f32_eq = |a: f32, b: f32, e: f32| f32::abs(a - b) < e;

    // make sure all values are positive.
    if let Some(min) = xs.iter().min_by(|a, b| f32::total_cmp(a, b)).cloned()
        && min < 0.
    {
        for policy in xs.iter_mut() {
            *policy += -min;
        }
    }

    let sum = xs.iter().sum();
    if f32_eq(sum, 0., 0.001) {
        // Fallback to uniform distribution
        let uniform = 1.0 / xs.len() as f32;
        for policy in xs.iter_mut() {
            *policy = uniform;
        }
    }
    else {
        for policy in xs.iter_mut() {
            *policy /= sum;
        }
    }

    {
        let sum = xs.iter().sum::<f32>();
        debug_assert!(
            f32_eq(sum, 1., 0.001),
            "Evaluator should return a probability distribution. Sum was expected to be 1, but \
             was {sum}"
        );
    }
}

fn softmax(xs: &mut [f32], temperature: f32) {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| ((x - max) / temperature).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for (x, e) in xs.iter_mut().zip(exps) {
        *x = e / sum;
    }
}

/// A value in range [-1; 1]
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Quality(f32);

impl Quality {
    pub fn new(v: f32) -> Self {
        debug_assert!(v >= Self::min().v() && v <= Self::max().v());
        Self(v)
    }

    /// Squishes v in range [-1; 1], whatever it's value is.
    fn squish(v: f32) -> Self {
        Self::new(v.tanh())
    }

    /// Inverses the value in it's range
    pub fn inverse(&self) -> Self {
        Self(-self.0)
    }

    pub fn v(&self) -> f32 {
        self.0
    }

    pub const fn max() -> Self {
        Self(1.)
    }

    pub const fn min() -> Self {
        Self(-1.)
    }

    // these are functions, because maybe later we want to have different values for
    // e.g. a win that is close to the root node or further.

    pub const fn draw() -> Self {
        Self(0.0)
    }

    pub const fn win() -> Self {
        Self::max()
    }

    pub const fn loss() -> Self {
        Self::min()
    }
}

/// tmp
// pub type Value = Quality;

impl From<Value> for Quality {
    fn from(v: Value) -> Self {
        Self::new((v.0 - 0.5) * 2.)
    }
}

/// A value in range [0; 1]
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Value(f32);

impl Value {
    pub fn new(v: f32) -> Self {
        debug_assert!(v >= Self::min().v() && v <= Self::max().v());
        Self(v)
    }

    /// Inverses the value in it's range
    pub fn inverse(&self) -> Value {
        Self(1. - self.0)
    }

    pub fn v(&self) -> f32 {
        self.0
    }

    pub const fn max() -> Self {
        Self(1.)
    }

    pub const fn min() -> Self {
        Self(0.)
    }

    // these are functions, because maybe later we want to have different values for
    // // e.g. a win that is close to the root node or further.

    pub const fn draw() -> Self {
        Self(0.5)
    }

    pub const fn win() -> Self {
        Self::max()
    }

    pub const fn loss() -> Self {
        Self::min()
    }
}

impl From<Quality> for Value {
    fn from(q: Quality) -> Self {
        Self::new((q.0 + 1.) / 2.)
    }
}
