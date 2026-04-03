use crate::core::{
    Position,
    color::Color,
    depth::Depth,
    search::mcts::{
        nn::POLICY_OUTPUTS,
        node::{
            NodeId, Tree, WinRate,
            node_state::{HasBranches, Terminal, Valid},
        },
        search::{BatchItem, Selection},
    },
};
use core::fmt;
use std::ops::ControlFlow;

#[cfg(test)]
pub mod test;

pub mod nn;
pub mod playout;
pub mod r#static;

pub trait Evaluator {
    type TraceData;

    /// Note a trace of a branching node during the selection phase.
    /// `node` may or may not be the node that was just expanded during the
    /// selection phase.
    fn trace<S: const Valid + HasBranches>(
        &self,
        node: NodeId<S>,
        tree: &Tree,
        pos: &mut Position,
    ) -> Self::TraceData;

    /// Evaluate a node's terminal state. If the node is terminal, return the
    /// evaluation, else return None.
    fn eval_terminal(
        _node: NodeId<Terminal>,
        _tree: &Tree,
        depth: Depth,
        pos: &Position,
    ) -> Evaluation {
        let game_result = pos.search_result(depth);
        let game_result = game_result.expect("Input requires a terminal node.");
        Evaluation::Terminal(game_result)
    }

    fn eval_batch<const X: usize>(
        &mut self,
        tree: &Tree,
        selection: &Selection<X, Self::TraceData>,
        leafs: &[&BatchItem<Self::TraceData>],
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
    pub policy: Policy,
}

impl Guess {
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// Returns the value of the guess relative to `turn`.
    pub fn to_value(&self, relative_to: Color) -> Value {
        let value = Value::from(self.quality);

        if self.relative_to == relative_to {
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
    pub fn to_value(self, color: Color) -> Value {
        match self {
            Self::Win { relative_to } if relative_to == color => Value::win(),
            Self::Win { .. } => Value::loss(),
            Self::Draw => Value::draw(),
        }
    }
}

impl Evaluation {
    /// Returns the value of the evaluation relative to `turn`.
    pub fn to_value(&self, relative_to: Color) -> Value {
        match self {
            Self::Terminal(result) => result.to_value(relative_to),
            Self::Nope => Value::draw(),
            Self::Guess(guess) => guess.to_value(relative_to),
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

pub trait PolicySource {
    fn into<I: IntoIterator<Item = usize>>(&self, move_indices: I) -> Policy;
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

    /// Returns an immutable view of the underlying policy values.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
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

    pub fn inner_mut(&mut self) -> &mut [f32] {
        &mut self.0
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

    pub fn sum(&self) -> f32 {
        self.0.iter().sum::<f32>()
    }

    pub fn new_even(len: usize) -> Self {
        debug_assert!(len > 0, "Policy::new_even called with len == 0");
        if len == 0 {
            // Avoid division by zero; callers should generally not request an
            // even policy over zero elements, but returning an empty policy
            // is safer than producing infinities.
            return Self(Vec::new());
        }

        let len_f = len as f32;
        let probability = 1. / len_f;
        Self(vec![probability; len])
    }

    /// Construct a `Policy` from probability-like values by normalizing them to
    /// sum to 1.
    pub fn from_raw<I>(raw_policy: &RawPolicy, indeces_of_interest: I) -> Option<Self>
    where
        I: Iterator<Item = usize>,
    {
        let mut policy = Vec::<f32>::new();
        for index in indeces_of_interest {
            policy.push(raw_policy.get(index)?);
        }

        let mut result = Self(policy);

        normalize(&mut result.0);

        Some(result)
    }

    /// Construct a `Policy` from logits by applying softmax.
    pub fn from_logits(logits: Vec<f32>) -> Self {
        debug_assert!(!logits.is_empty(), "Should have at least one policy item");

        let mut result = Self(logits);
        softmax(&mut result.0, 10.);
        result
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> {
        self.0.iter().cloned()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.0.iter_mut()
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

pub fn softmax(xs: &mut [f32], temperature: f32) {
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

/// Centi pawns
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Cp {
    v: TCp,
}

pub type TCp = i16;

impl Cp {
    const SCALE: f32 = 350.;
}

impl fmt::Display for Cp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.v)
    }
}

impl From<WinRate> for Cp {
    fn from(win_rate: WinRate) -> Self {
        let w = win_rate.0.clamp(0.001, 0.999);
        let cp = Cp::SCALE * (w / (1.0 - w)).ln();
        Self { v: cp as TCp }
    }
}

impl From<Cp> for Quality {
    fn from(cp: Cp) -> Self {
        let q = (cp.v as f32 / (Cp::SCALE * 2.)).tanh();
        Self::new(q)
    }
}
