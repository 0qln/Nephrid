use crate::{
    core::{
        Position,
        color::Color,
        depth::Depth,
        r#move::Move,
        search::mcts::{
            nn::POLICY_OUTPUTS,
            node::{
                NodeId, Tree, WinRate,
                node_state::{HasBranches, Terminal, Valid},
            },
            search::{BatchItem, Selection},
        },
    },
    misc::{CheckHealth, CheckHealthResult},
};
use core::fmt;
use std::ops::ControlFlow;

#[cfg(test)]
pub mod test;

pub mod hce;
pub mod nn;
pub mod playout;

/// Evaluate a node's terminal state. If the node is terminal, return the
/// evaluation, else return None.
pub fn eval_terminal(
    _node: NodeId<Terminal>,
    _tree: &Tree,
    depth: Depth,
    pos: &Position,
) -> GameResult {
    let game_result = pos.search_result(depth);
    game_result.expect("Input is a terminal node and thus there has to be a search result.")
}

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

    fn eval_batch(
        &mut self,
        tree: &Tree,
        selection: &Selection<Self::TraceData>,
        leafs: &[&BatchItem<Self::TraceData>],
    ) -> impl Iterator<Item = Guess>;
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
    Guess(Guess),
    /// we cannot go any further.
    Terminal(GameResult),
}

impl fmt::Display for Evaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // todo: this is the same as debug impl?
        match self {
            Self::Guess(x) => write!(f, "Evaluation::Guess({x:?})"),
            Self::Terminal(x) => write!(f, "Evaluation::Terminal({x:?})"),
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

/// Probability distribution over possible moves, as output by the network
/// before filtering the moves to the legal ones.
#[derive(Clone)]
pub struct RawPolicy([f32; POLICY_OUTPUTS]);

impl RawPolicy {
    const EPS: f32 = 1e-4;

    pub fn new(p: [f32; POLICY_OUTPUTS]) -> Self {
        debug_assert!(
            p.iter().all(|&x| x >= -Self::EPS),
            "policy probabilities must be non-negative"
        );
        debug_assert!(
            (p.iter().sum::<f32>() - 1.0).abs() < Self::EPS,
            "policy probabilities must sum to 1",
        );

        Self(p)
    }

    pub fn into_inner(self) -> [f32; POLICY_OUTPUTS] {
        self.0
    }
}

impl CheckHealth for RawPolicy {
    type Error = String;
    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        if self.0.iter().any(|&x| x.is_nan()) {
            return Err("policy probabilities must not be NaN".to_string());
        }

        if self.0.iter().any(|&x| x.is_infinite()) {
            return Err("policy probabilities must not be infinite".to_string());
        }

        if !self.0.iter().all(|&x| x >= -Self::EPS) {
            return Err("policy probabilities must be non-negative".to_string());
        }

        let sum = self.0.iter().sum::<f32>();
        if (sum - 1.0).abs() >= Self::EPS {
            return Err(format!(
                "policy probabilities must sum to 1, but was: {sum}",
            ));
        }
        Ok(())
    }
}

impl std::fmt::Debug for RawPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RawPolicy").field(&"...").finish()
    }
}

// todo: this better belongs in `nn/mod.rs`, right?
/// Raw logits outputs of the network.
#[derive(PartialEq, Clone)]
pub struct RawLogits(pub [f32; POLICY_OUTPUTS]);

impl std::fmt::Debug for RawLogits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RawLogits").field(&"...").finish()
    }
}

impl Default for RawLogits {
    fn default() -> Self {
        Self::null()
    }
}

impl RawLogits {
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

/// Visit counts for each legal move in a position.
#[derive(Debug, PartialEq, Clone)]
pub struct VisitCounts(pub Vec<(Move, u32)>);

/// Some kind of logits for each legal move in a position.
#[derive(Debug, PartialEq, Clone)]
pub struct Logits(pub Vec<f32>);

/// A probability distribution over moves.
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

    /// Construct a `Policy` from raw logits by normalizing them.
    pub fn from_raw_logits<I>(
        raw_logits: &RawLogits,
        indeces_of_interest: I,
        temp: f32,
    ) -> Option<Self>
    where
        I: Iterator<Item = usize>,
    {
        let mut policy = Vec::<f32>::new();
        for index in indeces_of_interest {
            policy.push(raw_logits.get(index)?);
        }

        softmax(&mut policy, temp);

        Some(Self(policy))
    }

    /// Construct a `Policy` from a raw policy.
    pub fn from_raw_policy<I>(raw_policy: &RawPolicy, indeces_of_interest: I) -> Option<Self>
    where
        I: Iterator<Item = usize>,
    {
        let mut policy = Vec::<f32>::new();
        for index in indeces_of_interest {
            policy.push(*raw_policy.0.get(index)?);
        }

        Some(Self(policy))
    }

    /// Construct a `Policy` from logits by applying softmax.
    pub fn from_logits(logits: Logits, temp: f32) -> Self {
        let mut values = logits.0;
        softmax(&mut values, temp);
        Self(values)
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> {
        self.0.iter().cloned()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.0.iter_mut()
    }

    pub fn as_slice(&self) -> &[f32] {
        self.0.as_slice()
    }
}

/// Applies the softmax function to `xs` in-place, yielding a policy.
pub fn softmax(xs: &mut [f32], temperature: f32) {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| ((x - max) / temperature).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for (x, e) in xs.iter_mut().zip(exps) {
        *x = e / sum;
    }
}

/// Construct a policy from visit counts by applying the AlphaZero
/// temperature formula.
pub fn normalize_visits<const N: usize>(xs: &[f32; N], temp: f32) -> [f32; N] {
    debug_assert!(!xs.is_empty(), "Should have at least one visit count");

    let mut probabilities = [0.0; N];

    // Temperature of 0 means pure greedy play (competitive play)
    if temp == 0.0 {
        let mut max_visits = 0.;
        let mut best_idx = 0;
        for (i, &v) in xs.iter().enumerate() {
            if v > max_visits {
                max_visits = v;
                best_idx = i;
            }
        }
        probabilities[best_idx] = 1.0;
        return probabilities;
    }

    // AlphaZero visit distribution: P_i = N_i^(1/T) / sum(N_j^(1/T))
    let inv_temp = 1.0 / temp;
    let mut sum = 0.0;

    for (i, &v) in xs.iter().enumerate() {
        if v > 0. {
            let powered = v.powf(inv_temp);
            probabilities[i] = powered;
            sum += powered;
        }
    }

    // Normalize so they sum to 1.0
    if sum > 0.0 {
        for p in probabilities.iter_mut() {
            *p /= sum;
        }
    }
    else {
        // Fallback if all visits were 0 (shouldn't happen in a valid MCTS)
        let uniform = 1.0 / xs.len() as f32;
        for p in probabilities.iter_mut() {
            *p = uniform;
        }
    }

    probabilities
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

impl CheckHealth for Quality {
    type Error = String;
    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        if self.0.is_nan() {
            return Err("Quality value was NaN".to_string());
        }

        if self.0.is_infinite() {
            return Err("Quality value was infinite".to_string());
        }

        if self.0 < Self::min().v() || self.0 > Self::max().v() {
            return Err(format!(
                "Quality value {} was out of range [{}, {}]",
                self.0,
                Self::min().v(),
                Self::max().v()
            ));
        }

        Ok(())
    }
}

impl From<Value> for Quality {
    fn from(v: Value) -> Self {
        Self::new((v.0 - 0.5) * 2.)
    }
}

impl fmt::Display for Quality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
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

    pub fn v(&self) -> TCp {
        self.v
    }
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
