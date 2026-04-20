use core::fmt;
use std::{cmp::max, ops};

use crate::core::search::mcts::node::{Branch, NodeData};

pub mod puct;
pub mod ucb;

pub trait Selector {
    // type Score: Ord + ops::Neg<Output = Self::Score> + Into<f32>;

    // note: we take the policy as an argument, because if we later convert this
    // tree structure to a graph, we have to consider different policies from
    // different parents. same reason that we have different struct for node and
    // branch.

    /// # Selector::score
    ///
    /// The score that the selector would assign to a branch.
    ///
    /// ## Params
    ///
    /// branch: The branch to be scored.
    /// cap_n_i: The number of times that the parent node has been visited.
    fn score(&self, node: &NodeData, branch: &Branch, cap_n_i: u32) -> Score;

    fn budget(&self, remaining_budget: usize) -> usize {
        max(1, (remaining_budget as f32 * 0.3) as usize)
    }

    fn min_score(&self) -> Score;
}

#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Score(pub f32);

impl Score {
    pub fn new(_0: f32) -> Self {
        Self(_0)
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl_op!(-|x: Score| -> Score { Score(-x.0) });

impl Eq for Score {}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("This shouldn't happen for scores.")
    }
}

impl From<Score> for f32 {
    fn from(val: Score) -> Self {
        val.0
    }
}
