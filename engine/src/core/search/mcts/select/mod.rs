use core::fmt;
use std::ops;

use crate::core::search::mcts::node::{BranchId, NodeId, Tree, node_state::Evaluated};

pub mod puct;
pub mod ucb;

pub trait Selector {
    // note: we take the policy as an argument, because if we later convert this
    // tree structure to a graph, we have to consider different policies from
    // different parents. same reason that we have different struct for node and
    // branch.

    fn exploitation(&self, tree: &Tree, branch_id: BranchId, parent_id: NodeId<Evaluated>)
    -> Score;
    fn exploration(&self, tree: &Tree, branch_id: BranchId, parent_id: NodeId<Evaluated>) -> Score;

    fn virtual_loss(&self) -> u32 {
        1
    }
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
impl_op!(+|x: Score, y: Score| -> Score { Score(x.0 + y.0) });
impl_op!(*|x: Score, y: f32| -> Score { Score(x.0 * y) });

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
