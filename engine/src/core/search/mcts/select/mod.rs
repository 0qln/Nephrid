use std::{cmp::max, ops};

use crate::core::search::mcts::node::{Branch, NodeData};

pub mod puct;
pub mod ucb;

pub trait Selector {
    type Score: Ord + ops::Neg<Output = Self::Score>;

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
    fn score(&self, node: &NodeData, branch: &Branch, cap_n_i: u32) -> Self::Score;

    fn budget(&self, remaining_budget: usize, ) -> usize {
        // todo: maybe make this relative to the branch's puct score.
        max(1, (remaining_budget as f32 * 0.6) as usize)
    }

    fn min_score(&self) -> Self::Score;
}
