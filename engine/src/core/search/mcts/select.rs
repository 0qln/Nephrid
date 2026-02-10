use std::{cmp::max, ops};

use crate::core::search::mcts::node::Branch;

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
    fn score(&self, branch: &Branch, cap_n_i: u32) -> Self::Score;

    fn budget(&self, remaining_budget: usize) -> usize;
}

pub struct PuctSelector {
    // todo: fine tune c. or make a uci option out of it idk
    c: f32,
}

impl PuctSelector {
    pub fn new(c: f32) -> Self {
        Self { c }
    }
}

impl Default for PuctSelector {
    fn default() -> Self {
        Self { c: f32::sqrt(2.0) }
    }
}

impl Selector for PuctSelector {
    type Score = Score;

    fn score(&self, branch: &Branch, cap_n_i: u32) -> Score {
        let n_i = branch.visits() as f32;

        // The quality is updated incrementally as the tree is explored.
        // Because of this, we have to divide by the number of playouts
        // to get the average quality of this node.
        // If this node has not yet been visited, we set the quality to 0.
        let value = branch.node().borrow().value();
        let exploitation = if n_i == 0.0 { 0.0 } else { value / n_i };

        let exploration = self.c * branch.policy() * (cap_n_i as f32).sqrt() / (1f32 + n_i);

        Score(exploitation + exploration)
    }

    fn budget(&self, remaining_budget: usize) -> usize {
        // todo: maybe make this relative to the branch's puct score.
        max(1, (remaining_budget as f32 * 0.3) as usize)
    }
}

pub struct UcbSelector {
    c: f32,
}

impl UcbSelector {
    pub fn new(c: f32) -> Self {
        Self { c }
    }
}

impl Default for UcbSelector {
    fn default() -> Self {
        Self { c: f32::sqrt(2.0) }
    }
}

impl Selector for UcbSelector {
    type Score = Score;

    fn score(&self, branch: &Branch, cap_n_i: u32) -> Score {
        match branch.visits() {
            0 => Score(f32::INFINITY),
            n_i => {
                let w_i = branch.value();
                let n_i = n_i as f32;
                let exploitation = w_i / n_i;
                let exploration = self.c * f32::sqrt((cap_n_i as f32).ln() / n_i);
                Score(exploitation + exploration)
            }
        }
    }

    fn budget(&self, remaining_budget: usize) -> usize {
        max(1, (remaining_budget as f32 * 0.3) as usize)
    }
}

#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Score(pub f32);

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
            .expect("This shouldn't happen for puct scores.")
    }
}
