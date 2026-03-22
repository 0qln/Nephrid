use core::fmt;

use crate::core::search::mcts::eval;

use super::*;

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
        // If this node has not yet been visited, we set the quality to 0 and rely
        // completely on exploration factor.
        let value = branch.node().borrow().value();
        let exploitation = if n_i == 0. {
            eval::Value::draw().v()
        }
        else {
            value / n_i
        };

        let exploration = self.c * branch.policy() * (cap_n_i as f32).sqrt() / (1. + n_i);

        Score(exploitation + exploration)
    }

    fn min_score(&self) -> Self::Score {
        Score(f32::NEG_INFINITY)
    }
}

#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Score(pub f32);

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
            .expect("This shouldn't happen for puct scores.")
    }
}
