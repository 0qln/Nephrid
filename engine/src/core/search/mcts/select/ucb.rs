use crate::core::search::mcts::node::{BranchId, Tree};

use super::*;

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
    fn score(&self, tree: &Tree, branch_id: BranchId, cap_n_i: u32) -> Score {
        let branch = tree.branch(branch_id);
        let node = tree.node(branch.node());

        match node.visits() {
            0 => Score(f32::INFINITY),
            n_i => {
                let w_i = node.value();
                let n_i = n_i as f32;
                let exploitation = w_i / n_i;
                let exploration = self.c * f32::sqrt((cap_n_i as f32).ln() / n_i);
                Score(exploitation + exploration)
            }
        }
    }

    fn min_score(&self) -> Score {
        Score::new(f32::NEG_INFINITY)
    }
}
