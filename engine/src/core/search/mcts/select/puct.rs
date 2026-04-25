use crate::core::search::mcts::{
    eval,
    node::{BranchId, Tree},
    select::Score,
};

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

impl super::Selector for PuctSelector {
    fn score(&self, tree: &Tree, branch_id: BranchId, cap_n_i: u32) -> Score {
        let branch = tree.branch(branch_id);
        let node = tree.node(branch.node());

        let n_i = node.visits() as f32;

        // The quality is updated incrementally as the tree is explored.
        // Because of this, we have to divide by the number of playouts
        // to get the average quality of this node.
        // If this node has not yet been visited, we set the quality to 0 and rely
        // completely on exploration factor.
        let value = node.value();
        let exploitation = if n_i == 0. {
            eval::Value::draw().v()
        }
        else {
            value / n_i
        };

        let exploration = self.c * branch.policy() * (cap_n_i as f32).sqrt() / (1. + n_i);

        Score::new(exploitation + exploration)
    }

    fn min_score(&self) -> Score {
        Score::new(f32::NEG_INFINITY)
    }
}
