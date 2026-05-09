use crate::core::search::mcts::{
    node::{BranchId, NodeId, Tree, node_state::Evaluated},
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
    fn score(&self, tree: &Tree, branch_id: BranchId, parent_id: NodeId<Evaluated>) -> Score {
        let branch = tree.branch(branch_id);
        let node = tree.node(branch.node());
        let parent = tree.node(parent_id);

        let cap_n_i = parent.visits().0 as f32;
        let n_i = node.visits().0 as f32;

        let value = node.value();
        let exploitation = if n_i == 0. {
            // fallback to parent q-value for unvisited nodes. note that the parent node
            // cannot be 0 because we only expand a node after visiting it at
            // least once.
            parent.value() / (cap_n_i)
        }
        else {
            value / (n_i)
        };

        let policy = branch.policy().v();
        let exploration = self.c * (policy) * cap_n_i.sqrt() / (1_f32 + (n_i));

        Score::new(exploitation + exploration)
    }

    fn min_score(&self) -> Score {
        Score::new(f32::NEG_INFINITY)
    }
}
