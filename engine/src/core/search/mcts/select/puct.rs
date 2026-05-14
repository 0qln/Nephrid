use crate::core::search::mcts::{
    node::{BranchId, NodeId, Tree, node_state::Evaluated},
    select::{Score, Selector},
};

pub trait PuctParams {
    fn select_cpuct(&self) -> f32;
}

pub struct PuctSelector {
    c: f32,
}

impl PuctSelector {
    pub fn new(c: f32) -> Self {
        Self { c }
    }

    #[inline]
    pub fn score(&self, tree: &Tree, branch_id: BranchId, parent_id: NodeId<Evaluated>) -> Score {
        let exploitation = self.exploitation(tree, branch_id, parent_id);
        let exploration = self.exploration(tree, branch_id, parent_id);
        exploitation + exploration
    }

    pub const fn c(&self) -> f32 {
        self.c
    }
}

impl Default for PuctSelector {
    fn default() -> Self {
        Self { c: 1.12 }
    }
}

impl super::Selector for PuctSelector {
    #[inline]
    fn exploration(&self, tree: &Tree, branch_id: BranchId, parent_id: NodeId<Evaluated>) -> Score {
        let branch = tree.branch(branch_id);
        let node = tree.node(branch.node());
        let parent = tree.node(parent_id);

        let c = self.c;
        let p = branch.policy().v();
        let cap_n_i = parent.visits().0 as f32;
        let n_i = node.visits().0 as f32;

        Score(c * p * cap_n_i.sqrt() / (1_f32 + (n_i)))
    }

    #[inline]
    fn exploitation(
        &self,
        tree: &Tree,
        branch_id: BranchId,
        parent_id: NodeId<Evaluated>,
    ) -> Score {
        let branch = tree.branch(branch_id);
        let node = tree.node(branch.node());
        let parent = tree.node(parent_id);
        let cap_n_i = parent.visits().0 as f32;
        let n_i = node.visits().0 as f32;
        if n_i == 0. {
            // fallback to parent q-value for unvisited nodes. note that the parent node
            // cannot be 0 because we only expand a node after visiting it at
            // least once.
            Score(parent.value() / cap_n_i)
        }
        else {
            Score(node.value() / n_i)
        }
    }
}
