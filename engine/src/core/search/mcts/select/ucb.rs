use crate::core::search::mcts::node::{BranchId, Tree, VisitCount};

use super::*;

pub struct UcbSelector {
    c: f32,
}

impl UcbSelector {
    pub fn new(c: f32) -> Self { Self { c } }
}

impl Default for UcbSelector {
    fn default() -> Self { Self { c: f32::sqrt(2.0) } }
}

impl Selector for UcbSelector {
    fn exploitation(&self, tree: &Tree, branch_id: BranchId, _parent_id: NodeId<Evaluated>) -> Score {
        let branch = tree.branch(branch_id);
        let node = tree.node(branch.node());

        match node.visits() {
            VisitCount(0) => Score(f32::INFINITY),
            VisitCount(n_i) => {
                let w_i = node.value();
                let n_i = n_i as f32;
                Score(w_i / n_i)
            }
        }
    }

    fn exploration(&self, tree: &Tree, branch_id: BranchId, parent_id: NodeId<Evaluated>) -> Score {
        let branch = tree.branch(branch_id);
        let node = tree.node(branch.node());

        match node.visits() {
            VisitCount(0) => Score(f32::INFINITY),
            VisitCount(n_i) => {
                let VisitCount(cap_n_i) = tree.node(parent_id).visits();
                let n_i = n_i as f32;
                Score(self.c * f32::sqrt((cap_n_i as f32).ln() / n_i))
            }
        }
    }
}
