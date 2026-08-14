use core::fmt;
use std::ops;

use crate::core::{
    color::Perspective, depth::Depth, position::Position, search::mcts::{
        node::{BranchId, NodeId, Tree, node_state::Evaluated},
        search::MctsParams,
    }
};

pub mod hpuct;
pub mod puct;
pub mod ucb;

pub trait Selector {
    fn exploitation(&self, tree: &Tree, branch_id: BranchId, parent_id: NodeId<Evaluated>) -> Score;
    fn exploration(&self, tree: &Tree, branch_id: BranchId, parent_id: NodeId<Evaluated>) -> Score;

    fn pick_branch<P: Perspective>(
        &mut self,
        tree: &Tree,
        parent_id: NodeId<Evaluated>,
        _depth: Depth,
        _position: &Position,
        _params: &impl MctsParams,
    ) -> BranchId {
        let mut best_score = Score(f32::NEG_INFINITY);
        let mut best_branch = None;

        for branch_id in tree.branch_ids(parent_id) {
            let loit = self.exploitation(tree, branch_id, parent_id);
            let lora = self.exploration(tree, branch_id, parent_id);
            let score = loit + lora;

            if score >= best_score {
                best_score = score;
                best_branch = Some(branch_id);
            }
        }

        best_branch.expect("Evaluated node must have at least one branch")
    }

    fn virtual_loss(&self) -> u32 { 1 }
}

#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Score(pub f32);

impl Score {
    pub fn new(_0: f32) -> Self { Self(_0) }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}

impl_op!(-|x: Score| -> Score { Score(-x.0) });
impl_op!(+|x: Score, y: Score| -> Score { Score(x.0 + y.0) });
impl_op!(*|x: Score, y: f32| -> Score { Score(x.0 * y) });
impl_op!(+|x: Score, y: f32| -> Score { Score(x.0 + y) });

impl Eq for Score {}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or_else(|| panic!("This shouldn't happen for scores. Failed to compare scores {self:?} and {other:?}"))
    }
}

impl From<Score> for f32 {
    fn from(val: Score) -> Self { val.0 }
}
