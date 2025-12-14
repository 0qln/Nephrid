use super::utils::*;
use crate::core::{search::mcts::node::Node, turn::Turn};

pub trait Backpropagater {
    /// Update the node with the result of an evaluation.
    fn update(node: &mut Node, value: f32) -> ();
}

pub struct BackupInfo {
    /// whose turn it is at the node
    turn: Turn,
}

pub type BackupNode = DoubleLinkedNode<BackupInfo>;

#[derive(Default)]
pub struct DefaultBackuper {}

impl Backpropagater for DefaultBackuper {
    fn update(node: &mut Node, value: f32) -> () {
        node.visits += 1;
        node.value += value;
    }
}
