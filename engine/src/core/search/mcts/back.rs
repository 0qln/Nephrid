use crate::core::search::mcts::node::Node;

pub trait Backpropagater {
    /// Update the node with the result of an evaluation.
    fn update(node: &mut Node, value: f32);
}

#[derive(Default)]
pub struct DefaultBackuper {}

impl Backpropagater for DefaultBackuper {
    fn update(node: &mut Node, value: f32) {
        node.visits += 1;
        node.value += value;
    }
}
