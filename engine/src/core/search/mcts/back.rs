use std::ops::ControlFlow;
use crate::core::search::mcts::eval::Value;

use crate::core::search::mcts::{
    eval::Evaluation,
    node::{Node, NodeState},
    search::{SelectionNode, SelectionNodeRef},
};

pub trait Backpropagater {
    /// Update the node with the result of an evaluation.
    fn update(node: &mut Node, value: Value);

    /// Backpropagate the [eval].
    fn backpropagate<T>(&self, leaf: SelectionNodeRef<T>, eval: &Evaluation);
}

#[derive(Default)]
pub struct DefaultBackuper {}

impl Backpropagater for DefaultBackuper {
    fn update(node: &mut Node, value: Value) {
        node.visits += 1;
        node.value += value;
    }

    fn backpropagate<T>(&self, leaf: SelectionNodeRef<T>, eval: &Evaluation) {
        // Traverse the selected node in reverse, updating the parents along the way.
        _ = SelectionNode::try_fold_up_mut(leaf.clone(), (), |_, node| {
            let turn = node.borrow().data().turn;
            let value = eval.to_value(turn);
            Self::update(&mut node.borrow_mut().data().node.borrow_mut(), value);

            ControlFlow::Continue::<(), ()>(())
        });

        // If the eval was a guess make sure to also set the policies of the selected
        // leaf.
        if let Evaluation::Guess(guess) = eval {
            debug_assert_ne!(
                leaf.borrow().data().node.borrow().state(),
                NodeState::Terminal,
                "We received a guess for a terminal node? What the fuck? Either this is a bad \
                 unit test, or some indeces are being handled incorrectly between the selector \
                 and the evaluator."
            );

            let policy = &guess.policy;
            let leaf_sel = leaf.borrow_mut();
            let leaf_node = &mut leaf_sel.data().node.borrow_mut();
            leaf_node.set_policy_raw(policy);
        }
    }
}
