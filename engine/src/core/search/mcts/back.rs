use std::ops::ControlFlow;

use crate::core::search::mcts::{
    eval::Evaluation,
    node::NodeState,
    search::{SelectionNode, SelectionNodeRef},
};

pub trait Backpropagater {
    /// Backpropagate the [eval].
    fn backpropagate<T>(&self, leaf: SelectionNodeRef<T>, eval: &Evaluation);
}

#[derive(Default)]
pub struct DefaultBackuper {}

impl Backpropagater for DefaultBackuper {
    fn backpropagate<T>(&self, leaf: SelectionNodeRef<T>, eval: &Evaluation) {
        // println!("[eval: {eval}]");

        // Traverse the selected node in reverse, updating the parents along the way.
        _ = SelectionNode::try_fold_up_mut(leaf.clone(), (), |_, node| {
            let turn = node.borrow().data().turn;
            let depth = node.borrow().data().depth;
            let value = eval.to_value(turn);
            let node = node.borrow_mut().data().node.borrow_mut();
            node.update(value);

            // println!("[node: ({turn}, {depth}), update: {value:?}]");

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
