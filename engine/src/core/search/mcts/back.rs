use std::ops::ControlFlow;

use crate::core::search::mcts::{
    eval::Evaluation,
    node::Tree,
    search::{EvalItem, Selection, SelectionLeaf},
};

pub trait Backpropagater {
    /// Backpropagate the [eval].
    fn backpropagate<const X: usize, T>(
        &self,
        tree: &mut Tree,
        selection: &Selection<X, T>,
        leaf: &SelectionLeaf<T>,
    );
}

#[derive(Default)]
pub struct DefaultBackuper {}

impl Backpropagater for DefaultBackuper {
    fn backpropagate<const X: usize, T>(
        &self,
        tree: &mut Tree,
        selection: &Selection<X, T>,
        leaf: &SelectionLeaf<T>,
    ) {
        // Extract the evaluation (we skip if it's batched and not yet evaluated)
        let eval = match &leaf.eval {
            EvalItem::Evaluated(e) => e,
            // this shouldn't happen, log an error or something
            EvalItem::Batched => return,
        };

        // Traverse the selected node in reverse, updating the parents along the way.
        _ = selection.try_fold_up(leaf.parent_id, (), |_, node| {
            let turn = node.item.turn;
            let value = eval.to_value(turn);

            let node = node.item.node.clone();
            tree.update_node(node, value);

            ControlFlow::Continue::<(), ()>(())
        });

        // If the eval was a guess, make sure to also set the policies of the selected
        // leaf.
        if let Evaluation::Guess(guess) = eval {
            // Terminal nodes don't have trace data (leaf_data is None), so we only
            // set policies for Branching leaf nodes!
            if let Some(data) = &leaf.leaf_data {
                let policy = &guess.policy;
                let node = data.node.clone();
                tree.set_policy_raw(node, policy);
            }
        }
    }
}
