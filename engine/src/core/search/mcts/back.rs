use std::ops::ControlFlow;

use crate::core::search::mcts::{
    eval::Evaluation,
    search::{EvalItem, Selection, SelectionLeaf},
};

pub trait Backpropagater {
    /// Backpropagate the [eval].
    fn backpropagate<const X: usize, T>(
        &self,
        selection: &Selection<X, T>,
        leaf: &SelectionLeaf<T>,
    );
}

#[derive(Default)]
pub struct DefaultBackuper {}

impl Backpropagater for DefaultBackuper {
    fn backpropagate<const X: usize, T>(
        &self,
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

            let mut node = node.item.node.borrow_mut();
            node.update(value);

            ControlFlow::Continue::<(), ()>(())
        });

        // If the eval was a guess, make sure to also set the policies of the selected
        // leaf.
        if let Evaluation::Guess(guess) = eval {
            // Terminal nodes don't have trace data (leaf_data is None), so we only
            // set policies for Branching leaf nodes!
            if let Some(leaf_data) = &leaf.leaf_data {
                let policy = &guess.policy;
                let leaf_node = leaf_data.node.clone();
                leaf_node.set_policy_raw(policy);
            }
        }
    }
}
