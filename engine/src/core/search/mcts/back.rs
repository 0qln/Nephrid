use std::ops::ControlFlow;

use crate::core::search::mcts::{
    eval::{Evaluation, RawPolicy},
    node::{
        Tree,
        node_state::{self, Branching, Evaluated, Terminal},
    },
    search::{SelNode, Selection},
};

pub trait Backpropagater {
    /// Backpropagate the [eval].
    fn backpropagate<const X: usize, T, S: const node_state::Valid>(
        &self,
        tree: &mut Tree,
        selection: &Selection<X, T>,
        leaf: &SelNode<Evaluation, S>,
    );
}

#[derive(Default)]
pub struct DefaultBackuper {}

impl Backpropagater for DefaultBackuper {
    fn backpropagate<const X: usize, T, S: const node_state::Valid>(
        &self,
        tree: &mut Tree,
        selection: &Selection<X, T>,
        leaf: &SelNode<Evaluation, S>,
    ) {
        let eval = &leaf.data;
        let policy = if let Evaluation::Guess(guess) = eval {
            &guess.policy
        }
        else {
            &RawPolicy::null()
        };

        // Update the leaf itself.
        {
            let node = &leaf.node;
            let value = eval.to_value(!leaf.turn);

            if let Some(unevaluated) = node.clone().try_into::<Branching>() {
                let evaluated = tree.set_policy_raw(unevaluated, policy);
                tree.update_node(evaluated, value);
            }
            else if let Some(evaluated) = node.clone().try_into::<Evaluated>() {
                tree.update_node(evaluated, value);
            }
            else if let Some(terminal) = node.clone().try_into::<Terminal>() {
                tree.update_node(terminal, value);
            }
        }

        // Update the parents until root.
        _ = selection.try_fold_up(leaf.parent, (), |_, item| {
            let node = item.node.clone();
            let value = eval.to_value(!item.turn);
            tree.update_node(node.clone(), value);

            ControlFlow::Continue::<(), ()>(())
        });
    }
}
