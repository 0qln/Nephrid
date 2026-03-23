use std::ops::ControlFlow;

use crate::core::search::mcts::{
    eval::{self, Evaluation, GameResult},
    node::{
        Tree,
        node_state::{self, NodeSwitch},
        proven,
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

        // Update the leaf itself.
        {
            let node = &leaf.node;
            let value = eval.to_value(!leaf.turn);

            match node.clone().into_ct() {
                NodeSwitch::Branching(node) => {
                    let evaluated = if let Evaluation::Guess(guess) = eval {
                        tree.set_policy(node, &guess.policy)
                    }
                    else {
                        tree.skip_policy(node)
                    };
                    tree.update_node(evaluated, value);
                }
                NodeSwitch::Evaluated(node) => tree.update_node(node, value),
                NodeSwitch::Terminal(node) => tree.update_node(node, value),
                NodeSwitch::Leaf(node) => tree.update_node(node, value),
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

#[derive(Default)]
pub struct MctsSolver {}

impl Backpropagater for MctsSolver {
    fn backpropagate<const X: usize, T, S: const node_state::Valid>(
        &self,
        tree: &mut Tree,
        selection: &Selection<X, T>,
        leaf: &SelNode<Evaluation, S>,
    ) {
        let eval = &leaf.data;

        // Update the leaf itself.
        {
            let node = &leaf.node;
            let turn = leaf.turn;
            let value = eval.to_value(!turn);

            match node.clone().into_ct() {
                NodeSwitch::Branching(node) => {
                    let evaluated = if let Evaluation::Guess(guess) = eval {
                        tree.set_policy(node, &guess.policy)
                    }
                    else {
                        tree.skip_policy(node)
                    };
                    tree.update_node(evaluated, value);
                }
                NodeSwitch::Evaluated(node) => tree.update_node(node, value),
                NodeSwitch::Terminal(node) => match eval {
                    Evaluation::Terminal(game_result) => match *game_result {
                        GameResult::Win { relative_to } if relative_to == turn => {
                            tree.set_proven(node, proven::LOSS)
                        }
                        GameResult::Win { relative_to } if relative_to != turn => {
                            tree.set_proven(node, proven::WIN)
                        }
                        _ => tree.update_node(node, value),
                    },
                    _ => tree.update_node(node, value),
                },

                // There could be a leaf node here (e.g. in the case of a 2fold-shortcut.) Make sure
                // that we also update leaf node values. Otherwise the exploration score of such a
                // node will be overinflated and the searcher will get stuck picking
                // this leaf node.
                NodeSwitch::Leaf(node) => tree.update_node(node, value),
            }
        }

        // Update the parents until root.
        _ = selection.try_fold_up(leaf.parent, (), |_, item| {
            let node = item.node.clone();
            let turn = item.turn;

            match eval {
                Evaluation::Terminal(game_result) => {
                    match *game_result {
                        GameResult::Win { .. }
                            if node
                                .borrow()
                                .branches()
                                .iter()
                                .any(|b| b.node().borrow().value().is_proven_win()) =>
                        {
                            tree.set_proven(node.clone(), proven::LOSS)
                        }

                        GameResult::Win { .. }
                            if node
                                .borrow()
                                .branches()
                                .iter()
                                .all(|b| b.node().borrow().value().is_proven_loss()) =>
                        {
                            tree.set_proven(node.clone(), proven::WIN)
                        }

                        x => tree.update_node(node.clone(), x.to_value(!turn)),
                    };
                }
                Evaluation::Guess(guess) => tree.update_node(node.clone(), guess.to_value(!turn)),
                Evaluation::Nope => tree.update_node(node.clone(), eval::Value::draw()),
            }

            ControlFlow::Continue::<(), ()>(())
        });
    }
}
