use std::ops::ControlFlow;

use crate::core::search::mcts::{
    eval::{self, Evaluation, GameResult},
    node::{
        Tree,
        node_state::{self, Branching, Evaluated, Terminal},
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

            if let Some(unevaluated) = node.clone().try_into::<Branching>() {
                let evaluated = if let Evaluation::Guess(guess) = eval {
                    tree.set_policy(unevaluated, &guess.policy)
                }
                else {
                    tree.skip_policy(unevaluated)
                };
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

            if let Some(unevaluated) = node.clone().try_into::<Branching>() {
                let evaluated = if let Evaluation::Guess(guess) = eval {
                    tree.set_policy(unevaluated, &guess.policy)
                }
                else {
                    tree.skip_policy(unevaluated)
                };
                tree.update_node(evaluated, value);
            }
            else if let Some(evaluated) = node.clone().try_into::<Evaluated>() {
                tree.update_node(evaluated, value);
            }
            else if let Some(terminal) = node.clone().try_into::<Terminal>() {
                match eval {
                    Evaluation::Terminal(game_result) => match *game_result {
                        GameResult::Win { relative_to } if relative_to == turn => {
                            tree.set_proven(terminal, proven::LOSS)
                        }
                        GameResult::Win { relative_to } if relative_to != turn => {
                            tree.set_proven(terminal, proven::WIN)
                        }
                        _ => tree.update_node(terminal, value),
                    },
                    _ => tree.update_node(terminal, value),
                };
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
