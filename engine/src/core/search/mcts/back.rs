use crate::core::search::mcts::{
    eval::{self, Evaluation, GameResult},
    node::{
        Tree,
        node_state::{self, Evaluated, Switch},
        proven,
    },
    search::SelNode,
};

pub trait Backpropagater {
    /// Backpropagate the [eval].
    fn backpropagate<'a, T: 'a, S: const node_state::Valid>(
        &self,
        tree: &mut Tree,
        path: impl Iterator<Item = &'a SelNode<T, Evaluated>>,
        leaf: SelNode<Evaluation, S>,
    );
}

#[derive(Default)]
pub struct DefaultBackuper {}

impl Backpropagater for DefaultBackuper {
    fn backpropagate<'a, T: 'a, S: const node_state::Valid>(
        &self,
        tree: &mut Tree,
        path: impl Iterator<Item = &'a SelNode<T, Evaluated>>,
        leaf: SelNode<Evaluation, S>,
    ) {
        let eval = &leaf.data;
        let weight = 1.;

        // Update the leaf itself.
        {
            let node = &leaf.node;
            let value = eval.to_value(!leaf.turn);

            match node.into_ct() {
                Switch::Branching(node) => {
                    let evaluated = if let Evaluation::Guess(guess) = eval {
                        tree.set_policy(node, &guess.policy)
                    }
                    else {
                        tree.skip_policy(node)
                    };
                    tree.update_node(evaluated, value, weight);
                }
                Switch::Evaluated(node) => tree.update_node(node, value, weight),
                Switch::Terminal(node) => tree.update_node(node, value, weight),
                Switch::Leaf(node) => tree.update_node(node, value, weight),
            }
        }

        // Update the parents until root.
        for item in path {
            let node = item.node;
            let value = eval.to_value(!item.turn);
            tree.update_node(node, value, weight);
        }
    }
}

#[derive(Default)]
pub struct MctsSolver {}

impl Backpropagater for MctsSolver {
    fn backpropagate<'a, T: 'a, S: const node_state::Valid>(
        &self,
        tree: &mut Tree,
        path: impl Iterator<Item = &'a SelNode<T, Evaluated>>,
        leaf: SelNode<Evaluation, S>,
    ) {
        let eval = &leaf.data;
        let weight = 1.;

        // Update the leaf itself.
        {
            let node = &leaf.node;
            let turn = leaf.turn;
            let value = eval.to_value(!turn);

            match node.into_ct() {
                Switch::Branching(node) => {
                    let evaluated = if let Evaluation::Guess(guess) = eval {
                        tree.set_policy(node, &guess.policy)
                    }
                    else {
                        tree.skip_policy(node)
                    };
                    tree.update_node(evaluated, value, weight);
                }
                Switch::Evaluated(node) => tree.update_node(node, value, weight),
                Switch::Terminal(node) => match eval {
                    Evaluation::Terminal(game_result) => match *game_result {
                        GameResult::Win { relative_to } if relative_to == turn => {
                            tree.set_proven(node, proven::LOSS, weight)
                        }
                        GameResult::Win { relative_to } if relative_to != turn => {
                            tree.set_proven(node, proven::WIN, weight)
                        }
                        _ => tree.update_node(node, value, weight),
                    },
                    _ => tree.update_node(node, value, weight),
                },

                // There could be a leaf node here (e.g. in the case of a 2fold-shortcut.) Make sure
                // that we also update leaf node values. Otherwise the exploration score of such a
                // node will be overinflated and the searcher will get stuck picking
                // this leaf node.
                Switch::Leaf(node) => tree.update_node(node, value, weight),
            }
        }

        // Update the parents until root.
        for item in path {
            let node = item.node;
            let turn = item.turn;

            match eval {
                Evaluation::Terminal(game_result) => {
                    match *game_result {
                        GameResult::Win { .. }
                            if tree
                                .branches(node)
                                .iter()
                                .any(|b| tree.node(b.node()).value().is_proven_win()) =>
                        {
                            tree.set_proven(node, proven::LOSS, weight)
                        }

                        GameResult::Win { .. }
                            if tree
                                .branches(node)
                                .iter()
                                .all(|b| tree.node(b.node()).value().is_proven_loss()) =>
                        {
                            tree.set_proven(node, proven::WIN, weight)
                        }

                        x => tree.update_node(node, x.to_value(!turn), weight),
                    };
                }
                Evaluation::Guess(guess) => tree.update_node(node, guess.to_value(!turn), weight),
                Evaluation::Nope => tree.update_node(node, eval::Value::draw(), weight),
            }
        }
    }
}
