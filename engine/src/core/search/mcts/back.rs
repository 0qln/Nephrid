use std::hint::{self};

use crate::core::{
    color::Color,
    search::mcts::{
        eval::{self, Evaluation, GameResult, Guess},
        node::{NodeId, Tree, node_state::*, proven},
    },
    turn::Turn,
};

pub trait RelativeValue {
    fn relative_to(&self, relative_to: Color) -> eval::Value;
}

impl RelativeValue for GameResult {
    fn relative_to(&self, relative_to: Color) -> eval::Value {
        match *self {
            GameResult::Win { relative_to: winner } if winner == relative_to => eval::Value::win(),
            GameResult::Win { relative_to: winner } if winner != relative_to => eval::Value::loss(),
            GameResult::Draw => eval::Value::draw(),
            _ => unreachable!(),
        }
    }
}

impl RelativeValue for Evaluation {
    fn relative_to(&self, relative_to: Color) -> eval::Value {
        self.to_value(relative_to)
    }
}

impl RelativeValue for Guess {
    fn relative_to(&self, relative_to: Color) -> eval::Value {
        self.to_value(relative_to)
    }
}

pub fn update_branching(
    tree: &mut Tree,
    node: NodeId<Branching>,
    turn: Turn,
    guess: &Guess,
    weight: f32,
) {
    let value = guess.to_value(!turn);
    let evaluated = tree.set_policy(node, &guess.policy);
    tree.update_node(evaluated, value, weight);
}

pub fn update_terminal(
    tree: &mut Tree,
    node: NodeId<Terminal>,
    turn: Turn,
    game_result: GameResult,
    weight: f32,
) {
    match game_result {
        GameResult::Win { relative_to } if relative_to == turn => {
            tree.set_proven(node, proven::LOSS, weight);
        }
        GameResult::Win { relative_to } if relative_to != turn => {
            tree.set_proven(node, proven::WIN, weight);
        }
        GameResult::Draw => {
            // todo: implement proven draws
            let value = eval::Value::draw();
            tree.update_node(node, value, weight);
        }
        _ => unsafe { hint::unreachable_unchecked() },
    }
}

pub fn update_shortcut(
    tree: &mut Tree,
    node: NodeId<Leaf>,
    weight: f32,
) -> impl RelativeValue + Copy + use<> {
    #[derive(Copy, Clone)]
    struct Draw;
    impl RelativeValue for Draw {
        fn relative_to(&self, _relative_to: Color) -> eval::Value {
            eval::Value::draw()
        }
    }
    let value = eval::Value::draw();
    tree.update_node(node, value, weight);
    Draw
}

pub fn update_skip(
    tree: &mut Tree,
    node: NodeId<Evaluated>,
    turn: Turn,
    eval: &Evaluation,
    weight: f32,
) {
    // (e.g. in the case of a 2fold.) Make sure that we also update leaf
    // node values. Otherwise the exploration score of such a node will be
    // overinflated and the searcher will get stuck picking this leaf node.
    let value = eval.to_value(!turn);
    tree.update_node(node, value, weight);
}

pub fn update_parent(
    tree: &mut Tree,
    node: NodeId<Evaluated>,
    turn: Turn,
    child_value: &impl RelativeValue,
    weight: f32,
) {
    tree.update_node(node, child_value.relative_to(!turn), weight);
}

pub fn try_prove_parent(tree: &mut Tree, node: NodeId<Evaluated>, weight: f32) {
    let branches = tree.branches(node);
    if branches
        .iter()
        .all(|b| tree.node(b.node()).value().is_proven_loss())
    {
        tree.set_proven(node, proven::WIN, weight);
    }
    else if branches
        .iter()
        .any(|b| tree.node(b.node()).value().is_proven_win())
    {
        tree.set_proven(node, proven::LOSS, weight);
    }
}

pub fn backpropagate_up(
    tree: &mut Tree,
    path: impl IntoIterator<Item = (NodeId<Evaluated>, Turn)>,
    leaf_value: &impl RelativeValue,
    weight: f32,
) {
    for (node, turn) in path {
        update_parent(tree, node, turn, leaf_value, weight);
        try_prove_parent(tree, node, weight);
    }
}
