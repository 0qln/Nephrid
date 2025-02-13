use itertools::Itertools;
use rand::{thread_rng, Rng};
use std::assert_matches::assert_matches;
use std::cmp::Ordering;
use std::ops::ControlFlow;
use std::ptr::NonNull;

use crate::engine::move_iter::king::King;
use crate::engine::piece::IPieceType;
use crate::engine::{color::Color, move_iter::fold_legal_moves, position::Position, r#move::Move};

#[cfg(test)]
mod test;

#[derive(Debug, PartialEq, Eq)]
pub enum PlayoutResult {
    Win { relative_to: Color },
    Draw,
}

impl PlayoutResult {
    pub fn maybe_new(pos: &Position, moves: &[Move]) -> Option<Self> {
        if pos.has_threefold_repetition() || pos.fifty_move_rule() {
            return Some(Self::Draw);
        }

        if moves.is_empty() {
            return Some({
                let us = pos.get_turn();
                let king = pos.get_bitboard(King::ID, us);
                let nstm_attacks = pos.get_nstm_attacks();
                let in_check = !(king & nstm_attacks).is_empty();
                if in_check {
                    // If in check and no moves, it's a loss for the current player
                    Self::Win { relative_to: !us }
                } else {
                    Self::Draw
                }
            });
        }

        None
    }
}

pub struct Tree {
    root: Node,
}

impl Tree {
    pub fn new(pos: &Position) -> Self {
        let root = Node::root(pos);
        Self { root }
    }

    pub fn best_move(&self) -> Option<Move> {
        if self.root.children.iter().any(|n| n.score.v().is_none()) {
            return None;
        }
        Some(
            self.root
                .children
                .iter()
                .max_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .expect("not all root nodes have been searched yet")
                })
                .expect("Root need's to have children.")
                .mov,
        )
    }

    pub fn dbg(&self) {
        println!("root: {:#?}", self.root);
        println!("\r\n");
    }

    pub fn grow(&mut self, pos: &mut Position) {
        let dbg = pos.clone();

        let mut stack = self.select_leaf_mut(pos);
        let leaf = unsafe {
            stack
                .last_mut()
                .expect("This is only None, if root.state == Leaf, which is not the case.")
                .as_mut()
        };

        let result = leaf.simulate(pos);

        self.backpropagate(pos, stack, result);

        assert_eq!(dbg, *pos, "Position was not reset correctly.");
    }

    fn backpropagate(
        &mut self,
        pos: &mut Position,
        mut stack: Vec<NonNull<Node>>,
        result: PlayoutResult,
    ) {
        unsafe {
            for node in stack.iter_mut().rev().map(|n| n.as_mut()) {
                pos.unmake_move(node.mov);

                node.score.playouts += 1;
                node.score.wins += match result {
                    PlayoutResult::Win { relative_to } if relative_to == pos.get_turn() => 1,
                    PlayoutResult::Win { relative_to: _ } => 0,
                    PlayoutResult::Draw => 0,
                };
            }
        }
        self.root.score.playouts += 1;
        self.root.score.wins += match result {
            PlayoutResult::Win { relative_to } if relative_to == pos.get_turn() => 1,
            PlayoutResult::Win { relative_to: _ } => 0,
            PlayoutResult::Draw => 0,
        };
    }

    fn select_leaf_mut(&mut self, pos: &mut Position) -> Vec<NonNull<Node>> {
        let mut stack: Vec<NonNull<Node>> = vec![];
        let mut current = unsafe { NonNull::from_ref(&self.root).as_mut() };
        loop {
            match current.state {
                NodeState::Root | NodeState::Branch => {
                    current = current.select_mut();
                    pos.make_move(current.mov);
                    stack.push(NonNull::from_ref(current));
                }
                NodeState::Leaf if current.score.playouts != 0 => {
                    current.expand(pos);
                }
                NodeState::Leaf | NodeState::Terminal => {
                    return stack;
                }
            }
        }
    }
}

#[derive(Debug, Default, Clone)]
struct Score {
    playouts: u32,
    wins: u32,
}

impl Score {
    pub fn v(&self) -> Option<f32> {
        match self.playouts {
            0 => None,
            _ => Some(self.wins as f32 / self.playouts as f32),
        }
    }
}

impl PartialEq for Score {
    fn eq(&self, other: &Self) -> bool {
        self.v().eq(&other.v())
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.v()?.partial_cmp(&other.v()?)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum NodeState {
    Root,
    Leaf,
    Branch,
    Terminal,
}

struct Node {
    score: Score,
    mov: Move,
    state: NodeState,
    children: Vec<Self>,
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("score", &self.score)
            .field("mov", &self.mov)
            .field("state", &self.state)
            .field(
                "children",
                &self
                    .children
                    .iter()
                    .filter(|c| c.score.playouts != 0)
                    .collect_vec(),
            )
            .finish()
    }
}

impl Node {
    pub fn root(pos: &Position) -> Self {
        let mut children = Vec::new();
        fold_legal_moves(pos, &mut children, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Node::leaf(m));
                acc
            })
        });
        assert!(children.len() > 0, "A root node cannot be a terminal node.");

        Self {
            score: Score::default(),
            mov: Move::null(),
            state: NodeState::Root,
            children,
        }
    }

    pub fn leaf(mov: Move) -> Self {
        Self {
            score: Score::default(),
            mov,
            state: NodeState::Leaf,
            children: Vec::new(),
        }
    }

    fn ucb(&self, cap_n_i: u32) -> f32 {
        match self.score.playouts {
            0 => f32::INFINITY,
            n_i => {
                let w_i = self.score.wins as f32;
                let n_i = n_i as f32;
                let exploitation = w_i / n_i;

                let c = f32::sqrt(2.0);
                let exploration = c * f32::sqrt((cap_n_i as f32).ln() / n_i);

                exploitation + exploration
            }
        }
    }

    fn select_mut(&mut self) -> &mut Self {
        assert_matches!(self.state, NodeState::Branch | NodeState::Root);

        self.children
            .iter_mut()
            .max_by(|a, b| {
                let a_ucb = a.ucb(self.score.playouts);
                let b_ucb = b.ucb(self.score.playouts);
                a_ucb.partial_cmp(&b_ucb).expect(&format!(
                    "UCB comparison failed: ({}, {}) <=> ({}, {})",
                    a_ucb, a.score.playouts, b_ucb, b.score.playouts
                ))
            })
            .expect("This is either a branch or a root node, which implies that this is not a terminal node, so there has to be atleast on child.")
    }

    fn simulate(&self, pos: &mut Position) -> PlayoutResult {
        assert_matches!(self.state, NodeState::Leaf | NodeState::Terminal);

        let mut rng = thread_rng();
        let mut stack: Vec<Move> = Vec::new();

        loop {
            let mut moves = Vec::new();
            fold_legal_moves(pos, &mut moves, |acc, m| {
                ControlFlow::Continue::<(), _>({
                    acc.push(m);
                    acc
                })
            });
            if let Some(result) = PlayoutResult::maybe_new(pos, &moves) {
                while let Some(m) = stack.pop() {
                    pos.unmake_move(m);
                }
                return result;
            }

            let mov = moves[rng.gen_range(0..moves.len())];
            pos.make_move(mov);
            stack.push(mov);
        }
    }

    fn expand(&mut self, pos: &Position) {
        assert_matches!(self.state, NodeState::Leaf);

        fold_legal_moves(pos, &mut self.children, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Node::leaf(m));
                acc
            })
        });
        self.state = if self.children.is_empty() {
            NodeState::Terminal
        } else {
            NodeState::Branch
        };
    }
}
