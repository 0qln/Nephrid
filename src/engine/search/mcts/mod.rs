use rand::{thread_rng, Rng};
use std::ops::ControlFlow;
use std::ptr::NonNull;

use crate::engine::move_iter::king::King;
use crate::engine::piece::IPieceType;
use crate::engine::{color::Color, move_iter::fold_legal_moves, position::Position, r#move::Move};

#[derive(Debug, PartialEq, Eq)]
pub enum PlayoutResult {
    Win { relative_to: Color },
    Draw,
}

impl PlayoutResult {
    pub fn new(pos: &Position) -> Self {
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
    }
    
    pub fn maybe_new(pos: &Position, move_cnt: usize) -> Option<Self> {
        if pos.has_threefold_repetition() || pos.fifty_move_rule() {
            return Some(Self::Draw);
        }
        
        if move_cnt == 0 {
            return Some(Self::new(pos));
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

    pub fn best_move(&self) -> Move {
        self.root.select().mov
    }

    pub fn grow(&mut self, pos: &mut Position) {
        let mut stack = self.select_leaf_mut(pos);
        let leaf = unsafe { stack.last_mut().unwrap().as_mut() };
        let result = leaf.simulate(pos);
        Self::backpropagate(pos, stack, result);
    }

    fn backpropagate(pos: &mut Position, mut stack: Vec<NonNull<Node>>, result: PlayoutResult) {
        unsafe {
            for node in stack.iter_mut().map(|n| n.as_mut()) {
                pos.unmake_move(node.mov);
                node.score.playouts += 1;
                node.score.wins += match result {
                    PlayoutResult::Win { relative_to } if relative_to == pos.get_turn() => 1,
                    PlayoutResult::Win { relative_to: _ } => 0,
                    PlayoutResult::Draw => 0,
                };
            }
        }
    }

    fn select_leaf_mut(&mut self, pos: &mut Position) -> Vec<NonNull<Node>> {
        let mut stack = vec![NonNull::from_ref(&self.root)];
        loop {
            let current = unsafe { stack.last_mut().unwrap().as_mut() };
            if current.state == NodeState::Leaf {
                if current.score.playouts == 0 {
                    current.expand(pos);
                    let next = current.select_mut();
                    pos.make_move(next.mov);
                    stack.push(NonNull::from_ref(next));
                    break;
                } else {
                    break;
                }
            } else {
                let next = current.select_mut();
                pos.make_move(next.mov);
                stack.push(NonNull::from_ref(next));
            }
        }
        stack
    }
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
struct Score {
    playouts: u32,
    wins: u32,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum NodeState {
    Root,
    Leaf,
    Branch,
}

#[derive(Debug)]
struct Node {
    score: Score,
    mov: Move,
    state: NodeState,
    children: Vec<Self>,
}

impl Node {
    pub fn root(pos: &Position) -> Self {
        let mut children = Vec::new();
        fold_legal_moves(pos, &mut children, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Node::new(m, NodeState::Leaf));
                acc
            })
        });
        Self {
            score: Score::default(),
            mov: Move::null(),
            state: NodeState::Root,
            children,
        }
    }

    pub fn new(mov: Move, state: NodeState) -> Self {
        Self {
            score: Score::default(),
            mov,
            state,
            children: Vec::new(),
        }
    }

    fn ucb(&self, cap_n_i: u32) -> f32 {
        if self.score.playouts == 0 {
            f32::INFINITY
        } else {
            let w_i = self.score.wins as f32;
            let n_i = self.score.playouts as f32;
            let exploitation = w_i / n_i;
            let c = f32::sqrt(2.0);
            let exploration = c * f32::sqrt((cap_n_i as f32).ln() / n_i);
            exploitation + exploration
        }
    }

    fn select_mut(&mut self) -> &mut Self {
        self.children
            .iter_mut()
            .max_by(|a, b| {
                let a_ucb = a.ucb(self.score.playouts);
                let b_ucb = b.ucb(self.score.playouts);
                a_ucb.partial_cmp(&b_ucb).unwrap()
            })
            .expect("No children to select from")
    }

    fn select(&self) -> &Self {
        self.children
            .iter()
            .max_by(|a, b| {
                let a_ucb = a.ucb(self.score.playouts);
                let b_ucb = b.ucb(self.score.playouts);
                a_ucb.partial_cmp(&b_ucb).unwrap()
            })
            .expect("No children to select from")
    }

    fn simulate(&self, pos: &mut Position) -> PlayoutResult {
        let mut rng = thread_rng();
        let mut moves_stack = Vec::new();
        loop {
            let mut moves = Vec::new();
            fold_legal_moves(pos, &mut moves, |acc, m| {
                ControlFlow::Continue::<(), _>({
                    acc.push(m);
                    acc
                })
            });
            if let Some(result) = PlayoutResult::maybe_new(pos, moves.len()) {
                while let Some(m) = moves_stack.pop() {
                    pos.unmake_move(m);
                }
                return result;
            }
            let mov = moves[rng.gen_range(0..moves.len())];
            pos.make_move(mov);
            moves_stack.push(mov);
        }
    }

    fn expand(&mut self, pos: &mut Position) {
        if self.state != NodeState::Leaf {
            return;
        }
        fold_legal_moves(pos, &mut self.children, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Node::new(m, NodeState::Leaf));
                acc
            })
        });
        self.state = NodeState::Branch;
    }
}
