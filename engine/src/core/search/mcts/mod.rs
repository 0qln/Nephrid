use itertools::Itertools;
use rand::{rng, Rng};
use std::assert_matches::assert_matches;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{AddAssign, ControlFlow};
use std::ptr::NonNull;

use crate::core::r#move::MoveList;
use crate::core::move_iter::king::King;
use crate::core::piece::IPieceType;
use crate::core::{color::Color, move_iter::fold_legal_moves, position::Position, r#move::Move};

#[cfg(test)]
mod test;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PlayoutResult {
    Win { relative_to: Color },
    Draw,
}

impl PlayoutResult {
    pub fn maybe_new(pos: &Position, move_cnt: u8) -> Option<Self> {
        if pos.has_threefold_repetition() || pos.fifty_move_rule() {
            return Some(Self::Draw);
        }

        if move_cnt == 0 {
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

#[derive(Default, Debug, Clone)]
pub struct Tree {
    /// Root of the tree.
    root: Node,
    
    /// Stack of nodes that were selected during the selection phase.
    selection_buffer: Vec<NonNull<Node>>,
    
    /// Buffers used during simulation.
    simulation_stack_buffer: Vec<Move>,
    simulation_moves_buffer: MoveList,
}

impl Tree {
    pub fn new(pos: &Position) -> Self {
        let root = Node::root(pos);
        Self {
            root,
            ..Default::default()
        }
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
        self.select_leaf_mut(pos);
        let leaf = unsafe { self.selected_leaf().as_mut() };
        let result = leaf.simulate(
            pos,
            &mut self.simulation_stack_buffer,
            &mut self.simulation_moves_buffer,
            &mut rng(),
        );
        self.backpropagate(pos, result);
    }
    
    pub fn selected_leaf(&mut self) -> NonNull<Node> {
        *self.selection_buffer
                .last_mut()
                .expect("This is only None, if root.state == Leaf, which is not the case.")
    }

    pub fn advance(&mut self, mov: Move) {
        self.root = self
            .root
            .children
            .iter()
            .find(|n| n.mov == mov)
            .unwrap()
            .clone();
    }

    fn backpropagate(&mut self, pos: &mut Position, result: PlayoutResult) {
        unsafe {
            for node in self.selection_buffer.iter_mut().rev().map(|n| n.as_mut()) {
                pos.unmake_move(node.mov);
                node.score += (result, pos.get_turn());
            }
        }
        self.root.score += (result, pos.get_turn());
    }

    pub fn select_leaf_mut(&mut self, pos: &mut Position) {
        self.selection_buffer.clear();
        let mut current = unsafe { NonNull::from_ref(&self.root).as_mut() };
        loop {
            match current.state {
                NodeState::Branch => {
                    current = current.select_mut();
                    pos.make_move(current.mov);
                    self.selection_buffer.push(NonNull::from_ref(current));
                }
                NodeState::Leaf if current.score.playouts != 0 => {
                    current.expand(pos);
                }
                NodeState::Leaf | NodeState::Terminal => {
                    break;
                }
            }
        }
    }
}

#[derive(Debug, Default, Clone)]
struct Score {
    playouts: u32,
    wins: f32,
}

impl Score {
    pub fn v(&self) -> Option<f32> {
        match self.playouts {
            0 => None,
            _ => Some(self.wins / self.playouts as f32),
        }
    }
}

impl AddAssign<(PlayoutResult, Color)> for Score {
    #[inline(always)]
    fn add_assign(&mut self, rhs: (PlayoutResult, Color)) {
        self.playouts += 1;
        self.wins += match rhs.0 {
            PlayoutResult::Win { relative_to } if relative_to == rhs.1 => 1.0,
            PlayoutResult::Win { relative_to: _ } => 0.0,
            PlayoutResult::Draw => 0.5,
        };
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

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
enum NodeState {
    Leaf,
    Branch,
    #[default]
    Terminal,
}

#[derive(Clone, Default)]
pub struct Node {
    score: Score,
    mov: Move,
    state: NodeState,
    children: Vec<Self>,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        assert!(!children.is_empty(), "A root node cannot be a terminal node.");

        Self {
            score: Score::default(),
            mov: Move::null(),
            state: NodeState::Branch,
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
                let w_i = self.score.wins;
                let n_i = n_i as f32;
                let exploitation = w_i / n_i;

                let c = f32::sqrt(2.0);
                let exploration = c * f32::sqrt((cap_n_i as f32).ln() / n_i);

                exploitation + exploration
            }
        }
    }

    pub fn select_mut(&mut self) -> &mut Self {
        assert_matches!(self.state, NodeState::Branch);

        self.children
            .iter_mut()
            .max_by(|a, b| {
                let a_ucb = a.ucb(self.score.playouts);
                let b_ucb = b.ucb(self.score.playouts);
                a_ucb.partial_cmp(&b_ucb).expect("UCB comparison failed!")
            })
            .expect("This is either a branch or a root node, which implies that this is not a terminal node, so there has to be atleast on child.")
    }

    pub fn simulate(
        &self,
        pos: &mut Position,
        stack: &mut Vec<Move>,
        moves: &mut MoveList,
        rng: &mut impl Rng,
    ) -> PlayoutResult {
        assert_matches!(self.state, NodeState::Leaf | NodeState::Terminal);

        stack.clear();

        loop {
            let mut move_cnt = 0;
            fold_legal_moves(pos, &mut *moves, |acc, m| {
                ControlFlow::Continue::<(), _>({
                    acc[move_cnt] = m;
                    move_cnt += 1;
                    acc
                })
            });

            if let Some(result) = PlayoutResult::maybe_new(pos, move_cnt) {
                while let Some(m) = stack.pop() {
                    pos.unmake_move(m);
                }
                return result;
            }

            let mov = moves[rng.random_range(0..move_cnt)];
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
