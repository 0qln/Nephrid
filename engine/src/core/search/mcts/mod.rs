use burn::prelude::Backend;
use eval::model::{board_input, state_input, Model};
use itertools::Itertools;
use std::assert_matches::assert_matches;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::ops::{AddAssign, ControlFlow};
use std::ptr::NonNull;

use crate::core::move_iter::king::King;
use crate::core::piece::IPieceType;
use crate::core::{color::Color, move_iter::fold_legal_moves, position::Position, r#move::Move};

#[cfg(test)]
mod test;

pub mod eval;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum PlayoutResult {
    Win { relative_to: Color },
    Score { relative_to: Color, quality: f32 },
    Draw,
}

impl PlayoutResult {
    pub fn maybe_new(pos: &Position, state: NodeState) -> Option<Self> {
        if pos.has_threefold_repetition() || pos.fifty_move_rule() {
            return Some(Self::Draw);
        }

        if state == NodeState::Terminal {
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
}

impl Tree {
    pub fn new(pos: &Position) -> Self {
        let root = Node::root(pos);
        Self {
            root,
            ..Default::default()
        }
    }

    pub fn best_move(&self) -> Move {
        self.root
            .children
            .iter()
            .max_by(|a, b| a.score.playouts.partial_cmp(&b.score.playouts).expect("not all root nodes have been searched yet"))
            .expect("Root need's to have children.")
            .mov
    }

    pub fn dbg(&self) {
        println!("root: {:#?}", self.root);
        println!("\r\n");
    }

    pub fn grow<B: Backend>(&mut self, pos: &mut Position, model: &mut Model<B>) {
        self.select_leaf_mut(pos);
        let leaf = unsafe { self.selected_leaf().as_mut() };
        let result = leaf.eval(pos, model);
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

#[derive(Debug, Default, Clone, Copy)]
struct Score {
    playouts: u32,
    quality: f32,
}

impl AddAssign<(PlayoutResult, Color)> for Score {
    #[inline(always)]
    fn add_assign(&mut self, rhs: (PlayoutResult, Color)) {
        self.playouts += 1;
        self.quality += match rhs.0 {
            PlayoutResult::Win { relative_to } if relative_to == rhs.1 => 1.0,
            PlayoutResult::Win { relative_to: _ } => 0.0,
            PlayoutResult::Draw => 0.5,
            PlayoutResult::Score { quality, relative_to } => {
                if relative_to == rhs.1 { quality } else { 1.0 - quality }
            },
        };
    }
}

impl Score {
    pub fn v(&self) -> Option<f32> {
        match self.playouts {
            0 => None,
            _ => Some(self.quality / self.playouts as f32),
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

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub enum NodeState {
    Leaf,
    Branch,
    #[default]
    Terminal,
}

#[derive(Clone, Default)]
pub struct Node {
    score: Score,
    policy: Option<HashMap<Move, f32>>, //todo: better implementation
    mov: Move,
    state: NodeState,
    children: Vec<Node>,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("score", &self.score)
            .field("mov", &self.mov)
            .field("state", &self.state)
            .field("policy", &self.policy)
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
        _ = fold_legal_moves(pos, &mut children, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Node::leaf(m));
                acc
            })
        });

        assert!(!children.is_empty(), "A root node cannot be a terminal node.");
        
        Self {
            mov: Move::null(),
            state: NodeState::Branch,
            children,
            score: Score::default(),
            policy: None,
        }
    }

    pub fn leaf(mov: Move) -> Self {
        Self {
            mov,
            state: NodeState::Leaf,
            children: Vec::new(),
            score: Score::default(),
            policy: None,
        }
    }
    
    // note: we take the policy as an argument, because if we later convert this 
    // tree structure to a graph, we have to consider different policies from different parents.
    fn puct(&self, cap_n_i: u32, policy: f32) -> f32 {
        let n_i = self.score.playouts as f32;
        
        // The quality is updated incrementally as the tree is explored.
        // Because of this, we have to divide by the number of playouts 
        // to get the average quality of this node.
        let exploitation = self.score.quality / n_i;

        // todo: fine tune c.
        let c = f32::sqrt(2.0);
        let exploration = c * policy * (cap_n_i as f32).sqrt() / (1f32 + n_i);

        exploitation + exploration
    }

    pub fn select_mut(&mut self) -> &mut Self {
        assert_matches!(self.state, NodeState::Branch);
        let policy = self.policy.as_ref().expect("Policy has not been initialized yet!");

        self.children
            .iter_mut()
            .max_by(|a, b| {
                let a_ucb = a.puct(self.score.playouts, *policy.get(&a.mov).expect("Child node was not present in policy map!"));
                let b_ucb = b.puct(self.score.playouts, *policy.get(&b.mov).expect("Child node was not present in policy map!"));
                a_ucb.partial_cmp(&b_ucb).expect("Node comparison failed!")
            })
            .expect("This is either a branch or a root node, which implies that this is not a terminal node, so there has to be atleast on child.")
    }
    
    pub fn eval<B: Backend>(&mut self, pos: &Position, model: &mut Model<B>) -> PlayoutResult {
        assert_matches!(self.state, NodeState::Leaf | NodeState::Terminal);
        
        PlayoutResult::maybe_new(pos, self.state).unwrap_or_else(|| {
            // todo: evaluation of quality and policy using the model.
            // todo: the quality is between -1 and 1, so we have to convert it to a 0 to 1 range.
            // todo: return the score of this node, relative to the current player of this node.
            
            let b_in = board_input(pos).into();
            let s_in = state_input(pos).into();
            let (quality, policy) = model.forward(b_in, s_in);
            
            let policy = policy.to_data();
            println!("{policy:#?}");
            todo!();
            
            self.policy = Some(self.children.iter().map(|c| (c.mov, 0f32)).collect_into(&mut HashMap::<Move, f32>::new()).to_owned());

            PlayoutResult::Score {
                relative_to: pos.get_turn(),
                quality: todo!(),
            }
        })
    }

    fn expand(&mut self, pos: &Position) {
        assert_matches!(self.state, NodeState::Leaf);

        _ = fold_legal_moves(pos, &mut self.children, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Node::leaf(m));
                acc
            })
        });

        self.state = if self.children.is_empty() { NodeState::Terminal } else { NodeState::Branch };
    }
}