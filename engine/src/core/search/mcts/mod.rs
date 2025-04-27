use burn::prelude::Backend;
use eval::model::{Model, board_input, state_input};
use itertools::Itertools;
use std::assert_matches::assert_matches;
use std::fmt;
use std::ops::ControlFlow;
use std::ptr::NonNull;

use crate::core::move_iter::king::King;
use crate::core::piece::IPieceType;
use crate::core::turn::Turn;
use crate::core::{color::Color, r#move::Move, move_iter::fold_legal_moves, position::Position};

#[cfg(test)]
mod test;

pub mod eval;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Evaluation {
    Win { relative_to: Color },
    Ongoing { relative_to: Color, quality: f32 },
    Draw,
}

impl Evaluation {
    /// Returns a number between 0 and 1, where 0 is a loss and 1 is a win.
    fn to_value(&self, turn: Color) -> f32 {
        match *self {
            Evaluation::Win { relative_to } if relative_to == turn => 1.0,
            Evaluation::Win { relative_to: _ } => 0.0,
            Evaluation::Draw => 0.5,
            Evaluation::Ongoing {
                quality,
                relative_to,
            } => {
                // The quality is between -1 and 1, so we have to convert it to a 0 to 1 range.
                let quality = (quality + 1.0) / 2.0;
                if relative_to == turn {
                    quality
                } else {
                    1.0 - quality
                }
            }
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Tree {
    /// Root of the tree.
    root: Node,

    /// Stack of nodes that were selected during the selection phase.
    selection_buffer: Vec<NonNull<Branch>>,
}

impl Tree {
    pub fn new<B: Backend>(pos: &Position, model: &Model<B>) -> Self {
        Self {
            root: Node::root(pos, model),
            ..Default::default()
        }
    }

    pub fn best_move(&self) -> Move {
        self.root
            .branches
            .iter()
            .max_by(|a, b| a.visits().cmp(&b.visits()))
            .expect("Root need's to have children.")
            .mov()
    }

    pub fn grow<B: Backend>(&mut self, pos: &mut Position, model: &Model<B>) {
        let result = self.select_leaf_mut(pos, model);
        self.backpropagate(pos, result);
    }

    pub fn selected_leaf(&mut self) -> NonNull<Branch> {
        *self
            .selection_buffer
            .last_mut()
            .expect("This is only None, if root.state == Leaf, which is not the case.")
    }

    fn backpropagate(&mut self, pos: &mut Position, result: Evaluation) {
        unsafe {
            for branch in self.selection_buffer.iter_mut().rev().map(|n| n.as_mut()) {
                pos.unmake_move(branch.mov());
                branch.update(result, pos.get_turn());
            }
        }
        self.root.update(result, pos.get_turn());
    }

    pub fn select_leaf_mut<B: Backend>(
        &mut self,
        pos: &mut Position,
        model: &Model<B>,
    ) -> Evaluation {
        self.selection_buffer.clear();
        let mut current = unsafe { NonNull::from_ref(&self.root).as_mut() };
        loop {
            match current.state() {
                NodeState::Expanded => {
                    let branch = current.select_mut();
                    pos.make_move(branch.mov());
                    self.selection_buffer.push(NonNull::from_ref(branch));
                    current = branch.traverse_mut();
                }
                NodeState::Leaf => {
                    return current.expand_and_eval(pos, model);
                }
            }
        }
    }
}

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub enum NodeState {
    #[default]
    Leaf,
    Expanded,
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct ChildNode {
    node: Node,
    mov: Move,
}

impl ChildNode {
    /// Create a new leaf node.
    pub fn leaf(m: Move) -> Self {
        Self {
            node: Node::leaf(),
            mov: m,
        }
    }

    // note: we take the policy as an argument, because if we later convert this
    // tree structure to a graph, we have to consider different policies from different parents.
    fn puct(&self, cap_n_i: u32, policy: f32) -> f32 {
        let n_i = self.node.visits() as f32;

        // The quality is updated incrementally as the tree is explored.
        // Because of this, we have to divide by the number of playouts
        // to get the average quality of this node.
        // If this node has not yet been visited, we set the quality to 0.
        let exploitation = if n_i == 0.0 {
            0.0
        } else {
            self.node.value() / n_i
        };

        // todo: fine tune c.
        let c = f32::sqrt(2.0);
        let exploration = c * policy * (cap_n_i as f32).sqrt() / (1f32 + n_i);

        exploitation + exploration
    }

    fn mov(&self) -> Move {
        self.mov
    }

    fn visits(&self) -> u32 {
        self.node.visits()
    }

    fn update(&mut self, result: Evaluation, get_turn: Color) {
        self.node.update(result, get_turn)
    }
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct Branch {
    node: ChildNode,
    policy: f32,
}

impl Branch {
    fn puct(&self, cap_n_i: u32) -> f32 {
        self.node.puct(cap_n_i, self.policy)
    }

    fn new(m: Move, policy: f32) -> Self {
        Self {
            node: ChildNode::leaf(m),
            policy,
        }
    }

    fn visits(&self) -> u32 {
        self.node.visits()
    }

    fn mov(&self) -> Move {
        self.node.mov()
    }

    fn update(&mut self, result: Evaluation, get_turn: Color) {
        self.node.update(result, get_turn)
    }

    fn traverse(&self) -> &Node {
        &self.node.node
    }

    fn traverse_mut(&mut self) -> &mut Node {
        &mut self.node.node
    }

    fn node_state(&self) -> NodeState {
        self.traverse().state()
    }
}

#[derive(Clone, Default, PartialEq)]
pub struct Node {
    visits: u32,
    value: f32,
    state: NodeState,
    branches: Vec<Branch>,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("value", &self.value())
            .field("visits", &self.visits())
            .field("state", &self.state())
            .field(
                "branches",
                &self
                    .branches
                    .iter()
                    .filter(|c| c.visits() != 0)
                    .collect_vec(),
            )
            .finish()
    }
}

impl Node {
    /// Create a new root node.
    pub fn root<B: Backend>(pos: &Position, model: &Model<B>) -> Self {
        let mut result = Self::leaf();
        result.expand_and_eval(pos, model);
        result
    }

    /// Create a new leaf node.
    pub fn leaf() -> Self {
        debug_assert_eq!(
            Self::default(),
            Self {
                state: NodeState::Leaf,
                branches: Vec::new(),
                visits: 0,
                value: 0.0
            }
        );

        Self {
            state: NodeState::Leaf,
            branches: Vec::new(),
            visits: 0,
            value: 0.0,
        }
    }

    pub fn update(&mut self, eval: Evaluation, turn: Turn) {
        self.visits += 1;
        self.value += eval.to_value(turn);
    }

    /// Select the branch with the highest PUCT score.
    pub fn select_mut(&mut self) -> &mut Branch {
        assert_matches!(self.state(), NodeState::Expanded);

        self.branches
            .iter_mut()
            .max_by(|a, b| {
                let a_ucb = a.puct(self.visits);
                let b_ucb = b.puct(self.visits);
                a_ucb.partial_cmp(&b_ucb).expect("Node comparison failed!")
            })
            .expect("This is either a branch or a root node, which implies that this is not a terminal node, so there has to be atleast on child.")
    }

    /// Returns true if the node is a terminal node.
    fn is_terminal(&self) -> bool {
        self.state() == NodeState::Expanded && self.branches.is_empty()
    }

    /// Expand the node and return an evaluation of the position.
    fn expand_and_eval<B: Backend>(&mut self, pos: &Position, model: &Model<B>) -> Evaluation {
        assert_matches!(self.state(), NodeState::Leaf);

        _ = fold_legal_moves(pos, &mut self.branches, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Branch::new(m, 0.0));
                acc
            })
        });

        self.state = NodeState::Expanded;

        if self.is_terminal() {
            if pos.has_threefold_repetition()
                || pos.fifty_move_rule()
                || pos.is_insufficient_material()
            {
                return Evaluation::Draw;
            }

            let us = pos.get_turn();
            let king = pos.get_bitboard(King::ID, us);
            let nstm_attacks = pos.get_nstm_attacks();
            let in_check = !(king & nstm_attacks).is_empty();
            if in_check {
                // If in check and no moves, it's a loss for the current player
                Evaluation::Win { relative_to: !us }
            } else {
                // Stalemate
                Evaluation::Draw
            }
        } else {
            let b_in = [board_input(pos)].into();
            let s_in = [state_input(pos)].into();
            let (quality, policy) = model.forward(b_in, s_in);

            let policy = policy
                .to_data()
                .to_vec::<f32>()
                .expect("Policy could not be converted to vec.");
            let quality = quality
                .to_data()
                .to_vec::<f32>()
                .expect("Quality could not be converted to vec.");

            for branch in &mut self.branches {
                branch.policy = policy[usize::from(branch.node.mov)];
            }

            // Renormalize the policy
            let policy_sum = self.branches.iter().map(|b| b.policy).sum::<f32>();

            // Fallback to uniform distribution
            let policy_sum = if policy_sum == 0.0 {
                self.branches.len() as f32
            } else {
                policy_sum
            };

            for branch in &mut self.branches {
                branch.policy /= policy_sum;
            }

            Evaluation::Ongoing {
                relative_to: pos.get_turn(),
                quality: quality[0],
            }
        }
    }

    fn visits(&self) -> u32 {
        self.visits
    }

    fn value(&self) -> f32 {
        self.value
    }

    fn state(&self) -> NodeState {
        self.state
    }
}
