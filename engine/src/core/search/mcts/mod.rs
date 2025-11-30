use itertools::Itertools;
use std::assert_matches::assert_matches;
use std::fmt;
use std::ops::ControlFlow;
use std::ptr::NonNull;

use crate::core::move_iter::king::King;
use crate::core::piece::IPieceType;
use crate::core::search::mcts::eval::model::POLICY_OUTPUTS;
use crate::core::{color::Color, r#move::Move, move_iter::fold_legal_moves, position::Position};

pub mod eval;

#[cfg(test)]
pub mod test;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GameResult {
    Win { relative_to: Color },
    Draw,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Evaluation {
    Guess {
        relative_to: Color,
        quality: f32,
        policies: Vec<f32>,
    },
    Terminal(GameResult),
}

impl GameResult {
    /// Returns a number between 0 and 1, where 0 is a loss and 1 is a win.
    fn to_value(self, turn: Color) -> f32 {
        match self {
            Self::Win { relative_to } => {
                if relative_to == turn {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Draw => 0.5,
        }
    }
}

impl Evaluation {
    /// Returns a number between 0 and 1, where 0 is a loss and 1 is a win.
    fn to_value(&self, turn: Color) -> f32 {
        match self {
            Self::Terminal(result) => result.to_value(turn),
            Evaluation::Guess {
                quality,
                relative_to,
                policies: _policies,
            } => {
                // The quality is between -1 and 1, so we have to convert it to a 0 to 1 range.
                let quality = (quality + 1.0) / 2.0;
                if *relative_to == turn { quality } else { 1.0 - quality }
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
    pub fn new<E: Evaluator>(pos: &Position, eval: &E) -> Self {
        Self {
            root: Node::root(pos, eval),
            ..Default::default()
        }
    }

    /// Returns None if there are no moves.
    pub fn best_move(&self) -> Option<Move> {
        let best = self.root.select_best()?;
        Some(best.mov())
    }

    /// Returns the current principal variation.
    pub fn principal_variation(&self) -> Vec<&Branch> {
        let mut buf = Vec::new();
        let mut current = &self.root;
        loop {
            match current.state() {
                NodeState::Expanded => {
                    debug_assert!(
                        !current.branches.is_empty(),
                        "Contradiction: NodeState == Expanded, but there are no branches."
                    );

                    // SAFETY: This branch is only reached when NodeState == Expanded
                    let branch = unsafe { current.select_best().unwrap_unchecked() };
                    buf.push(branch);
                    current = branch.traverse();
                }
                NodeState::Leaf | NodeState::Terminal => {
                    break;
                }
            }
        }
        buf
    }

    pub fn grow<E: Evaluator>(&mut self, pos: &mut Position, eval: &E) {
        let evaluation = self.select_leaf_mut(pos, eval);
        self.backpropagate(pos, &evaluation);
    }

    fn backpropagate(&mut self, pos: &mut Position, eval: &Evaluation) {
        unsafe {
            for branch in self.selection_buffer.iter_mut().rev().map(|n| n.as_mut()) {
                pos.unmake_move(branch.mov());
                let value = eval.to_value(pos.get_turn());
                branch.update_node(value);
            }
        }
        let value = eval.to_value(pos.get_turn());
        self.root.update(value);
    }

    /// Walk down the tree and select the branches with the highest score, until we find a leaf
    /// node. sono-ato, expand the leaf and return the evaluation of that leaf.
    pub fn select_leaf_mut<E: Evaluator>(&mut self, pos: &mut Position, model: &E) -> Evaluation {
        self.selection_buffer.clear();
        let mut current = unsafe { NonNull::from_ref(&self.root).as_mut() };
        loop {
            match current.state() {
                NodeState::Expanded => {
                    debug_assert!(
                        !current.branches.is_empty(),
                        "Contradiction: NodeState == Expanded, but there are no branches."
                    );

                    // SAFETY: This branch is only reached when NodeState == Expanded
                    let branch = unsafe { current.select_puct_mut().unwrap_unchecked() };
                    pos.make_move(branch.mov());
                    self.selection_buffer.push(NonNull::from_ref(branch));
                    current = branch.traverse_mut();
                }
                NodeState::Leaf => {
                    return current.expand_and_eval(pos, model);
                }
                NodeState::Terminal => {
                    return current.eval(pos, model);
                }
            }
        }
    }

    pub fn get_root(&self) -> &Node {
        &self.root
    }
}

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub enum NodeState {
    /// A leaf is an untouched node.
    #[default]
    Leaf,
    /// An expanded node is a node which has been analized and to be found to have children.
    Expanded,
    /// A terminal node is a node which has been analized and to be found to have no children.
    Terminal,
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct ChildNode {
    node: Node,
    mov: Move,
}

impl ChildNode {
    /// Create a new leaf node with the move that leads to this child node.
    pub fn leaf(m: Move) -> Self {
        Self { node: Node::leaf(), mov: m }
    }

    // note: we take the policy as an argument, because if we later convert this
    // tree structure to a graph, we have to consider different policies from different parents.
    fn puct(&self, cap_n_i: u32, policy: f32) -> f32 {
        let n_i = self.node.visits() as f32;

        // The quality is updated incrementally as the tree is explored.
        // Because of this, we have to divide by the number of playouts
        // to get the average quality of this node.
        // If this node has not yet been visited, we set the quality to 0.
        let exploitation = if n_i == 0.0 { 0.0 } else { self.node.value() / n_i };

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

    fn update(&mut self, value: f32) {
        self.node.update(value)
    }
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct Branch {
    node: ChildNode,
    policy: f32,
}

impl Branch {
    pub fn puct(&self, cap_n_i: u32) -> f32 {
        self.node.puct(cap_n_i, self.policy)
    }

    pub fn new(m: Move, policy: f32) -> Self {
        Self { node: ChildNode::leaf(m), policy }
    }

    pub fn visits(&self) -> u32 {
        self.node.visits()
    }

    pub fn mov(&self) -> Move {
        self.node.mov()
    }

    pub fn update_node(&mut self, value: f32) {
        self.node.update(value)
    }

    pub fn traverse(&self) -> &Node {
        &self.node.node
    }

    pub fn traverse_mut(&mut self) -> &mut Node {
        &mut self.node.node
    }

    pub fn node_state(&self) -> NodeState {
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

pub trait Evaluator {
    /// Returns: (quality [-1;1], policy [over ALL moves])
    fn evaluate(&self, pos: &Position) -> (f32, [f32; POLICY_OUTPUTS]);
}

impl Node {
    /// Create a new root node.
    pub fn root<E: Evaluator>(pos: &Position, eval: &E) -> Self {
        let mut result = Self::leaf();
        result.expand_and_eval(pos, eval);
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

    /// Update the node with the result of an evaluation.
    pub fn update(&mut self, value: f32) {
        self.visits += 1;
        self.value += value;
    }

    /// Select the branch with the highest PUCT score.
    /// Returns None if there are no branches.
    pub fn select_puct_mut(&mut self) -> Option<&mut Branch> {
        let visits = self.visits();
        self.select_mut(|b| b.puct(visits))
    }

    /// Select the branch with the most visits.
    /// Returns None if there are no branches.
    pub fn select_best(&self) -> Option<&Branch> {
        self.select(|b| b.visits())
    }

    /// Returns None if there are no branches.
    pub fn select<F, T>(&self, transform: F) -> Option<&Branch>
    where
        F: Fn(&Branch) -> T,
        T: PartialOrd,
    {
        self.branches.iter().max_by(|a, b| {
            let a = transform(a);
            let b = transform(b);
            a.partial_cmp(&b).expect("Node comparison failed!")
        })
    }

    /// Returns None if there are no branches.
    pub fn select_mut<F, T>(&mut self, transform: F) -> Option<&mut Branch>
    where
        F: Fn(&Branch) -> T,
        T: PartialOrd,
    {
        self.branches.iter_mut().max_by(|a, b| {
            let a = transform(a);
            let b = transform(b);
            a.partial_cmp(&b).expect("Node comparison failed!")
        })
    }

    /// Returns an evaluation of the position.
    pub fn eval<E: Evaluator>(&self, pos: &Position, evaluator: &E) -> Evaluation {
        assert_matches!(self.state(), NodeState::Expanded | NodeState::Terminal);

        if pos.has_threefold_repetition() || pos.fifty_move_rule() || pos.is_insufficient_material()
        {
            return Evaluation::Terminal(GameResult::Draw);
        }

        if self.branches.is_empty() {
            let us = pos.get_turn();
            let king = pos.get_bitboard(King::ID, us);
            let nstm_attacks = pos.get_nstm_attacks();
            let in_check = !(king & nstm_attacks).is_empty();
            if in_check {
                // If in check and no moves, it's a loss for the current player
                Evaluation::Terminal(GameResult::Win { relative_to: !us })
            } else {
                // Stalemate
                Evaluation::Terminal(GameResult::Draw)
            }
        } else {
            let (quality, raw_policy) = evaluator.evaluate(pos);

            let mut policies = Vec::<f32>::new();
            for branch in &self.branches {
                policies.push(raw_policy[usize::from(branch.node.mov)]);
            }

            // Renormalize the policy to a sum of 1, since not all of the probabilities
            // were assigned to moves that are actually playable in this position:

            let policy_sum = {
                let sum = policies.iter().sum();
                if sum == 0.0 {
                    // Fallback to uniform distribution
                    policies.len() as f32
                } else {
                    sum
                }
            };
            for policy in &mut policies {
                *policy /= policy_sum;
            }

            // Evaluator should return a probability distribution.
            let f32_eq = |a: f32, b: f32, e: f32| f32::abs(a - b) < e;
            debug_assert!(f32_eq(policies.iter().sum::<f32>(), 1., 0.001));

            Evaluation::Guess {
                relative_to: pos.get_turn(),
                quality,
                policies,
            }
        }
    }

    /// Expand the node.
    fn expand(&mut self, pos: &Position) {
        assert_matches!(self.state(), NodeState::Leaf);

        _ = fold_legal_moves(pos, &mut self.branches, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Branch::new(m, 0.0));
                acc
            })
        });

        self.state = if self.branches.is_empty() {
            NodeState::Terminal
        } else {
            NodeState::Expanded
        };
    }

    /// Expand the node and return an evaluation of the position.
    fn expand_and_eval<E: Evaluator>(&mut self, pos: &Position, e: &E) -> Evaluation {
        self.expand(pos);

        let eval = self.eval(pos, e);

        if let Evaluation::Guess { policies, .. } = &eval {
            for (i, branch) in self.branches.iter_mut().enumerate() {
                branch.policy = policies[i];
            }
        }

        eval
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
