use itertools::Itertools;
use ringbuf::{StaticRb, traits::*};
use std::assert_matches::assert_matches;
use std::fmt;
use std::ops::ControlFlow;
use std::ptr::NonNull;

use crate::core::depth::Depth;
use crate::core::position::CheckState;
use crate::core::search::mcts::eval::model::POLICY_OUTPUTS;
use crate::core::{color::Color, r#move::Move, move_iter::fold_legal_moves, position::Position};

pub mod eval;

pub mod test;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GameResult {
    Win { relative_to: Color },
    Draw,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Guess {
    relative_to: Color,
    quality: f32,
    policies: Vec<f32>,
}

impl Guess {
    pub fn policies(&self) -> &[f32] {
        &self.policies
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Evaluation {
    // we will go further and have a gues about this game.
    Guess(Guess),
    // we cannot go any further.
    Terminal(GameResult),
    // we don't feel like going any further.
    Nope,
}

impl GameResult {
    /// Returns a number between 0 and 1, where 0 is a loss and 1 is a win.
    const fn to_value(self, turn: Color) -> f32 {
        match self {
            Self::Win { relative_to } => {
                if relative_to.v() == turn.v() {
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
            Self::Nope => GameResult::Draw.to_value(turn),
            Self::Guess(Guess {
                quality,
                relative_to,
                policies: _policies,
            }) => {
                // The quality is between -1 and 1, so we have to convert it to a 0 to 1 range.
                let quality = (quality + 1.0) / 2.0;
                if *relative_to == turn { quality } else { 1.0 - quality }
            }
        }
    }
}

pub trait Searcher {}

/// # ja;lsdkjfh
///
/// This statemachine should hold all the info that is required to search.
///
/// This is the implementation for the nn-backed eval model searcher. (e.g. lc0, a0)
///
/// ## Multi pv.
///
/// Use multi pv lines with batched evaluation, if we have access to GPU and can parallelize. If we
/// don't have access to hardware accell, resort to using a backend like WebGPU, or NdArray, and just do
/// TreeSearcher<MPV=1>.
///
/// The pv lines are the top few lines sorted by puct. (e.g.:
/// ```rust
/// self.multi_select(|b| b.puct(self))
/// ```
/// )
///
/// Idea: Maybe we can have lines that are a lot more explorative next to the main puct line?
pub struct TreeSearcher<'a, B: Backend, E: Evaluator, L: Limiter, const MPV: usize> {
    /// Stack of nodes that were selected during the selection phase, for each principal line.
    selection_buffer: [Vec<NonNull<Branch>>; MPV],

    /// History info that the eval model can use.
    history_info: [StaticRb<BoardInputTensor<B>, BOARD_INPUT_HISTORY>; MPV],

    /// The position that will be edited during the selection and backpropagatation.
    pos: Position,

    /// Evaluator to evaluate non-terminating nodes.
    evaluater: E,

    /// Limiter to dicide whether to keep searching non-terminating nodes.
    limiter: L,

    /// The tree to be searched.
    tree: &'a mut Tree,
}

impl<const MPV: usize> TreeSearcher<MPV> {
    pub fn grow<E: Evaluator, L: Limiter>(&mut self) {
        let evaluation = self.select_leaf_mut(pos, eval, &mut line_history, l);
        self.backpropagate(pos, &evaluation);
    }

    /// Walk down the tree and select the branches with the highest score, until we find a leaf
    /// node. sono-ato, expand the leaf and return the evaluation of that leaf.
    fn select_leafs_mut<E: Evaluator, L: Limiter>(&mut self) -> [(Evaluation); MPV] {
        self.selection_buffer.clear();

        let mut current = unsafe { NonNull::from_ref(&self.tree.root()).as_mut() };
        let mut depth = Depth::MIN;
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
                    model.push(pos);
                    self.selection_buffer.push(NonNull::from_ref(branch));
                    current = branch.traverse_mut();
                }
                NodeState::Leaf => {
                    return current.expand_and_eval(pos, model, l, depth);
                }
                NodeState::Terminal => {
                    return current.eval(pos, model, l, depth);
                }
            }
            depth += 1;
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Tree {
    /// Root of the tree.
    root: Node,
}

impl Tree {
    pub fn new<E: Evaluator, L: Limiter>(pos: &Position, eval: &E, l: &L) -> Self {
        Self {
            root: Node::root(pos, eval, l),
            ..Default::default()
        }
    }

    pub fn advance_best(self) -> Option<Tree> {
        let best = self.root.take_best()?;
        Some(Tree { root: best.node.node })
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

    pub fn grow<E: Evaluator, L: Limiter>(&mut self, pos: &mut Position, eval: &mut E, l: &L) {
        let evaluation = self.select_leaf_mut(pos, eval, l);
        eval.clear();
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
    pub fn select_leaf_mut<E: Evaluator, L: Limiter>(
        &mut self,
        pos: &mut Position,
        model: &mut E,
        l: &L,
    ) -> Evaluation {
        self.selection_buffer.clear();
        model.push(pos);
        let mut current = unsafe { NonNull::from_ref(&self.root).as_mut() };
        let mut depth = Depth::MIN;
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
                    model.push(pos);
                    self.selection_buffer.push(NonNull::from_ref(branch));
                    current = branch.traverse_mut();
                }
                NodeState::Leaf => {
                    return current.expand_and_eval(pos, model, l, depth);
                }
                NodeState::Terminal => {
                    return current.eval(pos, model, l, depth);
                }
            }
            depth += 1;
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

pub trait Evaluator<const X: usize> {
    /// prepare the newest position that needs to be evaluated.
    fn push(&mut self, pos: &Position) -> ();

    /// pop the latest position that was prepared.
    fn pop(&mut self) -> ();

    /// clear all prepaered
    fn clear(&mut self) -> ();

    /// Returns: (quality [-1;1], policy [over ALL moves])
    fn evaluate(&self) -> [(f32, [f32; POLICY_OUTPUTS]); X];
}

pub trait Limiter {
    /// Returns: Whether to stop searching.
    fn should_stop(&self, pos: &Position, depth: Depth) -> bool;
}

#[derive(Default, Debug)]
pub struct NoopLimiter;

impl Limiter for NoopLimiter {
    fn should_stop(&self, _pos: &Position, _depth: Depth) -> bool {
        false
    }
}

impl Node {
    /// Create a new root node.
    pub fn root<E: Evaluator, L: Limiter>(pos: &Position, eval: &E, l: &L) -> Self {
        let mut result = Self::leaf();
        result.expand_and_eval(pos, eval, l, Depth::MIN);
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

    pub fn take_best(self) -> Option<Branch> {
        self.take(|b| b.visits())
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

    /// Returns None if there are no branches.
    pub fn take<F, T>(self, transform: F) -> Option<Branch>
    where
        F: Fn(Branch) -> T,
        T: PartialOrd,
    {
        self.branches.into_iter().max_by(|a, b| {
            let a = transform(a);
            let b = transform(b);
            a.partial_cmp(&b).expect("Node comparison failed!")
        })
    }

    /// Returns an evaluation of the position.
    pub fn eval<E: Evaluator, L: Limiter>(
        &self,
        pos: &Position,
        evaluator: &E,
        limiter: &L,
        depth: Depth,
    ) -> Evaluation {
        assert_matches!(self.state(), NodeState::Expanded | NodeState::Terminal);

        // First check if the position is a normal game ending.
        if self.branches.is_empty() {
            return if pos.get_check_state() != CheckState::None {
                // If in check and no moves, it's a loss for the current player
                Evaluation::Terminal(GameResult::Win { relative_to: !pos.get_turn() })
            } else {
                // Stalemate
                Evaluation::Terminal(GameResult::Draw)
            };
        }

        // Then check if the position has reached some of the extra-rule endings.
        if pos.has_threefold_repetition() || pos.fifty_move_rule() || pos.is_insufficient_material()
        {
            return Evaluation::Terminal(GameResult::Draw);
        }

        // Then check if we are even interested in searching this line any further.
        if limiter.should_stop(pos, depth) {
            return Evaluation::Nope;
        }

        // Otherwise guess a score.
        {
            let (quality, raw_policy) = evaluator.evaluate();

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

            Evaluation::Guess(Guess {
                relative_to: pos.get_turn(),
                quality,
                policies,
            })
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
    fn expand_and_eval<E: Evaluator, L: Limiter>(
        &mut self,
        pos: &Position,
        e: &E,
        l: &L,
        depth: Depth,
    ) -> Evaluation {
        self.expand(pos);

        let eval = self.eval(pos, e, l, depth);

        if let Evaluation::Guess(Guess { policies, .. }) = &eval {
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
