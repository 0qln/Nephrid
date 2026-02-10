use itertools::Itertools;

use crate::core::Move;
use crate::core::Position;
use crate::core::move_iter::fold_legal_moves;
use crate::core::search::mcts::eval::Policy;
use crate::core::search::mcts::eval::RawPolicy;
use crate::core::search::mcts::node::ops::ControlFlow;
use std::assert_matches::assert_matches;
use std::assert_matches::debug_assert_matches;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt;
use std::ops;
use std::rc::Rc;

#[cfg(test)]
pub mod test;

#[derive(Default, Debug, Clone)]
pub struct Tree {
    /// Root of the tree.
    root: Rc<RefCell<Node>>,
}

impl Tree {
    pub fn new() -> Self {
        Self {
            root: Rc::new(RefCell::new(Node::leaf())),
        }
    }

    pub fn advance_best(&mut self) {
        let node = {
            let root = self.root.borrow();
            let branch = root.select_best();
            branch.map(|b| b.node())
        };
        if let Some(node) = node {
            self.root = node;
        }
    }

    pub fn advance_to<F: Fn(&Branch) -> bool>(&mut self, pred: F) {
        let node = {
            let root = self.root.borrow();
            let branch = root.iter_branches().find(|x| pred(x));
            branch.map(|b| b.node())
        };
        if let Some(node) = node {
            self.root = node;
        }
    }

    /// Returns None if there are no moves.
    pub fn best_move(&self) -> Option<Move> {
        let root = self.root.borrow();
        let best = root.select_best()?;
        Some(best.mov())
    }

    pub fn best_moves(&self, threshold: Value) -> Vec<Move> {
        let root = self.root.borrow();
        root.iter_branches()
            .filter(|b| b.value() > threshold)
            .map(|b| b.mov())
            .collect_vec()
    }

    /// Returns the current principal variation.
    pub fn principal_variation(&self) -> Vec<Branch> {
        let mut buf = Vec::new();
        let mut current = self.root.clone();
        loop {
            let state = current.borrow().state();
            match state {
                NodeState::Expanded => {
                    debug_assert!(
                        !{ current.borrow().branches.is_empty() },
                        "Contradiction: NodeState == Expanded, but there are no branches."
                    );

                    // we can clone here since branch struct is very small
                    // SAFETY: This branch is only reached when NodeState == Expanded
                    let branch =
                        unsafe { current.borrow().select_best().unwrap_unchecked().clone() };
                    let node = branch.node();
                    buf.push(branch);
                    current = node;
                }
                NodeState::Leaf | NodeState::Terminal => {
                    break;
                }
            }
        }
        buf
    }

    pub fn get_root(&self) -> Rc<RefCell<Node>> {
        self.root.clone()
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
pub struct Branch {
    /// The node that this branch leads to.
    node: Rc<RefCell<Node>>,

    /// The policy of picking this branch.
    policy: f32,

    /// The move that lead to this node.
    mov: Move,
}

impl Branch {
    pub fn new(m: Move, policy: f32) -> Self {
        Self {
            node: Rc::new(RefCell::new(Node::leaf())),
            policy,
            mov: m,
        }
    }

    pub fn mov(&self) -> Move {
        self.mov
    }

    pub fn policy(&self) -> f32 {
        self.policy
    }

    pub fn node(&self) -> Rc<RefCell<Node>> {
        self.node.clone()
    }

    pub fn visits(&self) -> u32 {
        self.node.borrow().visits()
    }

    pub fn value(&self) -> Value {
        self.node.borrow().value()
    }

    pub fn node_state(&self) -> NodeState {
        self.node.borrow().state()
    }

    pub fn set_policy(&mut self, policy: f32) {
        self.policy = policy;
    }
}

#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Value(pub f32);

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl_op!(/ |l: Value, r: f32| -> f32 { l.0 / r });
impl_op!(+= |l: &mut Value, r: f32| { l.0 += r } );

impl Eq for Value {}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        f32::partial_cmp(&self.0, &other.0).unwrap_or(Ordering::Equal)
    }
}

pub type NodeRef = Rc<RefCell<Node>>;

#[derive(Clone, Default, PartialEq)]
pub struct Node {
    /// The number of times this node was visited.
    pub visits: u32,

    /// The value of this node. (~sums all the values of it's children)
    pub value: Value,

    // todo: put this behind a generic PAYLOAD parameter or something, this is currently only used
    // for training and should thus not be here in production.
    // /// win/draw/loss count
    // pub terminal_wdl: WDL,
    /// The current state of this node.
    state: NodeState,

    /// All the branches from this node.
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
    /// Sort the branches in ascending order.
    pub fn sort_by<T: Ord>(&mut self, f: impl Fn(&Branch) -> T) {
        // todo: the sorting can be done a lot more efficiently:
        // The puct score does not change very often later on, only as we start the search.
        // Also we might only need the first few branches if MPV is low.
        self.branches.sort_by_key(f);
    }

    pub fn get_branch(&self, index: usize) -> Option<&Branch> {
        self.branches.get(index)
    }

    pub fn set_branches(&mut self, branches: Vec<Branch>) {
        self.branches = branches;
    }

    pub fn iter_branches(&self) -> impl Iterator<Item = &Branch> {
        self.branches.iter()
    }

    pub fn iter_branches_mut(&mut self) -> impl Iterator<Item = &mut Branch> {
        self.branches.iter_mut()
    }

    /// Whether this node has branches.
    pub fn has_branches(&self) -> bool {
        !self.branches.is_empty()
    }

    /// The number of branches this node has
    pub fn num_branches(&self) -> usize {
        self.branches.len()
    }

    /// Create a new leaf node.
    pub fn leaf() -> Self {
        Self {
            state: NodeState::Leaf,
            branches: Vec::new(),
            visits: 0,
            value: Value(0.0),
        }
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
        F: Fn(&Branch) -> T,
        T: PartialOrd,
    {
        self.branches.into_iter().max_by(|a, b| {
            let a = transform(a);
            let b = transform(b);
            a.partial_cmp(&b).expect("Node comparison failed!")
        })
    }

    /// Returns None if there are no branches.
    pub fn take_branch<F>(self, pred: F) -> Option<Branch>
    where
        F: Fn(&Branch) -> bool,
    {
        self.branches.into_iter().find(pred)
    }

    /// Expand the node.
    pub fn expand(&mut self, pos: &Position) {
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

    /// Sets the policies of the branches.
    pub fn set_policy(&mut self, policy: &Policy) {
        assert_eq!(
            self.branches.len(),
            policy.len(),
            "There has to be exactly one policy for each branch."
        );

        for (i, branch) in self.branches.iter_mut().enumerate() {
            branch.policy = policy.get(i).unwrap();
        }
    }

    /// Sets the policies of the branches.
    pub fn set_policy_raw(&mut self, raw_policy: &RawPolicy) {
        debug_assert_matches!(
            self.state(),
            NodeState::Expanded,
            "Node has to be expanded. Terminal nodes do not have a policy."
        );

        let moves = self.branches.iter().map(|b| usize::from(b.mov()));
        let policy = Policy::from_raw(raw_policy, moves)
            .expect("Shouldn't be None, since the moves are correct for this node.");

        self.set_policy(&policy);
    }

    pub fn visits(&self) -> u32 {
        self.visits
    }

    pub fn value(&self) -> Value {
        self.value
    }

    pub fn state(&self) -> NodeState {
        self.state
    }

    pub fn set_state(&mut self, state: NodeState) {
        self.state = state
    }

    // /// The amount of wins in this and all subtrees.
    // pub fn wins(&self) -> usize {
    //     match self.state() {
    //         NodeState::Terminal => {
    //             let game_result = Evaluator::eval_terminal(self, pos)
    //         }
    //         NodeState::Expanded => self.iter_branches().map(
    //             |b| b.node().wins()
    //         ).sum()
    //     }
    // }

    ///// Applies `f` to this and all child nodes, until no more child is found or `f` returns
    ///// residual.
    //pub fn try_fold_down<B, F, R>(this: Rc<RefCell<Self>>, mut init: B, mut f: F) -> R
    //where
    //    F: FnMut(B, Rc<RefCell<Self>>) -> R,
    //    R: Try<Output = B>,
    //{
    //    init = f(init, this.clone())?;
    //    self.iter_branches()
    //        .try_fold(init, f);
    //    //while let Some(parent) = { this.borrow_mut().parent() } {
    //    //        this = parent;
    //    //        init = f(init, this.clone())?;
    //    //}
    //    R::from_output(init)
    //}
}
