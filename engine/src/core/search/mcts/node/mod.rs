use crate::core::search::mcts::node::node_state::{
    ExpandedRefSwitch, ExpandedSwitch, NodeState, NodeSwitch, Unknown,
};
use itertools::Itertools;
use std::mem;

use crate::core::{
    Move, Position,
    move_iter::fold_legal_moves,
    search::mcts::{
        eval::{self, Policy, RawPolicy},
        node::{
            node_state::{Branching, Evaluated, Leaf, Terminal},
            ops::ControlFlow,
        },
    },
};
use std::{cell::RefCell, cmp::Ordering, fmt, marker::PhantomData, ops, rc::Rc};

#[cfg(test)]
pub mod test;

#[derive(Default, Debug, Clone)]
pub struct Tree {
    /// Root of the tree.
    root: RtNodeRef,
}

impl Tree {
    pub fn new(root: RtNodeRef) -> Self {
        Self { root }
    }

    pub fn advance_best(&mut self) {
        let node = {
            let root_borrow = self.root.borrow();
            if let Some(branched) = root_borrow.branching() {
                Some(branched.select_best().node())
            }
            else if let Some(evaluated) = root_borrow.evaluated() {
                Some(evaluated.select_best().node())
            }
            else {
                None
            }
        };

        if let Some(node) = node {
            self.root = node;
        }
    }

    pub fn advance_to<F: Fn(&Branch) -> bool>(&mut self, pred: F) {
        let node = {
            let root_borrow = self.root.borrow();
            let branch = root_borrow.branches().iter().find(|x| pred(x));
            branch.map(|b| b.node())
        };

        if let Some(node) = node {
            self.root = node;
        }
    }

    /// Returns None if there are no moves.
    pub fn best_move(&self) -> Option<Move> {
        let root = self.root.borrow();

        // Downcast securely using the typestate pattern
        if let Some(branching) = root.branching() {
            Some(branching.select_best().mov())
        }
        else if let Some(evaluated) = root.evaluated() {
            Some(evaluated.select_best().mov())
        }
        else {
            None
        }
    }

    pub fn best_moves(&self, threshold: Value) -> Vec<Move> {
        let root = self.root.borrow();
        root.branches()
            .iter()
            .filter(|b| b.value() > threshold)
            .map(|b| b.mov())
            .collect_vec()
    }

    /// Returns the current principal variation.
    pub fn principal_variation(&self) -> Path {
        let mut buf = Vec::new();
        let mut current = self.root.clone();

        loop {
            let next_branch = {
                let borrow = current.borrow();
                if borrow.branches().is_empty() {
                    None
                }
                else {
                    borrow.branches().iter().max_by_key(|b| b.visits()).cloned()
                }
            };

            match next_branch {
                Some(branch) => {
                    let node = branch.node();
                    buf.push(branch);
                    current = node;
                }
                None => break,
            }
        }
        Path(buf)
    }

    /// Returns the number of nodes in this tree
    pub fn size(&self) -> usize {
        self.get_root().borrow().subtree_size()
    }

    /// Returns the max depth of the tree.
    pub fn maxdepth(&self) -> usize {
        self.get_root().borrow().subtree_maxdepth()
    }

    /// Returns the min depth of the tree.
    pub fn mindepth(&self) -> usize {
        self.get_root().borrow().subtree_mindepth()
    }

    pub fn get_root(&self) -> RtNodeRef {
        self.root.clone()
    }
}

pub struct Path(pub Vec<Branch>);

impl Path {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut moves = self.0.iter().map(|x| x.mov().to_string());
        f.write_str(&moves.join(" "))
    }
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct Branch {
    /// The node that this branch leads to.
    node: RtNodeRef,

    /// The policy of picking this branch.
    policy: f32,

    /// The move that lead to this node.
    mov: Move,
}

impl Branch {
    pub fn new(mov: Move, policy: f32, node: RtNodeRef) -> Self {
        Self { node, policy, mov }
    }

    pub fn mov(&self) -> Move {
        self.mov
    }

    pub fn policy(&self) -> f32 {
        self.policy
    }

    pub fn visits(&self) -> u32 {
        self.node.borrow().visits()
    }

    pub fn value(&self) -> Value {
        self.node.borrow().value()
    }

    pub fn set_policy(&mut self, policy: f32) {
        self.policy = policy;
    }

    pub fn node(&self) -> RtNodeRef {
        self.node.clone()
    }
}

/// The value of a node.
#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Value(pub f32);

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl_op!(/ |l: Value, r: f32| -> f32 { l.0 / r });
impl_op!(+= |l: &mut Value, r: eval::Value| { l.0 += r.v() } );

impl Eq for Value {}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        f32::partial_cmp(&self.0, &other.0).unwrap_or(Ordering::Equal)
    }
}

/// A node reference with compile time information about the state.
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct CtNodeRef<S: node_state::Any> {
    node: Rc<RefCell<Node<S>>>,
}

impl<S: node_state::Any> CtNodeRef<S> {
    pub fn new(node: Node<S>) -> Self {
        Self {
            node: Rc::new(RefCell::new(node)),
        }
    }
}

impl CtNodeRef<Leaf> {
    pub fn expand(mut self, pos: &Position) -> ExpandedRefSwitch {
        let leaf = self.node.replace(Default::default());
        let expanded = leaf.expand(pos);
        match expanded {
            ExpandedSwitch::Terminal(node) => ExpandedRefSwitch::Terminal(Self { node }),
            ExpandedSwitch::Branching(node) => ExpandedRefSwitch::Branching(Self { node }),
        }
    }
}

/// A node reference with runtime infomration about the state.
#[derive(Clone, Debug)]
pub struct RtNodeRef {
    node: CtNodeRef<Unknown>,
    state: NodeState,
}

impl RtNodeRef {
    pub fn new<S: node_state::Valid>(node: Node<S>) -> Self {
        Self::from_ct(CtNodeRef::new(node))
    }

    pub fn from_ct<S: node_state::Valid>(node: CtNodeRef<S>) -> Self {
        Self {
            // SAFETY: `Node<>` is #[repr(transparent)], so we can just transmute it into
            // another state.
            node: unsafe { mem::transmute(node) },
            state: S::state(),
        }
    }

    pub fn into_ct(self) -> NodeSwitch {
        // SAFETY: `Node<>` is #[repr(transparent)], so we can just transmute it into
        // another state.
        unsafe {
            match self.node_state() {
                NodeState::Leaf => NodeSwitch::Leaf(mem::transmute(self.node)),
                NodeState::Branching => NodeSwitch::Branching(mem::transmute(self.node)),
                NodeState::Terminal => NodeSwitch::Terminal(mem::transmute(self.node)),
                NodeState::Evaluated => NodeSwitch::Evaluated(mem::transmute(self.node)),
            }
        }
    }

    pub fn state(&self) -> NodeState {
        self.state
    }

    // pub fn expand(&mut self) {
    //     if let NodeSwitch::Leaf(leaf) = self.into_ct() {
    //         match leaf.expand(pos) {
    //             ExpandedSwitch::Terminal(node) => {
    //                 self.node_state = NodeState::Terminal;
    //                 self.node = node;
    //             }
    //             ExpandedSwitch::Branching(node) => {
    //                 self.node_state = NodeState::Branching;
    //                 self.node = node;
    //             }
    //         }
    //     }
    //     else {
    //         // todo: maybe log an error or something?
    //     }
    // }
}

#[derive(Clone, Default, PartialEq)]
pub struct NodeData {
    /// All the branches from this node.
    branches: Vec<Branch>,

    /// The number of times this node was visited.
    visits: u32,

    /// The value of this node. (~sums all the values of it's children)
    value: Value,
}

impl NodeData {
    pub fn visits(&self) -> u32 {
        self.visits
    }

    pub fn value(&self) -> Value {
        self.value
    }

    pub fn branches(&self) -> &[Branch] {
        &self.branches
    }

    pub fn update(&mut self, value: eval::Value) {
        self.visits += 1;
        self.value += value;
    }

    pub fn subtree_size(&self) -> usize {
        1 + self
            .branches()
            .iter()
            .map(|b| b.node().borrow().subtree_size())
            .sum::<usize>()
    }

    pub fn subtree_maxdepth(&self) -> usize {
        self.branches()
            .iter()
            .map(|b| 1 + b.node().borrow().subtree_maxdepth())
            .max()
            .unwrap_or(0)
    }

    pub fn subtree_mindepth(&self) -> usize {
        self.branches()
            .iter()
            .map(|b| 1 + b.node().borrow().subtree_mindepth())
            .min()
            .unwrap_or(0)
    }

    fn new_leaf() -> NodeData {
        Self::default()
    }
}

pub mod node_state {
    use super::{CtNodeRef, Node};

    #[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
    pub enum NodeState {
        #[default]
        Leaf,
        Branching,
        Terminal,
        Evaluated,
    }

    #[derive(Debug, PartialEq, Eq, Clone)]
    pub enum NodeSwitch {
        Leaf(CtNodeRef<Leaf>),
        Branching(CtNodeRef<Branching>),
        Terminal(CtNodeRef<Terminal>),
        Evaluated(CtNodeRef<Evaluated>),
    }

    pub enum ExpandedSwitch {
        Terminal(Node<Terminal>),
        Branching(Node<Branching>),
    }

    pub enum ExpandedRefSwitch {
        Terminal(CtNodeRef<Terminal>),
        Branching(CtNodeRef<Branching>),
    }

    pub trait Any {}

    pub trait Valid: Any {
        fn state() -> NodeState;
    }

    pub trait Expanded: Any {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Leaf;
    impl Any for Leaf {}
    impl Valid for Leaf {
        fn state() -> NodeState {
            NodeState::Leaf
        }
    }

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Terminal;
    impl Any for Terminal {}
    impl Valid for Terminal {
        fn state() -> NodeState {
            NodeState::Terminal
        }
    }
    impl Expanded for Terminal {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Branching;
    impl Any for Branching {}
    impl Valid for Branching {
        fn state() -> NodeState {
            NodeState::Branching
        }
    }
    impl Expanded for Branching {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Evaluated;
    impl Any for Evaluated {}
    impl Valid for Evaluated {
        fn state() -> NodeState {
            NodeState::Evaluated
        }
    }

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Unknown;
    impl Any for Unknown {}
}

#[repr(transparent)]
#[derive(Clone, Default, PartialEq)]
pub struct Node<State: node_state::Any> {
    /// The data.
    data: NodeData,

    /// The current state of this node.
    _state: PhantomData<State>,
}

impl<S: node_state::Any> fmt::Debug for Node<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("value", &self.value())
            .field("visits", &self.visits())
            .field(
                "branches",
                &self
                    .data
                    .branches
                    .iter()
                    .filter(|c| c.visits() != 0)
                    .collect_vec(),
            )
            .finish()
    }
}

impl Node<Leaf> {
    /// Expand the node, consuming the Leaf and returning the dynamic AnyNode
    /// variant.
    pub fn expand(mut self, pos: &Position) -> ExpandedSwitch {
        _ = fold_legal_moves(pos, &mut self.data.branches, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Branch::new(
                    m,
                    0.0,
                    RtNodeRef::new(Node::<Leaf>::new_leaf()),
                ));
                acc
            })
        });

        if self.data.branches.is_empty() {
            // SAFETY: We just expanded the moves and there are none, so the data has to be
            // of a Terminal node.
            ExpandedSwitch::Terminal(unsafe { Node::<Terminal>::new(self.data) })
        }
        else {
            // SAFETY: We just expanded the moves and there are some, so the data has to be
            // of a Branching node.
            ExpandedSwitch::Branching(unsafe { Node::<Branching>::new(self.data) })
        }
    }

    fn new_leaf() -> Self {
        // SAFETY: The Node data is of a leaf.
        unsafe { Node::<Leaf>::new(NodeData::new_leaf()) }
    }
}

impl<S: node_state::Expanded> Node<S> {
    pub fn has_branches(&self) -> bool {
        !self.data.branches.is_empty()
    }
}

impl Node<Branching> {
    pub fn branches(&self) -> &[Branch] {
        &self.data.branches
    }

    pub fn sort_by<T: Ord>(&mut self, f: impl Fn(&Branch) -> T) {
        self.data.branches.sort_by_key(f);
    }

    pub fn get_branch(&self, index: usize) -> Option<&Branch> {
        self.data.branches.get(index)
    }

    pub fn select_best(&self) -> &Branch {
        self.select(|b| b.visits())
    }

    pub fn take_best(self) -> Branch {
        self.take(|b| b.visits())
    }

    pub fn select<F, T>(&self, transform: F) -> &Branch
    where
        F: Fn(&Branch) -> T,
        T: PartialOrd,
    {
        self.data
            .branches
            .iter()
            .max_by(|a, b| {
                let a = transform(a);
                let b = transform(b);
                a.partial_cmp(&b).expect("Node comparison failed!")
            })
            .expect("An expanded node has to have atleast one branch.")
    }

    pub fn select_mut<F, T>(&mut self, transform: F) -> &mut Branch
    where
        F: Fn(&Branch) -> T,
        T: PartialOrd,
    {
        self.data
            .branches
            .iter_mut()
            .max_by(|a, b| {
                let a = transform(a);
                let b = transform(b);
                a.partial_cmp(&b).expect("Node comparison failed!")
            })
            .expect("An expanded node has to have atleast one branch.")
    }

    pub fn take<F, T>(self, transform: F) -> Branch
    where
        F: Fn(&Branch) -> T,
        T: PartialOrd,
    {
        self.data
            .branches
            .into_iter()
            .max_by(|a, b| {
                let a = transform(a);
                let b = transform(b);
                a.partial_cmp(&b).expect("Node comparison failed!")
            })
            .expect("An expanded node has to have atleast one branch.")
    }
}

impl Node<Branching> {
    pub fn set_policy(mut self, policy: &Policy) -> Node<Evaluated> {
        assert_eq!(
            self.branches().len(),
            policy.len(),
            "There has to be exactly one policy for each branch."
        );

        for (i, branch) in self.data.branches.iter_mut().enumerate() {
            branch.set_policy(policy.get(i).unwrap());
        }

        // SAFETY: we just set the policy for each branch. It has to be valid.
        unsafe { Node::<Evaluated>::new(self.data) }
    }

    pub fn set_policy_raw(self, raw_policy: &RawPolicy) -> Node<Evaluated> {
        let policy = {
            let moves = self.branches().iter().map(|b| usize::from(b.mov()));
            Policy::from_raw(raw_policy, moves)
                .expect("Shouldn't be None, since the moves are correct for this node.")
        };

        self.set_policy(&policy)
    }
}

impl<S: node_state::Valid> Node<S> {
    pub fn state() -> NodeState {
        S::state()
    }
}

impl<S: node_state::Any> Node<S> {
    /// Construct a new node.
    ///
    /// # Safety
    ///
    /// The caller has to make sure that `data` contains valid data to be in
    /// state `S`.
    unsafe fn new(data: NodeData) -> Self {
        Self { data, _state: PhantomData }
    }
}
