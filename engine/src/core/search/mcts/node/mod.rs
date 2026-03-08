use crate::core::search::mcts::node::node_state::{
    ExpandedRefSwitch, ExpandedSwitch, HasBranches, NodeState, NodeSwitch, Unknown,
};
use itertools::Itertools;
use std::{cell::Cell, mem, ops::Deref};

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

#[derive(Debug, Clone)]
pub struct Tree {
    /// Root of the tree.
    root: RtNodeRef,

    // todo: incrementally computed versions of the compute_subtree_* functions.
    size: usize,
    mindepth: u16,
    maxdepth: u16,
}

impl Default for Tree {
    fn default() -> Self {
        Self {
            root: Default::default(),
            size: 1,
            mindepth: 1,
            maxdepth: 1,
        }
    }
}

impl Tree {
    pub fn advance_best(&mut self) {
        fn map(node: CtNodeRef<impl HasBranches>) -> RtNodeRef {
            node.borrow().select_best().node()
        }

        let node = {
            match self.root.clone().into_ct() {
                NodeSwitch::Branching(node) => Some(map(node)),
                NodeSwitch::Evaluated(node) => Some(map(node)),
                _ => None,
            }
        };

        if let Some(node) = node {
            self.root = node;
        }
    }

    pub fn advance_to<F: Fn(&Branch) -> bool>(&mut self, pred: F) {
        fn map(
            node: CtNodeRef<impl HasBranches>,
            pred: impl Fn(&Branch) -> bool,
        ) -> Option<RtNodeRef> {
            node.borrow().find_branch(|x| pred(x)).map(|b| b.node())
        }

        let node = {
            match self.root.clone().into_ct() {
                NodeSwitch::Branching(node) => Some(map(node, pred)),
                NodeSwitch::Evaluated(node) => Some(map(node, pred)),
                _ => None,
            }
        };

        if let Some(node) = node.flatten() {
            self.root = node;
        }
    }

    /// Returns None if there are no moves.
    pub fn best_move(&self) -> Option<Move> {
        fn map(node: CtNodeRef<impl HasBranches>) -> Move {
            node.borrow().select_best().mov()
        }

        match self.root.clone().into_ct() {
            NodeSwitch::Branching(node) => Some(map(node)),
            NodeSwitch::Evaluated(node) => Some(map(node)),
            _ => None,
        }
    }

    pub fn best_moves(&self, threshold: Value) -> Vec<Move> {
        fn map(node: CtNodeRef<impl HasBranches>, threshold: Value) -> Vec<Move> {
            node.borrow()
                .branches()
                .iter()
                .filter(|b| b.value() > threshold)
                .map(|b| b.mov())
                .collect_vec()
        }

        match self.root.clone().into_ct() {
            NodeSwitch::Branching(node) => map(node, threshold),
            NodeSwitch::Evaluated(node) => map(node, threshold),
            _ => vec![],
        }
    }

    /// Returns the current principal variation. The branches are clones, but
    /// contain valid references to their nodes.
    pub fn principal_variation(&self) -> Path {
        let mut buf = Vec::with_capacity(self.mindepth().into());
        let mut current = self.root.clone();

        loop {
            let branch = match current.into_ct() {
                NodeSwitch::Branching(node) => Some(node.borrow().select_best().clone()),
                NodeSwitch::Evaluated(node) => Some(node.borrow().select_best().clone()),
                _ => None,
            };

            match branch {
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
    pub fn compute_size(&self) -> usize {
        self.get_root().borrow().data.compute_subtree_size()
    }

    /// Returns the max depth of the tree.
    pub fn compute_maxdepth(&self) -> usize {
        self.get_root().borrow().data.compute_subtree_maxdepth()
    }

    /// Returns the min depth of the tree.
    pub fn compute_mindepth(&self) -> usize {
        self.get_root().borrow().data.compute_subtree_mindepth()
    }

    pub fn get_root(&self) -> RtNodeRef {
        self.root.clone()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn mindepth(&self) -> u16 {
        self.mindepth
    }

    pub fn maxdepth(&self) -> u16 {
        self.maxdepth
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

#[derive(Clone, Default, Debug)]
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

#[derive(Debug)]
pub struct NodeInner<S: node_state::Any> {
    state: Cell<NodeState>,
    data: RefCell<Node<S>>,
}

/// A node reference with compile time information about the state.
#[repr(transparent)]
#[derive(Debug)]
pub struct CtNodeRef<S: node_state::Any> {
    inner: Rc<NodeInner<S>>,
}

impl<S: node_state::Any + Clone> Clone for CtNodeRef<S> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl<S: node_state::Any> Deref for CtNodeRef<S> {
    type Target = RefCell<Node<S>>;

    fn deref(&self) -> &Self::Target {
        &self.inner.data
    }
}

impl<S: node_state::Valid> CtNodeRef<S> {
    pub fn new(node: Node<S>) -> Self {
        Self {
            inner: Rc::new(NodeInner {
                state: Cell::new(S::state()),
                data: RefCell::new(node),
            }),
        }
    }
}

impl<S: node_state::Valid> CtNodeRef<S> {
    /// Try to transmute this into the specified state. Retruns none if the
    /// state is not the target state.
    pub fn try_into<Target: node_state::Valid>(self) -> Option<CtNodeRef<Target>> {
        if S::state() == Target::state() {
            Some(unsafe { mem::transmute(self) })
        }
        else {
            None
        }
    }
}

impl<S: node_state::Any + Default> CtNodeRef<S> {
    #[inline]
    unsafe fn transform_with<TargetState: node_state::Valid>(
        self,
        mut transform: impl FnMut(Node<S>) -> Node<TargetState>,
    ) -> CtNodeRef<TargetState> {
        let inner_node = self.replace(Default::default());
        let transformed = transform(inner_node);

        self.inner.state.set(TargetState::state());

        let ret: CtNodeRef<TargetState> = unsafe { mem::transmute(self) };
        ret.replace(transformed);
        ret
    }
}

impl CtNodeRef<Leaf> {
    pub fn expand(self, pos: &Position) -> ExpandedRefSwitch {
        let leaf = self.replace(Default::default());
        let expanded = leaf.expand(pos);

        match expanded {
            ExpandedSwitch::Terminal(node) => {
                self.inner.state.set(NodeState::Terminal);
                let ret: CtNodeRef<Terminal> = unsafe { mem::transmute(self) };
                ret.replace(node);
                ExpandedRefSwitch::Terminal(ret)
            }
            ExpandedSwitch::Branching(node) => {
                self.inner.state.set(NodeState::Branching);
                let ret: CtNodeRef<Branching> = unsafe { mem::transmute(self) };
                ret.replace(node);
                ExpandedRefSwitch::Branching(ret)
            }
        }
    }
}

impl CtNodeRef<Branching> {
    pub fn set_policy_raw(self, raw_policy: &RawPolicy) -> CtNodeRef<Evaluated> {
        unsafe { self.transform_with(|node| node.set_policy_raw(raw_policy)) }
    }
}

/// A node reference with runtime infomration about the state.
#[derive(Debug, Clone)]
pub struct RtNodeRef {
    node: CtNodeRef<node_state::Unknown>,
}

impl Deref for RtNodeRef {
    type Target = CtNodeRef<Unknown>;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl Default for RtNodeRef {
    fn default() -> Self {
        Self::new(Node::new_leaf())
    }
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
        }
    }

    pub fn into_ct(self) -> NodeSwitch {
        // SAFETY: `Node<>` is #[repr(transparent)], so we can just transmute it into
        // another state.
        unsafe {
            match self.state() {
                NodeState::Leaf => NodeSwitch::Leaf(mem::transmute(self.node)),
                NodeState::Branching => NodeSwitch::Branching(mem::transmute(self.node)),
                NodeState::Terminal => NodeSwitch::Terminal(mem::transmute(self.node)),
                NodeState::Evaluated => NodeSwitch::Evaluated(mem::transmute(self.node)),
            }
        }
    }

    pub fn state(&self) -> NodeState {
        self.node.inner.state.get()
    }
}

#[derive(Clone, Default)]
pub struct NodeData {
    /// All the branches from this node.
    branches: Vec<Branch>,

    /// The number of times this node was visited.
    visits: u32,

    /// The value of this node. (~sums all the values of it's children)
    value: Value,
}

impl fmt::Debug for NodeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("value", &self.value())
            .field("visits", &self.visits())
            .field("branches", &self.branches)
            .finish()
    }
}

impl fmt::Display for NodeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("value", &self.value())
            .field("visits", &self.visits())
            .field(
                "branches",
                &self
                    .branches
                    .iter()
                    .map(|b| format!("{}", b.mov()))
                    .collect_vec(),
            )
            .finish()
    }
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

    pub fn compute_subtree_size(&self) -> usize {
        1 + self
            .branches()
            .iter()
            .map(|b| b.node().borrow().data.compute_subtree_size())
            .sum::<usize>()
    }

    pub fn compute_subtree_maxdepth(&self) -> usize {
        self.branches()
            .iter()
            .map(|b| 1 + b.node().borrow().data.compute_subtree_maxdepth())
            .max()
            .unwrap_or(0)
    }

    pub fn compute_subtree_mindepth(&self) -> usize {
        self.branches()
            .iter()
            .map(|b| 1 + b.node().borrow().data.compute_subtree_mindepth())
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

    #[derive(Debug, Clone)]
    pub enum NodeSwitch {
        Leaf(CtNodeRef<Leaf>),
        Branching(CtNodeRef<Branching>),
        Terminal(CtNodeRef<Terminal>),
        Evaluated(CtNodeRef<Evaluated>),
    }

    #[derive(Debug)]
    pub enum ExpandedSwitch {
        Terminal(Node<Terminal>),
        Branching(Node<Branching>),
    }

    #[derive(Debug)]
    pub enum ExpandedRefSwitch {
        Terminal(CtNodeRef<Terminal>),
        Branching(CtNodeRef<Branching>),
    }

    pub trait Any {}

    pub const trait Valid: Any {
        fn state() -> NodeState;
    }

    pub trait Expanded: Any {}

    pub trait HasBranches: Any + Valid {}

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
    impl HasBranches for Branching {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Evaluated;
    impl Any for Evaluated {}
    impl Valid for Evaluated {
        fn state() -> NodeState {
            NodeState::Evaluated
        }
    }
    impl HasBranches for Evaluated {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Unknown;
    impl Any for Unknown {}
}

#[repr(transparent)]
#[derive(Debug, Clone, Default)]
pub struct Node<State: node_state::Any> {
    /// The data.
    data: NodeData,

    /// The current state of this node.
    _state: PhantomData<State>,
}

impl Node<Leaf> {
    /// Expand the node, consuming the Leaf.
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
            // println!("DATA: {}", self.data.branches.len());
            ExpandedSwitch::Branching(unsafe { Node::<Branching>::new(self.data) })
        }
    }

    pub fn new_leaf() -> Self {
        // SAFETY: The Node data is of a leaf.
        unsafe { Node::<Leaf>::new(NodeData::new_leaf()) }
    }
}

impl<S: node_state::HasBranches> Node<S> {
    pub fn branches(&self) -> &[Branch] {
        &self.data.branches
    }

    pub fn sort_by<T: Ord>(&mut self, f: impl Fn(&Branch) -> T) {
        self.data.branches.sort_by_key(f);
    }

    pub fn get_branch(&self, index: usize) -> Option<&Branch> {
        self.data.branches.get(index)
    }

    fn select_best(&self) -> &Branch {
        self.select(|b| b.visits())
    }

    fn find_branch(&self, mut pred: impl FnMut(&Branch) -> bool) -> Option<&Branch> {
        self.data.branches.iter().find(|&b| pred(b))
    }

    fn select<F, T>(&self, transform: F) -> &Branch
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

impl Node<Evaluated> {
    pub fn apply_policy_noise(&mut self, noise: &[f32], eps: f32) {
        let total = noise.iter().sum::<f32>();
        for (branch, noise) in self.data.branches.iter_mut().zip(noise) {
            let norm_noise = noise / total;
            let policy = branch.policy();
            branch.set_policy(policy * (1. - eps) + eps * norm_noise);
        }
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

    pub fn visits(&self) -> u32 {
        self.data.visits()
    }

    pub fn value(&self) -> Value {
        self.data.value()
    }

    pub fn update(&mut self, value: eval::Value) {
        self.data.update(value)
    }
}
