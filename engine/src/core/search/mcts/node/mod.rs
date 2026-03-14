use crate::core::{
    depth::Depth,
    search::mcts::node::node_state::{
        ExpandedRefSwitch, ExpandedSwitch, HasBranches, HasValue, NodeState, NodeSwitch, Unknown,
    },
};
use itertools::Itertools;
use std::{cell::Cell, mem::transmute, ops::Deref};

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

/// The height of the tree. The root is at height 1, and the height of an empty
/// tree is 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Height(pub u16);

impl Height {
    /// The height of an empty.
    pub const EMPTY: Height = Height(0);

    /// The height.
    pub const ROOT: Height = Height(1);
}

impl From<Depth> for Height {
    fn from(value: Depth) -> Self {
        Height(value.v() as u16 + 1)
    }
}

impl fmt::Display for Height {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl_op!(+|a: Height, b: Height| -> Height { Height(a.0 + b.0) });
impl_op!(+|a: Height, b: u16| -> Height { Height(a.0 + b) });

#[derive(Debug, Clone)]
pub struct Tree {
    /// Root of the tree.
    root: RtNodeRef,

    // incrementally computed versions of the compute_subtree_* functions.
    /// The size of the tree (total number of nodes).
    size: usize,

    // todo: figure out a way to track the minheight stat, such that we don't have to recompute it
    // each time we expand a node. maybe we can track the minheight of each subtree in the node
    // data, and update it when we expand a node? (That'd would be specially expensive tho)
    /// The maximum height of the tree.
    maxheight: Height,
}

impl Default for Tree {
    fn default() -> Self {
        Self {
            root: Default::default(),
            size: 1,
            maxheight: Height::ROOT,
        }
    }
}

impl Tree {
    pub fn new(root: RtNodeRef) -> Self {
        let mut ret = Self { root, ..Default::default() };
        ret.compute_stats();
        ret
    }
}

// Node mutable operations delegeated through the tree, such that the statistics
// are always up to date.
impl Tree {
    /// Expands a leaf node, creating branches and updating tree statistics.
    pub fn expand_node(
        &mut self,
        node: CtNodeRef<Leaf>,
        pos: &Position,
        search_depth: Depth,
    ) -> ExpandedRefSwitch {
        let expanded = node.expand(pos, search_depth);

        let height: Height = search_depth.into();

        match &expanded {
            ExpandedRefSwitch::Branching(b) => {
                let branches_count = b.borrow().branches().len();
                self.size += branches_count;
                self.maxheight = self.maxheight.max(height + 1);
            }
            ExpandedRefSwitch::Terminal(_) => {}
        }

        expanded
    }

    pub fn set_policy_raw(
        &mut self,
        node: CtNodeRef<Branching>,
        raw_policy: &RawPolicy,
    ) -> CtNodeRef<Evaluated> {
        node.set_policy_raw(raw_policy)
    }

    pub fn set_policy(
        &mut self,
        node: CtNodeRef<Branching>,
        policy: &Policy,
    ) -> CtNodeRef<Evaluated> {
        node.set_policy(policy)
    }

    pub fn apply_policy_noise(&mut self, node: CtNodeRef<Evaluated>, noise: &[f32], eps: f32) {
        node.borrow_mut().apply_policy_noise(noise, eps);
    }

    pub fn update_node<S: HasValue>(&mut self, node: CtNodeRef<S>, value: eval::Value) {
        node.borrow_mut().update(value);
    }
}

// Tree mutating operations, which will require recompute of all stats.
impl Tree {
    /// Advnace to the best branch.
    /// Will cause a costly recompute of the stats.
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
            self.compute_stats();
        }
    }

    /// Advnace to the first branch that matches `pred`.
    /// Will cause a costly recompute of the stats.
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
            self.compute_stats();
        }
    }
}

impl Tree {
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
        // let mut buf = Vec::with_capacity(self.mindepth().v().into());
        let mut buf = Vec::new();
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

    /// Computes the number of nodes in this tree
    pub fn compute_size(&self) -> usize {
        self.get_root().borrow().data.compute_subtree_size()
    }

    /// Computes the max height of the tree.
    pub fn compute_maxheight(&self) -> Height {
        self.get_root().borrow().data.compute_subtree_maxheight()
    }

    /// Computes the min height of the tree.
    pub fn compute_minheight(&self) -> Height {
        self.get_root().borrow().data.compute_subtree_minheight()
    }

    /// Recompute all stats.
    fn compute_stats(&mut self) {
        self.size = self.compute_size();
        self.maxheight = self.compute_maxheight();
        // self.mindepth = self.compute_mindepth();
    }

    pub fn get_root(&self) -> RtNodeRef {
        self.root.clone()
    }

    /// Returns the number of nodes in this tree
    pub fn size(&self) -> usize {
        self.size
    }

    // /// Returns the min depth of the tree.
    // pub fn mindepth(&self) -> Depth {
    //     self.mindepth
    // }

    /// Returns the max height of the tree.
    pub fn maxheight(&self) -> Height {
        self.maxheight
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

    pub(self) fn set_policy(&mut self, policy: f32) {
        self.policy = policy;
    }

    pub fn node(&self) -> RtNodeRef {
        self.node.clone()
    }
}

/// The value of a node.
/// - high ~> Winning for the parent node.
/// - low ~> Losing for the parent node.
#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Value(pub f32);

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

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

impl<S: node_state::Any> Clone for CtNodeRef<S> {
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

impl<S: const node_state::Valid> CtNodeRef<S> {
    const STATE: NodeState = S::state();

    /// Try to transmute this into the specified state. Retruns none if the
    /// state is not the target state.
    pub fn try_into<Target: node_state::Valid>(self) -> Option<CtNodeRef<Target>> {
        // idk i hope the compiler is smart enough to see that this is resolvable at
        // comp time x3
        if Self::STATE == Target::state() {
            Some(unsafe { transmute::<Self, CtNodeRef<Target>>(self) })
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

        let ret: CtNodeRef<TargetState> = unsafe { transmute(self) };
        ret.replace(transformed);
        ret
    }
}

impl CtNodeRef<Leaf> {
    pub(self) fn expand(self, pos: &Position, search_depth: Depth) -> ExpandedRefSwitch {
        let leaf = self.replace(Default::default());
        let expanded = leaf.expand(pos, search_depth);

        match expanded {
            ExpandedSwitch::Terminal(node) => {
                self.inner.state.set(NodeState::Terminal);
                let ret: CtNodeRef<Terminal> = unsafe { transmute(self) };
                ret.replace(node);
                ExpandedRefSwitch::Terminal(ret)
            }
            ExpandedSwitch::Branching(node) => {
                self.inner.state.set(NodeState::Branching);
                let ret: CtNodeRef<Branching> = unsafe { transmute(self) };
                ret.replace(node);
                ExpandedRefSwitch::Branching(ret)
            }
        }
    }
}

impl CtNodeRef<Branching> {
    pub(self) fn set_policy_raw(self, raw_policy: &RawPolicy) -> CtNodeRef<Evaluated> {
        unsafe { self.transform_with(|node| node.set_policy_raw(raw_policy)) }
    }

    pub(self) fn set_policy(self, policy: &Policy) -> CtNodeRef<Evaluated> {
        unsafe { self.transform_with(|node| node.set_policy(policy)) }
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
            node: unsafe { transmute::<CtNodeRef<S>, CtNodeRef<Unknown>>(node) },
        }
    }

    pub fn into_ct(self) -> NodeSwitch {
        type Switch = NodeSwitch;
        type State = NodeState;
        type Ref<S> = CtNodeRef<S>;
        type AnyRef = CtNodeRef<Unknown>;

        // SAFETY: `Node<>` is #[repr(transparent)], so we can just transmute it into
        // another state.
        unsafe {
            let state = self.state();
            let node = self.node;
            match state {
                State::Leaf => Switch::Leaf(transmute::<AnyRef, Ref<Leaf>>(node)),
                State::Branching => Switch::Branching(transmute::<AnyRef, Ref<Branching>>(node)),
                State::Terminal => Switch::Terminal(transmute::<AnyRef, Ref<Terminal>>(node)),
                State::Evaluated => Switch::Evaluated(transmute::<AnyRef, Ref<Evaluated>>(node)),
            }
        }
    }

    pub fn state(&self) -> NodeState {
        self.node.inner.state.get()
    }
}

#[derive(Clone, Default)]
pub struct NodeData {
    // todo: use a boxed slice instead. the branches will not change once set.
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

    pub(self) fn update(&mut self, value: eval::Value) {
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

    pub fn compute_subtree_maxheight(&self) -> Height {
        Height::ROOT
            + self
                .branches()
                .iter()
                .map(|b| b.node().borrow().data.compute_subtree_maxheight())
                .max()
                .unwrap_or(Height::EMPTY)
    }

    pub fn compute_subtree_minheight(&self) -> Height {
        // we ourselves are 1-nodes deep.
        Height::ROOT
            + self
                .branches()
                .iter()
                .map(|b| b.node().borrow().data.compute_subtree_minheight())
                .min()
                .unwrap_or(Height::EMPTY)
    }

    fn new_leaf() -> NodeData {
        Self::default()
    }
}

pub mod node_state {
    use std::fmt;

    use super::{CtNodeRef, Node};

    #[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
    pub enum NodeState {
        #[default]
        Leaf,
        Branching,
        Terminal,
        Evaluated,
    }

    impl fmt::Display for NodeState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                NodeState::Leaf => write!(f, "Leaf"),
                NodeState::Branching => write!(f, "Branching"),
                NodeState::Terminal => write!(f, "Terminal"),
                NodeState::Evaluated => write!(f, "Evaluated"),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub enum NodeSwitch {
        Leaf(CtNodeRef<Leaf>),
        Branching(CtNodeRef<Branching>),
        Terminal(CtNodeRef<Terminal>),
        Evaluated(CtNodeRef<Evaluated>),
    }
    impl NodeSwitch {
        pub fn evaluated(&self) -> Option<&CtNodeRef<Evaluated>> {
            match self {
                NodeSwitch::Evaluated(x) => Some(x),
                _ => None,
            }
        }
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
    impl ExpandedRefSwitch {
        pub fn branching(&self) -> Option<&CtNodeRef<Branching>> {
            match self {
                ExpandedRefSwitch::Terminal(_) => None,
                ExpandedRefSwitch::Branching(x) => Some(x),
            }
        }

        pub fn terminal(&self) -> Option<&CtNodeRef<Terminal>> {
            match self {
                ExpandedRefSwitch::Terminal(x) => Some(x),
                ExpandedRefSwitch::Branching(_) => None,
            }
        }
    }

    pub const trait Any {}

    pub const trait Valid: Any {
        fn state() -> NodeState;
    }

    pub const trait HasBranches: Any + Valid {}

    pub const trait HasValue: Any + Valid {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Leaf;
    impl Any for Leaf {}
    impl const Valid for Leaf {
        fn state() -> NodeState {
            NodeState::Leaf
        }
    }

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Terminal;
    impl Any for Terminal {}
    impl const Valid for Terminal {
        fn state() -> NodeState {
            NodeState::Terminal
        }
    }
    impl HasValue for Terminal {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Branching;
    impl const Any for Branching {}
    impl const Valid for Branching {
        fn state() -> NodeState {
            NodeState::Branching
        }
    }
    impl const HasBranches for Branching {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Evaluated;
    impl Any for Evaluated {}
    impl const Valid for Evaluated {
        fn state() -> NodeState {
            NodeState::Evaluated
        }
    }
    impl const HasValue for Evaluated {}
    impl const HasBranches for Evaluated {}

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

impl<S: node_state::HasBranches> fmt::Display for Node<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("value", &self.value())
            .field("visits", &self.visits())
            .field(
                "branches",
                &self
                    .branches()
                    .iter()
                    .map(|b| format!("{}", b.mov()))
                    .collect_vec(),
            )
            .finish()
    }
}

impl Node<Leaf> {
    /// Expand the node, consuming the Leaf.
    /// variant.
    pub(self) fn expand(mut self, pos: &Position, search_depth: Depth) -> ExpandedSwitch {
        // todo: save the game_result in the node data, such that it doesn't have to be
        // evaluated each time we encounter the terminal node in the search.
        if pos.search_result(search_depth).is_some() {
            // SAFETY: If there's a gameresult, we can be sure that this is a terminal node.
            return ExpandedSwitch::Terminal(unsafe { Node::new(self.data) });
        }

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
    pub(self) fn set_policy(mut self, policy: &Policy) -> Node<Evaluated> {
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

    pub(self) fn set_policy_raw(self, raw_policy: &RawPolicy) -> Node<Evaluated> {
        let policy = {
            let moves = self.branches().iter().map(|b| usize::from(b.mov()));
            Policy::from_raw(raw_policy, moves)
                .expect("Shouldn't be None, since the moves are correct for this node.")
        };

        self.set_policy(&policy)
    }
}

impl Node<Evaluated> {
    pub(self) fn apply_policy_noise(&mut self, noise: &[f32], eps: f32) {
        let total = noise.iter().sum::<f32>();
        for (branch, noise) in self.data.branches.iter_mut().zip(noise) {
            let norm_noise = noise / total;
            let policy = branch.policy();
            let new_policy = policy * (1. - eps) + eps * norm_noise;
            branch.set_policy(new_policy);
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
}

impl<S: HasValue> Node<S> {
    pub(self) fn update(&mut self, value: eval::Value) {
        self.data.update(value)
    }
}
