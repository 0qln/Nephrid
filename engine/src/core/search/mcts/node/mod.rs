use crate::core::{
    depth::Depth,
    search::mcts::node::node_state::{ExpandedRefSwitch, HasValue, NodeState, NodeSwitch},
};
use itertools::Itertools;
use std::mem::transmute;

use crate::core::{
    Move, Position,
    move_iter::fold_legal_moves,
    search::mcts::{
        eval::{self, Policy},
        node::{
            node_state::{Branching, Evaluated, Leaf},
            ops::ControlFlow,
        },
    },
};
use std::{cmp::Ordering, fmt, marker::PhantomData, ops};

#[cfg(test)]
pub mod test;

/// The height of the tree. The root is at height 1, and the height of an empty
/// tree is 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Height(pub u16);

impl Height {
    /// The height of an empty tree.
    pub const EMPTY: Height = Height(0);

    /// The height of a tree with only a root node.
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

// todo: figure out a way to track the minheight stat, such that we don't have
// to recompute it each time we expand a node. maybe we can track the minheight
// of each subtree in the node data, and update it when we expand a node?
// (That'd would be specially expensive tho)
#[derive(Debug, Clone)]
pub struct Tree {
    // nodes arena
    nodes: Vec<NodeData>,

    /// The size of the tree (total number of nodes).
    size: usize,

    /// The maximum height of the tree.
    maxheight: Height,
}

impl Default for Tree {
    fn default() -> Self {
        Self {
            size: 1,
            nodes: vec![NodeData::new_leaf()],
            maxheight: Height::ROOT,
        }
    }
}

impl Tree {
    /// Root of the tree, always at index 0.
    const ROOT: RtNodeId = RtNodeId::new(0);

    pub fn new(root: NodeData) -> Self {
        let mut ret = Self {
            nodes: vec![root],
            ..Default::default()
        };
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
        node_id: CtNodeId<Leaf>,
        pos: &Position,
        search_depth: Depth,
    ) -> ExpandedRefSwitch {
        // todo: save the game_result in the node data, such that it doesn't have to be
        // evaluated each time we encounter the terminal node in the search.
        //
        // note: do not switch this for search_result. The tree has to remain an
        // accurate representation of the game rules.
        // e.g. if we mark this as Terminal after 2 repetitions and the opponent
        // actually plays this later on down the line, the tree root will be
        // Terminal, even though it should be Branching.
        if pos.game_result().is_some() {
            // SAFETY: If there's a gameresult, we can be sure that this is a terminal node.
            unsafe {
                return ExpandedRefSwitch::Terminal(node_id.cast());
            }
        }

        // todo: could replace this allocation with a static mut buffer or something...
        let mut moves = Vec::with_capacity(35);
        _ = fold_legal_moves(pos, &mut moves, |acc, m| {
            acc.push(m);
            ControlFlow::Continue::<(), _>(acc)
        });

        let branches_count = moves.len();

        if branches_count == 0 {
            unreachable!(
                "If there are no branches for this node, it has to be terminal, in which case the \
                 first guard clause should've triggered!"
            )
        }

        let mut branches = Vec::with_capacity(branches_count);
        for m in moves {
            let child_id = self.new_leaf();
            // Assuming your down_cast / Option logic here
            branches.push(Branch::new(m, 0.0, child_id.down_cast()));
        }
        let node = self.node_mut(node_id);
        node.data.branches = branches;

        // update stats
        let height: Height = search_depth.into();
        self.size += branches_count;
        self.maxheight = self.maxheight.max(height + 1);

        // SAFETY: We just checked that this is a branching node.
        unsafe { ExpandedRefSwitch::Branching(node_id.cast()) }
    }

    fn new_leaf(&mut self) -> CtNodeId<Leaf> {
        let id = self.nodes.len() as u32;
        self.nodes.push(NodeData::new_leaf());
        // SAFETY: We just pushed a leaf node into the arena, so this is a valid leaf
        // node id.
        unsafe { CtNodeId::new(id) }
    }

    pub fn skip_policy(
        &mut self,
        node: CtNodeId<Branching>,
    ) -> CtNodeId<Evaluated> {
        self.node_mut(node).skip_policy();
        // SAFETY: we just set the policy for each branch. It has to be valid.
        unsafe { node.cast() }
    }

    pub fn set_policy(
        &mut self,
        node: CtNodeId<Branching>,
        policy: &Policy,
    ) -> CtNodeId<Evaluated> {
        self.node_mut(node).set_policy(policy);
        // SAFETY: we just set the policy for each branch. It has to be valid.
        unsafe { node.cast() }
    }

    pub fn apply_policy_noise(&mut self, node: CtNodeId<Evaluated>, noise: &[f32], eps: f32) {
        self.node_mut(node).apply_policy_noise(noise, eps);
    }

    pub fn update_node<S: HasValue>(&mut self, node: CtNodeId<S>, value: eval::Value) {
        self.node_mut(node).update(value);
    }
}

// Tree mutating operations, which will require recompute of all stats.
impl Tree {
    /// Advnace to the best branch.
    /// Will cause a costly recompute of the stats.
    pub fn advance_best(&mut self) {
        fn map(node: CtNodeId<impl HasBranches>) -> RtNodeId {
            node.borrow().select_best().node()
        }

        let node = {
            match self.root().clone().into_ct() {
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

    // /// Advnace to the first branch that matches `pred`.
    // /// Will cause a costly recompute of the stats.
    // pub fn advance_to<F: Fn(&Branch) -> bool>(&mut self, pred: F) {
    //     fn map(
    //         node: CtNodeId<impl HasBranches>,
    //         pred: impl Fn(&Branch) -> bool,
    //     ) -> Option<RtNodeId> {
    //         node.borrow().find_branch(|x| pred(x)).map(|b| b.node())
    //     }

    //     let node = {
    //         match self.root.clone().into_ct() {
    //             NodeSwitch::Branching(node) => Some(map(node, pred)),
    //             NodeSwitch::Evaluated(node) => Some(map(node, pred)),
    //             _ => None,
    //         }
    //     };

    //     if let Some(node) = node.flatten() {
    //         self.root = node;
    //         self.compute_stats();
    //     }
    // }
}

impl Tree {
    pub fn best_branch<'a, S: node_state::HasBranches + 'a>(
        &'a self,
        node_id: CtNodeId<S>,
    ) -> &'a Branch {
        self.node(node_id).select(|b| {
            let branch_node = self.node_data(b.node());
            branch_node.visits()
        })
    }

    pub fn best_move<S: node_state::HasBranches>(&self, node_id: CtNodeId<S>) -> Move {
        self.best_branch(node_id).mov()
    }

    // pub fn best_moves<S: node_state::HasBranches>(&self, node_id: CtNodeId<S>,
    // threshold: Value) -> Vec<Move> {     fn map(node: CtNodeId<impl
    // HasBranches>, threshold: Value) -> Vec<Move> {         node.borrow()
    //             .branches()
    //             .iter()
    //             .filter(|b| b.value() > threshold)
    //             .map(|b| b.mov())
    //             .collect_vec()
    //     }
    // }

    // pub fn map_root_if_has_branches<T>(&self, transform: FnMut(CtNodeId<impl
    // HasBranches>) -> T) {     match self.root().clone().into_ct() {
    //         NodeSwitch::Branching(node) => map(node, threshold),
    //         NodeSwitch::Evaluated(node) => map(node, threshold),
    //         _ => vec![],
    //     }
    // }

    /// Returns the current principal variation. The branches are clones, but
    /// contain valid references to their nodes.
    pub fn principal_variation(&self) -> Path {
        // let mut buf = Vec::with_capacity(self.mindepth().v().into());
        let mut buf = Vec::new();
        let mut current = Self::ROOT;

        loop {
            let branch = match self.node_switch(current) {
                NodeSwitch::Branching(node) => Some(self.best_branch(node)),
                NodeSwitch::Evaluated(node) => Some(self.best_branch(node)),
                _ => None,
            };

            match branch {
                Some(branch) => {
                    let node = branch.node();
                    buf.push(branch.clone());
                    current = node;
                }
                None => break,
            }
        }
        Path(buf)
    }

    /// Computes the number of nodes in this tree
    pub fn compute_size(&self) -> usize {
        self.root().data.compute_subtree_size(self)
    }

    /// Computes the max height of the tree.
    pub fn compute_maxheight(&self) -> Height {
        self.root().data.compute_subtree_maxheight(self)
    }

    /// Computes the min height of the tree.
    pub fn compute_minheight(&self) -> Height {
        self.node_data(Self::ROOT).compute_subtree_minheight(self)
    }

    /// Recompute all stats.
    fn compute_stats(&mut self) {
        self.size = self.compute_size();
        self.maxheight = self.compute_maxheight();
        // self.mindepth = self.compute_mindepth();
    }

    // private accessors, which will be used by the node operations to access the
    // nodes in the arena.

    fn node<S: node_state::Any>(&self, node: CtNodeId<S>) -> &Node<S> {
        // SAFETY: Node<> is #[repr(transparent)]
        unsafe { transmute::<&NodeData, &Node<S>>(&self.nodes[node.index as usize]) }
    }

    fn node_data(&self, node: RtNodeId) -> &NodeData {
        &self.nodes[node.index as usize]
    }

    pub fn node_mut<'a, S: node_state::Any>(&'a mut self, node: CtNodeId<S>) -> Node<'a, S> {
        Node {
            data: &mut self.nodes[node.index as usize],
            _state: PhantomData,
        }
    }

    pub fn node_switch(&self, node_id: RtNodeId) -> NodeSwitch {
        type Switch = NodeSwitch;
        type State = NodeState;

        // SAFETY: We check here the state.
        unsafe {
            match self.node_data(node_id).state() {
                State::Leaf => Switch::Leaf(node_id.up_cast()),
                State::Branching => Switch::Branching(node_id.up_cast()),
                State::Terminal => Switch::Terminal(node_id.up_cast()),
                State::Evaluated => Switch::Evaluated(node_id.up_cast()),
            }
        }
    }

    fn root(&self) -> &Node<node_state::Unknown> {
        // SAFETY: Node<> is #[repr(transparent)]
        unsafe {
            transmute::<&NodeData, &Node<node_state::Unknown>>(
                &self.nodes[Self::ROOT.index as usize],
            )
        }
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

#[derive(Clone, Debug)]
pub struct Branch {
    /// The node that this branch leads to.
    node: RtNodeId,

    /// The policy of picking this branch.
    policy: f32,

    /// The move that lead to this node.
    mov: Move,
}

impl Branch {
    pub fn new(mov: Move, policy: f32, node: RtNodeId) -> Self {
        Self { node, policy, mov }
    }

    pub fn mov(&self) -> Move {
        self.mov
    }

    pub fn policy(&self) -> f32 {
        self.policy
    }

    pub fn node(&self) -> RtNodeId {
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

#[derive(Debug, Clone, Copy)]
pub struct RtNodeId {
    index: u32,
}

impl RtNodeId {
    const fn new(index: u32) -> Self {
        Self { index }
    }

    /// SAFETY: The caller has to make sure that the node at `index` is in state
    /// `S`.
    pub unsafe fn up_cast<S: node_state::Valid>(&self) -> CtNodeId<S> {
        unsafe { CtNodeId::new(self.index) }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CtNodeId<S: node_state::Any> {
    index: u32,
    _state: PhantomData<S>,
}

impl<S: node_state::Any> CtNodeId<S> {
    /// SAFETY: The caller has to make sure that the node at `index` is in state
    /// `S`.
    unsafe fn new(index: u32) -> Self {
        Self { index, _state: PhantomData }
    }

    /// SAFETY: The caller has to make sure that the node at `index` is in state
    /// `S2`.
    unsafe fn cast<S2: node_state::Any>(self) -> CtNodeId<S2> {
        CtNodeId::<S2> {
            index: self.index,
            _state: PhantomData,
        }
    }

    pub fn down_cast(&self) -> RtNodeId {
        RtNodeId { index: self.index }
    }
}

impl<S: const node_state::Valid> CtNodeId<S> {
    const STATE: NodeState = S::state();

    pub fn try_into<Target: node_state::Valid>(self) -> Option<CtNodeId<Target>> {
        // idk i hope the compiler is smart enough to see that this is resolvable at
        // comp time x3
        if Self::STATE == Target::state() {
            Some(unsafe { CtNodeId::new(self.index) })
        }
        else {
            None
        }
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

    /// The state of the node (explicitly encoded such that we can figure it out
    /// at runtime).
    ///
    /// todo:
    /// can this be removed?
    /// can this be a descriminated enum that then also contains like either the
    /// branches or the value, such that we don't have to do unsafe transmutes?
    state: NodeState,
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
    pub fn state(&self) -> NodeState {
        self.state
    }

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

    pub fn compute_subtree_size(&self, tree: &Tree) -> usize {
        1 + self
            .branches()
            .iter()
            .map(|b| tree.node_data(b.node()).compute_subtree_size(tree))
            .sum::<usize>()
    }

    pub fn compute_subtree_maxheight(&self, tree: &Tree) -> Height {
        Height::ROOT
            + self
                .branches()
                .iter()
                .map(|b| tree.node_data(b.node()).compute_subtree_maxheight(tree))
                .max()
                .unwrap_or(Height::EMPTY)
    }

    pub fn compute_subtree_minheight(&self, tree: &Tree) -> Height {
        // we ourselves are 1-nodes deep.
        Height::ROOT
            + self
                .branches()
                .iter()
                .map(|b| tree.node_data(b.node()).compute_subtree_minheight(tree))
                .min()
                .unwrap_or(Height::EMPTY)
    }

    fn new_leaf() -> NodeData {
        Self::default()
    }
}

pub mod node_state {
    use std::fmt;

    use crate::core::search::mcts::node::CtNodeId;

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

    #[derive(Debug, Clone, Copy)]
    pub enum NodeSwitch {
        Leaf(CtNodeId<Leaf>),
        Branching(CtNodeId<Branching>),
        Terminal(CtNodeId<Terminal>),
        Evaluated(CtNodeId<Evaluated>),
    }
    impl NodeSwitch {
        pub fn evaluated(&self) -> Option<&CtNodeId<Evaluated>> {
            match self {
                NodeSwitch::Evaluated(x) => Some(x),
                _ => None,
            }
        }
    }

    #[derive(Debug)]
    pub enum ExpandedRefSwitch {
        Terminal(CtNodeId<Terminal>),
        Branching(CtNodeId<Branching>),
    }
    impl ExpandedRefSwitch {
        pub fn branching(&self) -> Option<&CtNodeId<Branching>> {
            match self {
                ExpandedRefSwitch::Terminal(_) => None,
                ExpandedRefSwitch::Branching(x) => Some(x),
            }
        }

        pub fn terminal(&self) -> Option<&CtNodeId<Terminal>> {
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

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    pub struct Leaf;
    impl Any for Leaf {}
    impl const Valid for Leaf {
        fn state() -> NodeState {
            NodeState::Leaf
        }
    }

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    pub struct Terminal;
    impl Any for Terminal {}
    impl const Valid for Terminal {
        fn state() -> NodeState {
            NodeState::Terminal
        }
    }
    impl HasValue for Terminal {}

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    pub struct Branching;
    impl const Any for Branching {}
    impl const Valid for Branching {
        fn state() -> NodeState {
            NodeState::Branching
        }
    }
    impl const HasBranches for Branching {}

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    pub struct Evaluated;
    impl Any for Evaluated {}
    impl const Valid for Evaluated {
        fn state() -> NodeState {
            NodeState::Evaluated
        }
    }
    impl const HasValue for Evaluated {}
    impl const HasBranches for Evaluated {}

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    pub struct Unknown;
    impl Any for Unknown {}
}

#[derive(Debug)]
pub struct Node<'a, S: node_state::Any> {
    pub(crate) data: &'a mut NodeData,
    _state: PhantomData<S>,
}

impl<S: node_state::HasBranches> fmt::Display for Node<'_, S> {
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

impl<S: node_state::HasBranches> Node<'_, S> {
    pub fn branches(&self) -> &[Branch] {
        &self.data.branches
    }

    pub fn move_indices(&self) -> impl Iterator<Item = usize> {
        self.branches().iter().map(|b| usize::from(b.mov()))
    }

    pub fn sort_by<T: Ord>(&mut self, f: impl Fn(&Branch) -> T) {
        self.data.branches.sort_by_key(f);
    }

    pub fn get_branch(&self, index: usize) -> Option<&Branch> {
        self.data.branches.get(index)
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

impl<'a> Node<'a, Branching> {
    pub(self) fn set_policy<'b>(self, policy: &'b Policy) -> Node<'a, Evaluated> {
        assert_eq!(
            self.branches().len(),
            policy.len(),
            "There has to be exactly one policy for each branch."
        );

        for (i, branch) in self.data.branches.iter_mut().enumerate() {
            let p = policy.get(i).expect("Policy should contain this move.");
            branch.set_policy(p);
        }

        // SAFETY: we just set the policy for each branch. It has to be valid.
        unsafe { Node::<'a, Evaluated>::new(self.data) }
    }

    /// Sets the policy to an even probability for all branches.
    pub(self) fn skip_policy(self) -> Node<Evaluated> {
        let policy = Policy::new_even(self.data.branches.len());
        self.set_policy(&policy)
    }
}

impl Node<'_, Evaluated> {
    pub(self) fn apply_policy_noise(&mut self, noise: &[f32], eps: f32) {
        let total = noise.iter().sum::<f32>();
        for (branch, noise) in self.data.branches.iter_mut().zip(noise) {
            let norm_noise = noise / total;
            let policy = branch.policy();
            let new_policy = policy * (1. - eps) + eps * norm_noise;
            branch.policy = new_policy;
        }
    }
}

impl<S: node_state::Valid> Node<'_, S> {
    pub fn state() -> NodeState {
        S::state()
    }
}

impl<'a, S: node_state::Any> Node<'a, S> {
    /// Construct a new node.
    ///
    /// # Safety
    ///
    /// The caller has to make sure that `data` contains valid data to be in
    /// state `S`.
    unsafe fn new(data: &'a mut NodeData) -> Self {
        Self { data, _state: PhantomData }
    }

    pub fn visits(&self) -> u32 {
        self.data.visits()
    }

    pub fn value(&self) -> Value {
        self.data.value()
    }
}

impl<S: HasValue> Node<'_, S> {
    pub(self) fn update(&mut self, value: eval::Value) {
        self.data.update(value)
    }
}
