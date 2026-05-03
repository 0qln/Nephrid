use crate::{
    core::{
        depth::Depth,
        r#move::MoveIndex,
        search::mcts::{
            eval::{Probability, Ratio},
            nn::PolicyHeadIndex,
            node::node_state::{ExpandedSwitch, HasBranches, HasValue, NodeState, Switch},
        },
    },
    impl_variants,
};
use itertools::Itertools;
use std::{
    cmp::Ordering,
    collections::VecDeque,
    fmt,
    marker::PhantomData,
    ops::{self, Deref},
    ptr,
};

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

#[cfg(test)]
pub mod test;

/// The height of the tree. The root is at height 1, and the height of an empty
/// tree is 0.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

/// The value of a node.
/// - high ~> Winning for the parent node.
/// - low ~> Losing for the parent node.
/// - +inf ~> Proven win for the parent node.
/// - -inf ~> Proven loss for the parent node.
#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Value(f32);

impl Value {
    pub const fn proven_win() -> Self {
        Self(f32::INFINITY)
    }

    pub const fn proven_loss() -> Self {
        Self(f32::NEG_INFINITY)
    }

    pub fn is_proven_win(&self) -> bool {
        *self == Self::proven_win()
    }

    pub fn is_proven_loss(&self) -> bool {
        *self == Self::proven_loss()
    }

    pub fn is_proven(&self) -> bool {
        self.is_proven_win() || self.is_proven_loss()
    }

    pub fn v(&self) -> f32 {
        self.0
    }
}

impl Deref for Value {
    type Target = f32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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
impl_op!(+= |l: &mut Value, r: f32| { l.0 += r } );

impl Eq for Value {}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        f32::partial_cmp(&self.0, &other.0).unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct VisitCount(pub TVisitCount);

impl_op!(+= |l: &mut VisitCount, r: u32| { l.0 += r } );
impl_op!(-= |l: &mut VisitCount, r: u32| { l.0 -= r } );

type TVisitCount = u32;

impl ops::DerefMut for VisitCount {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl ops::Deref for VisitCount {
    type Target = TVisitCount;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for VisitCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Proven {
    v: i32,
}

impl_variants! {
    i32 as Proven in proven {
        LOSS = -1,
        DRAW = 0,
        WIN = 1,
    }
}

impl From<Proven> for Value {
    fn from(x: Proven) -> Self {
        match x.v() {
            proven::WIN_C => Value::proven_win(),
            proven::LOSS_C => Value::proven_loss(),
            _ => Value(0.),
        }
    }
}

impl TryFrom<Value> for Proven {
    type Error = ();

    fn try_from(val: Value) -> Result<Self, Self::Error> {
        if val.is_proven_win() {
            Ok(proven::WIN)
        }
        else if val.is_proven_loss() {
            Ok(proven::LOSS)
        }
        else {
            Err(())
        }
    }
}

pub mod node_state {
    use std::fmt;

    use crate::core::search::mcts::node::NodeId;

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

    impl NodeState {
        pub const fn has_branches(&self) -> bool {
            matches!(self, NodeState::Branching | NodeState::Evaluated)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum Switch {
        Leaf(NodeId<Leaf>),
        Branching(NodeId<Branching>),
        Terminal(NodeId<Terminal>),
        Evaluated(NodeId<Evaluated>),
    }

    impl Switch {
        pub fn new(node_id: super::RtNodeId, state: NodeState) -> Self {
            use Switch as S;
            match state {
                NodeState::Leaf => S::Leaf(unsafe { node_id.cast() }),
                NodeState::Branching => S::Branching(unsafe { node_id.cast() }),
                NodeState::Terminal => S::Terminal(unsafe { node_id.cast() }),
                NodeState::Evaluated => S::Evaluated(unsafe { node_id.cast() }),
            }
        }

        pub fn get<T: Valid>(&self) -> Option<NodeId<T>> {
            use Switch as S;
            match self {
                S::Leaf(x) if T::state() == Leaf::state() => Some(unsafe { x.cast() }),
                S::Branching(x) if T::state() == Branching::state() => Some(unsafe { x.cast() }),
                S::Terminal(x) if T::state() == Terminal::state() => Some(unsafe { x.cast() }),
                S::Evaluated(x) if T::state() == Evaluated::state() => Some(unsafe { x.cast() }),
                _ => None,
            }
        }
    }

    #[derive(Debug)]
    pub enum ExpandedSwitch {
        Terminal(NodeId<Terminal>),
        Branching(NodeId<Branching>),
    }

    impl ExpandedSwitch {
        pub fn get<T: Expanded>(&self) -> Option<NodeId<T>> {
            use ExpandedSwitch as S;
            match self {
                S::Branching(x) if T::state() == Branching::state() => Some(unsafe { x.cast() }),
                S::Terminal(x) if T::state() == Terminal::state() => Some(unsafe { x.cast() }),
                _ => None,
            }
        }
    }

    pub const trait Any: Clone + Copy + PartialEq + Eq + std::hash::Hash {}

    pub const trait Valid: Any {
        fn state() -> NodeState;
        fn has_value() -> bool;
    }

    pub const trait HasBranches: Any {}

    pub const trait HasValue: Any + Valid {}

    pub const trait Expanded: Any + Valid {}

    #[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
    pub struct Leaf;
    impl Any for Leaf {}
    impl const Valid for Leaf {
        fn state() -> NodeState {
            NodeState::Leaf
        }
        fn has_value() -> bool {
            true
        }
    }
    impl const HasValue for Leaf {}

    #[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
    pub struct Terminal;
    impl Any for Terminal {}
    impl const Valid for Terminal {
        fn state() -> NodeState {
            NodeState::Terminal
        }
        fn has_value() -> bool {
            true
        }
    }
    impl const HasValue for Terminal {}
    impl const Expanded for Terminal {}

    #[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
    pub struct Branching;
    impl const Any for Branching {}
    impl const Valid for Branching {
        fn state() -> NodeState {
            NodeState::Branching
        }
        fn has_value() -> bool {
            false
        }
    }
    impl const HasBranches for Branching {}
    impl const Expanded for Branching {}

    #[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
    pub struct Evaluated;
    impl Any for Evaluated {}
    impl const Valid for Evaluated {
        fn state() -> NodeState {
            NodeState::Evaluated
        }
        fn has_value() -> bool {
            true
        }
    }
    impl const HasValue for Evaluated {}
    impl const HasBranches for Evaluated {}

    #[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
    pub struct Unknown;
    impl Any for Unknown {}
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

// -----------------------------------------------------------------------------
// Architecture: Double-Buffered Flat Arena MCTS Tree
// -----------------------------------------------------------------------------

/// The fundamental unit of storage, containing perfectly packed contiguous
/// arrays. Entirely private to encapsulate the DOD architecture.
#[derive(Clone, Default)]
struct ArenaBuffer {
    nodes: Vec<NodeData>,
    branches: Vec<Branch>,
}

impl ArenaBuffer {
    fn new(node_cap: usize, branch_cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(node_cap),
            branches: Vec::with_capacity(branch_cap),
        }
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.branches.clear();
    }
}

/// Completely flattened, index-based node representation.
#[derive(Clone, Default, Debug)]
pub struct NodeData {
    branch_start: u32,
    branch_count: MoveIndex,

    visits: VisitCount,
    value: Value,
    state: NodeState,
}

impl NodeData {
    fn new_leaf() -> Self {
        Self {
            state: NodeState::Leaf,
            ..Default::default()
        }
    }

    pub fn value(&self) -> Value {
        self.value
    }

    pub fn visits(&self) -> VisitCount {
        self.visits
    }
}

/// A relational representation connecting a parent node to a child node via
/// indices.
#[derive(Clone, Debug, PartialEq)]
pub struct Branch {
    node: RtNodeId,
    policy: Probability,
    mov: Move,
}

impl Branch {
    #[inline]
    pub fn mov(&self) -> Move {
        self.mov
    }

    #[inline]
    pub fn policy(&self) -> Probability {
        self.policy
    }

    #[inline]
    pub fn node(&self) -> RtNodeId {
        self.node
    }
}

/// The unified container encapsulating the active arena and tracking global
/// stats.
#[derive(Clone)]
pub struct Tree {
    arena: ArenaBuffer,
    size: usize,
    maxheight: Height,
    terminal_nodes: usize,
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    const ROOT_IDX: RtNodeId = RtNodeId::new(0);

    pub fn new() -> Self {
        let mut arena = ArenaBuffer::new(10000, 30000); // Sensible pre-alloc defaults
        arena.nodes.push(NodeData::new_leaf());
        Self {
            arena,
            size: 1,
            terminal_nodes: 0,
            maxheight: Height::ROOT,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn terminal_nodes(&self) -> usize {
        self.terminal_nodes
    }

    pub fn compute_subtree_size(&self, node_id: RtNodeId) -> usize {
        1 + self
            .branches_rt(node_id)
            .iter()
            .map(|b| self.compute_subtree_size(b.node))
            .sum::<usize>()
    }

    pub fn compute_subtree_terminal_nodes_count(&self, node_id: RtNodeId) -> usize {
        // todo: benchmark if using the fn below is slower, if not use that.
        let node = self.node(node_id);
        let terminal_node = if node.state() == NodeState::Terminal { 1 } else { 0 };
        terminal_node
            + self
                .branches_rt(node_id)
                .iter()
                .map(|b| self.compute_subtree_terminal_nodes_count(b.node))
                .sum::<usize>()
    }

    /// Counts the wins of the root node player.
    pub fn count_wins(&self) -> usize {
        self.count_subtree_nodes(Depth::ROOT, Self::ROOT_IDX, &|_node, _depth| todo!())
    }

    pub fn count_nodes(
        &self,
        pred: &impl Fn(NodeView<node_state::Unknown>, Depth) -> bool,
    ) -> usize {
        self.count_subtree_nodes(Depth::ROOT, Self::ROOT_IDX, pred)
    }

    pub fn count_subtree_nodes(
        &self,
        depth: Depth,
        node_id: RtNodeId,
        pred: &impl Fn(NodeView<node_state::Unknown>, Depth) -> bool,
    ) -> usize {
        let node = self.node(node_id);
        let count = if pred(node, depth) { 1 } else { 0 };
        count
            + self
                .branches_rt(node_id)
                .iter()
                .map(|b| self.count_subtree_nodes(depth + 1, b.node, pred))
                .sum::<usize>()
    }

    /// Iterates the nodes in the tree under `root` in some order.
    pub fn iter_nodes(
        &self,
        stack: &mut Vec<RtNodeId>,
    ) -> impl Iterator<Item = NodeView<'_, node_state::Unknown>> {
        // todo: for the implementation, we could just iterate the arena, which would
        // speed things up by a lot, however that might be unexpected by the
        // caller in some contexts.

        struct TreeIter<'a, 'b> {
            tree: &'a Tree,
            stack: &'b mut Vec<RtNodeId>,
        }

        impl<'a, 'b> Iterator for TreeIter<'a, 'b> {
            type Item = NodeView<'a, node_state::Unknown>;

            fn next(&mut self) -> Option<Self::Item> {
                let node_id = self.stack.pop()?;
                let branches = self.tree.branches_rt(node_id);
                for branch in branches.iter().rev() {
                    self.stack.push(branch.node());
                }

                Some(self.tree.node(node_id))
            }
        }

        stack.clear();
        stack.push(self.root());
        TreeIter { tree: self, stack }
    }

    pub fn maxheight(&self) -> Height {
        self.maxheight
    }

    pub fn compute_subtree_maxheight(&self, node: RtNodeId) -> Height {
        Height::ROOT
            + self
                .branches_rt(node)
                .iter()
                .map(|b| self.compute_subtree_maxheight(b.node))
                .max()
                .unwrap_or(Height::EMPTY)
    }

    pub fn compute_minheight(&self) -> Height {
        self.compute_subtree_minheight(self.root())
    }

    pub fn compute_subtree_minheight(&self, node: RtNodeId) -> Height {
        let mut queue = VecDeque::new();
        queue.push_back((node, Height::ROOT));

        while let Some((curr, height)) = queue.pop_front() {
            let branches = self.branches_rt(curr);

            // first time we encounter a node without branches, we've found the minimum
            // height.
            if branches.is_empty() {
                return height;
            }

            // if this subtree goes deeper, queue up the children to be checked later
            for branch in branches {
                queue.push_back((branch.node, height + 1));
            }
        }

        // SAFETY: This should be unreachable because every node should eventually lead
        // to a leaf node with no branches, at which point we return from the function.
        // If we exhaust the queue without finding such a node, it means there's a cycle
        // in the tree structure, which should never happen in a well-formed MCTS tree.
        // Therefore, we can safely assume this code is unreachable.
        unreachable!()
    }

    pub fn root(&self) -> RtNodeId {
        Self::ROOT_IDX
    }

    #[inline]
    pub fn branch_ids<S: HasBranches>(&self, node_id: NodeId<S>) -> impl Iterator<Item = BranchId> {
        let node = self.node(node_id);
        let data = node.data();
        let start = data.branch_start as usize;
        let end = start + data.branch_count.v as usize;
        (start..end).map(|i| BranchId::new(i as u32))
    }

    #[inline]
    pub fn branches<S: HasBranches>(&self, node_id: NodeId<S>) -> &[Branch] {
        let node = self.node(node_id);
        let data = node.data();
        let start = data.branch_start as usize;
        let end = start + data.branch_count.v as usize;
        &self.arena.branches[start..end]
    }

    #[inline]
    pub fn branches_rt(&self, node_id: RtNodeId) -> &[Branch] {
        const EMPTY_SLICE: &[Branch] = &[];
        match self.node_switch(node_id) {
            Switch::Leaf(_) => EMPTY_SLICE,
            Switch::Terminal(_) => EMPTY_SLICE,
            Switch::Branching(node) => self.branches(node),
            Switch::Evaluated(node) => self.branches(node),
        }
    }

    #[inline]
    pub fn branch_count<S: HasBranches>(&self, node_id: NodeId<S>) -> MoveIndex {
        let node = self.node(node_id);
        let data = node.data();
        data.branch_count
    }

    /// Safely provides mutable access to a node's branches, enforced by
    /// typestate. This allows external sorting without leaking the inner
    /// ArenaBuffer arrays.
    #[inline]
    pub fn branches_mut<S: HasBranches>(&mut self, node_id: NodeId<S>) -> &mut [Branch] {
        let data = &self.arena.nodes[node_id.index as usize];
        let start = data.branch_start as usize;
        let end = start + data.branch_count.v as usize;
        &mut self.arena.branches[start..end]
    }

    /// Sorts the branches of a given node in-place without allocating.
    /// Safely splits the borrow of the arena so we can mutate branches while
    /// reading nodes.
    pub fn sort_branches_by<F>(&mut self, parent_id: NodeId<Evaluated>, mut compare: F)
    where
        // The closure takes: (child_a_data, branch_a, child_b_data, branch_b)
        F: FnMut(&NodeData, &Branch, &NodeData, &Branch) -> Ordering,
    {
        // 1. Get the slice bounds
        let parent_data = &self.arena.nodes[parent_id.index as usize];
        let b_start = parent_data.branch_start as usize;
        let b_end = b_start + parent_data.branch_count.v as usize;

        // 2. Split the borrow!
        // `nodes` is borrowed immutably, `branches` is borrowed mutably.
        let nodes = &self.arena.nodes;
        let branches_slice = &mut self.arena.branches[b_start..b_end];

        // 3. Sort in-place using the disjoint slices
        branches_slice.sort_unstable_by(|branch_a, branch_b| {
            // Read the child data directly from the immutable `nodes` slice
            let child_a_data = &nodes[branch_a.node.index()];
            let child_b_data = &nodes[branch_b.node.index()];

            compare(child_a_data, branch_a, child_b_data, branch_b)
        });
    }

    pub fn policy_indeces<S: HasBranches>(
        &self,
        node: NodeId<S>,
    ) -> impl Iterator<Item = PolicyHeadIndex> {
        self.branches(node)
            .iter()
            .map(|b| PolicyHeadIndex::from(b.mov()))
    }

    /// Backpropagation exclusively mutates the tree sequentially using a
    /// recorded path slice.
    pub fn backpropagate(&mut self, path: &[u32], final_value: eval::Value) {
        for &idx in path {
            let node = &mut self.arena.nodes[idx as usize];
            node.visits.0 += 1;
            node.value += final_value;
        }
    }

    /// Expands a leaf node, creating branches and updating tree statistics
    /// immutably.
    pub fn expand_node(
        &mut self,
        node_id: NodeId<Leaf>,
        pos: &Position,
        search_depth: Depth,
    ) -> ExpandedSwitch {
        if pos.game_result().is_some() {
            self.arena.nodes[node_id.index as usize].state = NodeState::Terminal;
            self.terminal_nodes += 1;
            unsafe {
                return ExpandedSwitch::Terminal(node_id.cast());
            }
        }

        let mut moves = Vec::with_capacity(35);
        _ = fold_legal_moves(pos, &mut moves, |acc, m| {
            acc.push(m);
            ControlFlow::Continue::<(), _>(acc)
        });

        let branches_count = MoveIndex::try_from(moves.len())
            .expect("fold_legal_moves will return all moves in a legal position and no more.");
        if branches_count == MoveIndex::from(0) {
            unreachable!("If there are no branches for this node, it has to be terminal.");
        }

        let branch_start = self.arena.branches.len() as u32;
        let branch_count = branches_count;

        for m in moves {
            let child = RtNodeId::from(self.arena.nodes.len());
            self.arena.nodes.push(NodeData::new_leaf());

            self.arena.branches.push(Branch {
                node: child,
                policy: Probability::zero(),
                mov: m,
            });
        }

        // Link parent to branches
        let parent = &mut self.arena.nodes[node_id.index as usize];
        parent.branch_start = branch_start;
        parent.branch_count = branch_count;
        parent.state = NodeState::Branching;

        let height: Height = search_depth.into();
        self.size += branches_count.v as usize;
        self.maxheight = self.maxheight.max(height + 1);

        unsafe { ExpandedSwitch::Branching(node_id.cast()) }
    }

    pub fn skip_policy(&mut self, node: NodeId<Branching>) -> NodeId<Evaluated> {
        let data = &self.arena.nodes[node.index as usize];
        let start = data.branch_start as usize;
        let count = data.branch_count.v as usize;

        let even_prob = Probability::new((count as f32).recip());
        for branch in &mut self.arena.branches[start..start + count] {
            branch.policy = even_prob;
        }

        self.arena.nodes[node.index as usize].state = NodeState::Evaluated;
        unsafe { node.cast() }
    }

    pub fn set_policy(&mut self, node: NodeId<Branching>, policy: &Policy) -> NodeId<Evaluated> {
        let data = &self.arena.nodes[node.index as usize];
        let start = data.branch_start as usize;
        let count = data.branch_count.v as usize;

        debug_assert_eq!(
            count,
            policy.len(),
            "There has to be exactly one policy for each branch."
        );

        for (branch, p) in self.arena.branches[start..start + count]
            .iter_mut()
            .zip(policy.iter())
        {
            branch.policy = p;
        }

        self.arena.nodes[node.index as usize].state = NodeState::Evaluated;
        unsafe { node.cast() }
    }

    pub fn apply_policy_noise(&mut self, node: NodeId<Evaluated>, noise: &Policy, eps: Ratio) {
        let data = &self.arena.nodes[node.index as usize];
        let start = data.branch_start as usize;
        let count = data.branch_count.v as usize;

        for (branch, noise) in self.arena.branches[start..start + count]
            .iter_mut()
            .zip(noise.iter())
        {
            branch.policy.mix(noise, eps);
        }

        // todo: it is not guaranteed by the Policy type or something, that the
        // branch policies still sum to 1. it is implied by the math
        // here, but i would like this to be explicit.
    }

    pub fn update_node<S: HasValue>(&mut self, node: NodeId<S>, value: eval::Value, weight: f32) {
        self.arena.nodes[node.index as usize].visits += weight as u32;
        self.arena.nodes[node.index as usize].value += value.v() * weight;
    }

    pub fn set_proven<S: HasValue>(&mut self, node: NodeId<S>, state: Proven, weight: f32) {
        self.arena.nodes[node.index as usize].visits += weight as u32;
        self.arena.nodes[node.index as usize].value = Value::from(state);
    }

    pub fn apply_virtual_loss(&mut self, node: RtNodeId, amount: u32) {
        self.arena.nodes[node.index as usize].visits += amount;
    }

    pub fn revert_virtual_loss(&mut self, node: RtNodeId, amount: u32) {
        self.arena.nodes[node.index as usize].visits -= amount;
    }

    /// Double-buffering Garbage Collection implementation.
    /// Discards dead branches and retains only the subtree of the committed
    /// move.
    pub fn advance_to(&mut self, back_buffer: &mut Tree, new_root_index: RtNodeId) {
        back_buffer.arena.clear();
        let (_, new_size, terminal_nodes, new_height) =
            self.copy_subtree(new_root_index, &mut back_buffer.arena, 1);

        std::mem::swap(&mut self.arena, &mut back_buffer.arena);

        self.size = new_size;
        self.terminal_nodes = terminal_nodes;
        self.maxheight = new_height;
    }

    fn copy_subtree(
        &self,
        old_idx: RtNodeId,
        back_buffer: &mut ArenaBuffer,
        current_height: u16,
    ) -> (u32, usize, usize, Height) {
        let old_node = &self.arena.nodes[old_idx.index as usize];
        let new_idx = back_buffer.nodes.len() as u32;

        back_buffer.nodes.push(old_node.clone());

        let mut total_size = 1;
        let mut total_terminal_nodes = if old_node.state == NodeState::Terminal { 1 } else { 0 };
        let mut max_h = current_height;

        if old_node.branch_count > MoveIndex::from(0) {
            let branch_start = old_node.branch_start as usize;
            let branch_end = branch_start + old_node.branch_count.v as usize;
            let new_branch_start = back_buffer.branches.len() as u32;

            for branch_idx in branch_start..branch_end {
                let mut b = self.arena.branches[branch_idx].clone();
                b.node.index = 0;
                back_buffer.branches.push(b);
            }

            back_buffer.nodes[new_idx as usize].branch_start = new_branch_start;

            for (i, branch_idx) in (branch_start..branch_end).enumerate() {
                let old_branch = &self.arena.branches[branch_idx];
                let (child_new_idx, sub_size, sub_terminal_nodes, sub_height) =
                    self.copy_subtree(old_branch.node, back_buffer, current_height + 1);
                total_size += sub_size;
                total_terminal_nodes += sub_terminal_nodes;
                max_h = max_h.max(sub_height.0);

                back_buffer.branches[(new_branch_start as usize) + i]
                    .node
                    .index = child_new_idx;
            }
        }

        (new_idx, total_size, total_terminal_nodes, Height(max_h))
    }

    pub fn best_branch(&self, node_id: NodeId<Evaluated>) -> &Branch {
        self.branches(node_id)
            .iter()
            .max_by(|a, b| {
                self.node(a.node())
                    .partial_cmp(&self.node(b.node()))
                    .unwrap_or(Ordering::Equal)
            })
            .expect("Branching node should have branches")
    }

    pub fn best_move(&self, node_id: NodeId<Evaluated>) -> Move {
        self.best_branch(node_id).mov()
    }

    pub fn maybe_best_move(&self, node_id: RtNodeId) -> Option<Move> {
        self.node_switch(node_id)
            .get::<Evaluated>()
            .map(|node| self.best_move(node))
    }

    pub fn best_moves<S: HasBranches>(
        &self,
        node_id: NodeId<S>,
        threshold: Value,
    ) -> impl Iterator<Item = Move> {
        self.branches(node_id)
            .iter()
            .filter(move |b| self.node(b.node()).value() > threshold)
            .map(|b| b.mov())
    }

    pub fn line(&self, mut cmp: impl FnMut(&Branch, &Branch) -> Ordering) -> Path {
        let mut buf = Vec::new();
        let mut current = Self::ROOT_IDX;

        loop {
            let best_branch_opt = self.branches_rt(current).iter().max_by(|a, b| cmp(a, b));

            if let Some(branch) = best_branch_opt {
                buf.push(branch.clone());
                current = branch.node;
            }
            else {
                break;
            }
        }

        Path(buf)
    }

    pub fn principal_line(&self) -> Path {
        self.line(|a, b| {
            let a = self.node(a.node());
            let b = self.node(b.node());
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        })
    }

    pub fn node_switch(&self, node_id: RtNodeId) -> Switch {
        let rt_state = self.arena.nodes[node_id.index as usize].state;
        Switch::new(node_id, rt_state)
    }

    pub fn try_node<S: node_state::Valid>(&'_ self, node_id: RtNodeId) -> Option<NodeView<'_, S>> {
        self.node_switch(node_id).get::<S>().map(|x| self.node(x))
    }

    pub fn node<S: node_state::Any>(&'_ self, node_id: NodeId<S>) -> NodeView<'_, S> {
        NodeView::new(self, node_id)
    }

    pub fn branch_id<S: HasBranches>(
        &self,
        node_id: NodeId<S>,
        mov_index: MoveIndex,
    ) -> Option<BranchId> {
        let node = self.node(node_id);
        let data = node.data();
        if mov_index < data.branch_count {
            Some(BranchId::new(data.branch_start + mov_index.v as u32))
        }
        else {
            None
        }
    }

    pub fn branch(&self, branch: BranchId) -> &Branch {
        unsafe { self.arena.branches.get_unchecked(branch.index()) }
    }
}

// -----------------------------------------------------------------------------
// Typestates & Zero-Cost Views
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BranchId {
    index: u32,
}

impl BranchId {
    fn new(index: u32) -> Self {
        Self { index }
    }

    pub fn index(&self) -> usize {
        self.index as usize
    }
}

pub type RtNodeId = NodeId<node_state::Unknown>;

/// The Typestate ID is just a zero-cost wrapper around a `u32` index.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId<S: node_state::Any> {
    pub index: u32,
    _marker: PhantomData<S>,
}

impl From<usize> for NodeId<node_state::Unknown> {
    fn from(value: usize) -> Self {
        NodeId::new(value as u32)
    }
}

impl<S: node_state::Any> NodeId<S> {
    pub const fn new(index: u32) -> Self {
        Self { index, _marker: PhantomData }
    }

    /// # Safety
    /// The caller must ensure that the underlying node at this index is
    /// actually of the target typestate `T`. This is guaranteed by the internal
    /// logic of the Tree, but cannot be enforced by the type system.
    pub unsafe fn cast<T: node_state::Any>(self) -> NodeId<T> {
        NodeId::new(self.index)
    }

    pub fn down_cast(self) -> RtNodeId {
        RtNodeId::new(self.index)
    }

    pub fn index(&self) -> usize {
        self.index as usize
    }
}

impl<S: const node_state::Valid> NodeId<S> {
    const STATE: NodeState = S::state();

    pub fn try_into<Target: node_state::Valid>(self) -> Option<NodeId<Target>> {
        if Self::STATE == Target::state() {
            Some(NodeId::new(self.index))
        }
        else {
            None
        }
    }

    pub fn into_ct(self) -> Switch {
        unsafe {
            match Self::STATE {
                NodeState::Leaf => Switch::Leaf(self.cast()),
                NodeState::Branching => Switch::Branching(self.cast()),
                NodeState::Terminal => Switch::Terminal(self.cast()),
                NodeState::Evaluated => Switch::Evaluated(self.cast()),
            }
        }
    }
}

impl<S: node_state::Any> fmt::Debug for NodeId<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CtNodeId({})", self.index)
    }
}

#[derive(Clone, Copy)]
pub struct NodeView<'a, S: node_state::Any> {
    pub tree: &'a Tree,
    pub id: NodeId<S>,
}

impl<'a, S: node_state::Any> PartialEq for NodeView<'a, S> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.tree, other.tree) && self.id == other.id
    }
}

impl<S: node_state::Any> PartialOrd for NodeView<'_, S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // compare the proven-tiers (WIN > DRAW/UNPROVEN > LOSS)
        let tier_self = Proven::try_from(self.value()).ok().unwrap_or(proven::DRAW);
        let tier_other = Proven::try_from(other.value()).ok().unwrap_or(proven::DRAW);
        let tier_ord = tier_self.cmp(&tier_other);

        match tier_ord {
            // fall back to standard mcts visits comparison
            Ordering::Equal => {
                let visits_ord = self.visits().cmp(&other.visits());
                match visits_ord {
                    Ordering::Equal => self.value().partial_cmp(&other.value()),
                    ordering => Some(ordering),
                }
            }
            // otherwise, strictly obey the proven ranking.
            ordering => Some(ordering),
        }
    }
}

impl<'a, S: node_state::Any> NodeView<'a, S> {
    pub fn new(tree: &'a Tree, id: NodeId<S>) -> Self {
        Self { tree, id }
    }

    #[inline]
    pub fn id(&self) -> NodeId<S> {
        self.id
    }

    #[inline]
    fn data(&self) -> &NodeData {
        &self.tree.arena.nodes[self.id.index as usize]
    }

    #[inline]
    pub fn state(&self) -> NodeState {
        self.data().state
    }

    #[inline]
    pub fn value(&self) -> Value {
        self.data().value
    }

    #[inline]
    pub fn visits(&self) -> VisitCount {
        self.data().visits
    }
}

impl<'a, S: node_state::Valid> NodeView<'a, S> {
    pub fn expected_state() -> NodeState {
        S::state()
    }
}

impl<'a, S: HasBranches> NodeView<'a, S> {
    #[inline]
    pub fn branches(&self) -> &'a [Branch] {
        self.tree.branches(self.id)
    }

    pub fn branch_count(&self) -> MoveIndex {
        self.data().branch_count
    }
}

impl<'a, S: HasValue> NodeView<'a, S> {
    #[inline]
    pub fn proven(&self) -> Option<Proven> {
        self.data().value.try_into().ok()
    }
}

/// The winrate of a node in range [0; 1].
#[derive(Debug, Clone, Copy)]
pub struct WinRate(pub Probability);

impl ops::Deref for WinRate {
    type Target = Probability;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl WinRate {
    #[inline(always)]
    pub const fn win() -> Self {
        Self(Probability::one())
    }

    #[inline(always)]
    pub const fn loss() -> Self {
        Self(Probability::zero())
    }

    #[inline(always)]
    pub const fn draw() -> Self {
        Self(Probability::even())
    }

    #[inline(always)]
    pub const fn inv(&self) -> Self {
        Self(self.0.inv())
    }
}

impl Default for WinRate {
    fn default() -> Self {
        Self::draw()
    }
}

impl<'a> From<NodeView<'a, Evaluated>> for WinRate {
    fn from(node: NodeView<'a, Evaluated>) -> Self {
        let visits = node.visits();
        let value = node.value();

        if value.is_proven_win() {
            return Self::win();
        }

        if value.is_proven_loss() {
            return Self::loss();
        }

        if visits == VisitCount(0) {
            Self::default()
        }
        else {
            let prob = Probability::new(value.0 / (visits.0 as f32));
            Self(prob)
        }
    }
}

impl From<WinRate> for eval::Value {
    fn from(win_rate: WinRate) -> Self {
        Self::new(win_rate.0.v())
    }
}

impl fmt::Display for WinRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}%", self.0.v() * 100.)
    }
}
