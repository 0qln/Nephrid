use crate::{
    core::{
        depth::Depth,
        r#move::MoveIndex,
        search::mcts::{
            eval::{Probability, Ratio},
            nn::PolicyHeadIndex,
            node::node_state::{ExpandedSwitch, HasBranches, HasValue, NodeState, Switch},
        },
        zobrist::{self, ZobristBuildHasher},
    },
    impl_variants,
};
use itertools::Itertools;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
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

#[derive(Clone, Default)]
struct ArenaBuffer {
    nodes: HashMap<zobrist::Hash, NodeData, ZobristBuildHasher>,
    branches: Vec<Branch>,
}

impl ArenaBuffer {
    fn new(node_cap: usize, branch_cap: usize) -> Self {
        Self {
            nodes: HashMap::with_capacity_and_hasher(node_cap, ZobristBuildHasher::default()),
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
pub struct DAG {
    arena: ArenaBuffer,
    size: usize,
    maxheight: Height,
    terminal_nodes: usize,
    root: RtNodeId,
}

impl Default for DAG {
    fn default() -> Self {
        Self::new(&Position::default())
    }
}

impl DAG {
    pub fn new(pos: &Position) -> Self {
        let mut arena = ArenaBuffer::new(10000, 30000);
        let root_index = pos.get_key();
        arena.nodes.insert(root_index, NodeData::new_leaf());
        Self {
            arena,
            size: 1,
            terminal_nodes: 0,
            maxheight: Height::ROOT,
            root: RtNodeId::from(root_index),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn terminal_nodes(&self) -> usize {
        self.terminal_nodes
    }

    pub fn compute_subtree_size(&self, root: RtNodeId) -> usize {
        let mut stack = vec![root];
        let mut visited = std::collections::HashSet::new();
        let mut size = 0;

        while let Some(node_id) = stack.pop() {
            if !visited.insert(*node_id.index()) {
                continue;
            }
            size += 1;
            for branch in self.branches_rt(node_id) {
                stack.push(branch.node);
            }
        }
        size
    }

    pub fn compute_subtree_terminal_nodes_count(&self, root: RtNodeId) -> usize {
        let mut stack = vec![root];
        let mut visited = std::collections::HashSet::new();
        let mut count = 0;

        while let Some(node_id) = stack.pop() {
            if !visited.insert(*node_id.index()) {
                continue;
            }
            if self.node(node_id).state() == NodeState::Terminal {
                count += 1;
            }
            for branch in self.branches_rt(node_id) {
                stack.push(branch.node);
            }
        }
        count
    }

    // /// Counts the wins of the root node player.
    // pub fn count_wins(&self) -> usize {
    //     self.count_subtree_nodes(Depth::ROOT, Self::ROOT_IDX, &|_node, _depth|
    // todo!()) }

    pub fn count_nodes(
        &self,
        pred: &impl Fn(NodeView<node_state::Unknown>, Depth) -> bool,
        max: usize,
    ) -> usize {
        self.count_subtree_nodes(Depth::ROOT, self.root(), pred, max)
    }

    pub fn count_subtree_nodes(
        &self,
        depth: Depth,
        node_id: RtNodeId,
        pred: &impl Fn(NodeView<node_state::Unknown>, Depth) -> bool,
        max: usize,
    ) -> usize {
        if max == 0 {
            return 0;
        }

        let node = self.node(node_id);
        let mut total = if pred(node, depth) { 1 } else { 0 };

        for b in self.branches_rt(node_id) {
            if total >= max {
                break;
            }

            total += self.count_subtree_nodes(depth + 1, b.node, pred, max - total);
        }

        total
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
            tree: &'a DAG,
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

    pub fn compute_subtree_maxheight(&self, root: RtNodeId) -> Height {
        struct Frame {
            node: RtNodeId,
            next_child: usize,
            max_child_height: Height,
        }

        let mut stack = vec![Frame {
            node: root,
            next_child: 0,
            max_child_height: Height::EMPTY,
        }];
        let mut heights = std::collections::HashMap::<zobrist::Hash, Height>::new();

        while let Some(frame) = stack.last_mut() {
            let branches = self.branches_rt(frame.node);
            if frame.next_child < branches.len() {
                let child = branches[frame.next_child].node;
                frame.next_child += 1;
                if let Some(&h) = heights.get(child.index()) {
                    frame.max_child_height = frame.max_child_height.max(h);
                }
                else {
                    stack.push(Frame {
                        node: child,
                        next_child: 0,
                        max_child_height: Height::EMPTY,
                    });
                }
            }
            else {
                let node_height = Height::ROOT + frame.max_child_height;
                heights.insert(*frame.node.index(), node_height);
                stack.pop();
                if let Some(parent) = stack.last_mut() {
                    parent.max_child_height = parent.max_child_height.max(node_height);
                }
            }
        }

        heights.get(root.index()).copied().unwrap_or(Height::EMPTY)
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
        self.root
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
        let data = self.node_data(node_id);
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
        let parent_data = self.node_data(parent_id);
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
    pub fn backpropagate(&mut self, path: &[RtNodeId], final_value: eval::Value) {
        for &idx in path {
            let data = self.node_data_mut(idx);
            data.visits.0 += 1;
            data.value += final_value;
        }
    }

    /// Expands a leaf node, creating branches and updating tree statistics
    /// immutably.
    pub fn expand_node(
        &mut self,
        node_id: NodeId<Leaf>,
        pos: &mut Position,
        search_depth: Depth,
    ) -> ExpandedSwitch {
        if pos.game_result().is_some() {
            self.node_data_mut(node_id).state = NodeState::Terminal;
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

        let mut new_nodes = 0;
        for m in moves {
            // todo:
            // 1. this is super slow
            // 2. this should probably be made somewhere else down the line
            let index = {
                pos.make_move(m);
                let key = pos.get_key();
                pos.unmake_move(m);
                key
            };

            let child = RtNodeId::from(index);
            if !self.arena.nodes.contains_key(&index) {
                self.arena.nodes.insert(index, NodeData::new_leaf());
                new_nodes += 1;
            }

            self.arena.branches.push(Branch {
                node: child,
                policy: Probability::zero(),
                mov: m,
            });
        }

        // Link parent to branches
        let parent = self.node_data_mut(node_id);
        parent.branch_start = branch_start;
        parent.branch_count = branch_count;
        parent.state = NodeState::Branching;

        let height: Height = search_depth.into();
        self.size += new_nodes;
        self.maxheight = self.maxheight.max(height + 1);

        unsafe { ExpandedSwitch::Branching(node_id.cast()) }
    }

    pub fn skip_policy(&mut self, node: NodeId<Branching>) -> NodeId<Evaluated> {
        let data = self.node_data(node);
        let start = data.branch_start as usize;
        let count = data.branch_count.v as usize;

        let even_prob = Probability::new((count as f32).recip());
        for branch in &mut self.arena.branches[start..start + count] {
            branch.policy = even_prob;
        }

        self.node_data_mut(node).state = NodeState::Evaluated;
        unsafe { node.cast() }
    }

    pub fn set_policy(&mut self, node: NodeId<Branching>, policy: &Policy) -> NodeId<Evaluated> {
        let data = self.node_data(node);
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

        self.node_data_mut(node).state = NodeState::Evaluated;
        unsafe { node.cast() }
    }

    pub fn apply_policy_noise(&mut self, node: NodeId<Evaluated>, noise: &Policy, eps: Ratio) {
        let data = self.node_data(node);
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
        self.node_data_mut(node).visits += weight as u32;
        self.node_data_mut(node).value += value.v() * weight;
    }

    pub fn set_proven<S: HasValue>(&mut self, node: NodeId<S>, state: Proven, weight: f32) {
        self.node_data_mut(node).visits += weight as u32;
        self.node_data_mut(node).value = Value::from(state);
    }

    pub fn apply_virtual_loss(&mut self, node: RtNodeId, amount: u32) {
        self.node_data_mut(node).visits += amount;
    }

    fn node_data<S: node_state::Any>(&self, node: NodeId<S>) -> &NodeData {
        self.arena
            .nodes
            .get(node.index())
            .expect("Node ID should be valid")
    }

    fn node_data_mut<S: node_state::Any>(&mut self, node: NodeId<S>) -> &mut NodeData {
        self.arena
            .nodes
            .get_mut(node.index())
            .expect("Node ID should be valid")
    }

    pub fn revert_virtual_loss(&mut self, node: RtNodeId, amount: u32) {
        self.node_data_mut(node).visits -= amount;
    }

    /// Double-buffering Garbage Collection implementation.
    /// Discards dead branches and retains only the subtree of the committed
    /// move.
    pub fn advance_to(&mut self, back_buffer: &mut DAG, new_root_index: RtNodeId) {
        back_buffer.arena.clear();
        let (new_size, terminal_nodes, new_height) =
            self.copy_subtree(new_root_index, &mut back_buffer.arena, 1);
        std::mem::swap(&mut self.arena, &mut back_buffer.arena);
        self.size = new_size;
        self.terminal_nodes = terminal_nodes;
        self.maxheight = new_height;
        self.root = new_root_index;
    }

    // todo: this gc is not needed anymore when we use a fixed size arena/tt-table
    // instead for a dynamically sized hashmap
    fn copy_subtree(
        &self,
        old_idx: RtNodeId,
        back_buffer: &mut ArenaBuffer,
        current_height: u16,
    ) -> (usize, usize, Height) {
        let mut visited = HashSet::new();
        self.copy_subtree_internal(old_idx, back_buffer, current_height, &mut visited)
    }

    fn copy_subtree_internal(
        &self,
        old_idx: RtNodeId,
        back_buffer: &mut ArenaBuffer,
        current_height: u16,
        visited: &mut HashSet<zobrist::Hash>,
    ) -> (usize, usize, Height) {
        let key = *old_idx.index();

        // If already copied, skip (shared node in DAG)
        if visited.contains(&key) {
            return (0, 0, Height(0));
        }
        visited.insert(key);

        let old_node = &self.arena.nodes[&key];
        let is_terminal = old_node.state == NodeState::Terminal;

        let mut total_size = 1;
        let mut total_terminal = if is_terminal { 1 } else { 0 };
        let mut max_h = Height(current_height);

        // Copy branches if any
        let branch_start_old = old_node.branch_start as usize;
        let branch_count = old_node.branch_count.v as usize;
        let new_branch_start = back_buffer.branches.len() as u32;

        // Reserve space for the new branches
        if branch_count > 0 {
            // Extend with clones of the original branches (will be updated with child
            // results later)
            let old_branches =
                &self.arena.branches[branch_start_old..branch_start_old + branch_count];
            back_buffer.branches.extend(old_branches.iter().cloned());

            // Process each child and accumulate stats
            for branch in old_branches.iter() {
                let (child_size, child_terminal, child_height) = self.copy_subtree_internal(
                    branch.node,
                    back_buffer,
                    current_height + 1,
                    visited,
                );
                total_size += child_size;
                total_terminal += child_terminal;
                if child_height > max_h {
                    max_h = child_height;
                }
                // The branch already contains the correct child node hash, no
                // update needed
            }
        }

        // Create new NodeData pointing to the freshly copied branches
        let mut new_node = old_node.clone();
        new_node.branch_start = new_branch_start;
        new_node.branch_count = MoveIndex::from(branch_count as u8);
        back_buffer.nodes.insert(key, new_node);

        (total_size, total_terminal, max_h)
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
        let mut current = self.root();

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
        let rt_state = self.node(node_id).state();
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

pub type TNodeId = zobrist::Hash;

/// The Typestate ID is just a zero-cost wrapper around a `u32` index.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId<S: node_state::Any> {
    pub index: TNodeId,
    _marker: PhantomData<S>,
}

impl From<TNodeId> for NodeId<node_state::Unknown> {
    fn from(value: TNodeId) -> Self {
        NodeId::new(value)
    }
}

impl<S: node_state::Any> NodeId<S> {
    pub const fn new(index: TNodeId) -> Self {
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

    pub fn index(&self) -> &TNodeId {
        &self.index
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
    pub tree: &'a DAG,
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
    pub fn new(tree: &'a DAG, id: NodeId<S>) -> Self {
        Self { tree, id }
    }

    #[inline]
    pub fn id(&self) -> NodeId<S> {
        self.id
    }

    #[inline]
    fn data(&self) -> &NodeData {
        &self
            .tree
            .arena
            .nodes
            .get(self.id.index())
            .expect("NodeId should always point to a valid node")
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

impl From<eval::Value> for WinRate {
    fn from(value: eval::Value) -> Self {
        Self(Probability::new(value.v()))
    }
}

impl fmt::Display for WinRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}%", self.0.v() * 100.)
    }
}
