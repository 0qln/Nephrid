use itertools::Itertools;

use crate::core::{
    Move, Position,
    move_iter::fold_legal_moves,
    search::mcts::{
        eval::{self, Policy, RawPolicy},
        node::ops::ControlFlow,
    },
};
use std::{
    cell::{Ref, RefCell},
    cmp::Ordering,
    fmt,
    marker::PhantomData,
    ops,
    rc::Rc,
};

#[cfg(test)]
pub mod test;

#[derive(Default, Debug, Clone)]
pub struct Tree {
    /// Root of the tree.
    root: AnyNodeRef,
}

impl Tree {
    pub fn new(root: AnyNodeRef) -> Self {
        Self { root }
    }

    pub fn advance_best(&mut self) {
        let root = self.root.expanded();
        let node = root.map(|x| x.borrow().select_best().node());
        if let Some(node) = node {
            self.root = node;
        }
    }

    pub fn advance_to<F: Fn(&Branch) -> bool>(&mut self, pred: F) {
        let node = self
            .root
            .expanded()
            .map(|x| {
                let root = x.borrow();
                let branch = root.branches().iter().find(|x| pred(x));
                let node = branch.map(|b| b.node());
                node
            })
            .flatten();
        if let Some(node) = node {
            self.root = node;
        }
    }

    /// Returns None if there are no moves.
    pub fn best_move(&self) -> Option<Move> {
        let root = self.root.expanded()?;
        let best = Ref::map(root.borrow(), |n| n.select_best());
        Some(best.mov())
    }

    pub fn best_moves(&self, threshold: Value) -> Vec<Move> {
        let root = self.root.data();
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
            match current {
                AnyNodeRef::Expanded(expanded) => {
                    let branch = expanded.borrow().select_best().clone();
                    let node = branch.node();
                    buf.push(branch);
                    current = node;
                }
                _ => {
                    break;
                }
            }
        }
        Path(buf)
    }

    /// Retruns the number of nodes in this tree
    pub fn size(&self) -> usize {
        self.get_root().data().subtree_size()
    }

    /// Returns the max depth of the tree.
    /// e.g.
    /// when we only have root -> 0
    /// when root has 1 child -> 1
    /// when roto has 2 children, where 1 with a child -> 2
    pub fn maxdepth(&self) -> usize {
        self.get_root().data().subtree_maxdepth()
    }

    /// Returns the min depth of the tree.
    /// e.g.
    /// when we only have root -> 0
    /// when root has 1 child -> 1
    /// when roto has 2 children, where 1 with a child -> 1
    pub fn mindepth(&self) -> usize {
        self.get_root().data().subtree_mindepth()
    }

    pub fn get_root(&self) -> AnyNodeRef {
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

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub enum NodeState {
    /// A leaf is an untouched node.
    #[default]
    Leaf,
    /// An expanded node is a node which has been analized and to be found to
    /// have children.
    Expanded,
    /// A terminal node is a node which has been analized and to be found to
    /// have no children.
    Terminal,
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct Branch {
    /// The node that this branch leads to.
    node: AnyNodeRef,

    /// The policy of picking this branch.
    policy: f32,

    /// The move that lead to this node.
    mov: Move,
}

impl Branch {
    pub fn new(mov: Move, policy: f32, node: AnyNodeRef) -> Self {
        Self { node, policy, mov }
    }

    pub fn mov(&self) -> Move {
        self.mov
    }

    pub fn policy(&self) -> f32 {
        self.policy
    }

    pub fn node(&self) -> AnyNodeRef {
        self.node.clone()
    }

    pub fn visits(&self) -> u32 {
        self.node.data().visits()
    }

    pub fn value(&self) -> Value {
        self.node.data().value()
    }

    // pub fn node_state(&self) -> NodeState {
    //     self.node.data().state()
    // }

    pub fn set_policy(&mut self, policy: f32) {
        self.policy = policy;
    }
}

/// The value of a node.
/// positive ~> good for current player at this node
/// negative ~> bad for current player at this node
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

#[derive(Clone, Default, Debug, PartialEq)]
struct Leaf;

#[derive(Clone, Default, Debug, PartialEq)]
struct Terminal;

#[derive(Clone, Default, Debug, PartialEq)]
struct Expanded;

pub type NodeRef<S> = Rc<RefCell<Node<S>>>;

#[derive(Clone, Debug, PartialEq)]
pub enum AnyNodeRef {
    Terminal(NodeRef<Terminal>),
    Leaf(NodeRef<Leaf>),
    Expanded(NodeRef<Expanded>),
}

impl AnyNodeRef {
    pub fn new_terminal(data: NodeData) -> Self {
        Self::Terminal(Rc::new(RefCell::new(Node::<Terminal>::new(data))))
    }

    pub fn new_expanded(data: NodeData) -> Self {
        Self::Expanded(Rc::new(RefCell::new(Node::<Expanded>::new(data))))
    }

    pub fn new_leaf(data: NodeData) -> Self {
        Self::Leaf(Rc::new(RefCell::new(Node::<Leaf>::new(data))))
    }

    pub fn expanded(&self) -> Option<NodeRef<Expanded>> {
        match self {
            Self::Expanded(x) => Some(x.clone()),
            _ => None,
        }
    }

    pub fn leaf(&self) -> Option<NodeRef<Leaf>> {
        match self {
            Self::Leaf(x) => Some(x.clone()),
            _ => None,
        }
    }

    pub fn data(&self) -> Ref<'_, NodeData> {
        match self {
            AnyNodeRef::Terminal(ref_cell) => Ref::map(ref_cell.borrow(), |node| node.data()),
            AnyNodeRef::Leaf(ref_cell) => Ref::map(ref_cell.borrow(), |node| node.data()),
            AnyNodeRef::Expanded(ref_cell) => Ref::map(ref_cell.borrow(), |node| node.data()),
        }
    }
}

impl Default for AnyNodeRef {
    fn default() -> Self {
        Self::Leaf(Default::default())
    }
}

#[derive(Clone, Default, PartialEq)]
struct NodeData {
    // todo: could move this into [State]Expanded {...}, such that we don't carry this around in
    // leafs and terminal nodes.
    /// All the branches from this node.
    branches: Vec<Branch>,

    /// The number of times this node was visited.
    pub visits: u32,

    /// The value of this node. (~sums all the values of it's children)
    pub value: Value,
    // todo: put this behind a generic PAYLOAD parameter or something, this is currently only used
    // for training and should thus not be here in production.
    // /// win/draw/loss count
    // pub terminal_wdl: WDL,
}

impl NodeData {
    fn visits(&self) -> u32 {
        self.visits
    }

    fn value(&self) -> Value {
        self.value
    }

    fn branches(&self) -> &[Branch] {
        &self.branches
    }

    /// Returns the number of nodes in all subsequent branches + 1 (for this
    /// node).
    pub fn subtree_size(&self) -> usize {
        1 + self
            .branches()
            .iter()
            .map(|b| b.node().data().subtree_size())
            .sum::<usize>()
    }

    /// Retruns the max depth of this node's subtree.
    /// Returns 0 if there are no children.
    pub fn subtree_maxdepth(&self) -> usize {
        self.branches()
            .iter()
            .map(|b| 1 + b.node().data().subtree_maxdepth())
            .max()
            .unwrap_or(0)
    }

    /// Retruns the min depth of this node's subtree.
    /// Returns 0 if there are no children.
    pub fn subtree_mindepth(&self) -> usize {
        self.branches()
            .iter()
            .map(|b| 1 + b.node().data().subtree_mindepth())
            .min()
            .unwrap_or(0)
    }
}

#[derive(Clone, Default, PartialEq)]
pub struct Node<State> {
    /// The current state of this node.
    _state: PhantomData<State>,

    /// The data.
    data: NodeData,
}

// impl<S1, S2> From<Node<S1>> for Node<S2> {
//     fn from(Node { data, .. }: Node<S1>) -> Self {
//         Node::<S2> { data, _state: PhantomData }
//     }
// }

impl<S> fmt::Debug for Node<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("value", &self.value())
            .field("visits", &self.visits())
            // .field("state", &self.state())
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
    /// Expand the node.
    pub fn expand(mut self, pos: &Position) -> AnyNodeRef {
        _ = fold_legal_moves(pos, &mut self.data.branches, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Branch::new(
                    m,
                    0.0,
                    AnyNodeRef::new_leaf(Default::default()),
                ));
                acc
            })
        });

        if self.data.branches.is_empty() {
            AnyNodeRef::new_terminal(self.data)
        }
        else {
            AnyNodeRef::new_expanded(self.data)
        }
    }
}

impl Node<Expanded> {
    /// Sort the branches in ascending order.
    pub fn sort_by<T: Ord>(&mut self, f: impl Fn(&Branch) -> T) {
        // todo: the sorting can be done a lot more efficiently:
        // The puct score does not change very often later on, only as we start the
        // search. Also we might only need the first few branches if MPV is low.
        self.data.branches.sort_by_key(f);
    }

    pub fn get_branch(&self, index: usize) -> Option<&Branch> {
        self.data.branches.get(index)
    }

    pub fn set_branches(&mut self, branches: Vec<Branch>) {
        self.data.branches = branches;
    }

    /// Whether this node has branches.
    pub fn has_branches(&self) -> bool {
        !self.data.branches.is_empty()
    }

    /// The number of branches this node has
    pub fn num_branches(&self) -> usize {
        self.data.branches.len()
    }

    /// Select the branch with the most visits.
    pub fn select_best(&self) -> &Branch {
        self.select(|b| b.visits())
    }

    pub fn take_best(self) -> Branch {
        self.take(|b| b.visits())
    }

    /// Returns None if there are no branches.
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

    /// Sets the policies of the branches.
    pub fn set_policy(&mut self, policy: &Policy) {
        assert_eq!(
            self.data.branches.len(),
            policy.len(),
            "There has to be exactly one policy for each branch."
        );

        for (i, branch) in self.data.branches.iter_mut().enumerate() {
            branch.policy = policy.get(i).unwrap();
        }
    }

    /// Sets the policies of the branches.
    pub fn set_policy_raw(&mut self, raw_policy: &RawPolicy) {
        let moves = self.data.branches.iter().map(|b| usize::from(b.mov()));
        let policy = Policy::from_raw(raw_policy, moves)
            .expect("Shouldn't be None, since the moves are correct for this node.");

        self.set_policy(&policy);
    }
}

impl<S> Node<S> {
    pub fn new(data: NodeData) -> Self {
        Self { data, _state: PhantomData }
    }

    fn data(&self) -> &NodeData {
        &self.data
    }

    pub fn visits(&self) -> u32 {
        self.data.visits
    }

    pub fn value(&self) -> Value {
        self.data.value
    }

    pub fn branches(&self) -> &[Branch] {
        &self.data.branches
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

    ///// Applies `f` to this and all child nodes, until no more child is found or
    ///// `f` returns residual.
    //pub fn try_fold_down<B, F, R>(this: Rc<RefCell<Self>>, mut init: B, mut f: F)
    // -> R where
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
