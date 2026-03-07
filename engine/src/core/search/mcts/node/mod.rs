use itertools::Itertools;

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
    root: AnyNodeRef,
}

impl Tree {
    pub fn new(root: AnyNodeRef) -> Self {
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
    #[default]
    Leaf,
    Expanded,
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
        self.node.borrow().visits()
    }

    pub fn value(&self) -> Value {
        self.node.borrow().value()
    }

    pub fn set_policy(&mut self, policy: f32) {
        self.policy = policy;
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

pub type NodeRef<S> = Rc<RefCell<Node<S>>>;

pub type AnyNodeRef = Rc<RefCell<AnyNode>>;

#[derive(Clone, Debug, PartialEq)]
pub enum AnyNode {
    Terminal(Node<Terminal>),
    Leaf(Node<Leaf>),
    Branching(Node<Branching>),
    Evaluated(Node<Evaluated>),
}

impl AnyNode {
    pub fn new_terminal(data: NodeData) -> Self {
        Self::Terminal(Node::<Terminal>::new(data))
    }

    pub fn new_branching(data: NodeData) -> Self {
        Self::Branching(Node::<Branching>::new(data))
    }

    pub fn new_leaf(data: NodeData) -> Self {
        Self::Leaf(Node::<Leaf>::new(data))
    }

    pub fn new_evaluated(data: NodeData) -> Self {
        Self::Evaluated(Node::<Evaluated>::new(data))
    }

    pub fn branching(&self) -> Option<&Node<Branching>> {
        match self {
            Self::Branching(x) => Some(x),
            _ => None,
        }
    }

    pub fn leaf(&self) -> Option<&Node<Leaf>> {
        match self {
            Self::Leaf(x) => Some(x),
            _ => None,
        }
    }

    pub fn evaluated(&self) -> Option<&Node<Evaluated>> {
        match self {
            Self::Evaluated(x) => Some(x),
            _ => None,
        }
    }

    pub fn terminal(&self) -> Option<&Node<Terminal>> {
        match self {
            Self::Terminal(x) => Some(x),
            _ => None,
        }
    }

    pub fn data(&self) -> &NodeData {
        match self {
            Self::Terminal(n) => n.data(),
            Self::Leaf(n) => n.data(),
            Self::Branching(n) => n.data(),
            Self::Evaluated(n) => n.data(),
        }
    }

    pub fn data_mut(&mut self) -> &mut NodeData {
        match self {
            Self::Terminal(n) => n.data_mut(),
            Self::Leaf(n) => n.data_mut(),
            Self::Branching(n) => n.data_mut(),
            Self::Evaluated(n) => n.data_mut(),
        }
    }

    // Dynamic Delegation Methods
    pub fn visits(&self) -> u32 {
        self.data().visits()
    }
    pub fn value(&self) -> Value {
        self.data().value()
    }
    pub fn branches(&self) -> &[Branch] {
        self.data().branches()
    }
    pub fn subtree_size(&self) -> usize {
        self.data().subtree_size()
    }
    pub fn subtree_maxdepth(&self) -> usize {
        self.data().subtree_maxdepth()
    }
    pub fn subtree_mindepth(&self) -> usize {
        self.data().subtree_mindepth()
    }

    pub fn update(&mut self, value: eval::Value) {
        let data = self.data_mut();
        data.visits += 1;
        data.value += value;
    }
}

impl Default for AnyNode {
    fn default() -> Self {
        Self::Leaf(Default::default())
    }
}

#[derive(Clone, Default, PartialEq)]
pub struct NodeData {
    /// All the branches from this node.
    branches: Vec<Branch>,

    /// The number of times this node was visited.
    pub visits: u32,

    /// The value of this node. (~sums all the values of it's children)
    pub value: Value,
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
}

pub mod node_state {
    pub trait Any {}
    pub trait Expanded: Any {}
    pub trait Branched: Any {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Leaf;
    impl Any for Leaf {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Terminal;
    impl Any for Terminal {}
    impl Expanded for Terminal {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Branching;
    impl Any for Branching {}
    impl Expanded for Branching {}
    impl Branched for Branching {}

    #[derive(Clone, Default, Debug, PartialEq)]
    pub struct Evaluated;
    impl Any for Evaluated {}
    impl Expanded for Evaluated {}
    impl Branched for Evaluated {}
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
    pub fn expand(mut self, pos: &Position) -> AnyNode {
        _ = fold_legal_moves(pos, &mut self.data.branches, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Branch::new(
                    m,
                    0.0,
                    Rc::new(RefCell::new(AnyNode::new_leaf(Default::default()))),
                ));
                acc
            })
        });

        if self.data.branches.is_empty() {
            AnyNode::new_terminal(self.data)
        }
        else {
            AnyNode::new_branching(self.data)
        }
    }
}

impl<S: node_state::Expanded> Node<S> {
    pub fn has_branches(&self) -> bool {
        !self.data.branches.is_empty()
    }
}

impl<S: node_state::Branched> Node<S> {
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
    /// Consumes the Branching Node and strictly returns an Evaluated Node!
    pub fn set_policy(mut self, policy: &Policy) -> Node<Evaluated> {
        assert_eq!(
            self.branches().len(),
            policy.len(),
            "There has to be exactly one policy for each branch."
        );

        for (i, branch) in self.data.branches.iter_mut().enumerate() {
            branch.set_policy(policy.get(i).unwrap());
        }

        Node::<Evaluated>::new(self.data)
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

impl<S: node_state::Any> Node<S> {
    pub fn new(data: NodeData) -> Self {
        Self { data, _state: PhantomData }
    }

    pub fn data(&self) -> &NodeData {
        &self.data
    }

    pub(crate) fn data_mut(&mut self) -> &mut NodeData {
        &mut self.data
    }

    pub fn visits(&self) -> u32 {
        self.data.visits
    }

    pub fn value(&self) -> Value {
        self.data.value
    }

    pub fn update(&mut self, value: eval::Value) {
        self.data.visits += 1;
        self.data.value += value;
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
