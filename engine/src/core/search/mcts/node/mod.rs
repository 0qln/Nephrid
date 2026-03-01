use bumpalo::Bump;
use itertools::Itertools;

use crate::core::{
    Move, Position,
    move_iter::fold_legal_moves,
    search::mcts::{
        eval::{Policy, RawPolicy},
        node::ops::ControlFlow,
    },
};
use std::{
    assert_matches::{assert_matches, debug_assert_matches},
    cell::RefCell,
    cmp::Ordering,
    fmt, ops,
};

// todo: fix test
// #[cfg(test)]
// pub mod test;

#[derive(Debug)]
pub struct Tree<'bump> {
    /// Root of the tree.
    root: NodeRef<'bump>,
}

impl<'bump> Tree<'bump> {
    pub fn new_in(bump: &'bump Bump) -> Self {
        Self {
            // Allocates the root directly into the arena
            root: bump.alloc(RefCell::new(Node::leaf_in(bump))),
        }
    }

    /// Advances the root pointer *within the same arena*.
    /// Note: This leaks the dropped branches until the whole arena is
    /// dropped/reset.
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

    /// Consumes this tree, advances to the best branch, and deeply copies the
    /// retained subtree into a `new_bump` arena.
    /// This is the proper way to free memory of unselected branches.
    pub fn into_advance_best<'new>(self, new_bump: &'new Bump) -> Option<Tree<'new>> {
        let root = self.root.borrow();
        let branch = root.select_best()?;

        Some(Tree {
            root: branch.node().borrow().clone_into(new_bump),
        })
    }

    /// Advances the root pointer *within the same arena*.
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

    /// Consumes this tree, advances to the matched branch, and deeply copies
    /// the retained subtree into a `new_bump` arena.
    pub fn into_advance_to<'new: 'bump, F: Fn(&Branch) -> bool>(
        &mut self,
        new_bump: &'new Bump,
        pred: F,
    ) -> Option<()> {
        let root = self.root.borrow();
        let branch = root.iter_branches().find(|x| pred(x))?;

        self.root = branch.node().borrow().clone_into(new_bump);

        Some(())
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
    pub fn principal_variation(&self) -> Path<'bump> {
        let mut buf = Vec::new();
        let mut current = self.root;
        loop {
            let state = current.borrow().state();
            match state {
                NodeState::Expanded => {
                    debug_assert!(
                        !current.borrow().branches.is_empty(),
                        "Contradiction: NodeState == Expanded, but there are no branches."
                    );

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
        Path(buf)
    }

    /// Retruns the number of nodes in this tree
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

    pub fn get_root(&self) -> NodeRef<'bump> {
        self.root
    }
}

pub struct Path<'bump>(pub Vec<Branch<'bump>>);

impl<'bump> Path<'bump> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'bump> fmt::Display for Path<'bump> {
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

// Derived Copy since NodeRef<'bump> and Move are Copy.
#[derive(Clone, Debug, PartialEq)]
pub struct Branch<'bump> {
    /// The node that this branch leads to.
    node: NodeRef<'bump>,

    /// The policy of picking this branch.
    policy: f32,

    /// The move that lead to this node.
    mov: Move,
}

impl<'bump> Branch<'bump> {
    pub fn new_in(m: Move, policy: f32, bump: &'bump Bump) -> Self {
        Self {
            node: bump.alloc(RefCell::new(Node::leaf_in(bump))),
            policy,
            mov: m,
        }
    }

    /// Deep clones this branch and its descendants into a new bump arena
    pub fn clone_into<'new>(&self, new_bump: &'new Bump) -> Branch<'new> {
        Branch {
            node: self.node.borrow().clone_into(new_bump),
            policy: self.policy,
            mov: self.mov,
        }
    }

    pub fn mov(&self) -> Move {
        self.mov
    }

    pub fn policy(&self) -> f32 {
        self.policy
    }

    pub fn node(&self) -> NodeRef<'bump> {
        self.node
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

pub type NodeRef<'bump> = &'bump RefCell<Node<'bump>>;

pub type Branches<'bump> = bumpalo::collections::Vec<'bump, Branch<'bump>>;

#[derive(Clone, PartialEq)]
pub struct Node<'bump> {
    pub visits: u32,
    pub value: Value,
    state: NodeState,
    branches: Branches<'bump>,
}

impl<'bump> fmt::Debug for Node<'bump> {
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

impl<'bump> Node<'bump> {
    /// Deep clones this node and all of its descendants into a new bump arena
    pub fn clone_into<'new>(&self, new_bump: &'new Bump) -> NodeRef<'new> {
        let mut new_branches = Branches::new_in(new_bump);

        for branch in self.branches.iter() {
            new_branches.push(branch.clone_into(new_bump));
        }

        new_bump.alloc(RefCell::new(Node {
            visits: self.visits,
            value: self.value,
            state: self.state,
            branches: new_branches,
        }))
    }

    pub fn sort_by<T: Ord>(&mut self, f: impl Fn(&Branch) -> T) {
        self.branches.sort_by_key(f);
    }

    pub fn get_branch(&self, index: usize) -> Option<&Branch<'bump>> {
        self.branches.get(index)
    }

    pub fn set_branches(&mut self, branches: Branches<'bump>) {
        self.branches = branches;
    }

    pub fn iter_branches(&self) -> impl Iterator<Item = &Branch<'bump>> {
        self.branches.iter()
    }

    pub fn iter_branches_mut(&mut self) -> impl Iterator<Item = &mut Branch<'bump>> {
        self.branches.iter_mut()
    }

    pub fn has_branches(&self) -> bool {
        !self.branches.is_empty()
    }

    pub fn num_branches(&self) -> usize {
        self.branches.len()
    }

    pub fn leaf_in(bump: &'bump Bump) -> Self {
        Self {
            state: NodeState::Leaf,
            branches: Branches::new_in(bump),
            visits: 0,
            value: Value(0.0),
        }
    }

    pub fn select_best(&self) -> Option<&Branch<'bump>> {
        self.select(|b| b.visits())
    }

    pub fn select<F, T>(&self, transform: F) -> Option<&Branch<'bump>>
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

    pub fn select_mut<F, T>(&mut self, transform: F) -> Option<&mut Branch<'bump>>
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

    pub fn expand_in(&mut self, pos: &Position, bump: &'bump Bump) {
        assert_matches!(self.state(), NodeState::Leaf);

        _ = fold_legal_moves(pos, &mut self.branches, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Branch::new_in(m, 0.0, bump));
                acc
            })
        });

        self.state = if self.branches.is_empty() {
            NodeState::Terminal
        }
        else {
            NodeState::Expanded
        };
    }

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

    pub fn subtree_size(&self) -> usize {
        1 + self
            .iter_branches()
            .map(|b| b.node().borrow().subtree_size())
            .sum::<usize>()
    }

    pub fn subtree_maxdepth(&self) -> usize {
        self.iter_branches()
            .map(|b| 1 + b.node().borrow().subtree_maxdepth())
            .max()
            .unwrap_or(0)
    }

    pub fn subtree_mindepth(&self) -> usize {
        self.iter_branches()
            .map(|b| 1 + b.node().borrow().subtree_mindepth())
            .min()
            .unwrap_or(0)
    }
}
