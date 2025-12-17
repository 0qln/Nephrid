use std::{cell::RefCell, ops, rc::Rc};

use ringbuf::traits::Producer;

use crate::core::{
    depth::Depth,
    search::mcts::{
        node::{Branch, Node},
        utils::DoubleLinkedNode,
    },
    turn::Turn,
};

pub struct SelectionItem {
    /// The selected node.
    pub leaf: Rc<RefCell<Node>>,

    /// Depth from root
    pub depth: Depth,

    /// Current player's turn
    pub turn: Turn,
}

pub type SelectionNode = DoubleLinkedNode<SelectionItem>;

pub trait Selector {
    type Score: Ord + ops::Neg<Output = Self::Score>;

    // note: we take the policy as an argument, because if we later convert this
    // tree structure to a graph, we have to consider different policies from different parents.
    // same reason that we have different struct for node and branch.

    /// # Selector::score
    ///
    /// The score that the selector would assign to a branch.
    ///
    /// ## Params
    ///
    /// branch: The branch to be scored.
    /// cap_n_i: The number of times that the parent node has been visited.
    fn score(&self, branch: &Branch, cap_n_i: u32) -> Self::Score;

    fn set(&self, index: usize, leaf: Rc<RefCell<Node>>) -> ();

    fn iter(&self) -> impl Iterator<Item = Option<Rc<RefCell<SelectionNode>>>>;
}

pub struct PuctSelector<const X: usize> {
    // todo: fine tune c. or make a uci option out of it idk
    c: f32,

    /// Stack of nodes that were selected during the selection phase, for each principal line.
    selection: [Option<Rc<RefCell<SelectionNode>>>; X],
}

impl<const X: usize> PuctSelector<X> {
    pub fn new(c: f32) -> Self {
        Self { c, ..Default::default() }
    }
}

impl<const X: usize> Default for PuctSelector<X> {
    fn default() -> Self {
        Self {
            c: f32::sqrt(2.0),
            selection: [const { None }; X],
        }
    }
}

#[derive(PartialOrd, PartialEq, Clone, Copy, Debug, Default)]
pub struct PuctScore(pub f32);

impl_op!(-|x: PuctScore| -> PuctScore { PuctScore(-x.0) });

impl Eq for PuctScore {}

impl Ord for PuctScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        f32::partial_cmp(&self.0, &other.0).expect("This shouldn't happen for puct scores.")
    }
}

impl<const X: usize> Selector for PuctSelector<X> {
    type Score = PuctScore;

    fn score(&self, branch: &Branch, cap_n_i: u32) -> PuctScore {
        let n_i = branch.visits() as f32;

        // The quality is updated incrementally as the tree is explored.
        // Because of this, we have to divide by the number of playouts
        // to get the average quality of this node.
        // If this node has not yet been visited, we set the quality to 0.
        let value = branch.node().borrow().value();
        let exploitation = if n_i == 0.0 { 0.0 } else { value / n_i };

        let exploration = self.c * branch.policy() * (cap_n_i as f32).sqrt() / (1f32 + n_i);

        PuctScore(exploitation + exploration)
    }

    fn set(&self, index: usize, item: Rc<RefCell<Node>>) -> () {
        self.selection
            .try_push(item)
            .expect("The searcher tried to push more than was expected via `X`");
    }

    fn iter(&self) -> impl Iterator<Item = Option<Rc<RefCell<SelectionNode>>>> {
        self.selection.iter().cloned()
    }
}
