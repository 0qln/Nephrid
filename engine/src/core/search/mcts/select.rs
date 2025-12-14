use std::{cell::RefCell, rc::Weak};

use ringbuf::{
    StaticRb,
    traits::{Consumer, Producer},
};

use crate::core::{
    depth::Depth,
    search::mcts::node::{Branch, Node},
    turn::Turn,
};

pub struct SelectionItem {
    /// The selected node.
    pub leaf: Weak<RefCell<Node>>,

    /// Depth from root
    pub depth: Depth,

    /// Current player's turn
    pub turn: Turn,
}

pub trait Selector {
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
    fn score(&self, branch: &Branch, cap_n_i: u32) -> f32;

    fn push(&self, leaf: SelectionItem) -> ();

    fn iter(&self) -> impl Iterator<Item = &SelectionItem>;
}

#[derive(Debug)]
pub struct PuctSelector<const X: usize> {
    // todo: fine tune c. or make a uci option out of it idk
    c: f32,

    /// Stack of nodes that were selected during the selection phase, for each principal line.
    selection: StaticRb<SelectionItem, X>,
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
            ..Default::default()
        };
    }
}

impl<const X: usize> Selector for PuctSelector<X> {
    fn score(&self, branch: &Branch, cap_n_i: u32) -> f32 {
        let n_i = branch.visits() as f32;

        // The quality is updated incrementally as the tree is explored.
        // Because of this, we have to divide by the number of playouts
        // to get the average quality of this node.
        // If this node has not yet been visited, we set the quality to 0.
        let exploitation = if n_i == 0.0 { 0.0 } else { branch.value() / n_i };

        let exploration = self.c * branch.policy() * (cap_n_i as f32).sqrt() / (1f32 + n_i);

        exploitation + exploration
    }

    fn push(&self, item: SelectionItem) -> () {
        self.selection
            .try_push(item)
            .expect("The searcher tried to push more than was expected via `X`");
    }

    fn iter(&self) -> impl Iterator<Item = &SelectionItem> {
        self.selection.iter()
    }
}

// #[derive(Debug)]
// pub struct PuctWithTempSelector {
//     rng: SmallRng,
//     puct: PuctSelector
// }

// impl PuctWithTempSelector {
//     pub fn new(seed: u64) -> Self {
//         let rng = SmallRng::seed_from_u64(seed);
//         let puct =
//         Slef { rng, puct: }
//     }
// }
