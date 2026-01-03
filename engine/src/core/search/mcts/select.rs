use std::{cell::RefCell, cmp::max, ops, rc::Rc};

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

    /// Initializes and returns a reference to the root selection node.
    fn init(&mut self, root_node: Rc<RefCell<Node>>, turn: Turn) -> Rc<RefCell<SelectionNode>>;

    fn set(&mut self, index: usize, item: Rc<RefCell<SelectionNode>>);

    fn iter(&self) -> impl Iterator<Item = Option<Rc<RefCell<SelectionNode>>>>;

    fn budget(&self, remaining_budget: usize) -> usize;
}

// todo: be careful when we dereference the selection, there might be collisions in the
// tree if you switch up the way that the nodes are selected.
pub struct Selection<const X: usize> {
    pub root: Option<Rc<RefCell<SelectionNode>>>,
    pub leafs: [Option<Rc<RefCell<SelectionNode>>>; X],
}

impl<const X: usize> Default for Selection<X> {
    fn default() -> Self {
        Self {
            root: None,
            leafs: [const { None }; X],
        }
    }
}

pub struct PuctSelector<const X: usize> {
    // todo: fine tune c. or make a uci option out of it idk
    c: f32,

    /// Stack of nodes that were selected during the selection phase, for each principal line.
    selection: Selection<X>,
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
            selection: Default::default(),
        }
    }
}

impl<const X: usize> Selector for PuctSelector<X> {
    type Score = Score;

    fn score(&self, branch: &Branch, cap_n_i: u32) -> Score {
        let n_i = branch.visits() as f32;

        // The quality is updated incrementally as the tree is explored.
        // Because of this, we have to divide by the number of playouts
        // to get the average quality of this node.
        // If this node has not yet been visited, we set the quality to 0.
        let value = branch.node().borrow().value();
        let exploitation = if n_i == 0.0 { 0.0 } else { value / n_i };

        let exploration = self.c * branch.policy() * (cap_n_i as f32).sqrt() / (1f32 + n_i);

        Score(exploitation + exploration)
    }

    fn set(&mut self, index: usize, item: Rc<RefCell<SelectionNode>>) {
        self.selection.leafs[index] = Some(item);
    }

    fn iter(&self) -> impl Iterator<Item = Option<Rc<RefCell<SelectionNode>>>> {
        self.selection.leafs.iter().cloned()
    }

    fn init(&mut self, root_node: Rc<RefCell<Node>>, turn: Turn) -> Rc<RefCell<SelectionNode>> {
        let root = Rc::new(RefCell::new(SelectionNode::new_root(SelectionItem {
            leaf: root_node,
            depth: Depth::MIN,
            turn,
        })));

        self.selection.root = Some(root.clone());
        root
    }

    fn budget(&self, remaining_budget: usize) -> usize {
        // todo: maybe make this relative to the branch's puct score.
        max(1, (remaining_budget as f32 * 0.3) as usize)
    }
}

// pub struct UcbSelector<const X: usize> {
//     c: f32,
//     selection: Selection<X>,
// }

// impl<const X: usize> UcbSelector<X> {
//     pub fn new(c: f32) -> Self {
//         Self { c, ..Default::default() }
//     }
// }

// impl<const X: usize> Default for UcbSelector<X> {
//     fn default() -> Self {
//         Self {
//             c: f32::sqrt(2.0),
//             selection: Default::default(),
//         }
//     }
// }

// impl<const X: usize> Selector for UcbSelector<X> {
//     type Score = Score;

//     fn score(&self, branch: &Branch, cap_n_i: u32) -> Score {
//         match branch.visits() {
//             0 => Score(f32::INFINITY),
//             n_i => {
//                 let w_i = branch.value();
//                 let n_i = n_i as f32;
//                 let exploitation = w_i / n_i;
//                 let exploration = self.c * f32::sqrt((cap_n_i as f32).ln() / n_i);
//                 Score(exploitation + exploration)
//             }
//         }
//     }

//     fn set(&mut self, index: usize, item: Rc<RefCell<SelectionNode>>) {
//         self.selection.leafs[index] = Some(item);
//     }

//     fn iter(&self) -> impl Iterator<Item = Option<Rc<RefCell<SelectionNode>>>> {
//         self.selection.leafs.iter().cloned()
//     }

//     fn init(&mut self, root_node: Rc<RefCell<Node>>, turn: Turn) -> Rc<RefCell<SelectionNode>> {
//         let root = Rc::new(RefCell::new(SelectionNode::new_root(SelectionItem {
//             leaf: root_node,
//             depth: Depth::MIN,
//             turn,
//         })));

//         self.selection.root = Some(root.clone());
//         root
//     }
// }

#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Score(pub f32);

impl_op!(-|x: Score| -> Score { Score(-x.0) });

impl Eq for Score {}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("This shouldn't happen for puct scores.")
    }
}
