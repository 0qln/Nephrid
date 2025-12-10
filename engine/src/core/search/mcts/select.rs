use crate::core::search::mcts::node::Branch;

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
}

#[derive(Debug)]
pub struct PuctSelector {
    // todo: fine tune c. or make a uci option out of it idk
    c: f32,
}

impl PuctSelector {
    pub fn new(c: f32) -> Self {
        Self { c }
    }
}

impl Default for PuctSelector {
    fn default() -> Self {
        Self { c: f32::sqrt(2.0) };
    }
}

impl Selector for PuctSelector {
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
