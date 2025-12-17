use crate::core::{position::Position, search::mcts::eval::Evaluator};
use std::{cell::RefCell, rc::Rc};

use rand::{SeedableRng, rngs::SmallRng};

use super::{
    eval::{EvalInfoNode, Evaluation},
    node::Node,
};

#[cfg(test)]
pub mod fuzz;

#[cfg(test)]
pub mod node;

#[cfg(test)]
pub mod mpv;

// todo: returning none might break stuff,,.....
pub struct DummyEvaluator(RefCell<SmallRng>);
impl<const X: usize> Evaluator<X> for DummyEvaluator {
    // Prepare an eval_info_node with the required info for this evaluator.
    fn prepare_node(
        &mut self,
        _index: usize,
        _eval_node: Rc<RefCell<EvalInfoNode>>,
        _node: Rc<RefCell<Node>>,
        _pos: &Position,
    ) -> () {
    }

    /// Evluate all the nodes in the batch.
    /// (Which is, all the nodes that are eval `None`)
    fn eval_guesses(&mut self) -> () {}

    /// Evaluate a node's terminal state. If the node is terminal, return the evaluation, else
    /// return None.
    fn eval_terminal(_node: &Node, _pos: &Position) -> Option<Evaluation> {
        None
    }

    /// Set the evaluation at a specific index.
    fn set_eval(&mut self, _index: usize, _eval: Evaluation) -> () {}

    /// Clear the evaluation at a specific index.
    /// (Mark the node at `index` to be evaluated by `eval_guesses`.)
    fn clear_eval(&mut self, _index: usize) {}

    /// Get the evaluation at a specific index.
    fn get_eval(&self, _index: usize) -> Option<&Evaluation> {
        None
    }
}

impl Default for DummyEvaluator {
    fn default() -> Self {
        let seed = 0xdead_beef;
        let rng = SmallRng::seed_from_u64(seed);
        Self(RefCell::new(rng))
    }
}
