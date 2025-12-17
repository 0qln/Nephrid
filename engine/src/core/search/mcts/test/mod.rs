use crate::core::{
    color::colors,
    position::Position,
    search::mcts::{eval::Evaluator, nn::POLICY_OUTPUTS},
};
use std::{cell::RefCell, rc::Rc};

use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::{
    eval::{EvalInfoNode, Evaluation, Guess},
    node::Node,
};

#[cfg(test)]
pub mod fuzz;

#[cfg(test)]
pub mod mpv;

#[derive(Clone)]
pub struct DummyEvaluator<const X: usize>(RefCell<SmallRng>, [Option<Evaluation>; X]);

impl<const X: usize> DummyEvaluator<X> {
    fn fill(&mut self) -> () {
        let mut rng = self.0.borrow_mut();

        for i in 0..X {
            let quality = rng.random_range(-1.0..=1.0);

            let policies: [f32; POLICY_OUTPUTS] = {
                let mut p = [0.2; POLICY_OUTPUTS];
                let policy_idx = rng.random_range(0..POLICY_OUTPUTS);
                p[policy_idx] = 1.0;
                p
            };

            self.1[i] = Some(Evaluation::Guess(Guess {
                relative_to: colors::WHITE,
                quality,
                policies: policies.into(),
            }));
        }
    }
}

impl<const X: usize> Evaluator<X> for DummyEvaluator<X> {
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
    fn eval_guesses(&mut self) -> () {
        self.fill()
    }

    /// Set the evaluation at a specific index.
    fn set_eval(&mut self, _index: usize, _eval: Evaluation) -> () {}

    /// Clear the evaluation at a specific index.
    /// (Mark the node at `index` to be evaluated by `eval_guesses`.)
    fn clear_eval(&mut self, _index: usize) {}

    /// Get the evaluation at a specific index.
    /// (Dummy Code: returns a random value)
    fn get_eval(&self, index: usize) -> Option<&Evaluation> {
        if let Some(x) = self.1.get(index) {
            x.as_ref()
        } else {
            None
        }
    }
}

impl<const X: usize> Default for DummyEvaluator<X> {
    fn default() -> Self {
        let seed = 0xdead_beef;
        let rng = SmallRng::seed_from_u64(seed);
        let mut result = Self(RefCell::new(rng), [const { None }; X]);
        result.fill();
        result
    }
}
