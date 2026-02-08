use crate::core::{
    color::colors,
    position::Position,
    search::mcts::{
        eval::{
            Evaluator, RawPolicy,
            nn::{self, EvalInfoNode},
        },
        nn::POLICY_OUTPUTS,
    },
};
use std::{cell::RefCell, rc::Rc};

use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::{
    eval::{Evaluation, Guess},
    node::Node,
};

#[derive(Clone)]
pub struct DummyEvaluator<const X: usize>(RefCell<SmallRng>, [Option<Evaluation>; X]);

impl<const X: usize> DummyEvaluator<X> {
    fn fill(&mut self) {
        let mut rng = self.0.borrow_mut();

        for i in 0..X {
            let quality = rng.random_range(-1.0..=1.0);

            let raw_policy = RawPolicy::new({
                let mut p = [0.2; POLICY_OUTPUTS];
                let policy_idx = rng.random_range(0..POLICY_OUTPUTS);
                p[policy_idx] = 1.0;
                p
            });

            if self.get_eval(i).is_none() {
                self.1[i] = Some(Evaluation::Guess(Box::new(Guess {
                    relative_to: colors::WHITE,
                    quality,
                    policy: raw_policy,
                })));
            }
        }
    }
}

impl<const X: usize> Evaluator for DummyEvaluator<X> {
    type Node = nn::EvalInfoNode;

    // Prepare an eval_info_node with the required info for this evaluator.
    fn create_data(
        &mut self,
        parent: &mut Rc<RefCell<Self::Node>>,
        _node: Rc<RefCell<Node>>,
        _pos: &Position,
    ) -> Rc<RefCell<Self::Node>> {
        Self::Node::append(parent, None)
    }

    /// Evluate all the nodes in the batch.
    fn eval_guesses(&mut self) {
        self.fill()
    }

    /// Set the evaluation at a specific index.
    fn set_eval(&mut self, index: usize, eval: Evaluation) {
        self.1[index] = Some(eval);
    }

    /// Clear the evaluation at a specific index.
    /// (Mark the node at `index` to be evaluated by `eval_guesses`.)
    fn batch_eval(&mut self, _index: usize, _eval_node: Rc<RefCell<EvalInfoNode>>) {}

    /// Get the evaluation at a specific index.
    /// (Dummy Code: returns a random value)
    fn get_eval(&self, index: usize) -> Option<&Evaluation> {
        if let Some(x) = self.1.get(index) {
            x.as_ref()
        }
        else {
            None
        }
    }

    fn init(&mut self, _node: Rc<RefCell<Node>>, _pos: &Position) -> Rc<RefCell<Self::Node>> {
        Rc::new(RefCell::new(EvalInfoNode::new_root(None)))
    }

    fn iter(&self) -> impl Iterator<Item = &super::eval::EvalInfo<Self::Node>> {
        [].iter()
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
