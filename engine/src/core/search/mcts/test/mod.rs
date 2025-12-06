use super::*;
use std::cell::RefCell;

use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::core::{position::Position, search::mcts::eval::model::POLICY_OUTPUTS};

#[cfg(test)]
pub mod fuzz;

#[cfg(test)]
pub mod node;

pub struct DummyEvaluator(RefCell<SmallRng>);
impl Evaluator for DummyEvaluator {
    fn evaluate(&self) -> (f32, [f32; POLICY_OUTPUTS]) {
        let mut rng = self.0.borrow_mut();

        let quality = rng.random_range(-1.0..=1.0);

        let policies: [f32; POLICY_OUTPUTS] = {
            let mut p = [0.2; POLICY_OUTPUTS];
            let policy_idx = rng.random_range(0..POLICY_OUTPUTS);
            p[policy_idx] = 1.0;
            p
        };

        (quality, policies)
    }

    fn push(&mut self, _pos: &Position) -> () {
        ()
    }

    fn pop(&mut self) -> () {
        ()
    }

    fn clear(&mut self) -> () {
        ()
    }
}

impl Default for DummyEvaluator {
    fn default() -> Self {
        let seed = 0xdead_beef;
        let rng = SmallRng::seed_from_u64(seed);
        Self(RefCell::new(rng))
    }
}
