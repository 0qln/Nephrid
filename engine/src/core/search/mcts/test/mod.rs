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

pub struct DummyEvaluator(RefCell<SmallRng>);
impl<const X: usize> Evaluator<X> for DummyEvaluator {
    fn push(&mut self, _parent: Rc<RefCell<EvalInfoNode>>, _pos: &Position) -> () {
        todo!()
    }

    fn push_item(&mut self, _item: Rc<RefCell<EvalInfoNode>>) -> () {
        todo!()
    }

    fn eval_guess(&self) -> Vec<Evaluation> {
        todo!()
    }

    fn eval_terminal(_node: &Node, _pos: &Position) -> Option<Evaluation> {
        todo!()
    }
}

impl Default for DummyEvaluator {
    fn default() -> Self {
        let seed = 0xdead_beef;
        let rng = SmallRng::seed_from_u64(seed);
        Self(RefCell::new(rng))
    }
}
