use super::*;
use std::iter;

#[derive(Default)]
pub struct NoneEvaluator<const X: usize>;

impl<const X: usize> NoneEvaluator<X> {
    pub fn new() -> Self {
        Self {}
    }
}

impl<const X: usize> Evaluator for NoneEvaluator<X> {
    type Node = DoubleLinkedNode<()>;
    type NodeRef = DoubleLinkedNode<()>;

    fn init(&mut self, _node: Rc<RefCell<Node>>, _pos: &Position) -> Self::NodeRef {
        Default::default()
    }

    fn create_data(
        &mut self,
        _parent: &mut Self::NodeRef,
        _node: Rc<RefCell<Node>>,
        _pos: &Position,
    ) -> Self::NodeRef {
        Default::default()
    }

    fn get_eval(&self, _index: usize) -> Option<&Evaluation> {
        None
    }

    fn set_eval(&mut self, _index: usize, _eval: Evaluation) {}

    fn batch_eval(&mut self, _index: usize, _eval_node: Self::NodeRef) {}

    fn eval_guesses(&mut self) {}

    fn iter(&self) -> impl Iterator<Item = &EvalInfo<Self::Node>> {
        iter::empty()
    }
}
