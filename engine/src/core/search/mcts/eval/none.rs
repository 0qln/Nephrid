use std::iter;
use super::*;

#[derive(Default)]
pub struct NoneEvaluator<const X: usize>;

impl<const X: usize> NoneEvaluator<X> {
    pub fn new() -> Self {
        Self {}
    }
}

impl<const X: usize> Evaluator for NoneEvaluator<X> {
    type Node = DoubleLinkedNode<()>;

    fn init(&mut self, _node: Rc<RefCell<Node>>, _pos: &Position) -> Rc<RefCell<Self::Node>> {
        Rc::new(RefCell::new(Default::default()))
    }

    fn register_info(
        &mut self,
        _parent: &mut Rc<RefCell<Self::Node>>,
        _node: Rc<RefCell<Node>>,
        _pos: &Position,
    ) -> Rc<RefCell<Self::Node>> {
        Rc::new(RefCell::new(Default::default()))
    }

    fn get_eval(&self, _index: usize) -> Option<&Evaluation> {
        None
    }

    fn set_eval(&mut self, _index: usize, _eval: Evaluation) {}

    fn batch_eval(&mut self, _index: usize, _eval_node: Rc<RefCell<Self::Node>>) {}

    fn eval_guesses(&mut self) {}

    fn iter(&self) -> impl Iterator<Item = &EvalState<Self::Node>> {
        iter::empty()
    }
}
