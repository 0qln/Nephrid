use super::*;

#[derive(PartialEq, Debug)]
pub struct EvalInfo {
    /// The node that this eval info is for.
    node: Rc<RefCell<Node>>,

    /// Turn of the current player.
    turn: Turn,
}

impl EvalInfo {
    pub fn new(node: Rc<RefCell<Node>>, pos: &Position) -> Self {
        Self { node, turn: pos.get_turn() }
    }
}

pub type EvalInfoNode = DoubleLinkedNode<Option<EvalInfo>>;

/// X: batch size
pub struct StaticEvaluator<const X: usize> {
    /// Eval infos
    eval_infos: EvaluationInfos<X, EvalInfoNode>,
}

impl<const X: usize> StaticEvaluator<X> {
    pub fn new() -> Self {
        Self {
            eval_infos: EvaluationInfos {
                evals: [const { EvalState::None }; X],
                root: None,
            },
        }
    }

    fn iter_batch(&self) -> impl Iterator<Item = Rc<RefCell<EvalInfoNode>>> {
        self.eval_infos.evals.iter().filter_map(|x| {
            if let EvalState::OnBatch(eval_info) = x {
                Some(eval_info.clone())
            } else {
                None
            }
        })
    }
}

impl<const X: usize> Evaluator for StaticEvaluator<X> {
    type Node = EvalInfoNode;

    fn init(&mut self, node: Rc<RefCell<Node>>, pos: &Position) -> Rc<RefCell<Self::Node>> {
        let data = EvalInfo::new(node, pos);
        let root = Rc::new(RefCell::new(Self::Node::new_root(Some(data))));
        self.eval_infos.root = Some(root.clone());
        root
    }

    fn register_info(
        &mut self,
        parent: &mut Rc<RefCell<Self::Node>>,
        node: Rc<RefCell<Node>>,
        pos: &Position,
    ) -> Rc<RefCell<Self::Node>> {
        let data = EvalInfo { node, turn: pos.get_turn() };
        Self::Node::append(parent, Some(data))
    }

    fn get_eval(&self, index: usize) -> Option<&Evaluation> {
        if let Some(x) = self.eval_infos.evals.get(index)
            && let EvalState::Evaluated(eval) = x
        {
            Some(eval)
        } else {
            None
        }
    }

    fn set_eval(&mut self, index: usize, eval: Evaluation) {
        if let Some(x) = self.eval_infos.evals.get_mut(index) {
            *x = EvalState::Evaluated(eval);
        }
    }

    fn batch_eval(&mut self, index: usize, eval_node: Rc<RefCell<Self::Node>>) {
        if let Some(x) = self.eval_infos.evals.get_mut(index) {
            println!("[{index} prepare batch");
            *x = EvalState::OnBatch(eval_node);
        } else {
            panic!("Out of range");
        }
    }

    fn eval_guesses(&mut self) {
        let batch_size = self.iter_batch().count();
        // println!("batchsize: {batch_size}");
        if batch_size == 0 {
            return;
        }

        // Enumerate all eval_infos, which have not yet been assigned and assign them a their guess.
        for (index, eval_info) in self
            .eval_infos
            .evals
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                if let EvalState::OnBatch(eval_info) = x {
                    Some((i, eval_info.clone()))
                } else {
                    None
                }
            })
            // ---
            // todo: we allocate here bc if we borrow above, we cannot borrow mutably in the
            // for loop to set the eval_infos...
            // find a better solution
            .collect_vec()
            .into_iter()
        // ---
        {
            let eval_info = eval_info.borrow();
            let eval_info = eval_info
                .data()
                .as_ref()
                .expect("This should be a leaf and leafes should have data.");

            let eval = Evaluation::Guess(Box::new(Guess {
                relative_to: eval_info.turn,
                quality: todo!("Static quality analysis"),
                policy: todo!("Static policy analysis"),
            }));

            println!("[{index}] save guess: {eval}");
            self.eval_infos.evals[index] = EvalState::Evaluated(eval);
        }
    }

    fn iter(&self) -> impl Iterator<Item = &EvalState<Self::Node>> {
        self.eval_infos.evals.iter()
    }
}
