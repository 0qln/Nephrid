use super::*;

#[derive(Debug, PartialEq)]
pub struct InputFloats {
    board: BoardInputFloats,
    state: StateInputFloats,
}

impl InputFloats {
    pub fn new(pos: &Position) -> Self {
        InputFloats {
            board: board_input(pos),
            state: state_input(pos),
        }
    }
}

#[derive(Default, Debug)]
pub enum EvalInfo {
    OnBatch(BatchInfo),
    Evaluated(Evaluation),
    #[default]
    None,
}

impl EvalInfo {
    pub const fn new_none() -> Self {
        Self::None
    }
}

#[derive(PartialEq, Debug)]
pub struct BatchInfo {
    /// Input floats for the eval model.
    inputs: InputFloats,

    /// The node that this eval info is for.
    node: Rc<RefCell<Node>>,

    /// Turn of the current player.
    turn: Turn,
}

impl BatchInfo {
    pub fn new(node: Rc<RefCell<Node>>, pos: &Position) -> Self {
        Self {
            inputs: InputFloats::new(pos),
            node,
            turn: pos.get_turn(),
        }
    }
}

/// X: batch size
pub struct NNEvaluator<'a, 'b, B: Backend, const X: usize> {
    /// NN Model
    model: &'a Model<B>,

    // Device on which the nn will run.
    device: &'b B::Device,
}

impl<'a, 'b, B: Backend, const X: usize> NNEvaluator<'a, 'b, B, X> {
    pub fn new(model: &'a Model<B>, device: &'b B::Device) -> Self {
        Self { model, device }
    }

    fn device(&self) -> &B::Device {
        self.device
    }

    fn build_board_batch<B: Iterator<Item = Rc<RefCell<EvalInfoNode>>>>(&self, batch: B) -> Tensor<B, 4> {
        // concatenate the board inputs along the batch dimension.
        Tensor::cat(
            batch
                .map(|eval_node| Self::get_node_history(eval_node))
                // concatenate the board inputs along the channel dimension.
                .map(|history| nn::board_history_input(&history, self.device()))
                .collect_vec(),
            0,
        )
    }

    /// Returns the history in following order:
    /// - the oldest board state is the first index
    /// - the youngest board state is the last index
    fn get_node_history(eval_info: Rc<RefCell<EvalInfoNode>>) -> Vec<BoardInputFloats> {
        let mut vec: Vec<BoardInputFloats> = vec![];

        _ = EvalInfoNode::try_fold_up_mut(eval_info.clone(), (), |_, eval_info| {
            if vec.len() == BOARD_INPUT_HISTORY {
                return ControlFlow::Break(());
            }

            let board_input = eval_info
                .borrow()
                .data()
                .as_ref()
                .expect("Eval info is missing. Did you forget to call `init()`?")
                .inputs
                .board;

            vec.insert(0, board_input);

            ControlFlow::Continue::<(), ()>(())
        });

        vec
    }

    fn build_state_batch<B: Iterator<Item = Rc<RefCell<EvalInfoNode>>>>(&self, batch: B) -> Tensor<B, 2> {
        // concatenate the state inputs along the batch dimension.
        Tensor::cat(
            batch.map(|eval_info| {
                    let state_input = eval_info
                        .borrow()
                        .data()
                        .as_ref()
                        .expect("Eval info is missing")
                        .inputs
                        .state;

                    Tensor::from_floats([state_input], self.device())
                })
                .collect_vec(),
            0,
        )
    }
}

impl<'a, 'b, B: Backend, const X: usize> Evaluator for NNEvaluator<'a, 'b, B, X> {
    type TraceData = EvalInfo;

    fn create_data(&mut self, node: Rc<RefCell<Node>>, pos: &Position) -> Self::TraceData {
        let data = BatchInfo::new(node, pos);
        EvalInfo::OnBatch(data)
    }

    fn get_eval(&self, index: usize) -> Option<&Evaluation> {
        if let Some(x) = self.eval_infos.evals.get(index)
            && let EvalInfo::Evaluated(eval) = x
        {
            Some(eval)
        }
        else {
            None
        }
    }

    fn set_eval(&mut self, index: usize, eval: Evaluation) {
        if let Some(x) = self.eval_infos.evals.get_mut(index) {
            *x = EvalInfo::Evaluated(eval);
        }
    }

    fn batch_eval(&mut self, index: usize, eval_node: Rc<RefCell<Self::Node>>) {
        if let Some(x) = self.eval_infos.evals.get_mut(index) {
            *x = EvalInfo::OnBatch(eval_node);
        }
        else {
            panic!("Out of range");
        }
    }

    fn eval_batch(&mut self, ) {
        let batch_size = self.iter_batch().count();
        if batch_size == 0 {
            return;
        }

        let board_batch = self.build_board_batch();
        let state_batch = self.build_state_batch();
        let (values, raw_policies) = self.model.forward(board_batch, state_batch);

        assert_eq!(values.shape(), Shape::new([batch_size, VALUE_OUTPUTS]));
        assert_eq!(
            raw_policies.shape(),
            Shape::new([batch_size, POLICY_OUTPUTS])
        );

        let values = values.into_data();
        let values = values
            .as_slice::<f32>()
            .expect("Qualities could not be converted to vec.");
        let values = values.chunks(VALUE_OUTPUTS);

        let raw_policies = raw_policies.into_data();
        let raw_policies = raw_policies
            .as_slice::<f32>()
            .expect("Policy could not be converted to vec.");
        let raw_policies = raw_policies
            .chunks(POLICY_OUTPUTS)
            .map(|raw_policy| RawPolicy(raw_policy.try_into().unwrap()));

        // Enumerate all eval_infos, which have not yet been assigned and assign them a
        // their guess.
        for (index, eval_info, value, raw_policy) in self
            .eval_infos
            .evals
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                if let EvalInfo::OnBatch(eval_info) = x {
                    Some((i, eval_info.clone()))
                }
                else {
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
            .zip(values)
            .zip(raw_policies)
            .map(|(((a, b), c), d)| (a, b, c, d))
        {
            let eval_info = eval_info.borrow();
            let eval_info = eval_info
                .data()
                .as_ref()
                .expect("This should be a leaf and leafes should have data.");

            let eval = Evaluation::Guess(Box::new(Guess {
                relative_to: eval_info.turn,
                quality: value[0],
                policy: raw_policy,
            }));

            self.eval_infos.evals[index] = EvalInfo::Evaluated(eval);
        }
    }

    fn iter(&self) -> impl Iterator<Item = &EvalInfo<Rc<RefCell<Self::Node>>>> {
        self.eval_infos.evals.iter()
    }
}
