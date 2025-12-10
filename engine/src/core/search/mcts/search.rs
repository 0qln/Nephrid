/// # Tree searcher
///
/// This statemachine should hold all the info that is required to search.
///
/// This is the implementation for the nn-backed eval model searcher. (e.g. lc0, a0)
///
/// ## Multi pv.
///
/// Use multi pv lines with batched evaluation, if we have access to GPU and can parallelize. If we
/// don't have access to hardware accell, resort to using a backend like WebGPU, or NdArray, and just do
/// TreeSearcher<MPV=1>.
///
/// The pv lines are the top few lines sorted by puct. (e.g.:
/// ```rust
/// self.multi_select(|b| b.puct(self))
/// ```
/// )
///
/// Idea: Maybe we can have lines that are a lot more explorative next to the main puct line?
pub struct TreeSearcher<'a, const MPV: usize, E: Evaluator<MPV>, L: Limiter> {
    /// RNG
    rng: SmallRng,

    // todo: be careful when we dereference the selection, there might be collisions in the tree.
    /// Stack of nodes that were selected during the selection phase, for each principal line.
    selection: [(Weak<RefCell<Node>>, Rc<RefCell<EvalInfoNode>>); MPV],

    /// Evaluations of the selected lines
    evaluations: [Evaluation; MPV],

    // todo: also refactor out (??)
    /// Backup info
    backup_info_root: BackupInfoNode,

    /// The position that will be edited during the selection and backpropagatation.
    position: Position,

    /// Evaluator to evaluate non-terminating nodes.
    evaluater: E,

    /// Limiter to dicide whether to keep searching non-terminating nodes.
    limiter: L,

    /// The tree to be searched.
    tree: &'a mut Tree,
}

impl<'a, const MPV: usize, E: Evaluator<MPV>, L: Limiter> TreeSearcher<'a, MPV, E, L> {
    pub fn new() -> Self {
        Self {
            eval_info_root: EvalInfoNode::new_root(None),
            ..Default::default()
        }
    }

    pub fn grow(&mut self) -> () {
        self.prepare_grow();
        self.select_lines();
        self.eval_leafes_();
        self.backup_evals();
    }

    //
    // # Select the leafes.
    //
    fn select_lines(&mut self) -> () {
        // todo: convert to iterative approach -- or not if the stack frame is
        // small with this one ._.
        let root = self.tree.root();
        let eval_node = self.eval_info_root;
        self.follow_line(MPV, 0, Depth::MIN, root, eval_node);
    }

    /// Follows a branch and decides what to do depending on the current state of the branch's
    /// node.
    fn follow_line(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        node: Rc<RefCell<Node>>,
        eval_node: Rc<RefCell<EvalInfoNode>>,
    ) {
        // Push evaluation info for this node to the tree.
        // The info is written to this eval_node.
        self.push_eval_info(eval_node.clone());

        match node.borrow().state() {
            // Only if the node is already expanded we want to follow the branch.
            NodeState::Expanded => {
                // If the node is expanded, pick branches and follow the lines.
                self.pick_branches(budget, line_index, depth, node, eval_node);
            }
            NodeState::Leaf => {
                // If the node is a leaf, expand the node's branches for future mcts iterations.
                // And select it for mcts evaluation.
                //
                // this is fine to do now, since we split above and in the fn `select_branches` we
                // immidieatly increment the branch_index and thus during a single selection phase, this
                // match block is only reached once per node in a mcts iteration. Altough we should probably
                // unit test this somehow.
                node.borrow_mut()
                    .expand(self.pos, self.evaluator, self.limiter, depth);

                self.selection[line_index] = (node.downgrade(), eval_node);
            }
            NodeState::Terminal => {
                // select the node.
                self.selection[line_index] = (node.downgrade(), eval_node);
            }
        }
    }

    fn push_eval_info(&self, parent: &mut Rc<RefCell<EvalInfoNode>>) -> () {
        debug_assert_eq!(parent.borrow().data(), None);

        let board = board_input(&self.pos);
        let state = state_input(&self.pos);
        let input = InputFloats(board, state);
        parent.borrow_mut().set_data(input);
    }

    fn pick_branches(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        current: Rc<RefCell<Node>>,
        eval_node_parent: Rc<RefCell<EvalInfoNode>>,
    ) {
        // Split the budget up between this and the subsequent best nodes.
        let root_visits = current.borrow().visits();
        let selector = PuctSelector::default();
        current
            .borrow_mut()
            .sort_by(|b| -selector.score(b, root_visits));

        let mut budget = budget;
        let mut line_index = line_index;
        let mut branch_index = 0;
        while budget >= 1 {
            if let Some(branch) = current.get_branch(branch_index) {
                // todo: maybe make this relative to the branch's puct score.
                let current_budget = (budget as f32 * 0.3) as usize;

                // make the move of the current branch, such that we can follow the line.
                self.pos.make_move(branch.mov());

                // follow the line and decide what to do depending on the fucking state.
                let eval_info = EvalInfoNode::append(&mut eval_node_parent, None);
                self.follow_line(current_budget, line_index, depth + 1, branch, eval_info);

                // undo the move again
                self.pos.unmake_move(branch.mov());

                budget -= current_budget;
                line_index += current_budget;
                branch_index += 1;
            } else {
                // in this case there are no more branches to distribute the budget to.
                // todo:
                // this currently wastes some remaining budget, fix that.
                // or later if we make the selection and eval parallel, just go select as many
                // lines as possible, until we have a full (X==MPV) batch.
                break;
            }
        }
    }

    fn eval_leafes_(&mut self) -> () {
        let mut batch: Vec<usize> = vec![];

        // 1. Loop over the selection and
        // 1.a: Node is terminal => safe the evaluation
        // 1.b: Node is on-going => safe the nodes index into a batch buffer.
        for (leaf, i) in self.selection.enumerate() {
            let evaluation = &mut self.evaluation[i];
            if let Some(leaf) = leaf.upgrade() {
                // Check if the board has a terminal evaluation
                if let Some(terminal_eval) = E::eval_terminal(leaf.borrow(), self.pos, depth) {
                    *evaluation = terminal_eval;
                }
                // If not, note down that the need to guess this node's evaluation.
                else {
                    to_be_guessed.push((i,));
                }
            }
        }

        // 2. For all nodes in the batch buffer, build the input tensors and evaluate them using
        //    the eval model.
        //
        // todo: this belongs in the evaluator. refactor later.
        // self.evaluator.evaluate()
        //
        let board_input = self.build_board_batch();
        let state_input = self.build_state_batch();
        let evals = self.evaluator.evaluate(..);
        // save the evaluation result.
        for (eval, i) in evals.into_iter().enumerate() {
            self.evaluation[batch[i]] = eval;
        }
    }

    fn build_board_batch(&self, selection_indeces: &[usize]) -> Tensor<B, 4> {
        // concatenate the board inputs along the batch dimension.
        let board_batch = Tensor::cat(
            selection_indeces
                .map(|d_idx| self.get_node_history(d_idx))
                // concatenate the board inputs along the channel dimension.
                .map(|history| {
                    // convert input floats to tensors
                    let history_tensor = Tensor::cat(
                        history
                            .into_iter()
                            .map(|b| Tensor::from_floats([b]))
                            .collect_vec(),
                        1,
                    );

                    // pad missing history info with zeroes.
                    let padding_len = BOARD_INPUT_HISTORY - history.len();
                    let padding_tensor = BoardInputTensor::<B>::zeros(
                        [
                            X,
                            BOARD_INPUT_CHANNELS,
                            ranks::N_VARIANTS,
                            files::N_VARIANTS,
                        ],
                        device,
                    );

                    // concat padding with history
                    Tensor::cat(vec![padding_tensor, history], 1)
                })
                .collect_vec(),
            0,
        );
    }

    /// return the history where:
    /// the oldest board state is the first index
    /// the youngest board state is the last index
    fn get_node_history(&self, selected_node_index: usize) -> Vec<BoardInputFloats> {
        let mut vec: Vec<BoardInputFloats> = vec![];

        let selection = self.selection[selected_node_index];
        let mut eval_info = selection.1;

        let board_input = eval_info.borrow().data().expect("Eval info is missing").0;
        vec.prepend(board_input);

        while let Some(parent) = eval_info.borrow().parent {
            eval_info = parent.upgrade().expect("can't get a Rc from a Weak");

            let board_input = eval_info.borrow().data().expect("Eval info is missing").0;
            vec.prepend(board_input);
        }

        vec
    }

    fn build_state_batch(&self, selection_indeces: &[usize]) -> Tensor<B, 2> {
        // concatenate the state inputs along the batch dimension.
        let state_batch = Tensor::cat(
            selection_indeces
                .map(|d_idx| {
                    let selection = self.selection[selected_node_index];
                    let eval_info = selection.1;
                    let state_input = eval_info.borrow().data().expect("Eval info is missing").1;
                    [state_input]
                })
                .collect_vec(),
            0,
        );
    }

    fn backup_evals(&mut self) -> () {
        for (selected, i) in self.selection.enumerate() {
            let eval = self.evaluation[i];
            // 1. Traverse the selected node in reverse, updating the parents along the way.
            // e.g.:
            // for branch in selected.iter_mut().rev().map(|n| n.as_mut()) {
            //     let value = eval.to_value(pos.get_turn());
            //     branch.update_node(value);
            // }
            let node = selected.0.upgrade().expect("Failed to upgrade weak");
            let value = eval.to_value(pos.get_turn());
            while let Some(parent) = node.parent {}
            node.update(eval);

            // 2. If the eval was a guess make sure to also set the policies of the selected leaf.
            // todo
            // if let Evaluation::Guess(Guess { policies, .. }) = evaluation {
            //     current.set_policies(&policies);
            // }
        }
    }
}
