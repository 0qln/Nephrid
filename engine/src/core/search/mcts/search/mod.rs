use crate::core::Position;
use crate::core::search::mcts::back::Backpropagater;
use crate::core::search::mcts::back::DefaultBackuper;
use crate::core::search::mcts::eval::EvalState;
use crate::core::search::mcts::eval::Evaluation;
use crate::core::search::mcts::eval::Evaluator;
use crate::core::search::mcts::limiter;
use crate::core::search::mcts::limiter::DefaultLimiter;
use crate::core::search::mcts::limiter::Limiter;
use crate::core::search::mcts::node::Node;
use crate::core::search::mcts::node::NodeState;
use crate::core::search::mcts::node::Tree;
use crate::core::search::mcts::select::PuctSelector;
use crate::core::search::mcts::select::SelectionItem;
use crate::core::search::mcts::select::SelectionNode;
use crate::core::search::mcts::select::Selector;
use std::cell::RefCell;
use std::cmp::max;
use std::rc::Rc;

use crate::core::depth::Depth;

#[cfg(test)]
pub mod test;

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
/// )
pub struct TreeSearcher<
    'a,
    const MPV: usize,
    E: Evaluator,
    L: Limiter = DefaultLimiter,
    S: Selector = PuctSelector<MPV>,
    B: Backpropagater = DefaultBackuper,
> {
    /// The position that will be edited during the selection and backpropagatation.
    position: Position,

    /// Selector to select the leafes.
    selector: S,

    /// Limiter to dicide whether to keep searching non-terminating nodes.
    limiter: L,

    /// Evaluator to evaluate non-terminating nodes.
    evaluator: E,

    /// Backpropagater
    backpropagater: B,

    /// The tree to be searched.
    tree: &'a mut Tree,

    debug_arr: [usize; MPV],
}

impl<'a, const MPV: usize, E: Evaluator, L: Limiter, S: Selector, B: Backpropagater + Default>
    TreeSearcher<'a, MPV, E, L, S, B>
{
    pub fn new(
        tree: &'a mut Tree,
        position: Position,
        selector: S,
        limiter: L,
        evaluator: E,
    ) -> Self {
        Self {
            tree,
            position,
            selector,
            limiter,
            evaluator,
            backpropagater: Default::default(),
            debug_arr: [const { 0 }; MPV],
        }
    }
}

impl<'a, const MPV: usize, E: Evaluator, L: Limiter, S: Selector, B: Backpropagater>
    TreeSearcher<'a, MPV, E, L, S, B>
{
    pub fn grow(&mut self) {
        self.select_lines();
        // println!(
        //     "selected: {:?}",
        //     self.selector
        //         .iter()
        //         .enumerate()
        //         .filter_map(|(i, x)| Some((i, x?)))
        //         .map(|(i, x)| i)
        //         .count() // .collect_vec()
        // );

        self.eval_leafes_();
        // println!(
        //     "evaluated: {:?}",
        //     self.evaluator
        //         .iter()
        //         .enumerate()
        //         .filter_map(|(i, x)| match x {
        //             EvalState::None => None,
        //             x => Some((i, x)),
        //         })
        //         .map(|(i, x)| i)
        //         .count() // .collect_vec()
        // );

        self.backup_evals();
        // println!("backpropagated.");
    }

    //
    // # Select the leafes.
    //
    fn select_lines(&mut self) {
        // todo: convert to iterative approach -- or not if the stack frame is
        // small with this one ._.
        let node = self.tree.get_root();
        let depth = Depth::MIN;
        let turn = self.position.get_turn();
        let eval_node = self.evaluator.init(node.clone(), &self.position);
        let sel_root = self.selector.init(node.clone(), turn);
        self.process_node(MPV, 0, depth, node, eval_node, sel_root);
    }

    /// Follows a branch and decides what to do depending on the current state of the branch's
    /// node.
    fn process_node(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        node: Rc<RefCell<Node>>,
        eval_node: Rc<RefCell<E::Node>>,
        sel_node: Rc<RefCell<SelectionNode>>,
    ) {
        let state = node.borrow().state();
        match state {
            // Only if the node is already expanded we want to follow the branch.
            NodeState::Expanded => {
                // If the node is expanded, pick branches and follow the lines.
                self.pick_branches(budget, line_index, depth, node, eval_node, sel_node);
            }
            NodeState::Leaf => {
                // If the node is a leaf, expand the node's branches for future mcts iterations.
                // And select it for mcts evaluation.
                //
                // this is fine to do now, since we split above and in the fn `select_branches` we
                // immidieatly increment the branch_index and thus during a single selection phase, this
                // match block is only reached once per node in a mcts iteration. Altough we should probably
                // unit test this somehow.
                node.borrow_mut().expand(&self.position);

                // select the node.
                self.select_node(line_index, node, sel_node, eval_node, depth);
            }
            NodeState::Terminal => {
                // select the node.
                self.select_node(line_index, node, sel_node, eval_node, depth);
            }
        }
    }

    fn select_node(
        &mut self,
        line_index: usize,
        node: Rc<RefCell<Node>>,
        slct: Rc<RefCell<SelectionNode>>,
        eval: Rc<RefCell<E::Node>>,
        depth: Depth,
    ) {
        let pos = &self.position;

        if self.debug_arr[line_index] >= 1 {
            panic!("Each node should only be selected once atmost.");
        }

        // Check if the board has a terminal evaluation
        let terminal_eval = E::eval_terminal(&node.borrow(), pos);
        if let Some(terminal_eval) = terminal_eval {
            // println!("[{line_index}] terminal");
            node.borrow_mut().set_state(NodeState::Terminal);
            self.evaluator.set_eval(line_index, terminal_eval);
        }
        // Check if we are even interested in searching this line any further.
        else if self.limiter.should_stop(limiter::Params { pos, depth }) {
            self.evaluator.set_eval(line_index, Evaluation::Nope);
        }
        // Else note down that we need to guess this node's evaluation.
        else {
            // println!("[{line_index}] batched");
            self.evaluator.batch_eval(line_index, eval);
        }

        self.selector.set(line_index, slct);

        self.debug_arr[line_index] += 1;
    }

    fn pick_branches(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        parent_node: Rc<RefCell<Node>>,
        mut eval_node_parent: Rc<RefCell<E::Node>>,
        mut sel_node_parent: Rc<RefCell<SelectionNode>>,
    ) {
        // Split the budget up between this and the subsequent best nodes.
        let root_visits = parent_node.borrow().visits();
        parent_node
            .borrow_mut()
            .sort_by(|b| -self.selector.score(b, root_visits));

        let mut budget = budget;
        let mut line_index = line_index;
        let mut branch_index = 0;
        while budget >= 1 {
            if let Some(branch) = parent_node.borrow().get_branch(branch_index) {
                //
                // todo: maybe make this relative to the branch's puct score.
                //
                // todo: be careful when we dereference the selection, there might be collisions in the
                // tree if you switch up the way that the nodes are selected.
                //
                let current_budget = max(1, (budget as f32 * 0.3) as usize);

                // make the move of the current branch, such that we can follow the line.
                self.position.make_move(branch.mov());

                let depth = depth + 1;
                let node = branch.node();

                let selc_info = SelectionNode::append(
                    &mut sel_node_parent,
                    SelectionItem {
                        depth,
                        leaf: node.clone(),
                        turn: self.position.get_turn(),
                    },
                );

                let eval_info = self.evaluator.register_info(
                    &mut eval_node_parent,
                    node.clone(),
                    &self.position,
                );

                self.process_node(
                    current_budget,
                    line_index,
                    depth,
                    node,
                    eval_info,
                    selc_info,
                );

                // undo the move again
                self.position.unmake_move(branch.mov());

                // `process_node` should have used `current_budget` nodes. Thus we increase the
                // line_index by that amount.
                // todo: this is a bug. as seen below, the budget is not always used 100%. idk if
                // these gaps actually create problems, but better fix the indexing increment here
                // before it does.
                line_index += current_budget;
                budget -= current_budget;
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

    fn eval_leafes_(&mut self) {
        // For all nodes in the batch buffer, build the input tensors and evaluate them using
        // the eval model.
        self.evaluator.eval_guesses()
    }

    fn backup_evals(&mut self) {
        for (index, leaf) in self
            .selector
            .iter()
            .enumerate()
            .filter_map(|(i, x)| Some((i, x?)))
        {
            let eval = self
                .evaluator
                .get_eval(index)
                .unwrap_or_else(|| panic!("Evaluation missing for index {index}"));

            // println!("[{index}] backup: {eval}");

            self.backpropagater.backpropagate(leaf, eval);
        }
    }
}
