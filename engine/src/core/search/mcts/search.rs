use crate::core::Position;
use crate::core::search::mcts::back::Backpropagater;
use crate::core::search::mcts::eval::EvalInfoNode;
use crate::core::search::mcts::eval::Evaluation;
use crate::core::search::mcts::eval::Evaluator;
use crate::core::search::mcts::limiter;
use crate::core::search::mcts::limiter::Limiter;
use crate::core::search::mcts::node::Node;
use crate::core::search::mcts::node::NodeState;
use crate::core::search::mcts::node::Tree;
use crate::core::search::mcts::select::SelectionItem;
use crate::core::search::mcts::select::Selector;
use itertools::Itertools;
use std::cell::RefCell;
use std::rc::Rc;

use crate::core::depth::Depth;

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
pub struct TreeSearcher<
    'a,
    const MPV: usize,
    E: Evaluator<MPV>,
    L: Limiter,
    S: Selector,
    B: Backpropagater,
> {
    // todo: be careful when we dereference the selection, there might be collisions in the tree.
    selector: S,

    /// Evaluations of the selected lines
    evaluations: [Evaluation; MPV],

    /// The position that will be edited during the selection and backpropagatation.
    position: Position,

    /// Evaluator to evaluate non-terminating nodes.
    evaluator: E,

    /// Backpropagater
    backpropagater: B,

    /// Limiter to dicide whether to keep searching non-terminating nodes.
    limiter: L,

    /// The tree to be searched.
    tree: &'a mut Tree,
}

impl<
    'a,
    const MPV: usize,
    E: Evaluator<MPV>,
    L: Limiter + Default,
    S: Selector + Default,
    B: Backpropagater + Default,
> TreeSearcher<'a, MPV, E, L, S, B>
{
    pub fn new(tree: &'a mut Tree, evaluator: E) -> Self {
        Self {
            tree,
            evaluator,
            position
        }
    }
}

impl<'a, const MPV: usize, E: Evaluator<MPV>, L: Limiter, S: Selector, B: Backpropagater>
    TreeSearcher<'a, MPV, E, L, S, B>
{
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
        let eval_node = self.evaluator.info_root(root);
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
        self.evaluator.push(eval_node.clone(), &self.position);

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
                node.borrow_mut().expand(self.position);

                // select the node.
                self.selector.push(SelectionItem {
                    leaf: node.downgrade(),
                    depth,
                    turn: self.position.get_turn(),
                });
                self.evaluator.push_item(eval_node);
            }
            NodeState::Terminal => {
                // select the node.
                self.selector.push(SelectionItem {
                    leaf: node.downgrade(),
                    depth,
                    turn: self.position.get_turn(),
                });
                self.evaluator.push_item(eval_node);
            }
        }
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
        current
            .borrow_mut()
            .sort_by(|b| -self.selector.score(b, root_visits));

        let mut budget = budget;
        let mut line_index = line_index;
        let mut branch_index = 0;
        while budget >= 1 {
            if let Some(branch) = current.get_branch(branch_index) {
                // todo: maybe make this relative to the branch's puct score.
                let current_budget = (budget as f32 * 0.3) as usize;

                // make the move of the current branch, such that we can follow the line.
                self.position.make_move(branch.mov());

                // follow the line and decide what to do depending on the fucking state.
                let eval_info = EvalInfoNode::append(&mut eval_node_parent, None);
                self.follow_line(current_budget, line_index, depth + 1, branch, eval_info);

                // undo the move again
                self.position.unmake_move(branch.mov());

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
        for (x, i) in self.selector.iter().enumerate() {
            if let Some(leaf) = x.leaf.upgrade() {
                // Check if the board has a terminal evaluation
                if let Some(terminal_eval) = E::eval_terminal(leaf.borrow(), self.position) {
                    self.evaluation[i] = terminal_eval;
                }
                // Check if we are even interested in searching this line any further.
                else if self.limiter.should_stop(limiter::Params {
                    pos: self.position,
                    depth: x.depth,
                }) {
                    self.evaluation[i] = Evaluation::Nope;
                }
                // Note down that the need to guess this node's evaluation.
                else {
                    batch.push(i);
                }
            }
        }

        // 2. For all nodes in the batch buffer, build the input tensors and evaluate them using
        //    the eval model.
        let evals = self.evaluator.eval_guess();
        for (eval, i) in evals.into_iter().enumerate() {
            self.evaluation[batch[i]] = eval;
        }
    }

    fn backup_evals(&mut self) -> () {
        for (selected, i) in self.selector.iter().enumerate() {
            let eval = self.evaluation[i];
            // 1. Traverse the selected node in reverse, updating the parents along the way.
            // e.g.:
            // for branch in selected.iter_mut().rev().map(|n| n.as_mut()) {
            //     let value = eval.to_value(pos.get_turn());
            //     branch.update_node(value);
            // }
            let node = selected.leaf.upgrade().expect("Failed to upgrade weak");
            let value = eval.to_value(selected.turn);
            while let Some(parent) = node.parent() {
                node.update(eval);
            }

            // 2. If the eval was a guess make sure to also set the policies of the selected leaf.
            // todo
            // if let Evaluation::Guess(Guess { policies, .. }) = evaluation {
            //     current.set_policies(&policies);
            // }
        }
    }
}
