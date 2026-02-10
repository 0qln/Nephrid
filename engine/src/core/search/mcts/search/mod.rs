use itertools::Itertools;

use crate::core::{
    Position,
    depth::Depth,
    search::mcts::{
        back::{Backpropagater, DefaultBackuper},
        eval::{Evaluation, Evaluator},
        limiter::{self, Limiter},
        node::{Node, NodeRef, NodeState, Tree},
        noise::{DirichletNoiser, Noiser},
        select::Selector,
        utils::DoubleLinkedNode,
    },
    turn::Turn,
};
use std::{cell::RefCell, rc::Rc};

#[cfg(test)]
pub mod test;

pub struct SelectionItem<T> {
    /// The selected node.
    pub node: NodeRef,

    /// Depth from root
    pub depth: Depth,

    /// Current player's turn
    pub turn: Turn,

    /// Context specific data.
    pub trace_data: T,
}

#[derive(Debug)]
pub enum EvalItem {
    Batched,
    Evaluated(Evaluation),
}

impl EvalItem {
    fn is_batched(&self) -> bool {
        matches!(self, Self::Batched)
    }
}

pub type SelectionNode<T> = DoubleLinkedNode<SelectionItem<T>>;

pub type SelectionNodeRef<T> = Rc<RefCell<SelectionNode<T>>>;

pub type SelectionLeaf<T> = (SelectionNodeRef<T>, EvalItem);

pub struct Selection<const X: usize, T> {
    pub root: Option<SelectionNodeRef<T>>,
    pub leafs: [Option<SelectionLeaf<T>>; X],
}

impl<const X: usize, T> Default for Selection<X, T> {
    fn default() -> Self {
        const fn empty_leaf<T>() -> Option<SelectionLeaf<T>> {
            None
        }
        Self {
            root: None,
            leafs: [const { empty_leaf() }; X],
        }
    }
}

impl<const X: usize, T> Selection<X, T> {
    pub fn init_root(
        &mut self,
        root_node: Rc<RefCell<Node>>,
        turn: Turn,
        trace_data: T,
    ) -> SelectionNodeRef<T> {
        let root = Rc::new(RefCell::new(SelectionNode::new_root(SelectionItem {
            node: root_node,
            depth: Depth::MIN,
            turn,
            trace_data,
        })));

        self.root = Some(root.clone());
        root
    }

    pub fn set(&mut self, index: usize, item: SelectionLeaf<T>) {
        self.leafs[index] = Some(item);
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut SelectionLeaf<T>> {
        self.leafs[index].as_mut()
    }
}

/// # Tree searcher
///
/// This statemachine should hold all the info that is required to search.
///
/// ## Multi pv.
///
/// Use multi pv lines with batched evaluation, if we have access to GPU and can
/// parallelize. If we don't have access to hardware accell, resort to using a
/// backend like WebGPU, or NdArray, and just do TreeSearcher<MPV=1>.
pub struct TreeSearcher<
    'a,
    const MPV: usize,
    E: Evaluator,
    L: Limiter,
    S: Selector,
    B: Backpropagater,
    N: Noiser,
> {
    /// The position that will be edited during the selection and
    /// backpropagatation.
    position: Position,

    /// Selector to select the leafes.
    selector: S,

    /// Limiter to dicide whether to keep searching non-terminating nodes.
    limiter: L,

    /// Evaluator to evaluate non-terminating nodes.
    evaluator: E,

    /// Backpropagater
    backprop: B,

    /// Noiser
    noiser: N,

    /// The tree to be searched.
    tree: &'a mut Tree,

    /// Number of compeleted iterations.
    iterations: usize,

    /// Stack of nodes that were selected during the selection phase, for each
    /// principal line.
    selection: Selection<MPV, E::TraceData>,
}

impl<'a, const MPV: usize, E: Evaluator, L: Limiter, S: Selector, B: Backpropagater, N: Noiser>
    TreeSearcher<'a, MPV, E, L, S, B, N>
{
    pub fn new(
        tree: &'a mut Tree,
        position: Position,
        selector: S,
        limiter: L,
        evaluator: E,
        backprop: B,
        noiser: N,
    ) -> Self {
        Self {
            tree,
            position,
            selector,
            limiter,
            evaluator,
            backprop,
            noiser,
            iterations: 0,
            selection: Default::default(),
        }
    }
}

impl<'a, const MPV: usize, E: Evaluator, L: Limiter, S: Selector, B: Backpropagater, N: Noiser>
    TreeSearcher<'a, MPV, E, L, S, B, N>
{
    pub fn grow(&mut self) {
        self.select_lines();
        self.eval_batched();
        self.backup_evals();
        self.apply_noise();
        self.iterations += 1;
    }

    // Select the root leafes.
    fn select_lines(&mut self) {
        // todo: convert to iterative approach -- or not if the stack frame is
        // small with this one ._.
        let node = self.tree.get_root();
        let turn = self.position.get_turn();
        let eval_data = self.evaluator.trace(node.clone(), &self.position);
        let sel_root = self.selection.init_root(node.clone(), turn, eval_data);
        self.process_node(MPV, 0, Depth::MIN, node, sel_root);
    }

    /// Follows a branch and decides what to do depending on the current state
    /// of the branch's node.
    /// Returns: how much of the budget was used.
    fn process_node(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        node: Rc<RefCell<Node>>,
        sel_node: SelectionNodeRef<E::TraceData>,
    ) -> usize {
        let state = node.borrow().state();
        match state {
            // Only if the node is already expanded we want to follow the branch.
            NodeState::Expanded => {
                // If the node is expanded, pick branches and follow the lines.
                self.pick_branches(budget, line_index, depth, node, sel_node)
            }
            NodeState::Leaf => {
                // If the node is a leaf, expand the node's branches for future mcts iterations.
                // And select it for mcts evaluation.
                //
                // this is fine to do now, since we split above and in the fn `select_branches`
                // we immidieatly increment the branch_index and thus during a
                // single selection phase, this match block is only reached once
                // per node in a mcts iteration. Altough we should probably unit
                // test this somehow.
                node.borrow_mut().expand(&self.position);

                // select the node.
                self.select_node(line_index, node, sel_node, depth)
            }
            NodeState::Terminal => {
                // select the node.
                self.select_node(line_index, node, sel_node, depth)
            }
        }
    }

    /// Returns: how much of the budget was used.
    fn select_node(
        &mut self,
        line_index: usize,
        node: Rc<RefCell<Node>>,
        sel_node: SelectionNodeRef<E::TraceData>,
        depth: Depth,
    ) -> usize {
        let pos = &self.position;

        // Check if the board has a terminal evaluation.
        let terminal_eval = E::eval_terminal(&node.borrow(), pos);
        let (used_budget, eval) = if let Some(eval) = terminal_eval {
            node.borrow_mut().set_state(NodeState::Terminal);
            (1, EvalItem::Evaluated(eval))
        }
        // Check if we are even interested in searching this line any
        // further.
        else if self.limiter.should_stop(limiter::Params { pos, depth }) {
            // todo: maybe it's better to return 0 (skip this node) instead of using a draw
            (1, EvalItem::Evaluated(Evaluation::Nope))
        }
        // Else note down that we need to guess this node's evaluation.
        else {
            (1, EvalItem::Batched)
        };

        self.selection.set(line_index, (sel_node, eval));

        used_budget
    }

    /// Returns: how much of the budget was used.
    fn pick_branches(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        parent_node: Rc<RefCell<Node>>,
        mut sel_node_parent: SelectionNodeRef<E::TraceData>,
    ) -> usize {
        // Split the budget up between this and the subsequent best nodes.
        let root_visits = parent_node.borrow().visits();
        parent_node
            .borrow_mut()
            .sort_by(|b| -self.selector.score(b, root_visits));

        let mut budget = budget;
        let mut used_budget = 0;
        let mut line_index = line_index;
        let mut branch_index = 0;
        while budget >= 1 {
            if let Some(branch) = parent_node.borrow().get_branch(branch_index) {
                let current_budget = self.selector.budget(budget);
                if current_budget == 0 {
                    break;
                }

                // make the move of the current branch, such that we can follow the line.
                self.position.make_move(branch.mov());

                let depth = depth + 1;
                let node = branch.node();

                let eval_info = self.evaluator.trace(node.clone(), &self.position);

                let sel_info = SelectionNode::append(
                    &mut sel_node_parent,
                    SelectionItem {
                        depth,
                        node: node.clone(),
                        turn: self.position.get_turn(),
                        trace_data: eval_info,
                    },
                );

                let used = self.process_node(current_budget, line_index, depth, node, sel_info);

                // undo the move again
                self.position.unmake_move(branch.mov());

                budget -= current_budget;
                branch_index += 1;

                // `process_node` should have used `used` nodes. Thus we increase the
                // line_index by that amount.
                line_index += used;
                used_budget += used;
            }
            else {
                // in this case there are no more branches to distribute the budget to.
                // todo:
                // this currently wastes some remaining budget, fix that.
                // or later if we make the selection and eval parallel, just go select as many
                // lines as possible, until we have a full (X==MPV) batch.
                break;
            }
        }
        used_budget
    }

    fn eval_batched(&mut self) {
        let mut batched_leafs = self
            .selection
            .leafs
            .iter_mut()
            .filter_map(|l| match l {
                Some(l) if l.1.is_batched() => Some(l),
                _ => None,
            })
            .collect_vec();

        let evals = {
            let batch = batched_leafs.iter().map(|b| b.0.clone()).collect_vec();
            self.evaluator
                .eval_batch(&batch)
                // todo: idk why we need to collect this as vec here
                .collect_vec()
        };

        for (leaf, eval) in batched_leafs.iter_mut().zip(evals) {
            leaf.1 = EvalItem::Evaluated(eval);
        }
    }

    fn backup_evals(&mut self) {
        for sel in self.selection.leafs.iter().flatten() {
            let node = sel.0.clone();
            if let EvalItem::Evaluated(eval) = &sel.1 {
                self.backprop.backpropagate(node, eval);
            }
        }
    }

    fn apply_noise(&mut self) {
        // Only apply noise once
        if self.iterations > 0 {
            return;
        }

        let root = self.tree.get_root();
        _ = self.noiser.apply_noise(&mut root.borrow_mut());
    }
}
