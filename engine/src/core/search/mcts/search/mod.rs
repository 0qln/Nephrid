use bumpalo::Bump;
use itertools::Itertools;

use crate::core::{
    Position,
    depth::Depth,
    search::mcts::{
        back::Backpropagater,
        eval::{Evaluation, Evaluator},
        limiter::{self, Limiter},
        node::{NodeRef, NodeState, Tree},
        noise::Noiser,
        select::Selector,
        utils::{DoubleLinkedNode, DoubleLinkedNodeRef},
    },
    turn::Turn,
};
use std::{cell::RefCell, rc::Rc};

// todo: fix test
// #[cfg(test)]
// pub mod test;

pub struct SelectionItem<'bump, T> {
    /// The selected node.
    pub node: NodeRef<'bump>,

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

pub type SelectionNode<'bump, T> = DoubleLinkedNode<SelectionItem<'bump, T>>;

pub type SelectionNodeRef<'bump, T> = DoubleLinkedNodeRef<SelectionItem<'bump, T>>;

pub type SelectionLeaf<'bump, T> = (SelectionNodeRef<'bump, T>, EvalItem);

pub struct Selection<'bump, const X: usize, T> {
    pub root: Option<SelectionNodeRef<'bump, T>>,
    pub leafs: [Option<SelectionLeaf<'bump, T>>; X],
}

impl<'bump, const X: usize, T> Default for Selection<'bump, X, T> {
    fn default() -> Self {
        const fn empty_leaf<'bump, T>() -> Option<SelectionLeaf<'bump, T>> {
            None
        }
        Self {
            root: None,
            leafs: [const { empty_leaf() }; X],
        }
    }
}

impl<'bump, const X: usize, T> Selection<'bump, X, T> {
    pub fn init_root(
        &mut self,
        root_node: NodeRef<'bump>,
        turn: Turn,
        trace_data: T,
    ) -> SelectionNodeRef<'bump, T> {
        let root = Rc::new(RefCell::new(SelectionNode::new_root(SelectionItem {
            node: root_node,
            depth: Depth::MIN,
            turn,
            trace_data,
        })));

        self.root = Some(Rc::clone(&root));
        root
    }

    pub fn set(&mut self, index: usize, item: SelectionLeaf<'bump, T>) {
        self.leafs[index] = Some(item);
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut SelectionLeaf<'bump, T>> {
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
    'bump,
    const MPV: usize,
    E: Evaluator<'bump>,
    L: Limiter,
    S: Selector,
    B: Backpropagater,
    N: Noiser,
> {
    /// The arena where the MCTS tree is allocated.
    bump: &'bump Bump,

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

    /// Number of compeleted iterations.
    iterations: usize,

    /// Stack of nodes that were selected during the selection phase, for each
    /// principal line.
    selection: Selection<'bump, MPV, E::TraceData>,
}

impl<
    'bump,
    const MPV: usize,
    E: Evaluator<'bump>,
    L: Limiter,
    S: Selector,
    B: Backpropagater,
    N: Noiser,
> TreeSearcher<'bump, MPV, E, L, S, B, N>
{
    pub fn new_in(
        bump: &'bump Bump,
        position: Position,
        selector: S,
        limiter: L,
        evaluator: E,
        backprop: B,
        noiser: N,
    ) -> Self {
        Self {
            bump,
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

impl<
    'bump,
    const MPV: usize,
    E: Evaluator<'bump>,
    L: Limiter,
    S: Selector,
    B: Backpropagater,
    N: Noiser,
> TreeSearcher<'bump, MPV, E, L, S, B, N>
{
    pub fn grow(&mut self, tree: &mut Tree<'bump>) {
        self.select_lines(tree);
        self.eval_batched();
        self.backup_evals();
        self.apply_noise(tree);
        self.iterations += 1;
    }

    // Select the root leafes.
    fn select_lines(&mut self, tree: &mut Tree<'bump>) {
        // todo: convert to iterative approach -- or not if the stack frame is
        // small with this one ._.
        let node = tree.get_root();
        let turn = self.position.get_turn();
        let eval_data = self.evaluator.trace(&node, &self.position);
        let sel_root = self.selection.init_root(&node, turn, eval_data);
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
        node: NodeRef<'bump>,
        sel_node: SelectionNodeRef<'bump, E::TraceData>,
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
                node.borrow_mut().expand_in(&self.position, self.bump);

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
        node: NodeRef<'bump>,
        sel_node: SelectionNodeRef<'bump, E::TraceData>,
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
        parent_node: NodeRef<'bump>,
        mut sel_node_parent: SelectionNodeRef<'bump, E::TraceData>,
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

                let eval_info = self.evaluator.trace(node, &self.position);

                let sel_info = SelectionNode::append(
                    &mut sel_node_parent,
                    SelectionItem {
                        depth,
                        node,
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
            let batch = batched_leafs.iter().map(|b| Rc::clone(&b.0)).collect_vec();
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
            let node = Rc::clone(&sel.0);
            if let EvalItem::Evaluated(eval) = &sel.1 {
                self.backprop.backpropagate(node, eval);
            }
        }
    }

    fn apply_noise(&mut self, tree: &mut Tree<'bump>) {
        // Only apply noise once
        if self.iterations > 0 {
            return;
        }

        let root = tree.get_root();
        _ = self.noiser.apply_noise(&mut root.borrow_mut());
    }
}
