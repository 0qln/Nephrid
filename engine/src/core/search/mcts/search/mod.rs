use crate::core::{
    Position,
    search::mcts::{
        back::{Backpropagater, DefaultBackuper},
        eval::{Evaluation, Evaluator},
        limiter::{self, DefaultLimiter, Limiter},
        node::{Node, NodeState, Tree},
        noise::{DirichletNoiser, Noiser},
        select::{PuctSelector, Selector},
        utils::DoubleLinkedNode,
    },
    turn::Turn,
};
use std::{cell::RefCell, rc::Rc};

use crate::core::depth::Depth;

#[cfg(test)]
pub mod test;

pub struct SelectionItem<T> {
    /// The selected node.
    pub leaf: Rc<RefCell<Node>>,

    /// Depth from root
    pub depth: Depth,

    /// Current player's turn
    pub turn: Turn,

    /// Context specific data.
    pub trace_data: T,
}

pub type SelectionNode<T> = DoubleLinkedNode<SelectionItem<T>>;
pub type SelectionNodeRef<T> = Rc<RefCell<SelectionNode<T>>>;

pub struct Selection<const X: usize, T> {
    pub root: Option<SelectionNodeRef<T>>,
    pub leafs: [Option<SelectionNodeRef<T>>; X],
}

impl<const X: usize, T> Default for Selection<X, T> {
    fn default() -> Self {
        const fn empty_leaf<T>() -> Option<SelectionNodeRef<T>> {
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
            leaf: root_node,
            depth: Depth::MIN,
            turn,
            trace_data,
        })));

        self.root = Some(root.clone());
        root
    }

    pub fn set(&mut self, index: usize, item: SelectionNodeRef<T>) {
        self.leafs[index] = Some(item);
    }
}

/// # Tree searcher
///
/// This statemachine should hold all the info that is required to search.
///
/// This is the implementation for the nn-backed eval model searcher. (e.g. lc0,
/// a0)
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
    L: Limiter = DefaultLimiter,
    S: Selector = PuctSelector,
    B: Backpropagater = DefaultBackuper,
    N: Noiser = DirichletNoiser,
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
        self.evaluator.eval_guesses();
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
        let eval_data = self.evaluator.create_data(node.clone(), &self.position);
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

        // Check if the board has a terminal evaluation
        let terminal_eval = E::eval_terminal(&node.borrow(), pos);
        if let Some(terminal_eval) = terminal_eval {
            node.borrow_mut().set_state(NodeState::Terminal);
            self.evaluator.set_eval(line_index, terminal_eval);
        }
        // Check if we are even interested in searching this line any further.
        else if self.limiter.should_stop(limiter::Params { pos, depth }) {
            self.evaluator.set_eval(line_index, Evaluation::Nope);
        }
        // Else note down that we need to guess this node's evaluation.
        else {
            self.evaluator.batch_eval(line_index, eval_node);
        }

        self.selection.set(line_index, sel_node);

        1
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
                        trace_data: self.evaluator.create_data(node.clone(), &self.position),
                    },
                );

                let used = self.process_node(current_budget, line_index, depth, node, selc_info);

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

            self.backprop.backpropagate(leaf, eval);
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
