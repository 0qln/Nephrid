use std::{cmp::Ordering, ops::Try};

use itertools::Itertools;

use crate::core::{
    Position,
    depth::Depth,
    search::mcts::{
        back::Backpropagater,
        eval::{Evaluation, Evaluator, softmax},
        limiter::{self, Limiter},
        node::{
            BranchId, NodeId, NodeView, Tree,
            node_state::{self, *},
        },
        noise::Noiser,
        select::Selector,
    },
    turn::Turn,
};

use super::eval::GameResult;

#[cfg(test)]
pub mod test;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SelNodeId(pub usize);

/// Info about a selected node or its ascendant in the tree.
#[derive(Debug)]
pub struct SelNode<T, S: node_state::Any> {
    /// The node index wrapper
    pub node: NodeId<S>,

    /// Current player's turn
    pub turn: Turn,

    /// The sel parent node.
    pub parent: Option<SelNodeId>,

    /// Payload `T`
    pub data: T,

    /// Weight.
    pub weight: f32,
}

pub type BatchItem<T> = SelNode<T, Branching>;

pub type EvalItem = SelNode<Evaluation, Branching>;

pub type TerminalItem = SelNode<Evaluation, Terminal>;

pub type ShortcutItem = SelNode<Evaluation, Leaf>;

pub type SkipItem = SelNode<Evaluation, Evaluated>;

#[derive(Debug)]
pub enum PhaseItem<T> {
    Unused,
    Batched(BatchItem<T>),
    Evaluated(EvalItem),
    Terminal(TerminalItem),
    Shortcut(ShortcutItem),
    Skip(SkipItem),
}

impl<T> PhaseItem<T> {
    pub fn batch_item(&self) -> Option<&BatchItem<T>> {
        match self {
            PhaseItem::Batched(x) => Some(x),
            _ => None,
        }
    }
}

pub struct Selection<const X: usize, T> {
    pub arena: Vec<SelNode<T, Evaluated>>,
    pub root: Option<SelNodeId>,
    pub leafs: [PhaseItem<T>; X],
}

const fn empty_leaf<T>() -> PhaseItem<T> {
    PhaseItem::<T>::Unused
}

impl<const X: usize, T> Default for Selection<X, T> {
    fn default() -> Self {
        Self {
            arena: Vec::new(),
            root: None,
            leafs: [const { empty_leaf::<T>() }; X],
        }
    }
}

impl<const X: usize, T> Selection<X, T> {
    /// Initializes a new root node.
    pub fn init_root(
        &mut self,
        root_node: NodeId<Evaluated>,
        turn: Turn,
        trace_data: T,
    ) -> SelNodeId {
        self.clear();

        let root_id = SelNodeId(self.arena.len());
        self.arena.push(SelNode {
            node: root_node,
            turn,
            parent: None,
            data: trace_data,
            weight: 1.,
        });

        self.root = Some(root_id);
        root_id
    }

    /// Clear the arena and selection.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.leafs = [const { empty_leaf::<T>() }; X];
        self.root = None;
    }

    /// Allocates a new Parent node in the arena and attaches it to the parent.
    pub fn append_parent(
        &mut self,
        parent_id: SelNodeId,
        parent_node: NodeId<Evaluated>,
        turn: Turn,
        trace_data: T,
    ) -> SelNodeId {
        let child_id = SelNodeId(self.arena.len());

        self.arena.push(SelNode {
            node: parent_node,
            turn,
            parent: Some(parent_id),
            data: trace_data,
            weight: 1.,
        });

        child_id
    }

    pub fn set(&mut self, index: usize, item: PhaseItem<T>) {
        self.leafs[index] = item;
    }

    pub fn get_node(&self, id: SelNodeId) -> &SelNode<T, Evaluated> {
        &self.arena[id.0]
    }

    pub fn get_node_mut(&mut self, id: SelNodeId) -> &mut SelNode<T, Evaluated> {
        &mut self.arena[id.0]
    }

    /// Applies `f` to the given node and all parent nodes, moving up the tree.
    pub fn try_fold_up_mut<B, F, R>(&mut self, mut current: SelNodeId, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, &mut SelNode<T, Evaluated>) -> R,
        R: Try<Output = B>,
    {
        loop {
            let node = &mut self.arena[current.0];
            init = f(init, node)?;

            if let Some(parent_id) = node.parent {
                current = parent_id;
            }
            else {
                break;
            }
        }
        R::from_output(init)
    }

    pub fn try_fold_up<B, F, R>(&self, mut current: Option<SelNodeId>, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, &SelNode<T, Evaluated>) -> R,
        R: std::ops::Try<Output = B>,
    {
        while let Some(curr) = current {
            let node = &self.arena[curr.0];
            init = f(init, node)?;
            current = node.parent;
        }
        R::from_output(init)
    }
}

/// # Tree searcher
pub struct TreeSearcher<
    'pos,
    const MPV: usize,
    E: Evaluator,
    L: Limiter,
    S: Selector,
    B: Backpropagater,
    N: Noiser,
> {
    position: &'pos mut Position,
    selector: S,
    limiter: L,
    evaluator: E,
    backprop: B,
    noiser: N,
    selection: Selection<MPV, E::TraceData>,
}

impl<'pos, const MPV: usize, E: Evaluator, L: Limiter, S: Selector, B: Backpropagater, N: Noiser>
    TreeSearcher<'pos, MPV, E, L, S, B, N>
{
    pub fn new(
        position: &'pos mut Position,
        selector: S,
        limiter: L,
        evaluator: E,
        backprop: B,
        noiser: N,
    ) -> Self {
        Self {
            position,
            selector,
            limiter,
            evaluator,
            backprop,
            noiser,
            selection: Default::default(),
        }
    }

    /// Expands, evaluates, and applies noise to the root node, Such that the
    /// tree is prepared to be grown.
    pub fn init_root(&mut self, tree: &mut Tree) {
        loop {
            match tree.node_switch(tree.root()) {
                // If the root is a leaf, expand and transition to next phase.
                Switch::Leaf(node) => {
                    let _ = tree.expand_node(node, self.position, Depth::ROOT);
                }
                // If the root is branching, evaluate and transition to next phase.
                Switch::Branching(node) => {
                    // init selection
                    let turn = self.position.get_turn();
                    let trace_data = self.evaluator.trace(node, tree, self.position);
                    self.selection.clear();
                    self.selection.set(
                        0,
                        PhaseItem::Batched(SelNode {
                            node,
                            turn,
                            parent: None,
                            data: trace_data,
                            weight: 1.,
                        }),
                    );

                    // eval selection
                    let eval = {
                        let leaf = self.selection.leafs[0].batch_item().unwrap();
                        self.evaluator
                            .eval_batch(tree, &self.selection, &[leaf])
                            .next()
                            .unwrap()
                    };

                    // backpropagation for root
                    let _evaluated = match eval {
                        Evaluation::Guess(guess) => tree.set_policy(node, &guess.policy),
                        _ => tree.skip_policy(node),
                    };

                    self.selection.clear();
                }
                // If the node is evaluated, apply noise and we're done.
                Switch::Evaluated(node) => {
                    if let Err(err) = self.noiser.apply_noise(node, tree) {
                        println!("Failed to apply noise to root node: {err}");
                    }
                    break;
                }
                // If the root node is terminal, we cannot grow it... just break here.
                Switch::Terminal(_node) => {
                    break;
                }
            }
        }
    }

    pub fn grow(&mut self, tree: &mut Tree) {
        self.selection.clear();
        self.select_lines(tree);
        self.eval_batched(tree);
        self.backup_evals(tree);
    }

    fn select_lines(&mut self, tree: &mut Tree) {
        let root_id = tree.root();
        let turn = self.position.get_turn();

        let root = match tree.node_switch(root_id) {
            Switch::Evaluated(n) => n,
            Switch::Terminal(_) => {
                panic!("Root must not be terminal! Did you forget to abandon the search?")
            }
            Switch::Leaf(_) | Switch::Branching(_) => {
                panic!("Root must be evaluated before selecting lines! Did you call init_root?")
            }
        };

        let eval_data = self.evaluator.trace(root, tree, self.position);

        let sel_root_id = self.selection.init_root(root, turn, eval_data);
        self.pick_branches(MPV, 0, Depth::ROOT, root, tree, sel_root_id);
    }

    fn pick_branches(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        parent_node: NodeId<Evaluated>,
        tree: &mut Tree,
        sel_node_id: SelNodeId,
    ) -> usize {
        let parent_visits = NodeView::new(tree, parent_node).visits();

        tree.sort_branches_by(parent_node, |child_a, branch_a, child_b, branch_b| {
            let visit_threshold = 4; // todo: fine-tune this

            let score_a = if child_a.value().is_proven_loss() && child_a.visits() >= visit_threshold
            {
                self.selector.min_score()
            }
            else {
                self.selector.score(child_a, branch_a, parent_visits)
            };

            let score_b = if child_b.value().is_proven_loss() && child_b.visits() >= visit_threshold
            {
                self.selector.min_score()
            }
            else {
                self.selector.score(child_b, branch_b, parent_visits)
            };

            // Negative so we sort descending
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        let branch_count = tree.node(parent_node).branch_count();

        // todo:
        // it was not benchmarked that this is actually better than the previous
        // budgeting strategy. for sure it is better when training the
        // eval-model, but maybe not vor eval/hce and not when actually
        // inferencing? this should be benchmarked and optimized before merge
        // into main.
        let mut scores = (0..branch_count.v)
            .map(|branch_index| {
                let branch_id = tree
                    .branch_id(parent_node, branch_index.into())
                    .expect("The branch should exist...");

                let branch = tree.branch(branch_id);
                let child = tree.node(branch.node());

                // todo when making the budget relative to the score, maybe use something
                // similar to policy::from_visit_counts ?

                let score = self.selector.score(child.data(), branch, parent_visits);

                Into::<f32>::into(score)
            })
            .collect_vec();

        softmax(scores.as_mut_slice(), 1.);

        // 1. Calculate precise floating-point budgets
        let float_budgets: Vec<f32> = scores.iter().map(|&p| p * budget as f32).collect();

        // 2. Assign the guaranteed floor to each branch
        let mut allocated_budgets: Vec<usize> =
            float_budgets.iter().map(|&f| f.floor() as usize).collect();

        // 3. Calculate how much budget is left to distribute
        let remaining_budget = budget - allocated_budgets.iter().sum::<usize>();

        // 4. Distribute the remainder based on the highest fractional parts
        if remaining_budget > 0 {
            // Pair original index with its remainder
            let mut remainders: Vec<(usize, f32)> = float_budgets
                .iter()
                .enumerate()
                .map(|(i, &f)| (i, f - f.floor()))
                .collect();

            // Sort descending by remainder
            remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            // Give 1 extra budget to the top branches until we run out
            for (index, _) in remainders.into_iter().take(remaining_budget) {
                allocated_budgets[index] += 1;
            }
        }

        let mut line_index = line_index;
        let mut used_budget = 0;

        for (branch_index, curr_budget) in (0..branch_count.v).zip(allocated_budgets.into_iter()) {
            let branch_id = tree
                .branch_id(parent_node, branch_index.into())
                .expect("The branch should exist...");

            if curr_budget == 0 {
                continue;
            };

            let used =
                self.select_branch(curr_budget, line_index, depth, branch_id, tree, sel_node_id);

            line_index += used;
            used_budget += used;
        }

        used_budget
    }

    fn select_branch(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        branch: BranchId,
        tree: &mut Tree,
        parent_sel_id: SelNodeId,
    ) -> usize {
        let (mov, node) = {
            let branch = tree.branch(branch);
            (branch.mov(), branch.node())
        };

        self.position.make_move(mov);
        let depth = depth + 1;
        let turn = self.position.get_turn();

        let used = match tree.node_switch(node) {
            Switch::Branching(node) => {
                self.select_branching(budget, line_index, parent_sel_id, node, depth, tree)
            }
            Switch::Evaluated(node) => {
                let val = NodeView::new(tree, node).value();
                // skip further selection of proven_win/loss nodes.
                if val.is_proven_win() {
                    let eval = Evaluation::Terminal(GameResult::Win { relative_to: !turn });
                    self.select_skip(budget, line_index, parent_sel_id, node, eval, depth)
                }
                else if val.is_proven_loss() {
                    let eval = Evaluation::Terminal(GameResult::Win { relative_to: turn });
                    self.select_skip(budget, line_index, parent_sel_id, node, eval, depth)
                }
                // otherwise continue down the tree
                else {
                    let trace_data = self.evaluator.trace(node, tree, self.position);
                    let child_id =
                        self.selection
                            .append_parent(parent_sel_id, node, turn, trace_data);
                    self.pick_branches(budget, line_index, depth, node, tree, child_id)
                }
            }
            Switch::Leaf(node) => {
                // shortcut leaf selection using twofold repetition
                if depth > Depth::ROOT && self.position.has_twofold_repetition() {
                    let eval = Evaluation::Terminal(GameResult::Draw);
                    self.select_shortcut(budget, line_index, parent_sel_id, node, eval, depth)
                }
                // otherwise select this leaf for evaluation in the next phase
                else {
                    self.select_leaf(budget, line_index, parent_sel_id, node, tree, depth)
                }
            }
            Switch::Terminal(node) => {
                self.select_terminal(budget, line_index, parent_sel_id, node, depth, tree)
            }
        };

        self.position.unmake_move(mov);
        used
    }

    #[inline]
    fn select_skip(
        &mut self,
        budget: usize,
        line_index: usize,
        parent_id: SelNodeId,
        node: NodeId<Evaluated>,
        eval: Evaluation,
        _depth: Depth,
    ) -> usize {
        self.selection.set(
            line_index,
            PhaseItem::Skip(SelNode {
                node,
                turn: self.position.get_turn(),
                parent: Some(parent_id),
                data: eval,
                weight: budget as f32,
            }),
        );
        budget
    }

    #[inline]
    fn select_leaf(
        &mut self,
        budget: usize,
        line_index: usize,
        parent_sel_id: SelNodeId,
        node: NodeId<Leaf>,
        tree: &mut Tree,
        depth: Depth,
    ) -> usize {
        let expanded = tree.expand_node(node, self.position, depth);

        match expanded {
            ExpandedSwitch::Terminal(node) => {
                self.select_terminal(budget, line_index, parent_sel_id, node, depth, tree)
            }
            ExpandedSwitch::Branching(node) => {
                self.select_branching(budget, line_index, parent_sel_id, node, depth, tree)
            }
        }
    }

    /// Select a shortcut to a node that can be considered terminal.
    #[inline]
    fn select_shortcut(
        &mut self,
        budget: usize,
        line_index: usize,
        parent_id: SelNodeId,
        node: NodeId<Leaf>,
        eval: Evaluation,
        _depth: Depth,
    ) -> usize {
        self.selection.set(
            line_index,
            PhaseItem::Shortcut(SelNode {
                node,
                turn: self.position.get_turn(),
                parent: Some(parent_id),
                data: eval,
                weight: budget as f32,
            }),
        );
        budget
    }

    fn select_terminal(
        &mut self,
        budget: usize,
        line_index: usize,
        parent_id: SelNodeId,
        node: NodeId<Terminal>,
        depth: Depth,
        tree: &Tree,
    ) -> usize {
        let eval = E::eval_terminal(node, tree, depth, self.position);
        self.selection.set(
            line_index,
            PhaseItem::Terminal(SelNode {
                node,
                turn: self.position.get_turn(),
                parent: Some(parent_id),
                data: eval,
                weight: budget as f32,
            }),
        );
        budget
    }

    fn select_branching(
        &mut self,
        budget: usize,
        line_index: usize,
        parent_id: SelNodeId,
        node: NodeId<Branching>,
        depth: Depth,
        tree: &Tree,
    ) -> usize {
        let pos = &mut self.position;

        // todo: is this good? not even sure tbh... should be benchmarked
        let (used_budget, item) = if self.limiter.should_stop(limiter::Params { pos, depth }) {
            // todo: also add the budget as a weight here?
            (budget, PhaseItem::Unused)
        }
        else {
            let trace_data = self.evaluator.trace(node, tree, pos);
            (
                budget,
                PhaseItem::Batched(SelNode {
                    node,
                    turn: self.position.get_turn(),
                    parent: Some(parent_id),
                    data: trace_data,
                    weight: budget as f32,
                }),
            )
        };

        self.selection.set(line_index, item);

        used_budget
    }

    fn eval_batched(&mut self, tree: &Tree) {
        let batched_indices = self
            .selection
            .leafs
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l, PhaseItem::Batched(_)))
            .map(|(i, _)| i)
            .collect_vec();

        let evals: Vec<Evaluation> = {
            let leafs: Vec<&BatchItem<E::TraceData>> = batched_indices
                .iter()
                .filter_map(|&i| self.selection.leafs[i].batch_item())
                .collect_vec();

            self.evaluator
                .eval_batch(tree, &self.selection, &leafs)
                .collect()
        };

        for (i, eval) in batched_indices.into_iter().zip(evals) {
            let batch_item = self.selection.leafs[i].batch_item().unwrap();
            self.selection.set(
                i,
                PhaseItem::Evaluated(SelNode {
                    node: batch_item.node,
                    turn: batch_item.turn,
                    parent: batch_item.parent,
                    data: eval,
                    // todo: unlcear if wegiht 1 is correct here... maybe use the weight from
                    // budgeting...
                    weight: 1.0,
                }),
            )
        }
    }

    fn backup_evals(&mut self, tree: &mut Tree) {
        for leaf in self.selection.leafs.iter() {
            match leaf {
                // ignore unused slots.
                PhaseItem::Unused => {}

                // the evaluator is bad.
                PhaseItem::Batched(_) => todo!(
                    "the evaluator forgot the eval a batched item. this shouldn't happen, log an \
                     error or something"
                ),

                // backup terminals, guesses, etc.
                PhaseItem::Evaluated(x) => self.backprop.backpropagate(tree, &self.selection, x),
                PhaseItem::Terminal(x) => self.backprop.backpropagate(tree, &self.selection, x),
                PhaseItem::Shortcut(x) => self.backprop.backpropagate(tree, &self.selection, x),
                PhaseItem::Skip(x) => self.backprop.backpropagate(tree, &self.selection, x),
            }
        }
    }
}
