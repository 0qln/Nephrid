use std::ops::Try;

use itertools::Itertools;

use crate::{
    core::{
        Position,
        depth::Depth,
        search::mcts::{
            back::Backpropagater,
            eval::{Evaluation, Evaluator, RawPolicy},
            limiter::{self, Limiter},
            node::{
                Branch, CtNodeRef, Tree,
                node_state::{self, *},
            },
            noise::Noiser,
            select::Selector,
        },
        turn::Turn,
    },
    uci::sync::CancellationToken,
};

use super::eval::GameResult;

#[cfg(test)]
pub mod test;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Info about a selected node or it's ascendend in the tree.
#[derive(Debug)]
pub struct SelNode<T, S: node_state::Any> {
    /// The node.
    pub node: CtNodeRef<S>,

    /// Current player's turn
    pub turn: Turn,

    /// The sel parent node.
    pub parent: Option<NodeId>,

    pub data: T,
}

pub type BatchItem<T> = SelNode<T, Branching>;

pub type EvalItem = SelNode<Evaluation, Branching>;

pub type TerminalItem = SelNode<Evaluation, Terminal>;

pub type ShortcutItem = SelNode<Evaluation, Leaf>;

#[derive(Debug)]
pub enum PhaseItem<T> {
    Unused,
    Batched(BatchItem<T>),
    Evaluated(EvalItem),
    Terminal(TerminalItem),
    Shortcut(ShortcutItem),
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
    pub root: Option<NodeId>,
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
        root_node: CtNodeRef<Evaluated>,
        turn: Turn,
        trace_data: T,
    ) -> NodeId {
        self.clear();

        let root_id = NodeId(self.arena.len());
        self.arena.push(SelNode {
            node: root_node,
            turn,
            parent: None,
            data: trace_data,
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
        parent_id: NodeId,
        parent_node: CtNodeRef<Evaluated>,
        turn: Turn,
        trace_data: T,
    ) -> NodeId {
        let child_id = NodeId(self.arena.len());

        self.arena.push(SelNode {
            node: parent_node,
            turn,
            parent: Some(parent_id),
            data: trace_data,
        });

        child_id
    }

    pub fn set(&mut self, index: usize, item: PhaseItem<T>) {
        self.leafs[index] = item;
    }

    pub fn get_node(&self, id: NodeId) -> &SelNode<T, Evaluated> {
        &self.arena[id.0]
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> &mut SelNode<T, Evaluated> {
        &mut self.arena[id.0]
    }

    /// Applies `f` to the given node and all parent nodes, moving up the tree.
    pub fn try_fold_up_mut<B, F, R>(&mut self, mut current: NodeId, mut init: B, mut f: F) -> R
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

    pub fn try_fold_up<B, F, R>(&self, mut current: Option<NodeId>, mut init: B, mut f: F) -> R
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
            match tree.get_root().into_ct() {
                // If the root is a leaf, expand and transition to next phase.
                NodeSwitch::Leaf(node) => {
                    let _ = tree.expand_node(node, self.position, Depth::ROOT);
                }
                // If the root is branching, evaluate and transition to next phase.
                NodeSwitch::Branching(node) => {
                    // init selection
                    let turn = self.position.get_turn();
                    let trace_data = self.evaluator.trace(node.clone(), self.position);
                    self.selection.clear();
                    self.selection.set(
                        0,
                        PhaseItem::Batched(SelNode {
                            node: node.clone(),
                            turn,
                            parent: None,
                            data: trace_data,
                        }),
                    );

                    // eval selection
                    let eval = {
                        let leaf = self.selection.leafs[0].batch_item().unwrap();
                        self.evaluator
                            .eval_batch(&self.selection, &[leaf])
                            .next()
                            .unwrap()
                    };

                    // backpropagation for root
                    let policy = match eval {
                        Evaluation::Guess(guess) => guess.policy,
                        _ => {
                            // default to null policy, such that we can be sure the state advances
                            // from here on.
                            RawPolicy::null()
                        }
                    };
                    let _ = tree.set_policy_raw(node, &policy);

                    self.selection.clear();
                }
                // If the node is evaluated, apply noise and we're done.
                NodeSwitch::Evaluated(node) => {
                    let _ = self.noiser.apply_noise(node, tree);
                    break;
                }
                // If the root node is terminal, we cannot grow it... just break here.
                NodeSwitch::Terminal(_node) => {
                    break;
                }
            }
        }
    }

    pub fn grow(&mut self, tree: &mut Tree, ct: CancellationToken) {
        self.selection.clear();
        if ct.is_cancelled() {
            return;
        }
        self.select_lines(tree);
        if ct.is_cancelled() {
            return;
        }
        self.eval_batched();
        if ct.is_cancelled() {
            return;
        }
        self.backup_evals(tree);
    }

    fn select_lines(&mut self, tree: &mut Tree) {
        let root = tree.get_root().clone();
        let turn = self.position.get_turn();

        let root = match root.into_ct() {
            NodeSwitch::Evaluated(n) => n,
            _ => panic!("Root must be evaluated before selecting lines! Did you call init_root?"),
        };

        let eval_data = self.evaluator.trace(root.clone(), self.position);

        let sel_root_id = self.selection.init_root(root.clone(), turn, eval_data);
        self.pick_branches(MPV, 0, Depth::ROOT, root, tree, sel_root_id);
    }

    fn pick_branches(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        parent_node: CtNodeRef<Evaluated>,
        tree: &mut Tree,
        sel_node_id: NodeId,
    ) -> usize {
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
                let curr_budget = self.selector.budget(budget);
                if curr_budget == 0 {
                    break;
                };

                let used =
                    self.select_branch(curr_budget, line_index, depth, branch, tree, sel_node_id);

                budget -= curr_budget;
                branch_index += 1;

                line_index += used;
                used_budget += used;
            }
            else {
                break;
            }
        }
        used_budget
    }

    fn select_branch(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        branch: &Branch,
        tree: &mut Tree,
        parent_sel_id: NodeId,
    ) -> usize {
        self.position.make_move(branch.mov());
        let depth = depth + 1;
        let turn = self.position.get_turn();

        let used = match branch.node().into_ct() {
            NodeSwitch::Branching(node) => {
                self.select_branching(line_index, parent_sel_id, node, depth)
            }
            NodeSwitch::Evaluated(node) => {
                let trace_data = self.evaluator.trace(node.clone(), self.position);
                let child_id =
                    self.selection
                        .append_parent(parent_sel_id, node.clone(), turn, trace_data);
                self.pick_branches(budget, line_index, depth, node, tree, child_id)
            }
            NodeSwitch::Leaf(node) => {
                if depth > Depth::ROOT && self.position.has_twofold_repetition() {
                    let eval = Evaluation::Terminal(GameResult::Draw);
                    self.select_shortcut(line_index, parent_sel_id, node, eval, depth)
                }
                else {
                    self.select_leaf(line_index, parent_sel_id, node, tree, depth)
                }
            }
            NodeSwitch::Terminal(node) => {
                self.select_terminal(line_index, parent_sel_id, node, depth)
            }
        };

        self.position.unmake_move(branch.mov());
        used
    }

    fn select_leaf(
        &mut self,
        line_index: usize,
        parent_sel_id: NodeId,
        node: CtNodeRef<Leaf>,
        tree: &mut Tree,
        depth: Depth,
    ) -> usize {
        let expanded = tree.expand_node(node, self.position, depth);

        match expanded {
            ExpandedRefSwitch::Terminal(node) => {
                self.select_terminal(line_index, parent_sel_id, node, depth)
            }
            ExpandedRefSwitch::Branching(node) => {
                self.select_branching(line_index, parent_sel_id, node, depth)
            }
        }
    }

    /// Select a shortcut to a node that can be considered terminal.
    fn select_shortcut(
        &mut self,
        line_index: usize,
        parent_id: NodeId,
        node: CtNodeRef<Leaf>,
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
            }),
        );
        1
    }

    fn select_terminal(
        &mut self,
        line_index: usize,
        parent_id: NodeId,
        node: CtNodeRef<Terminal>,
        depth: Depth,
    ) -> usize {
        let eval = E::eval_terminal(node.clone(), depth, self.position);
        self.selection.set(
            line_index,
            PhaseItem::Terminal(SelNode {
                node,
                turn: self.position.get_turn(),
                parent: Some(parent_id),
                data: eval,
            }),
        );
        1
    }

    fn select_branching(
        &mut self,
        line_index: usize,
        parent_id: NodeId,
        node: CtNodeRef<Branching>,
        depth: Depth,
    ) -> usize {
        let pos = &self.position;

        let (used_budget, item) = if self.limiter.should_stop(limiter::Params { pos, depth }) {
            (1, PhaseItem::Unused)
        }
        else {
            let trace_data = self.evaluator.trace(node.clone(), pos);
            (
                1,
                PhaseItem::Batched(SelNode {
                    node,
                    turn: self.position.get_turn(),
                    parent: Some(parent_id),
                    data: trace_data,
                }),
            )
        };

        self.selection.set(line_index, item);

        used_budget
    }

    fn eval_batched(&mut self) {
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

            self.evaluator.eval_batch(&self.selection, &leafs).collect()
        };

        for (i, eval) in batched_indices.into_iter().zip(evals) {
            let batch_item = self.selection.leafs[i].batch_item().unwrap();
            self.selection.set(
                i,
                PhaseItem::Evaluated(SelNode {
                    node: batch_item.node.clone(),
                    turn: batch_item.turn,
                    parent: batch_item.parent,
                    data: eval,
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
            }
        }
    }
}
