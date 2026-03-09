use std::ops::Try;

use crate::core::{
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
};

#[cfg(test)]
pub mod test;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug)]
pub struct SelectionItem<T, S: node_state::Any> {
    /// The selected leaf that was just expanded, but has not yet been evaluated
    pub node: CtNodeRef<S>,
    /// Current player's turn
    pub turn: Turn,
    /// Context specific data
    pub trace_data: T,
}

#[derive(Debug)]
pub enum EvalItem {
    Batched,
    Evaluated(Evaluation),
}

impl EvalItem {
    pub fn is_batched(&self) -> bool {
        matches!(self, Self::Batched)
    }
}

pub type SelParentData<T> = SelectionItem<T, Evaluated>;
pub type SelLeafData<T> = SelectionItem<T, Branching>;

/// A node allocated within the Selection's bump arena.
/// Since leaves are stored separately, this is ALWAYS a parent node!
#[derive(Debug)]
pub struct SelectionNode<T> {
    pub item: SelParentData<T>,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
}

/// The new dedicated Leaf structure.
#[derive(Debug)]
pub struct SelectionLeaf<T> {
    /// Option because Terminal nodes don't have branching trace data!
    pub leaf_data: Option<SelLeafData<T>>,
    /// The parent node inside the bump arena
    pub parent_id: NodeId,
    pub eval: EvalItem,
}

pub struct Selection<const X: usize, T> {
    /// The Bump Arena. Now STRICTLY contains only Parent nodes.
    pub arena: Vec<SelectionNode<T>>,
    pub root: Option<NodeId>,
    /// The dedicated arena for our Leaf nodes.
    pub leafs: [Option<SelectionLeaf<T>>; X],
}

const fn empty_leaf<T>() -> Option<SelectionLeaf<T>> {
    None
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
        let root_id = NodeId(self.arena.len());
        self.arena.push(SelectionNode {
            item: SelectionItem {
                node: root_node,
                turn,
                trace_data,
            },
            parent: None,
            children: vec![],
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
    /// No more Results/Enums needed!
    pub fn append_parent(
        &mut self,
        parent_id: NodeId,
        parent_node: CtNodeRef<Evaluated>,
        turn: Turn,
        trace_data: T,
    ) -> NodeId {
        let child_id = NodeId(self.arena.len());

        // Add the child ID to the parent's children vector
        self.arena[parent_id.0].children.push(child_id);

        // Bump allocate the new child node
        self.arena.push(SelectionNode {
            item: SelectionItem {
                node: parent_node,
                turn,
                trace_data,
            },
            parent: Some(parent_id),
            children: vec![],
        });

        child_id
    }

    pub fn set(&mut self, index: usize, item: SelectionLeaf<T>) {
        self.leafs[index] = Some(item);
    }

    pub fn get_node(&self, id: NodeId) -> &SelectionNode<T> {
        &self.arena[id.0]
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> &mut SelectionNode<T> {
        &mut self.arena[id.0]
    }

    /// Applies `f` to the given node and all parent nodes, moving up the tree.
    pub fn try_fold_up_mut<B, F, R>(&mut self, mut current: NodeId, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, &mut SelectionNode<T>) -> R,
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

    pub fn try_fold_up<B, F, R>(&self, mut current: NodeId, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, &SelectionNode<T>) -> R,
        R: std::ops::Try<Output = B>,
    {
        loop {
            let node = &self.arena[current.0];
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
                    let _ = tree.expand_node(node, &self.position, Depth::ROOT);
                }
                // If the root is branching, evaluate and transition to next phase.
                NodeSwitch::Branching(node) => {
                    // init selection
                    let turn = self.position.get_turn();
                    let trace_data = self.evaluator.trace(node.clone(), &self.position);
                    self.selection.clear();
                    self.selection.set(
                        0,
                        SelectionLeaf {
                            leaf_data: Some(SelectionItem {
                                node: node.clone(),
                                turn,
                                trace_data,
                            }),
                            parent_id: NodeId(0),
                            eval: EvalItem::Batched,
                        },
                    );

                    // eval selection
                    let eval = {
                        let leaf = self.selection.leafs[0].as_ref().unwrap();
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

    pub fn grow(&mut self, tree: &mut Tree) {
        self.selection.clear();

        self.select_lines(tree);
        self.eval_batched();
        self.backup_evals(tree);
    }

    fn select_lines(&mut self, tree: &mut Tree) {
        let root = tree.get_root().clone();
        let turn = self.position.get_turn();

        let root = match root.into_ct() {
            NodeSwitch::Evaluated(n) => n,
            _ => panic!("Root must be evaluated before selecting lines! Did you call init_root?"),
        };

        let eval_data = self.evaluator.trace(root.clone(), &self.position);

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
                // Now directly goes to select_branching! No appending here.
                self.select_branching(line_index, parent_sel_id, node, depth)
            }
            NodeSwitch::Evaluated(node) => {
                let trace_data = self.evaluator.trace(node.clone(), &self.position);
                let child_id =
                    self.selection
                        .append_parent(parent_sel_id, node.clone(), turn, trace_data);
                self.pick_branches(budget, line_index, depth, node, tree, child_id)
            }
            NodeSwitch::Leaf(node) => {
                self.select_leaf(line_index, parent_sel_id, node, tree, depth)
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
        let expanded = tree.expand_node(node, &self.position, depth);

        match expanded {
            ExpandedRefSwitch::Terminal(node) => {
                self.select_terminal(line_index, parent_sel_id, node, depth)
            }
            ExpandedRefSwitch::Branching(node) => {
                self.select_branching(line_index, parent_sel_id, node, depth)
            }
        }
    }

    fn select_terminal(
        &mut self,
        line_index: usize,
        parent_sel_id: NodeId,
        node: CtNodeRef<Terminal>,
        _depth: Depth,
    ) -> usize {
        let eval = E::eval_terminal(node, &self.position);

        // Terminal nodes just set `leaf_data` to None!
        self.selection.set(
            line_index,
            SelectionLeaf {
                leaf_data: None,
                parent_id: parent_sel_id,
                eval: EvalItem::Evaluated(eval),
            },
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

        let (used_budget, eval) = if self.limiter.should_stop(limiter::Params { pos, depth }) {
            (1, EvalItem::Evaluated(Evaluation::Nope))
        }
        else {
            (1, EvalItem::Batched)
        };

        let turn = self.position.get_turn();
        let trace_data = self.evaluator.trace(node.clone(), pos);

        // Put the newly encountered leaf directly into the leafs array
        self.selection.set(
            line_index,
            SelectionLeaf {
                leaf_data: Some(SelectionItem { node, turn, trace_data }),
                parent_id,
                eval,
            },
        );

        used_budget
    }

    fn eval_batched(&mut self) {
        let mut batched_indices = Vec::new();
        for (i, leaf) in self.selection.leafs.iter().enumerate() {
            if let Some(l) = leaf {
                if l.eval.is_batched() {
                    batched_indices.push(i);
                }
            }
        }

        let evals: Vec<Evaluation> = {
            let leafs: Vec<&SelectionLeaf<E::TraceData>> = batched_indices
                .iter()
                .map(|&i| self.selection.leafs[i].as_ref().unwrap())
                .collect();

            self.evaluator.eval_batch(&self.selection, &leafs).collect()
        };

        for (i, eval) in batched_indices.into_iter().zip(evals) {
            if let Some(leaf) = &mut self.selection.leafs[i] {
                leaf.eval = EvalItem::Evaluated(eval);
            }
        }
    }

    fn backup_evals(&mut self, tree: &mut Tree) {
        for leaf in self.selection.leafs.iter().flatten() {
            self.backprop.backpropagate(tree, &self.selection, leaf);
        }
    }
}
