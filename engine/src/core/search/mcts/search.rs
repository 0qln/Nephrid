use std::mem;

use itertools::Itertools;

use crate::core::{
    Position,
    color::{Perspective, colors, perspectives},
    depth::Depth,
    search::mcts::{
        back::Backpropagater,
        eval::{Evaluation, Evaluator},
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

    /// The total virtual loss that was applied to this node during selection.
    pub virtual_loss: u32,
    //
    // todo:
    // the weight parameter was removed in the process of implementing virtual loss.
    // logically the effect remains the same: instead of using the remaining budget at
    // skip/terminal selections, we instead select that node multiple times when the outer
    // while loop hits that nodes multiple times. this however causes more backpropagation
    // passes. mayube the weight concept can be reintroduced somehow?
    //
    // todo:
    // also, i believe this currently evaluates the same node multiple times on low dephts instead
    // of overflowing remaining budget to other branches. this should be investigated.
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

    fn parent(&self) -> Option<SelNodeId> {
        match self {
            PhaseItem::Unused => None,
            PhaseItem::Batched(sel_node) => sel_node.parent,
            PhaseItem::Evaluated(sel_node) => sel_node.parent,
            PhaseItem::Terminal(sel_node) => sel_node.parent,
            PhaseItem::Shortcut(sel_node) => sel_node.parent,
            PhaseItem::Skip(sel_node) => sel_node.parent,
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
    /// Clear the arena and selection.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.leafs = [const { empty_leaf::<T>() }; X];
        self.root = None;
    }

    /// Initializes a new root node.
    pub fn init_root(
        &mut self,
        _tree: &mut Tree,
        root_node: NodeId<Evaluated>,
        turn: Turn,
        trace_data: T,
    ) -> SelNodeId {
        let id = SelNodeId(self.arena.len());

        self.arena.push(SelNode {
            node: root_node,
            turn,
            parent: None,
            data: trace_data,
            virtual_loss: 0,
        });

        self.root = Some(id);

        id
    }

    /// Allocates a new Parent node in the arena and attaches it to the parent.
    pub fn append_parent(
        &mut self,
        tree: &mut Tree,
        parent_id: SelNodeId,
        node: NodeId<Evaluated>,
        turn: Turn,
        virtual_loss: u32,
        trace_data: T,
    ) -> SelNodeId {
        tree.apply_virtual_loss(node, virtual_loss);

        let id = SelNodeId(self.arena.len());

        self.arena.push(SelNode {
            node,
            turn,
            parent: Some(parent_id),
            data: trace_data,
            virtual_loss,
        });

        id
    }

    pub fn append_leaf<S: Valid>(
        &mut self,
        tree: &mut Tree,
        node: NodeId<S>,
        index: usize,
        virtual_loss: u32,
        item: PhaseItem<T>,
    ) {
        tree.apply_virtual_loss(node, virtual_loss);

        self.leafs[index] = item;
    }

    pub fn update_leaf(&mut self, index: usize, item: PhaseItem<T>) {
        self.leafs[index] = item;
    }

    pub fn iter_path_up(
        &self,
        leaf: Option<SelNodeId>,
    ) -> impl Iterator<Item = &SelNode<T, Evaluated>> {
        Self::iter_up(leaf, &self.arena)
    }

    fn iter_up(
        current: Option<SelNodeId>,
        arena: &[SelNode<T, Evaluated>],
    ) -> impl Iterator<Item = &SelNode<T, Evaluated>> {
        pub struct IterUp<'a, T> {
            arena: &'a [SelNode<T, Evaluated>],
            current: Option<SelNodeId>,
        }

        pub struct IterItem<'a, T>(&'a SelNode<T, Evaluated>);

        impl<'a, T> Iterator for IterUp<'a, T> {
            type Item = IterItem<'a, T>;

            fn next(&mut self) -> Option<Self::Item> {
                let curr_id = self.current?;
                let node = &self.arena[curr_id.0];
                self.current = node.parent;
                Some(IterItem(node))
            }
        }

        IterUp { arena, current }.map(|item| item.0)
    }

    /// Returns an iterator over all leafs, and the selection path for each
    /// leaf.
    pub fn drain_leafs(
        &mut self,
    ) -> impl Iterator<Item = (PhaseItem<T>, impl Iterator<Item = &SelNode<T, Evaluated>>)> {
        self.leafs.iter_mut().filter_map(|leaf| {
            let leaf = mem::replace(leaf, const { empty_leaf::<T>() });
            let parent = leaf.parent();
            Some((leaf, Self::iter_up(parent, &self.arena)))
        })
    }

    fn revert_virtual_loss(&self, tree: &mut Tree) {
        for leaf in self.leafs.iter() {
            match leaf {
                PhaseItem::Unused => continue,
                PhaseItem::Batched(sel_node) => {
                    tree.revert_virtual_loss(sel_node.node, sel_node.virtual_loss)
                }
                PhaseItem::Evaluated(sel_node) => {
                    tree.revert_virtual_loss(sel_node.node, sel_node.virtual_loss)
                }
                PhaseItem::Terminal(sel_node) => {
                    tree.revert_virtual_loss(sel_node.node, sel_node.virtual_loss)
                }
                PhaseItem::Shortcut(sel_node) => {
                    tree.revert_virtual_loss(sel_node.node, sel_node.virtual_loss)
                }
                PhaseItem::Skip(sel_node) => {
                    tree.revert_virtual_loss(sel_node.node, sel_node.virtual_loss)
                }
            };
        }

        for parent in self.arena.iter() {
            let virtual_loss = parent.virtual_loss;
            let node = parent.node;
            tree.revert_virtual_loss(node, virtual_loss);
        }
    }
}

/// # Tree searcher
pub struct TreeSearcher<
    'pos,
    const MPV: usize,
    E: Evaluator,
    S: Selector,
    B: Backpropagater,
    N: Noiser,
> {
    position: &'pos mut Position,
    selector: S,
    evaluator: E,
    backprop: B,
    noiser: N,
    selection: Selection<MPV, E::TraceData>,
}

impl<'pos, const MPV: usize, E: Evaluator, S: Selector, B: Backpropagater, N: Noiser>
    TreeSearcher<'pos, MPV, E, S, B, N>
{
    pub fn new(
        position: &'pos mut Position,
        selector: S,
        evaluator: E,
        backprop: B,
        noiser: N,
    ) -> Self {
        Self {
            position,
            selector,
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
                    self.selection.append_leaf(
                        tree,
                        node,
                        0,
                        0,
                        PhaseItem::Batched(SelNode {
                            node,
                            turn,
                            parent: None,
                            data: trace_data,
                            virtual_loss: 0,
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

        match self.position.get_turn() {
            colors::WHITE => self.select_lines::<perspectives::White>(tree),
            colors::BLACK => self.select_lines::<perspectives::Black>(tree),
            _ => unsafe { std::hint::unreachable_unchecked() },
        }

        self.eval_batched(tree);

        self.revert_virtual_loss(tree);

        self.backup_evals(tree);
    }

    fn revert_virtual_loss(&mut self, tree: &mut Tree) {
        self.selection.revert_virtual_loss(tree)
    }

    fn select_lines<P: Perspective>(&mut self, tree: &mut Tree) {
        let root_id = tree.root();

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

        let sel_root_id = self.selection.init_root(tree, root, P::COLOR, eval_data);

        let mut line_index = 0;
        let mut budget = MPV as u32;
        while budget > 0 {
            // todo: first go down paths along the selection.parents arena, such that we can
            // just bump up the virtual loss there instead of inserting
            // duplicate parent nodes.
            budget -= self.pick_branch::<P>(line_index, Depth::ROOT, root, tree, sel_root_id);
            line_index += 1;
        }
    }

    // returns virtual loss
    fn pick_branch<P: Perspective>(
        &mut self,
        line_index: usize,
        depth: Depth,
        parent_node_id: NodeId<Evaluated>,
        tree: &mut Tree,
        sel_node_id: SelNodeId,
    ) -> u32 {
        let parent_node = tree.node(parent_node_id);
        let parent_visits = parent_node.visits();
        let visit_threshold = 4; // todo: fine-tune
        let branches = tree.branch_ids(parent_node_id);
        let best_branch_id = branches
            .max_by_key(|&branch_id| {
                let branch = tree.branch(branch_id);
                let child = tree.node(branch.node());
                if child.value().is_proven_loss() && child.visits() >= visit_threshold {
                    self.selector.min_score()
                }
                else {
                    self.selector.score(tree, branch_id, parent_visits)
                }
            })
            .expect("There has to be a branch on an evaluated node");

        self.select_branch::<P>(line_index, depth, best_branch_id, tree, sel_node_id)
    }

    fn select_branch<P: Perspective>(
        &mut self,
        line_index: usize,
        depth: Depth,
        branch: BranchId,
        tree: &mut Tree,
        parent_sel_id: SelNodeId,
    ) -> u32 {
        let (mov, node) = {
            let branch = tree.branch(branch);
            (branch.mov(), branch.node())
        };

        self.position.make_move_for::<P>(mov);
        let depth = depth + 1;
        let turn = self.position.get_turn();

        let used = match tree.node_switch(node) {
            Switch::Branching(node) => {
                self.select_branching(tree, line_index, parent_sel_id, node, depth)
            }
            Switch::Evaluated(node) => {
                let val = NodeView::new(tree, node).value();
                // skip further selection of proven_win/loss nodes.
                if val.is_proven_win() {
                    let eval = Evaluation::Terminal(GameResult::Win { relative_to: !turn });
                    self.select_skip(tree, line_index, parent_sel_id, node, eval, depth)
                }
                else if val.is_proven_loss() {
                    let eval = Evaluation::Terminal(GameResult::Win { relative_to: turn });
                    self.select_skip(tree, line_index, parent_sel_id, node, eval, depth)
                }
                // otherwise continue down the tree
                else {
                    let trace_data = self.evaluator.trace(node, tree, self.position);
                    let child_id = self.selection.append_parent(
                        tree,
                        parent_sel_id,
                        node,
                        turn,
                        1,
                        trace_data,
                    );
                    self.pick_branch::<P::Opponent>(line_index, depth, node, tree, child_id)
                }
            }
            Switch::Leaf(node) => {
                // shortcut leaf selection using twofold repetition
                if depth > Depth::ROOT && self.position.has_twofold_repetition() {
                    let eval = Evaluation::Terminal(GameResult::Draw);
                    self.select_shortcut(tree, line_index, parent_sel_id, node, eval, depth)
                }
                // otherwise select this leaf for evaluation in the next phase
                else {
                    self.select_leaf(tree, line_index, parent_sel_id, node, depth)
                }
            }
            Switch::Terminal(node) => {
                self.select_terminal(tree, line_index, parent_sel_id, node, depth)
            }
        };

        self.position.unmake_move_for::<P>(mov);
        used
    }

    #[inline]
    fn select_skip(
        &mut self,
        tree: &mut Tree,
        line_index: usize,
        parent_id: SelNodeId,
        node: NodeId<Evaluated>,
        eval: Evaluation,
        _depth: Depth,
    ) -> u32 {
        let virtual_loss = 1;
        self.selection.append_leaf(
            tree,
            node,
            line_index,
            virtual_loss,
            PhaseItem::Skip(SelNode {
                node,
                turn: self.position.get_turn(),
                parent: Some(parent_id),
                data: eval,
                virtual_loss,
            }),
        );
        virtual_loss
    }

    #[inline]
    fn select_leaf(
        &mut self,
        tree: &mut Tree,
        line_index: usize,
        parent_sel_id: SelNodeId,
        node: NodeId<Leaf>,
        depth: Depth,
    ) -> u32 {
        let expanded = tree.expand_node(node, self.position, depth);

        match expanded {
            ExpandedSwitch::Terminal(node) => {
                self.select_terminal(tree, line_index, parent_sel_id, node, depth)
            }
            ExpandedSwitch::Branching(node) => {
                self.select_branching(tree, line_index, parent_sel_id, node, depth)
            }
        }
    }

    /// Select a shortcut to a node that can be considered terminal.
    #[inline]
    fn select_shortcut(
        &mut self,
        tree: &mut Tree,
        line_index: usize,
        parent_id: SelNodeId,
        node: NodeId<Leaf>,
        eval: Evaluation,
        _depth: Depth,
    ) -> u32 {
        let virtual_loss = 1;
        self.selection.append_leaf(
            tree,
            node,
            line_index,
            virtual_loss,
            PhaseItem::Shortcut(SelNode {
                node,
                turn: self.position.get_turn(),
                parent: Some(parent_id),
                data: eval,
                virtual_loss,
            }),
        );
        virtual_loss
    }

    #[inline]
    fn select_terminal(
        &mut self,
        tree: &mut Tree,
        line_index: usize,
        parent_id: SelNodeId,
        node: NodeId<Terminal>,
        depth: Depth,
    ) -> u32 {
        let virtual_loss = 1;
        let eval = E::eval_terminal(node, tree, depth, self.position);
        self.selection.append_leaf(
            tree,
            node,
            line_index,
            virtual_loss,
            PhaseItem::Terminal(SelNode {
                node,
                turn: self.position.get_turn(),
                parent: Some(parent_id),
                data: eval,
                virtual_loss,
            }),
        );
        virtual_loss
    }

    #[inline]
    fn select_branching(
        &mut self,
        tree: &mut Tree,
        line_index: usize,
        parent_id: SelNodeId,
        node: NodeId<Branching>,
        depth: Depth,
    ) -> u32 {
        if depth > Depth::MAX {
            // return a loss of 0, such that the top-level while loop will try again and
            // because we applied virtual loss to the parents of this node, it
            // should try a different path next time.
            return 0;
        }

        let trace_data = self.evaluator.trace(node, tree, &mut self.position);
        let virtual_loss = 1;

        self.selection.append_leaf(
            tree,
            node,
            line_index,
            virtual_loss,
            PhaseItem::Batched(SelNode {
                node,
                turn: self.position.get_turn(),
                parent: Some(parent_id),
                data: trace_data,
                virtual_loss,
            }),
        );

        virtual_loss
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
            self.selection.update_leaf(
                i,
                PhaseItem::Evaluated(SelNode {
                    node: batch_item.node,
                    turn: batch_item.turn,
                    parent: batch_item.parent,
                    data: eval,
                    virtual_loss: batch_item.virtual_loss,
                }),
            )
        }
    }

    fn backup_evals(&mut self, tree: &mut Tree) {
        for (leaf, path) in self.selection.drain_leafs() {
            match leaf {
                // ignore unused slots.
                PhaseItem::Unused => {}

                // the evaluator is bad.
                PhaseItem::Batched(_) => panic!(
                    "the evaluator forgot the eval a batched item. this shouldn't happen, log an \
                     error or something"
                ),

                // backup terminals, guesses, etc.
                PhaseItem::Evaluated(x) => self.backprop.backpropagate(tree, path, x),
                PhaseItem::Terminal(x) => self.backprop.backpropagate(tree, path, x),
                PhaseItem::Shortcut(x) => self.backprop.backpropagate(tree, path, x),
                PhaseItem::Skip(x) => self.backprop.backpropagate(tree, path, x),
            }
        }
    }
}
