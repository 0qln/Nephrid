use std::{
    collections::{HashMap, hash_map::Entry},
    mem::MaybeUninit,
};

use itertools::Itertools;
use rustc_hash::FxBuildHasher;

use crate::core::{
    Position,
    color::{Perspective, colors, perspectives},
    depth::Depth,
    r#move::Move,
    search::mcts::{
        back::{self},
        eval::{Evaluation, Evaluator, Guess, eval_terminal},
        node::{BranchId, NodeId, NodeView, RtNodeId, Tree, VisitCount, node_state::*},
        noise::Noiser,
        select::{self, Selector},
    },
    turn::Turn,
    zobrist,
};

use super::eval::GameResult;

#[cfg(test)]
pub mod test;

// todo:
// the weight parameter was removed in the process of implementing virtual loss.
// logically the effect remains the same: instead of using the remaining budget
// at skip/terminal selections, we instead select that node multiple times when
// the outer while loop hits that nodes multiple times. this however causes more
// backpropagation passes. mayube the weight concept can be reintroduced
// somehow?
//
// todo:
// also, i believe this currently evaluates the same node multiple times on low
// dephts instead of overflowing remaining budget to other branches. this should
// be investigated.
//
// todo:
// this pipeline model was optimized for gpu infrerence...
// maybe it's better to have a work-stealing pipeline for cpu inrference (e.g.
// hce-eval) ?
//
// todo:
// instead of picking going each line down, while searching for the max puct
// score each time, we could calculate how much virtual loss the second-to-best
// puct score would need to surpass the best puct score. then we could use the
// prior budgeting technique. that would solve the problem of duplicate parents
// and duplicate evaluations.
//
// todo:
// first go down paths along the selection.parents arena, such that we can
// just bump up the virtual loss there instead of inserting
// duplicate parent nodes.
//
// todo:
// the HashMap can probably be replaced by some custom stack info clever
// indexing scheme...
//
// todo:
// when switch to iterative, replace the `iterations < BATCH * 2` check. we
// will keep a frontier of unexplorable nodes anyway (probably right?), and just
// check if that is empty instead of doing the safestop above.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParentNodeId(pub usize);

pub struct ParentItem<T> {
    parent: Option<ParentNodeId>,
    pub trace: T,
    sel_data: SelData,
    node: NodeId<Evaluated>,
}

#[derive(Clone, Copy)]
pub struct SelData {
    pub turn: Turn,
}

pub struct BatchItem<T> {
    pub parent: Option<ParentNodeId>,
    pub trace: T,
    pub node: NodeId<Branching>,
    pub sel_data: SelData,
    pub weight: f32,
}

pub struct EvalItem {
    parent: ParentNodeId,
    eval: Guess,
    node: NodeId<Branching>,
    sel_data: SelData,
    weight: f32,
}

pub struct TerminalItem {
    parent: ParentNodeId,
    pub eval: GameResult,
    pub node: NodeId<Terminal>,
    pub sel_data: SelData,
}

pub struct ShortcutItem {
    parent: ParentNodeId,
    node: NodeId<Leaf>,
}

pub struct SkipItem {
    parent: ParentNodeId,
    eval: Evaluation,
    node: NodeId<Evaluated>,
    sel_data: SelData,
}

pub struct Selection<T> {
    pub terminals: Vec<TerminalItem>,
    pub evaluations: Vec<EvalItem>,
    pub shortcuts: Vec<ShortcutItem>,
    pub skips: Vec<SkipItem>,
    pub batched: Vec<BatchItem<T>>,
    // todo: isntead of hashing, just do index=(id % BATCH_SIZE) and skip to the next free gap?
    pub batched_map: HashMap<NodeId<Branching>, usize, FxBuildHasher>,
    pub parents: Vec<ParentItem<T>>,
    pub virtual_loss: Vec<(RtNodeId, u32)>,
}

impl<T> Default for Selection<T> {
    fn default() -> Self {
        Self {
            terminals: Default::default(),
            evaluations: Default::default(),
            shortcuts: Default::default(),
            skips: Default::default(),
            batched: Default::default(),
            batched_map: Default::default(),
            parents: Default::default(),
            virtual_loss: Default::default(),
        }
    }
}

impl<T> Selection<T> {
    /// Clear the arena and selection.
    pub fn clear(&mut self) {
        self.terminals.clear();
        self.evaluations.clear();
        self.shortcuts.clear();
        self.skips.clear();
        self.batched.clear();
        self.batched_map.clear();
        self.parents.clear();
        self.virtual_loss.clear();
    }

    /// Allocates a new ParentItem in the arena and attaches it to the parent.
    pub fn push_parent(&mut self, item: ParentItem<T>) -> ParentNodeId {
        let id = ParentNodeId(self.parents.len());
        self.parents.push(item);
        id
    }

    pub fn iter_path_up(&self, leaf: Option<ParentNodeId>) -> impl Iterator<Item = &ParentItem<T>> {
        Self::iter_up(leaf, &self.parents)
    }

    fn iter_up(
        current: Option<ParentNodeId>,
        arena: &[ParentItem<T>],
    ) -> impl Iterator<Item = &ParentItem<T>> {
        pub struct IterUp<'a, T> {
            arena: &'a [ParentItem<T>],
            current: Option<ParentNodeId>,
        }

        pub struct IterItem<'a, T>(&'a ParentItem<T>);

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

    fn revert_virtual_loss(&self, tree: &mut Tree) {
        for &(node, loss) in self.virtual_loss.iter() {
            tree.revert_virtual_loss(node, loss);
        }
    }

    fn apply_virtual_loss(&mut self, tree: &mut Tree, node: RtNodeId, loss: u32) {
        tree.apply_virtual_loss(node, loss);
        self.virtual_loss.push((node, loss));
    }

    /// Returns the root parent id (the first parent in the arena, if any).
    pub fn root_id(&self) -> Option<ParentNodeId> {
        if self.parents.is_empty() {
            None
        }
        else {
            Some(ParentNodeId(0))
        }
    }
}

/// # Tree searcher
pub struct TreeSearcher<'pos, const BATCH_SIZE: usize, E: Evaluator, S: Selector, N: Noiser> {
    position: &'pos mut Position,
    selector: S,
    evaluator: E,
    noiser: N,
    selection: Selection<E::TraceData>,
    tt: Box<TranspositionTable<{ 2 << 10 }, TTData>>,
    ss: SearchStack,
}

impl<'pos, const BATCH: usize, E: Evaluator, S: Selector, N: Noiser>
    TreeSearcher<'pos, BATCH, E, S, N>
{
    pub fn new(position: &'pos mut Position, selector: S, evaluator: E, noiser: N) -> Self {
        Self {
            position,
            selector,
            evaluator,
            noiser,
            selection: Default::default(),
            tt: Default::default(),
            ss: SearchStack::new(),
        }
    }

    /// Prepares the root node for search (expand, evaluate, apply noise).
    pub fn init_root(&mut self, tree: &mut Tree) {
        loop {
            match tree.node_switch(tree.root()) {
                Switch::Leaf(node) => {
                    let _ = tree.expand_node(node, self.position, Depth::ROOT);
                }
                Switch::Branching(node) => {
                    // Evaluate the root as a batch of size 1
                    let turn = self.position.get_turn();
                    let batch_item = BatchItem {
                        parent: None, // dummy; will be replaced
                        trace: self.evaluator.trace(node, tree, self.position),
                        node,
                        sel_data: SelData { turn },
                        weight: 1.,
                    };
                    let evals: Vec<Guess> = self
                        .evaluator
                        .eval_batch(tree, &self.selection, &[&batch_item])
                        .collect();
                    back::update_branching(tree, node, turn, &evals[0], 1.);
                }
                Switch::Evaluated(node) => {
                    if let Err(err) = self.noiser.apply_noise(node, tree) {
                        eprintln!("Failed to apply noise to root node: {err}");
                    }
                    break;
                }
                Switch::Terminal(_) => break,
            }
        }
    }

    pub fn grow(&mut self, tree: &mut Tree) {
        self.selection.clear();
        self.select_lines(tree);
        self.eval_batched(tree);
        self.revert_virtual_loss(tree);
        self.backup_evals(tree);
    }

    fn revert_virtual_loss(&mut self, tree: &mut Tree) {
        self.selection.revert_virtual_loss(tree)
    }

    fn select_lines(&mut self, tree: &mut Tree) {
        match self.position.get_turn() {
            colors::WHITE => self.select_lines_for::<perspectives::White>(tree),
            colors::BLACK => self.select_lines_for::<perspectives::Black>(tree),
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    fn select_lines_for<P: Perspective>(&mut self, tree: &mut Tree) {
        let root_id = tree.root();
        let root = match tree.node_switch(root_id) {
            Switch::Evaluated(n) => n,
            Switch::Terminal(_) => panic!("Root is terminal – search abandoned?"),
            _ => panic!("Root must be evaluated before selection"),
        };

        // Create root parent item
        let trace_data = self.evaluator.trace(root, tree, self.position);
        let root_sel_id = self.selection.push_parent(ParentItem {
            parent: None,
            trace: trace_data,
            sel_data: SelData { turn: P::COLOR },
            node: root,
        });

        // todo
        let mut iterations = 0;
        let num_batchables = tree.count_nodes(&|node, _| node.state() == NodeState::Leaf, BATCH);
        while self.selection.batched.len() < num_batchables && iterations < BATCH * 2 {
            self.pick_branch::<P>(Depth::ROOT, root, tree, root_sel_id);
            iterations += 1;
        }

        // // We'll keep a worklist of nodes to expand (evaluated nodes that are
        // not // terminal) let mut frontier = vec![root_sel_id];

        // while self.selection.batched.len() < BATCH && !frontier.is_empty() {
        //     let mut next_frontier = Vec::new();
        //     for &parent_id in &frontier {
        //         self.expand_parent::<P>(tree, parent_id, &mut next_frontier);
        // // Batch became full – we can         if
        // self.selection.batched.len() >= BATCH {             // stop
        // early             break;
        //         }
        //     }
        //     frontier = next_frontier;
        // }
    }

    // fn expand_parent<P: Perspective>(
    //     &mut self,
    //     tree: &mut Tree,
    //     parent_sel_id: ParentNodeId,
    // ) -> bool {
    //     match tree.node_switch(child_node) {
    //         Switch::Evaluated(node) => {
    //             if value.is_proven_win() {
    //                 // ...
    //             }
    //             else if value.is_proven_loss() {
    //                 // ...
    //             }
    //             else {
    //                 // Continue deeper
    //                 let child_parent = self.selection.push_parent(ParentItem {
    //                     // ...
    //                 });
    //                 next_frontier.push(child_parent);
    //             }
    //         }
    //         Switch::Branching(node) => {
    //             // ...
    //         }
    //         Switch::Leaf(node) => {
    //             // Expand leaf
    //             // ...
    //         }
    //         Switch::Terminal(node) => {
    //             // ...
    //         }
    //     }
    // }

    fn pick_branch<P: Perspective>(
        &mut self,
        depth: Depth,
        parent_node_id: NodeId<Evaluated>,
        tree: &mut Tree,
        sel_node_id: ParentNodeId,
    ) {
        let key = self.position.get_key();

        let tt_entry = self.tt.get(key);
        let tt_best_move = tt_entry.and_then(|data| data.best_move);
        let tt_exploitation = tt_entry.map(|data| data.exploitation);

        let ss_entry = self.ss.get(depth);
        let killer_move = ss_entry.and_then(|entry| entry.killer_move);
        let killer_exploitation = ss_entry.and_then(|entry| entry.killer_exploitation);

        let visit_threshold = VisitCount(4); // todo: fine-tune

        let sel = &self.selector;
        const MIN: select::Score = select::Score(f32::NEG_INFINITY);
        let (best_branch_id, best_move, best_exploitation, _) = {
            // best score etc.
            let mut curr_score = MIN;
            let mut curr_exploitation = MaybeUninit::uninit();
            let mut curr_exploration = MaybeUninit::uninit();
            let mut curr_move = MaybeUninit::uninit();
            let mut curr_branch_id = MaybeUninit::uninit();

            for branch_id in tree.branch_ids(parent_node_id) {
                let branch = tree.branch(branch_id);
                let child = tree.node(branch.node());
                let mov = branch.mov();

                let (score, exploitation, exploration);

                // proven loss penalty
                if child.value().is_proven_loss() && child.visits() >= visit_threshold {
                    score = MIN;
                    exploitation = MIN;
                    exploration = MIN;
                }
                else {
                    let exploration_tt_move = {
                        // use the exploitation score from the tt best_move as guidance in the
                        // exploration factor.
                        if tt_best_move == Some(mov) {
                            tt_exploitation.unwrap().0 * 2.0 // todo: make this tunable
                        }
                        else {
                            1.
                        }
                    };

                    let killer_move = {
                        // if a quiete move from a sibling branch proved to be of high exploitation
                        // after some searching, consider that move here aswell.
                        if killer_move == Some(mov) && child.visits() <= VisitCount(2) {
                            killer_exploitation.unwrap().0 * 1.0 // todo: make this tunable
                        }
                        else {
                            0.
                        }
                    };

                    exploration = sel.exploration(tree, branch_id, parent_node_id);
                    exploitation = sel.exploitation(tree, branch_id, parent_node_id);
                    score = (exploitation + killer_move) + (exploration * exploration_tt_move);
                }

                if score >= curr_score {
                    curr_score = score;
                    curr_exploitation.write(exploitation);
                    curr_exploration.write(exploration);
                    curr_move.write(mov);
                    curr_branch_id.write(branch_id);
                }
            }

            // SAFETY: a first pass is guruanteed because parent_node_id is evaluated and
            // thus has to have atleast one branch.
            unsafe {
                (
                    curr_branch_id.assume_init(),
                    curr_move.assume_init(),
                    curr_exploitation.assume_init(),
                    curr_exploration.assume_init(),
                )
            }
        };

        // update tt
        self.tt.insert(
            key,
            TTData {
                key,
                best_move: Some(best_move),
                exploitation: best_exploitation,
            },
        );

        // update ss.killer
        if !best_move.get_flag().is_capture()
            && killer_exploitation.is_none_or(|e| e < best_exploitation)
        {
            let e = self.ss.entry(depth);
            e.killer_move = Some(best_move);
            e.killer_exploitation = Some(best_exploitation);
        }

        // todo: just return the branch id instead of recursing
        self.select_branch::<P>(depth, best_branch_id, tree, sel_node_id)
    }

    fn select_branch<P: Perspective>(
        &mut self,
        depth: Depth,
        branch: BranchId,
        tree: &mut Tree,
        parent_sel_id: ParentNodeId,
    ) {
        let (mov, node) = {
            let branch = tree.branch(branch);
            (branch.mov(), branch.node())
        };

        // todo: make this a debug assertion
        // let depth = Depth::new(self.selection.iter_path_up(parent_sel_id).count() as
        // u8);

        self.position.make_move_for::<P>(mov);
        let depth = depth + 1;
        let turn = self.position.get_turn();

        match tree.node_switch(node) {
            Switch::Branching(node) => self.select_branching(tree, parent_sel_id, node, depth),
            Switch::Evaluated(node) => {
                let val = NodeView::new(tree, node).value();
                // skip further selection of proven_win/loss nodes.
                if val.is_proven_win() {
                    let eval = Evaluation::Terminal(GameResult::Win { relative_to: !turn });
                    self.select_skip(tree, parent_sel_id, node, eval, depth)
                }
                else if val.is_proven_loss() {
                    let eval = Evaluation::Terminal(GameResult::Win { relative_to: turn });
                    self.select_skip(tree, parent_sel_id, node, eval, depth)
                }
                // otherwise continue down the tree
                else {
                    self.selection.apply_virtual_loss(
                        tree,
                        node.down_cast(),
                        self.selector.virtual_loss(),
                    );
                    let new_parent_id = self.selection.push_parent(ParentItem {
                        parent: Some(parent_sel_id),
                        trace: self.evaluator.trace(node, tree, self.position),
                        sel_data: SelData { turn },
                        node,
                    });

                    self.pick_branch::<P::Opponent>(depth, node, tree, new_parent_id)
                }
            }
            Switch::Leaf(node) => {
                // shortcut leaf selection using twofold repetition
                if depth > Depth::ROOT && self.position.has_twofold_repetition() {
                    self.select_shortcut(tree, parent_sel_id, node)
                }
                // otherwise select this leaf for evaluation in the next phase
                else {
                    self.select_leaf(tree, parent_sel_id, node, depth)
                }
            }
            Switch::Terminal(node) => self.select_terminal(tree, parent_sel_id, node, depth),
        };

        self.position.unmake_move_for::<P>(mov);
    }

    #[inline]
    fn select_skip(
        &mut self,
        tree: &mut Tree,
        parent_id: ParentNodeId,
        node: NodeId<Evaluated>,
        eval: Evaluation,
        _depth: Depth,
    ) {
        self.selection
            .apply_virtual_loss(tree, node.down_cast(), self.selector.virtual_loss());
        self.selection.skips.push(SkipItem {
            node,
            sel_data: SelData { turn: self.position.get_turn() },
            parent: parent_id,
            eval,
        });
    }

    #[inline]
    fn select_leaf(
        &mut self,
        tree: &mut Tree,
        parent_sel_id: ParentNodeId,
        node: NodeId<Leaf>,
        depth: Depth,
    ) {
        let expanded = tree.expand_node(node, self.position, depth);

        match expanded {
            ExpandedSwitch::Terminal(node) => {
                self.select_terminal(tree, parent_sel_id, node, depth)
            }
            ExpandedSwitch::Branching(node) => {
                self.select_branching(tree, parent_sel_id, node, depth)
            }
        }
    }

    /// Select a shortcut to a node that can be considered terminal.
    #[inline]
    fn select_shortcut(&mut self, tree: &mut Tree, parent: ParentNodeId, node: NodeId<Leaf>) {
        self.selection
            .apply_virtual_loss(tree, node.down_cast(), self.selector.virtual_loss());
        self.selection.shortcuts.push(ShortcutItem { parent, node })
    }

    #[inline]
    fn select_terminal(
        &mut self,
        tree: &mut Tree,
        parent: ParentNodeId,
        node: NodeId<Terminal>,
        depth: Depth,
    ) {
        self.selection
            .apply_virtual_loss(tree, node.down_cast(), self.selector.virtual_loss());
        self.selection.terminals.push(TerminalItem {
            parent,
            eval: eval_terminal(node, tree, depth, self.position),
            node,
            sel_data: SelData { turn: self.position.get_turn() },
        })
    }

    #[inline]
    fn select_branching(
        &mut self,
        tree: &mut Tree,
        parent: ParentNodeId,
        node: NodeId<Branching>,
        depth: Depth,
    ) {
        self.selection
            .apply_virtual_loss(tree, node.down_cast(), self.selector.virtual_loss());

        if depth > Depth::MAX {
            // return, such that the top-level while loop will try again and
            // because we applied virtual loss, it should try a different path next time.
            return;
        }

        match self.selection.batched_map.entry(node) {
            Entry::Occupied(entry) => self.selection.batched[*entry.get()].weight += 1.,
            Entry::Vacant(vacant_entry) => {
                let idx = self.selection.batched.len();
                self.selection.batched.push(BatchItem {
                    parent: Some(parent),
                    node,
                    trace: self.evaluator.trace(node, tree, self.position),
                    sel_data: SelData { turn: self.position.get_turn() },
                    weight: 1.,
                });
                vacant_entry.insert(idx);
            }
        };
    }

    fn eval_batched(&mut self, tree: &Tree) {
        // todo: clean this up
        let batch = self.selection.batched.iter().collect_vec();

        let evals: Vec<Guess> = self
            .evaluator
            .eval_batch(tree, &self.selection, &batch)
            .collect();

        for (item, eval) in batch.iter().zip(evals) {
            self.selection.evaluations.push(EvalItem {
                parent: item.parent.expect(
                    "Only the root has not parent and we don't grow if there the root is not \
                     initialized. So this shouldn't ever be None. todo: this is a bad solution. \
                     Refactor such that this case is impossible to hit.",
                ),
                node: item.node,
                sel_data: item.sel_data,
                eval,
                weight: item.weight,
            })
        }
    }

    fn backup_evals(&mut self, tree: &mut Tree) {
        for &TerminalItem { parent, eval, node, sel_data } in &self.selection.terminals {
            back::update_terminal(tree, node, sel_data.turn, eval, 1.0);
            let path = self
                .selection
                .iter_path_up(Some(parent))
                .map(|p| (p.node, p.sel_data.turn));
            back::backpropagate_up(tree, path, &eval, 1.0);
        }

        for &EvalItem {
            parent,
            ref eval,
            weight,
            node,
            sel_data,
        } in &self.selection.evaluations
        {
            back::update_branching(tree, node, sel_data.turn, eval, weight);
            let path = self
                .selection
                .iter_path_up(Some(parent))
                .map(|p| (p.node, p.sel_data.turn));
            back::backpropagate_up(tree, path, eval, 1.0);
        }

        for &ShortcutItem { parent, node, .. } in &self.selection.shortcuts {
            let eval = back::update_shortcut(tree, node, 1.0);
            let path = self
                .selection
                .iter_path_up(Some(parent))
                .map(|p| (p.node, p.sel_data.turn));
            back::backpropagate_up(tree, path, &eval, 1.0);
        }

        for &SkipItem { parent, ref eval, node, sel_data } in &self.selection.skips {
            back::update_skip(tree, node, sel_data.turn, eval, 1.0);
            let path = self
                .selection
                .iter_path_up(Some(parent))
                .map(|p| (p.node, p.sel_data.turn));
            back::backpropagate_up(tree, path, eval, 1.0);
        }
    }
}

pub struct TranspositionTable<const ENTRIES: usize, Data> {
    entries: [Option<Data>; ENTRIES],
}

impl<const ENTRIES: usize, Data> Default for TranspositionTable<ENTRIES, Data> {
    fn default() -> Self {
        const fn const_none<T>() -> Option<T> {
            None
        }
        Self {
            entries: [const { const_none() }; ENTRIES],
        }
    }
}

impl<const ENTRIES: usize, Data: ZKey> TranspositionTable<ENTRIES, Data> {
    /// Get data for the given key.
    #[inline]
    pub fn get(&self, key: zobrist::Hash) -> Option<&Data> {
        let idx = key.index(ENTRIES);
        let entry = self.entries[idx].as_ref();
        if let Some(data) = entry
            && data.key() == key
        {
            Some(data)
        }
        else {
            None
        }
    }

    /// Insert and overwrite in any case.
    #[inline]
    pub fn insert(&mut self, key: zobrist::Hash, data: Data) {
        let idx = key.index(ENTRIES);
        self.entries[idx] = Some(data);
    }

    /// Remove the entry for the given key, if it exists.
    #[inline]
    pub fn remove(&mut self, key: zobrist::Hash) {
        let idx = key.index(ENTRIES);

        // if there is no Some at the idx, there is no entry for this key anyhow.
        if let Some(data) = &self.entries[idx]
            // if the key doesn't match, there wasn't an entry for this key anyhow.
            && data.key() == key
        {
            self.entries[idx] = None;
        }
    }
}

pub trait ZKey {
    fn key(&self) -> zobrist::Hash;
}

pub struct TTData {
    key: zobrist::Hash,
    best_move: Option<Move>,
    exploitation: select::Score,
}

impl ZKey for TTData {
    fn key(&self) -> zobrist::Hash {
        self.key
    }
}

#[derive(Default)]
pub struct SearchStack {
    entries: Vec<SearchEntry>,
}

impl SearchStack {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn get(&self, depth: Depth) -> Option<&SearchEntry> {
        let idx = depth.v() as usize;
        self.entries.get(idx)
    }

    pub fn entry(&mut self, depth: Depth) -> &mut SearchEntry {
        let idx = depth.v() as usize;
        if idx >= self.entries.len() {
            self.entries.resize(idx + 1, SearchEntry::default());
        }
        &mut self.entries[idx]
    }
}

#[derive(Default, Clone)]
pub struct SearchEntry {
    killer_move: Option<Move>,
    killer_exploitation: Option<select::Score>,
}
