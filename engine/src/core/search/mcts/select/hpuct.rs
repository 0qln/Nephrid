use std::mem::MaybeUninit;

use crate::core::{
    color::Perspective,
    depth::Depth,
    r#move::Move,
    position::Position,
    search::mcts::{
        node::{BranchId, NodeId, Tree, VisitCount, node_state::Evaluated},
        search::MctsParams,
        select::puct::PuctSelector,
    },
    zobrist,
};

/// A puct that uses heuristics
pub struct HeuristicPuct {
    // the inner selector could also be generic
    puct: super::puct::PuctSelector,
    tt: Box<TranspositionTable<{ 2 << 10 }, TTData>>,
    ss: SearchStack,
}

impl HeuristicPuct {
    pub fn new(cpuct: f32) -> Self {
        Self {
            puct: PuctSelector::new(cpuct),
            ..Default::default()
        }
    }
}

impl Default for HeuristicPuct {
    fn default() -> Self {
        Self {
            puct: Default::default(),
            tt: Default::default(),
            ss: SearchStack::new(),
        }
    }
}

impl super::Selector for HeuristicPuct {
    fn pick_branch<P: Perspective>(
        &mut self,
        tree: &Tree,
        parent_node_id: NodeId<Evaluated>,
        depth: Depth,
        position: &Position,
        params: &impl MctsParams,
    ) -> BranchId {
        let key = position.get_key();

        let tt_entry = self.tt.get(key);
        let tt_best_move = tt_entry.and_then(|data| data.best_move);
        let tt_exploitation = tt_entry.map(|data| data.exploitation);

        let ss_entry = self.ss.get(depth);
        let killer_move = ss_entry.and_then(|entry| entry.killer_move);
        let killer_exploitation = ss_entry.and_then(|entry| entry.killer_exploitation);

        let visit_threshold = params.proven_loss_visit_threshold();

        const MIN: super::Score = super::Score(f32::NEG_INFINITY);
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

                let (score, loit, lora);

                // proven loss penalty
                if child.value().is_proven_loss() && child.visits() >= visit_threshold {
                    score = MIN;
                    loit = MIN;
                    lora = MIN;
                }
                else {
                    // tt-move bonus from
                    let tt_move_bonus = {
                        // use the exploitation score from the tt best_move as guidance in the
                        // exploration factor.
                        if tt_best_move == Some(mov) {
                            tt_exploitation.unwrap().0 * params.tt_best_move()
                        }
                        else {
                            1.
                        }
                    };

                    // killer move bonus for barely visited nodes
                    let killer_move_bonus = {
                        // if a quiet move from a sibling branch proved to be of high exploitation
                        // after some searching, consider that move here aswell.
                        if killer_move == Some(mov) && child.visits() <= VisitCount(2) {
                            killer_exploitation.unwrap().0 * params.killer_exploitation()
                        }
                        else {
                            0.
                        }
                    };

                    loit = self.exploitation(tree, branch_id, parent_node_id);
                    lora = self.exploration(tree, branch_id, parent_node_id);
                    score = (loit + killer_move_bonus) + (lora * tt_move_bonus);

                    debug_assert!(
                        !score.0.is_nan(),
                        "score is NAN! (tt_move_bonus={tt_move_bonus}, killer_move_bonus={killer_move_bonus}, exploration={lora}, \
                         exploitation={loit})"
                    );
                }

                if score >= curr_score {
                    curr_score = score;
                    curr_exploitation.write(loit);
                    curr_exploration.write(lora);
                    curr_move.write(mov);
                    curr_branch_id.write(branch_id);
                }
            }

            // SAFETY: a first pass is guaranteed because parent_node_id is evaluated and
            // thus has to have at least one branch.
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
        self.tt.insert(TTData {
            key,
            best_move: Some(best_move),
            exploitation: best_exploitation,
        });

        // update ss.killer
        if !best_move.get_flag().is_capture() && killer_exploitation.is_none_or(|e| e < best_exploitation) {
            let e = self.ss.entry(depth);
            e.killer_move = Some(best_move);
            e.killer_exploitation = Some(best_exploitation);
        }

        best_branch_id
    }

    fn exploitation(&self, tree: &Tree, branch_id: BranchId, parent_id: NodeId<Evaluated>) -> super::Score {
        self.puct.exploitation(tree, branch_id, parent_id)
    }

    fn exploration(&self, tree: &Tree, branch_id: BranchId, parent_id: NodeId<Evaluated>) -> super::Score {
        self.puct.exploration(tree, branch_id, parent_id)
    }
}

pub struct TranspositionTable<const ENTRIES: usize, Data> {
    entries: [Option<Data>; ENTRIES],
}

impl<const ENTRIES: usize, Data> Default for TranspositionTable<ENTRIES, Data> {
    fn default() -> Self {
        const fn const_none<T>() -> Option<T> { None }
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
    pub fn insert(&mut self, data: Data) {
        let key = data.key();
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
    exploitation: super::Score,
}

impl ZKey for TTData {
    fn key(&self) -> zobrist::Hash { self.key }
}

#[derive(Default)]
pub struct SearchStack {
    entries: Vec<SearchEntry>,
}

impl SearchStack {
    pub fn new() -> Self { Self { entries: Vec::new() } }

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
    killer_exploitation: Option<super::Score>,
}
