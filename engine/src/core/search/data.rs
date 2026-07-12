use std::{hint::unreachable_unchecked, marker::PhantomData};

use uom::si::{information::byte, u64::Information};

use crate::core::{
    color::{Color, Perspective, colors, perspectives},
    coordinates::{Square, squares},
    depth::Depth,
    r#move::Move,
    piece::{PieceType, piece_type},
    search::{id, ordering::MoveScore, score::AnyScore},
    zobrist,
};

pub const trait TTKey {
    fn key(&self) -> zobrist::Hash;
}

pub const trait TTMove {
    fn mov(&self) -> Move;
}

pub const trait TTDepth {
    fn depth(&self) -> Depth;
}

pub const trait TTBound {
    fn bound(&self) -> id::Bound;
}

pub const trait TTScore {
    fn score(&self) -> AnyScore;
}

pub const trait TTStaticEval {
    fn static_eval(&self) -> AnyScore;
    fn static_eval_mut(&mut self) -> &mut AnyScore;
}

pub trait ReplacementStrategy {
    type Data;
    fn should_replace(old: &Self::Data, new: &Self::Data) -> bool;
}

pub struct TranspositionTable<Data, Strat> {
    entries: Box<[Data]>,
    strat: PhantomData<Strat>,
}

impl<Data: Clone + const Default, S> TranspositionTable<Data, S> {
    pub fn new(size: usize) -> Self {
        const fn const_default<T: const Default>() -> T { T::default() }
        Self {
            entries: (vec![const { const_default() }; size]).into_boxed_slice(),
            strat: PhantomData,
        }
    }

    pub fn new_of_size(size: Information) -> Self {
        let bytes = size.get::<byte>() as usize;
        let entry_size = std::mem::size_of::<Option<Data>>();
        let num_entries = (bytes / entry_size).max(1);
        Self::new(num_entries)
    }

    pub fn clear(&mut self) { self.entries.fill(Data::default()); }
}

impl<Data, S> TranspositionTable<Data, S> {
    /// Number of entries
    #[inline]
    pub fn size(&self) -> usize { self.entries.len() }
}

impl<Data: TTKey, S> TranspositionTable<Data, S> {
    /// Get data for the given key.
    #[inline]
    pub fn get(&self, key: zobrist::Hash) -> Option<&Data> {
        let idx = key.index(self.size());
        let data = &self.entries[idx];
        if data.key() == key { Some(data) } else { None }
    }

    /// Get data for the given key.
    #[inline]
    pub fn get_mut(&mut self, key: zobrist::Hash) -> Option<&mut Data> {
        let idx = key.index(self.size());
        let data = &mut self.entries[idx];
        if data.key() == key { Some(data) } else { None }
    }
}

impl<Data: TTKey, Strat: ReplacementStrategy<Data = Data>> TranspositionTable<Data, Strat> {
    pub fn try_insert<T: Into<Data>>(&mut self, t: T) {
        let data = t.into();

        let key = data.key();
        let idx = key.index(self.size());
        let old_data = &self.entries[idx];

        if Strat::should_replace(old_data, &data) {
            self.entries[idx] = data;
        }
    }
}

#[derive(Clone, Copy)]
pub struct PieceHistory {
    /// For each piece type to it's destination square.
    scores: [[MoveScore; squares::N_VARIANTS]; piece_type::N_VARIANTS - 1],
}

impl const Default for PieceHistory {
    fn default() -> Self { Self::new() }
}

impl PieceHistory {
    pub const fn new() -> Self {
        Self {
            scores: [[0; squares::N_VARIANTS]; piece_type::N_VARIANTS - 1],
        }
    }
}

#[derive(Default, Clone)]
pub struct PieceHistories {
    histories: [PieceHistory; colors::N_VARIANTS],
}

impl PieceHistories {
    pub const fn new() -> Self {
        Self {
            histories: [PieceHistory::new(); colors::N_VARIANTS],
        }
    }

    pub const fn clear(&mut self) { self.histories = [PieceHistory::new(); colors::N_VARIANTS]; }

    pub const fn get(&self, c: Color, pt: PieceType, sq: Square) -> MoveScore {
        match c {
            colors::WHITE => self.get_for::<perspectives::White>(pt, sq),
            colors::BLACK => self.get_for::<perspectives::Black>(pt, sq),
            _ => unsafe { unreachable_unchecked() },
        }
    }

    pub const fn get_for<P: Perspective>(&self, pt: PieceType, sq: Square) -> MoveScore {
        debug_assert!(pt != piece_type::NONE, "Cannot get history for NONE piece type.");
        let c = P::COLOR.v() as usize;
        let pt = pt.v() as usize - 1;
        let sq = sq.v() as usize;

        unsafe { *self.histories.get_unchecked(c).scores.get_unchecked(pt).get_unchecked(sq) }
    }

    pub const fn update(&mut self, c: Color, pt: PieceType, sq: Square, val: MoveScore) {
        match c {
            colors::WHITE => self.update_for::<perspectives::White>(pt, sq, val),
            colors::BLACK => self.update_for::<perspectives::Black>(pt, sq, val),
            _ => unsafe { unreachable_unchecked() },
        }
    }

    pub const fn update_for<P: Perspective>(&mut self, pt: PieceType, sq: Square, val: MoveScore) {
        debug_assert!(pt != piece_type::NONE, "Cannot update history for NONE piece type.");
        let c = P::COLOR.v() as usize;
        let pt = pt.v() as usize - 1;
        let sq = sq.v() as usize;

        unsafe {
            *self.histories.get_unchecked_mut(c).scores.get_unchecked_mut(pt).get_unchecked_mut(sq) += val;
        }
    }
}
