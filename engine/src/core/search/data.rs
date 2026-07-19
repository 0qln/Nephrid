use std::{
    hint::{assert_unchecked, unreachable_unchecked},
    iter,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

use uom::si::{information::byte, u64::Information};

use crate::{
    core::{
        color::{Color, Perspective, colors, perspectives},
        coordinates::{Square, squares},
        depth::Depth,
        r#move::Move,
        piece::{PieceType, piece_type},
        search::{id, ordering::MoveScore, score::AnyScore},
        zobrist,
    },
    misc::List,
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

pub struct SearchStack<T> {
    entries: Vec<T>,
}

impl<T: Clone + Default> Default for SearchStack<T> {
    #[inline(always)]
    fn default() -> Self { Self::new() }
}

impl<T: Clone + Default> SearchStack<T> {
    pub const CAPACITY: usize = Depth::MAX.v() as usize + 2;

    #[inline(always)]
    pub fn new() -> Self {
        Self {
            entries: vec![T::default(); Self::CAPACITY],
        }
    }
}

impl<T: Clone + Default> From<Vec<T>> for SearchStack<T> {
    #[inline(always)]
    fn from(mut vec: Vec<T>) -> Self {
        vec.extend(iter::repeat_n(T::default(), Self::CAPACITY - vec.len()));
        Self { entries: vec }
    }
}

impl<T> SearchStack<T> {
    #[inline(always)]
    pub fn propagate(&mut self, old: Depth, new: Depth, mut f: impl FnMut(&T, &mut T)) {
        let old_idx = old.index();
        let new_idx = new.index();
        unsafe {
            let [parent, child] = self.entries.get_disjoint_unchecked_mut([old_idx, new_idx]);
            f(parent, child);
        }
    }

    #[inline(always)]
    pub fn propagate_forward(&mut self, old: Depth, f: impl FnMut(&T, &mut T)) {
        debug_assert!(old < Depth::MAX, "Cannot propagate forward from the last ply.");

        let old_idx = old.v() as usize;
        let new_idx = old_idx + 1;
        self.propagate(old, Depth::new(new_idx as u8), f);
    }

    /// # Safety: see slice::get_disjoint_unchecked_mut
    pub unsafe fn get_disjoint_unchecked_mut<const N: usize>(&mut self, indices: [Depth; N]) -> [&mut T; N] {
        // NB: This implementation is written as it is because any variation of
        // `indices.map(|i| self.get_unchecked_mut(i))` would make miri unhappy,
        // or generate worse code otherwise. This is also why we need to go
        // through a raw pointer here.
        let slice: *mut [T] = self.entries.as_mut_slice();
        let mut arr: MaybeUninit<[&mut T; N]> = MaybeUninit::uninit();
        let arr_ptr = arr.as_mut_ptr();

        // SAFETY: We expect `indices` to contain disjunct values that are
        // in bounds of `self`.
        unsafe {
            for i in 0..N {
                let idx = indices.get_unchecked(i).clone();
                arr_ptr.cast::<&mut T>().add(i).write(&mut *slice.get_unchecked_mut(idx.index()));
            }
            arr.assume_init()
        }
    }

    pub fn get_mut(&mut self, ply: Depth) -> &mut T {
        let idx = ply.index();

        // Safety: entries is atleast Self::CAPACITY, which is gt Depth::MAX
        unsafe { self.entries.get_unchecked_mut(idx) }
    }

    pub fn get(&self, ply: Depth) -> &T {
        let idx = ply.index();

        // Safety: entries is atleast Self::CAPACITY, which is gt Depth::MAX
        unsafe { self.entries.get_unchecked(idx) }
    }
}

/// A Ring Buffer Set of size `N`.
///
/// Maintains up to `N` unique elements. When an element is pushed:
/// - If it already exists, it is promoted to the front (index 0), and the
///   elements before its old position are shifted down.
/// - If it is new, all elements are shifted down, evicting the oldest.
///
/// # Examples
///
/// ```
/// # use engine::core::search::data::RbSet;
///
/// let mut killers = RbSet::<i32, 3>::new();
///
/// killers.push(10);
/// killers.push(20);
/// killers.push(30);
/// assert_eq!(killers, RbSet::from([30, 20, 10]));
///
/// // Pushing an existing element moves it to the front (Promotes it)
/// killers.push(20);
/// assert_eq!(killers, RbSet::from([20, 30, 10]));
///
/// // Pushing a new element evicts the oldest (10 drops off)
/// killers.push(40);
/// assert_eq!(killers, RbSet::from([40, 20, 30]));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RbSet<T, const N: usize> {
    items: [T; N],
}

const impl<T: const Default, const N: usize> Default for RbSet<T, N> {
    #[inline]
    fn default() -> Self {
        Self {
            items: [const { T::default() }; N],
        }
    }
}

const impl<T: const Default + Copy + Eq, const N: usize> From<[T; N]> for RbSet<T, N> {
    fn from(items: [T; N]) -> Self { Self { items } }
}

impl<T: const Default + Copy + Eq, const N: usize> RbSet<T, N> {
    #[inline(always)]
    pub const fn new() -> Self { Self { items: [T::default(); N] } }

    #[inline(always)]
    pub const fn len() -> usize { N }

    // todo: make sure this is unrolled for our N=2/3
    // todo: this is O(n) but i don't  think this matters for our n=2 lmao
    #[inline(always)]
    pub fn push(&mut self, item: T) {
        let pos = self.position(&item).unwrap_or(N - 1);

        // Safety: pos is either the index of the item in the set, or the last index if
        // the item is not
        unsafe {
            assert_unchecked(pos < N);
        }

        for i in (1..=pos).rev() {
            self.items[i] = self.items[i - 1];
        }

        self.items[0] = item;
    }

    #[inline(always)]
    pub fn position(&self, item: &T) -> Option<usize> { self.items.iter().position(|x| x == item) }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] { &self.items }
}

// todo: benchmark that this is actually faster...
/// spezialized version of the const generic impls.
impl<T: Default + Copy + Eq> RbSet<T, 2> {
    #[inline(always)]
    pub fn _push(&mut self, item: T) {
        if self.items[0] != item {
            self.items[1] = self.items[0];
            self.items[0] = item;
        }
    }

    #[inline(always)]
    pub fn _position(&self, item: &T) -> Option<usize> {
        if self.items[0] == *item {
            Some(0)
        }
        else if self.items[1] == *item {
            Some(1)
        }
        else {
            None
        }
    }

    #[inline(always)]
    pub fn _is_empty(&self) -> bool { self.items[0] == T::default() && self.items[1] == T::default() }
}

const MAX_HISTORY: MoveScore = 30_000;

#[derive(Clone, Copy)]
pub struct PieceHistory {
    /// For each piece type to it's destination square.
    scores: [[MoveScore; squares::N_VARIANTS]; piece_type::N_VARIANTS - 1],
}

const impl Default for PieceHistory {
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

        let curr_score = unsafe { self.histories.get_unchecked_mut(c).scores.get_unchecked_mut(pt).get_unchecked_mut(sq) };

        // This scales up history updates when a beta cutoff is unexpected, and scales
        // down history updates when a beta cutoff is expected. A beneficial side effect
        // is that this formula also clamps history values from -MAX_HISTORY to
        // MAX_HISTORY, which prevents oversaturated values.
        // ref: https://www.chessprogramming.org/History_Heuristic
        let max = MAX_HISTORY as i32;
        let clamped_val = i32::from(val.clamp(-MAX_HISTORY, MAX_HISTORY));
        let current_val = i32::from(*curr_score);
        let bonus = clamped_val - current_val * clamped_val.abs() / max;
        *curr_score += bonus.clamp(-max, max) as MoveScore;
        // todo: replace the last clamp with a debug assert?
    }

    pub fn updates_for<P: Perspective>(&mut self, items: impl Iterator<Item = (PieceType, Square)>, val: MoveScore) {
        for (pt, sq) in items {
            self.update_for::<P>(pt, sq, val);
        }
    }
}

const LINE_CAP: usize = Depth::MAX.index();

#[derive(Default, Clone, Debug)]
pub struct Line(List<LINE_CAP, Move>);

impl Deref for Line {
    type Target = List<LINE_CAP, Move>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for Line {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<'a> IntoIterator for &'a Line {
    type Item = Move;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, Move>>;

    fn into_iter(self) -> Self::IntoIter { self.iter().copied() }
}
