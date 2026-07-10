use std::marker::PhantomData;

use uom::si::{information::byte, u64::Information};

use crate::core::{
    depth::Depth,
    r#move::Move,
    search::{id, score::AnyScore},
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
