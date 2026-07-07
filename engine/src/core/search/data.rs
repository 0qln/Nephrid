use std::marker::PhantomData;

use uom::si::{information::byte, u64::Information};

use crate::core::{r#move::Move, search::score::AnyScore, zobrist};

pub const trait TTKey {
    fn key(&self) -> zobrist::Hash;
}

pub const trait TTMove {
    fn mov(&self) -> Move;
}

pub const trait TTStaticEval {
    fn static_eval(&self) -> AnyScore;
}

pub trait ReplacementStrategy {
    type Data;
    fn should_replace(old: &Self::Data, new: &Self::Data) -> bool;
}

pub struct TranspositionTable<Data, Strat> {
    entries: Box<[Option<Data>]>,
    strat: PhantomData<Strat>,
}

impl<Data: Clone, S> TranspositionTable<Data, S> {
    pub fn new(size: usize) -> Self {
        const fn const_none<T>() -> Option<T> { None }
        Self {
            entries: (vec![const { const_none() }; size]).into_boxed_slice(),
            strat: PhantomData,
        }
    }

    pub fn new_of_size(size: Information) -> Self {
        let bytes = size.get::<byte>() as usize;
        let entry_size = std::mem::size_of::<Option<Data>>();
        let num_entries = (bytes / entry_size).max(1);
        Self::new(num_entries)
    }
}

impl<Data: TTKey, S> TranspositionTable<Data, S> {
    /// Number of entries
    #[inline]
    pub fn size(&self) -> usize { self.entries.len() }

    /// Get data for the given key.
    #[inline]
    pub fn get(&self, key: zobrist::Hash) -> Option<&Data> {
        let idx = key.index(self.size());
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
        let idx = key.index(self.size());
        self.entries[idx] = Some(data);
    }

    /// Remove the entry for the given key, if it exists.
    #[inline]
    pub fn remove(&mut self, key: zobrist::Hash) {
        let idx = key.index(self.size());

        // if there is no Some at the idx, there is no entry for this key anyhow.
        if let Some(data) = &self.entries[idx]
            // if the key doesn't match, there wasn't an entry for this key anyhow.
            && data.key() == key
        {
            self.entries[idx] = None;
        }
    }

    pub fn entries_mut(&mut self) -> impl Iterator<Item = &mut Data> { self.entries.iter_mut().flatten() }
}

impl<Data: TTKey, Strat: ReplacementStrategy<Data = Data>> TranspositionTable<Data, Strat> {
    pub fn try_insert(&mut self, data: Data) {
        let key = data.key();
        let idx = key.index(self.size());

        if let Some(old_data) = &self.entries[idx] {
            if Strat::should_replace(old_data, &data) {
                self.entries[idx] = Some(data);
            }
        }
        else {
            self.entries[idx] = Some(data);
        }
    }
}
