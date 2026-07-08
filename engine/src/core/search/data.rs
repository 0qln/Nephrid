use std::marker::PhantomData;

use uom::si::{information::byte, u64::Information};

use crate::core::{r#move::Move, search::score::AnyScore, zobrist};

// pub const trait TTIsValid {
//     fn is_valid(&self) -> bool;
//     fn new_invalid() -> Self;

//     fn as_validated(&self) -> Option<&Self> { if self.is_valid() { Some(self)
// } else { None } }     fn as_validated_mut(&mut self) -> Option<&mut Self> {
// if self.is_valid() { Some(self) } else { None } } }

pub const trait TTKey {
    fn key(&self) -> zobrist::Hash;
}

pub const trait TTMove {
    fn mov(&self) -> Move;
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

    /// Get data for the given key.
    #[inline]
    pub fn raw_mut(&mut self, key: zobrist::Hash) -> Option<&mut Data> {
        let idx = key.index(self.size());
        let data = &mut self.entries[idx];
        if data.key() == key { Some(data) } else { None }
    }

    /// Insert and overwrite in any case.
    #[inline]
    #[deprecated(note = "Use `try_insert` with an always-true-strategy instead.")]
    pub fn insert(&mut self, data: Data) {
        let key = data.key();
        let idx = key.index(self.size());
        self.entries[idx] = data;
    }

    // /// Remove the entry for the given key, if it exists.
    // #[inline]
    // pub fn remove(&mut self, key: zobrist::Hash) {
    //     let idx = key.index(self.size());

    //     // if there is no Some at the idx, there is no entry for this key anyhow.
    //     if let Some(data) = &self.entries[idx].as_validated()
    //         // if the key doesn't match, there wasn't an entry for this key
    // anyhow.         && data.key() == key
    //     {
    //         self.entries[idx] = Data::new_invalid();
    //     }
    // }

    // pub fn entries_mut(&mut self) -> impl Iterator<Item = &mut Data> {
    // self.entries.iter_mut().flatten() }
}

impl<Data: TTKey, Strat: ReplacementStrategy<Data = Data>> TranspositionTable<Data, Strat> {
    pub fn try_insert(&mut self, data: Data) {
        let key = data.key();
        let idx = key.index(self.size());
        let old_data = &self.entries[idx];

        if Strat::should_replace(old_data, &data) {
            self.entries[idx] = data;
        }
    }
}
