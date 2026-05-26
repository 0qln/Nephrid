use crate::core::zobrist;

pub struct TranspositionTable<Data> {
    entries: Box<[Option<Data>]>,
}

impl<Data: ZKey + Clone> TranspositionTable<Data> {
    pub fn new(size: usize) -> Self {
        const fn const_none<T>() -> Option<T> {
            None
        }
        Self {
            entries: (vec![const { const_none() }; size]).into_boxed_slice(),
        }
    }

    /// Number of entries
    #[inline]
    pub fn size(&self) -> usize {
        self.entries.len()
    }

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
}

pub trait ZKey {
    fn key(&self) -> zobrist::Hash;
}
