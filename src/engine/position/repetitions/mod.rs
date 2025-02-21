use itertools::Itertools;

use crate::engine::zobrist;

#[cfg(test)]
mod test;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct TableEntry {
    occurances: u16,
    key: zobrist::Hash,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct TableBucket {
    // todo: replace with a ThinVec?
    entries: Vec<TableEntry>,
}

impl TableBucket {
    fn entry<'a>(&'a self, hash: zobrist::Hash) -> Option<&'a TableEntry> {
        self.entries.iter().find(|entry| entry.key == hash)
    }

    fn entry_mut<'a>(&'a mut self, hash: zobrist::Hash) -> Option<&'a mut TableEntry> {
        self.entries.iter_mut().find(|entry| entry.key == hash)
    }

    fn entry_with_index_mut<'a>(
        &'a mut self,
        hash: zobrist::Hash,
    ) -> Option<(&'a mut TableEntry, usize)> {
        self.entries
            .iter_mut()
            .enumerate()
            .find_map(|(idx, e)| (e.key == hash).then_some((e, idx)))
    }

    fn collisions(&self) -> usize {
        self.entries.len().saturating_sub(1)
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct RepetitionTable<const N: usize> {
    buckets: Box<[TableBucket; N]>,
    full: usize,
}

impl<const N: usize> Clone for RepetitionTable<N> {
    fn clone(&self) -> Self {
        Self::new(
            self.buckets
                .iter()
                .cloned()
                .collect_vec()
                .into_boxed_slice()
                .try_into()
                .expect("Failed to convert vec to array."),
        )
    }
}

impl<const N: usize> Default for RepetitionTable<N> {
    fn default() -> Self {
        Self::new(
            (0..N)
                .map(|_| TableBucket::default())
                .collect_vec()
                .into_boxed_slice()
                .try_into()
                .expect("Failed to convert vec to array."),
        )
    }
}

impl<const N: usize> RepetitionTable<N> {
    #[inline]
    fn new(buckets: Box<[TableBucket; N]>) -> Self {
        Self { buckets, full: 0 }
    }

    fn bucket<'a>(&'a self, hash: zobrist::Hash) -> &'a TableBucket {
        &self.buckets[hash.v() as usize % N]
    }

    fn bucket_mut<'a>(&'a mut self, hash: zobrist::Hash) -> &'a mut TableBucket {
        &mut self.buckets[hash.v() as usize % N]
    }

    pub fn push<const DBG: bool>(&mut self, hash: zobrist::Hash) {
        let bucket = self.bucket_mut(hash);
        match bucket.entry_mut(hash) {
            Some(e) => e.occurances += 1,
            None => bucket.entries.push(TableEntry {
                occurances: 1,
                key: hash,
            }),
        };
    }

    pub fn pop<const DBG: bool>(&mut self, hash: zobrist::Hash) {
        let bucket = self.bucket_mut(hash);
        match bucket.entry_with_index_mut(hash) {
            Some((e, idx)) if e.occurances <= 1 => _ = bucket.entries.swap_remove(idx),
            Some((e, _)) => e.occurances -= 1,
            None => (),
        }
    }

    pub fn get(&self, hash: zobrist::Hash) -> Option<u16> {
        self.bucket(hash).entry(hash).map(|e| e.occurances)
    }

    pub fn collisions(&self) -> usize {
        self.buckets.iter().map(TableBucket::collisions).sum()
    }

    pub fn free(&self) -> usize {
        self.buckets
            .iter()
            .filter(|b| TableBucket::is_empty(&b))
            .count()
    }

    pub fn len(&self) -> usize {
        self.capacity() - self.free()
    }

    pub fn capacity(&self) -> usize {
        N
    }
}
