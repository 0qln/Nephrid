use itertools::Itertools;

use crate::core::zobrist;

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
    #[inline]
    fn entry(&self, hash: zobrist::Hash) -> Option<&TableEntry> {
        self.entries.iter().find(|entry| entry.key == hash)
    }

    #[inline]
    fn entry_mut(&mut self, hash: zobrist::Hash) -> Option<&mut TableEntry> {
        self.entries.iter_mut().find(|entry| entry.key == hash)
    }

    #[inline]
    fn entry_with_index_mut(&mut self, hash: zobrist::Hash) -> Option<(&mut TableEntry, usize)> {
        self.entries
            .iter_mut()
            .enumerate()
            .find_map(|(idx, e)| (e.key == hash).then_some((e, idx)))
    }

    #[inline]
    fn collisions(&self) -> usize {
        self.entries.len().saturating_sub(1)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct RepetitionTable<const N: usize>
where
    [(); 1 << N]:,
{
    buckets: Box<[TableBucket; 1 << N]>,
}

// this is just a workaround.
// todo: use the blanket impl, when it works again
macro_rules! impl_clone {
    ($n:expr) => {
        impl Clone for RepetitionTable<$n> {
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
    };
}
//
impl_clone!(16);
impl_clone!(20);
//
// impl<const N: usize> Clone for RepetitionTable<N>
// where
//     [(); 1 << N]:,
// {
//     fn clone(&self) -> Self {
//         Self::new(
//             self.buckets
//                 .iter()
//                 .cloned()
//                 .collect_vec()
//                 .into_boxed_slice()
//                 .try_into()
//                 .expect("Failed to convert vec to array."),
//         )
//     }
// }
//

impl<const N: usize> Default for RepetitionTable<N>
where
    [(); 1 << N]:,
{
    fn default() -> Self {
        Self::new(
            (0..Self::capacity())
                .map(|_| TableBucket::default())
                .collect_vec()
                .into_boxed_slice()
                .try_into()
                .expect("Failed to convert vec to array."),
        )
    }
}

impl<const N: usize> RepetitionTable<N>
where
    [(); 1 << N]:,
{
    #[inline]
    fn new(buckets: Box<[TableBucket; 1 << N]>) -> Self {
        Self { buckets }
    }

    #[inline]
    fn bucket(&self, hash: zobrist::Hash) -> &TableBucket {
        &self.buckets[Self::index(hash)]
    }

    #[inline]
    const fn index(hash: zobrist::Hash) -> usize {
        hash.v() as usize & (Self::capacity() - 1)
    }

    #[inline]
    fn bucket_mut(&mut self, hash: zobrist::Hash) -> &mut TableBucket {
        &mut self.buckets[Self::index(hash)]
    }

    #[inline]
    pub fn push(&mut self, hash: zobrist::Hash) {
        let bucket = self.bucket_mut(hash);
        match bucket.entry_mut(hash) {
            Some(e) => e.occurances += 1,
            None => bucket.entries.push(TableEntry { occurances: 1, key: hash }),
        };
    }

    #[inline]
    pub fn pop(&mut self, hash: zobrist::Hash) {
        let bucket = self.bucket_mut(hash);
        match bucket.entry_with_index_mut(hash) {
            Some((e, idx)) if e.occurances <= 1 => _ = bucket.entries.swap_remove(idx),
            Some((e, _)) => e.occurances -= 1,
            None => (),
        }
    }

    #[inline]
    pub fn get(&self, hash: zobrist::Hash) -> Option<u16> {
        self.bucket(hash).entry(hash).map(|e| e.occurances)
    }

    #[inline]
    pub fn collisions(&self) -> usize {
        self.buckets.iter().map(TableBucket::collisions).sum()
    }

    #[inline]
    pub fn free(&self) -> usize {
        self.buckets
            .iter()
            .filter(|b| TableBucket::is_empty(b))
            .count()
    }

    #[inline]
    pub fn len(&self) -> usize {
        Self::capacity() - self.free()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub const fn capacity() -> usize {
        1 << N
    }
}
