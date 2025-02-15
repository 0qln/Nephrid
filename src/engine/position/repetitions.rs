use std::{array, collections::hash_map::Entry};

use crate::engine::zobrist;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepetitionTable<const N: usize> {
    buckets: [TableBucket; N]    
}

impl<const N: usize> Default for RepetitionTable<N> {
    fn default() -> Self {
        Self { buckets: array::from_fn(|_| Default::default()) }
    }
}

impl<const N: usize>  RepetitionTable<N> {
    pub fn push(&mut self, hash: zobrist::Hash) {
        match self.table.entry(hash) {
            Entry::Occupied(mut e) => *e.get_mut() += 1,
            Entry::Vacant(e) => { e.insert(1); },
        };
    }
    
    pub fn pop(&mut self, hash: zobrist::Hash) {
        match self.table.entry(hash) {
            Entry::Occupied(e) if e.get() <= &1 => { e.remove_entry(); },
            Entry::Occupied(mut e) => *e.get_mut() -= 1,
            Entry::Vacant(_) => (),
        }
    }
    
    pub fn get(&self, hash: zobrist::Hash) -> Option<u8> {
        self.table.get(&hash).copied()
    }
    
    pub fn contains(&mut self, hash: zobrist::Hash) -> bool {
        self.get(hash) > Some(0)
    }
}