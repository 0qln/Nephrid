use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;

use crate::engine::zobrist;


#[derive(Default, Clone)]
pub struct RepetitionTable {
    table: FxHashMap<zobrist::Hash, u8>    
}

impl RepetitionTable {
    pub fn push(&mut self, hash: zobrist::Hash) {
        match self.table.entry(hash) {
            Entry::Occupied(mut e) => *e.get_mut() += 1,
            Entry::Vacant(e) => { e.insert(1); },
        };
    }
    
    pub fn pop(&mut self, hash: zobrist::Hash) {
        match self.table.entry(hash) {
            Entry::Occupied(e) if e.get() == &0 => { e.remove_entry(); },
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