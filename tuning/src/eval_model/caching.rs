use std::collections::HashMap;
use burn::config::Config;
use engine::core::{r#move::Move, zobrist};
use crate::self_play::Outcome;

#[derive(Debug, Config)]
pub struct CachingConfig {
    pub proven_games: bool,
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: zobrist::Hash,
    pub outcome: Outcome,
    pub best_move: Move,
}

pub struct Cache {
    entries: HashMap<zobrist::Hash, CacheEntry>,
    config: CachingConfig,
}

impl Cache {
    pub fn new(config: CachingConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
        }
    }

    pub fn get(&self, hash: zobrist::Hash) -> Option<&CacheEntry> {
        if !self.config.proven_games {
            return None;
        }
        self.entries.get(&hash)
    }

    pub fn insert(&mut self, hash: zobrist::Hash, outcome: Outcome, best_move: Move) {
        if !self.config.proven_games {
            return;
        }
        // Only store discrete outcomes (win/loss/draw) – continuous outcomes are not "proven"
        if let Outcome::Discrete(_) = outcome {
            self.entries.insert(hash, CacheEntry { key: hash, outcome, best_move });
        }
    }
}
