use std::cmp::min;
use std::time::Duration;

use crate::core::depth::Depth;
use crate::core::r#move::Move;
use crate::core::{color::colors, position::Position};

/// A struct to hold all by the UCI defined search limits and targets.
#[derive(Debug, Clone)]
pub struct Limit {
    pub is_active: bool,
    pub wtime: u64,
    pub btime: u64,
    pub winc: u64,
    pub binc: u64,
    pub movestogo: u16,
    pub nodes: u64,
    pub movetime: u64,
    pub mate: Depth,
    pub depth: Depth,
    pub search_moves: Vec<Move>,
}

impl Limit {
    pub fn max() -> Self {
        Self {
            is_active: false,
            wtime: u64::MAX,
            btime: u64::MAX,
            winc: u64::MAX,
            binc: u64::MAX,
            movestogo: 0,
            nodes: u64::MAX,
            movetime: u64::MAX,
            mate: Depth::MAX,
            depth: Depth::MAX,
            search_moves: vec![],
        }
    }

    pub fn time_per_move(&self, pos: &Position) -> Duration {
        let time = match pos.get_turn() {
            colors::WHITE => self.wtime,
            colors::BLACK => self.btime,
            _ => unreachable!(),
        };
        let inc = match pos.get_turn() {
            colors::WHITE => self.winc,
            colors::BLACK => self.binc,
            _ => unreachable!(),
        };

        let lag_buf = 500;

        let time_per_move = time / 50 + inc;
        let time_per_move = min(time_per_move, self.movetime);

        let result = if time_per_move > lag_buf {
            time_per_move - lag_buf
        } else {
            0
        };

        Duration::from_millis(result)
    }
}

impl Default for Limit {
    fn default() -> Self {
        Self {
            is_active: true,
            wtime: u64::MAX,
            btime: u64::MAX,
            winc: 0,
            binc: 0,
            movestogo: 0,
            nodes: u64::MAX,
            movetime: u64::MAX,
            mate: Depth::MAX,
            depth: Depth::MAX,
            search_moves: vec![],
        }
    }
}
