use std::time::Duration;

use crate::core::{color::Color, position::Position};

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

        Duration::from_millis(time / 50 + inc)
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
        }
    }
}
