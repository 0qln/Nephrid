use std::{
    cmp::min,
    time::{Duration, Instant},
};

use crate::core::{color::colors, depth::Depth, r#move::Move, position::Position};

/// A struct to hold all by the UCI defined search limits and targets.
#[derive(Debug, Clone)]
pub struct UciLimit {
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
    pub iterations: u64,
    pub search_moves: Vec<Move>,
    pub lag_buf: u16,
}

impl UciLimit {
    pub fn max() -> Self {
        Self {
            is_active: false,
            wtime: u64::MAX,
            btime: u64::MAX,
            winc: u64::MAX,
            binc: u64::MAX,
            movestogo: u16::MAX,
            nodes: u64::MAX,
            movetime: u64::MAX,
            mate: Depth::MAX,
            depth: Depth::MAX,
            iterations: u64::MAX,
            search_moves: vec![],
            lag_buf: 0,
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

        let moves_to_go = min(self.movestogo, 50) as u64;
        let time_per_move = time / moves_to_go + inc;
        let time_per_move = min(time_per_move, self.movetime);

        let result = time_per_move.saturating_sub(self.lag_buf.into());

        Duration::from_millis(result)
    }

    pub fn is_active(&self) -> bool {
        self.is_active
    }

    pub fn is_reached(
        &self,
        nodes: u64,
        curr_time: Instant,
        time_limit: Instant,
        iterations: u64,
    ) -> bool {
        nodes >= self.nodes || curr_time >= time_limit || iterations >= self.iterations
    }
}

impl Default for UciLimit {
    fn default() -> Self {
        Self {
            is_active: true,
            wtime: u64::MAX,
            btime: u64::MAX,
            winc: 0,
            binc: 0,
            movestogo: u16::MAX,
            nodes: u64::MAX,
            movetime: u64::MAX,
            mate: Depth::MAX,
            depth: Depth::MAX,
            iterations: u64::MAX,
            search_moves: vec![],
            lag_buf: 500,
        }
    }
}
