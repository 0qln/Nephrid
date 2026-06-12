use std::{
    cmp::min,
    time::{Duration, Instant},
};

use crate::core::{
    color::colors,
    position::Position,
    search::{id, limit::UciLimit},
    turn::Turn,
};

#[derive(Debug)]
pub struct TimeMan {
    /// Begin of search
    time_start: Instant,

    /// Hard bound
    time_limit: Instant,

    /// Soft bound
    time_target: Instant,

    /// Time allocated per move
    time_per_move: Duration,
}

impl TimeMan {
    pub fn init(limit: &UciLimit, pos: &Position) -> Self {
        let time_per_move = Self::time_per_move(limit, pos.get_turn());
        let time_start = Instant::now();
        let time_limit = time_start + time_per_move;

        TimeMan {
            time_start,
            time_limit,
            // set this to the hard limit first. it will update later when we gather stats during
            // search.
            time_target: time_limit,
            time_per_move,
        }
    }

    pub fn reinit_limit(&mut self) {
        self.time_limit = Instant::now() + self.time_per_move;
    }

    pub fn time_per_move(limit: &UciLimit, turn: Turn) -> Duration {
        let (time, inc) = match turn {
            colors::WHITE => (limit.wtime, limit.winc),
            colors::BLACK => (limit.btime, limit.binc),
            _ => unreachable!(),
        };

        let moves_to_go = min(limit.movestogo, 20) as u64;
        let time_per_move = (time / moves_to_go).saturating_add(inc / 2);
        let time_per_move = min(time_per_move, limit.movetime);

        let result = time_per_move.saturating_sub(limit.lag_buf.into());

        Duration::from_millis(result)
    }

    pub fn set_time_limit(&mut self, duration: Duration) {
        self.time_limit = self.time_start + duration;
    }

    pub fn reached_limit(&self) -> bool {
        let curr_time = Instant::now();

        curr_time >= self.time_limit
    }

    pub fn reached_target(&self) -> bool {
        let curr_time = Instant::now();

        curr_time >= self.time_target
    }

    /// Hint to the time manager that right now would be a preferred (soft) stop
    /// point.
    pub fn hint_preferred_target(&mut self, stats: &id::SearchStats) {
        self.time_target = self.time_limit - stats.iter_time;
    }

    pub fn time_start(&self) -> Instant {
        self.time_start
    }

    pub fn search_time(&self) -> Duration {
        Instant::now() - self.time_start
    }
}
