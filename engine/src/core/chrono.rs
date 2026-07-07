use std::{
    cmp::min,
    time::{Duration, Instant},
};

use crate::{
    core::{color::colors, position::Position, search::limit::UciLimit, turn::Turn},
    math::NormalizedEntropy,
};

pub const trait ChronoParams {
    /// Fraction of the maximum possible root entropy below which the engine is
    /// considered confident enough to stop searching early.
    #[deprecated(
        note = "Elo improves but sometimes the engine blunders when a move on depth 1 looks like the only promising one. Not using this for now."
    )]
    fn entropy_target(&self) -> NormalizedEntropy;

    fn movestreak_target(&self) -> u32;
}

#[derive(Debug)]
pub struct TimeMan {
    /// Begin of search
    time_start: Instant,

    /// Hard bound
    time_limit: Instant,

    /// Soft bound
    time_target: Instant,

    /// Soft bound: stop once the current root entropy drops to/below this.
    entropy_target: NormalizedEntropy,

    /// Soft bound: stop once the one move has been the best for x times in a
    /// row.
    movestreak_target: u32,

    /// Time allocated per move
    time_per_move: Duration,

    /// Curr Stats
    curr_entropy: NormalizedEntropy,

    /// Current move streak
    curr_movestreak: u32,
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

            entropy_target: NormalizedEntropy::zero(),
            movestreak_target: u32::MAX,

            time_per_move,

            curr_entropy: NormalizedEntropy::one(),
            curr_movestreak: 0,
        }
    }

    pub fn reinit_limit(&mut self) { self.time_limit = Instant::now() + self.time_per_move; }

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

    pub fn set_time_limit(&mut self, duration: Duration) { self.time_limit = self.time_start + duration; }

    pub fn reached_limit(&self) -> bool {
        let curr_time = Instant::now();

        curr_time >= self.time_limit
    }

    pub fn reached_target(&self) -> bool {
        let curr_time = Instant::now();

        curr_time >= self.time_target || self.curr_entropy.v() <= self.entropy_target.v() || self.curr_movestreak >= self.movestreak_target
    }

    pub fn set_curr_entropy(&mut self, entropy: NormalizedEntropy) { self.curr_entropy = entropy; }
    pub fn set_curr_movestreak(&mut self, movestreak: u32) { self.curr_movestreak = movestreak; }

    pub fn hint_entropy_target(&mut self, entropy: NormalizedEntropy) { self.entropy_target = entropy; }
    pub fn hint_movestreak_target(&mut self, movestreak: u32) { self.movestreak_target = movestreak; }
    pub fn hint_time_target(&mut self, time: Instant) { self.time_target = time; }

    pub fn time_start(&self) -> Instant { self.time_start }
    pub fn time_limit(&self) -> Instant { self.time_limit }
    pub fn search_time(&self) -> Duration { Instant::now() - self.time_start }
}
