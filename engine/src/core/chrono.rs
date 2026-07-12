use std::{
    cmp::min,
    hint::unreachable_unchecked,
    time::{Duration, Instant},
};

use crate::{
    core::{color::colors, depth::Depth, position::Position, search::limit::UciLimit, turn::Turn},
    math::NormalizedEntropy,
};

pub const trait ChronoParams {
    /// Fraction of the maximum possible root entropy below which the engine is
    /// considered confident enough to stop searching early.
    fn entropy_target(&self) -> NormalizedEntropy;

    fn movestreak_target(&self) -> u32;
}

/// Soft bounds.
#[derive(Debug, Default)]
struct SoftTargets {
    /// Time
    time: Option<Instant>,

    /// stop once the current root entropy drops to/below this.
    entropy: Option<NormalizedEntropy>,

    /// stop once the one move has been the best for x times in a row.
    movestreak: Option<u32>,
}

impl SoftTargets {
    pub fn reached_time(&self) -> bool { self.time.is_some_and(|x| Instant::now() >= x) }
    pub fn reached_entropy(&self, curr: NormalizedEntropy) -> bool { self.entropy.is_some_and(|x| curr <= x) }
    pub fn reached_movestreak(&self, curr: u32) -> bool { self.movestreak.is_some_and(|x| curr >= x) }
}

#[derive(Debug, Default)]
struct HardLimits {
    time: Option<Instant>,
}

impl HardLimits {
    pub fn reached_time(&self) -> bool { self.time.is_some_and(|x| Instant::now() >= x) }
}

#[derive(Debug)]
pub struct TimeMan {
    /// Begin of search
    time_start: Instant,

    /// Hard limits
    limits: HardLimits,

    /// Soft target
    targets: SoftTargets,

    /// Whether to enable soft targets or not.
    enable_soft_targets: bool,

    /// Time allocated per move
    time_per_move: Duration,

    // Current Stats
    curr_entropy: Option<NormalizedEntropy>,
    curr_movestreak: Option<u32>,
}

impl Default for TimeMan {
    fn default() -> Self {
        TimeMan {
            time_start: Instant::now(),
            limits: Default::default(),

            enable_soft_targets: false,
            targets: Default::default(),

            time_per_move: Duration::MAX,
            curr_entropy: None,
            curr_movestreak: None,
        }
    }
}

impl TimeMan {
    pub fn new(limit: &UciLimit, pos: &Position) -> Self {
        let mut new = Self::default();
        new.init_limits(limit, pos);
        new
    }

    pub fn init_limits(&mut self, limit: &UciLimit, pos: &Position) {
        let time_per_move = Self::time_per_move(limit, pos.get_turn());
        let time_start = Instant::now();
        let time_limit = time_start + time_per_move;

        let limits = HardLimits { time: Some(time_limit) };

        self.time_start = time_start;
        self.limits = limits;
        self.time_per_move = time_per_move;

        // soft targets remain unchanged
    }

    pub fn time_per_move(limit: &UciLimit, turn: Turn) -> Duration {
        let (time, inc) = match turn {
            colors::WHITE => (limit.wtime, limit.winc),
            colors::BLACK => (limit.btime, limit.binc),
            _ => unsafe { unreachable_unchecked() },
        };

        let moves_to_go = min(limit.movestogo, 20) as u64;
        let time_per_move = (time / moves_to_go).saturating_add(inc / 2);
        let time_per_move = min(time_per_move, limit.movetime);

        let result = time_per_move.saturating_sub(limit.lag_buf.into());

        Duration::from_millis(result)
    }

    pub fn update_soft_targets(&mut self, depth: Depth, movestreak: u32, last_iter_time: Duration, entropy: NormalizedEntropy) {
        // don't start another iteration if we expect to run completely out of hard time
        // during it.
        let predict_target = self.limits.time.map(|x| x - last_iter_time);

        // set our baseline soft time target to 50% of the total allocated move time.
        let base_soft_duration = self.time_per_move.mul_f32(0.5);

        // scale down allowed thinking time linearly as the move streak stabilizes.
        // capped at a minimum of 35% of our baseline soft time budget.
        let stability_factor = (1.0 - (movestreak as f32 * 0.08)).max(0.35);

        let entropy_factor = if depth.v() < 5 {
            // At shallow depths, tactical mirages are common.
            1.0
        }
        else {
            // Map normalized entropy (0.0 = certain, 1.0 = highly uncertain)
            // to a time multiplier range of 0.5x to 1.5x.
            // Low entropy saves time; high entropy invests extra time.
            0.5 + (entropy.v() as f32 * 1.0)
        };

        let combined_factor = (stability_factor * entropy_factor).clamp(0.2, 1.5);

        let scaled_soft_duration = base_soft_duration.mul_f32(combined_factor);
        let stability_target = self.time_start + scaled_soft_duration;

        let final_soft_time = match predict_target {
            Some(predict_instant) => min(predict_instant, stability_target),
            None => stability_target,
        };

        self.targets.time = Some(final_soft_time);
        self.curr_movestreak = Some(movestreak);
        self.curr_entropy = Some(entropy);
        self.enable_soft_targets = true;
    }

    #[allow(clippy::needless_return)]
    pub fn reached_limit(&self) -> bool {
        if self.limits.reached_time() {
            return true;
        }

        return false;
    }

    pub fn reached_target(&self) -> bool {
        if !self.enable_soft_targets {
            return false;
        }

        if self.targets.reached_time() {
            return true;
        }

        if self.curr_movestreak.is_some_and(|curr| self.targets.reached_movestreak(curr)) {
            return true;
        }

        if self.curr_entropy.is_some_and(|curr| self.targets.reached_entropy(curr)) {
            return true;
        }

        false
    }

    pub fn set_curr_entropy(&mut self, entropy: NormalizedEntropy) { self.curr_entropy = Some(entropy); }
    pub fn set_curr_movestreak(&mut self, movestreak: u32) { self.curr_movestreak = Some(movestreak); }

    pub fn hint_entropy_target(&mut self, entropy: Option<NormalizedEntropy>) { self.targets.entropy = entropy; }
    pub fn hint_movestreak_target(&mut self, movestreak: Option<u32>) { self.targets.movestreak = movestreak; }
    pub fn hint_time_target(&mut self, time: Option<Instant>) { self.targets.time = time; }

    pub fn time_start(&self) -> Instant { self.time_start }
    pub fn time_limit(&self) -> Option<Instant> { self.limits.time }
    pub fn search_time(&self) -> Duration { Instant::now() - self.time_start }

    pub fn is_soft_targets_enabled(&self) -> bool { self.enable_soft_targets }
    pub fn enable_soft_targets(&mut self, enable: bool) { self.enable_soft_targets = enable }
}
