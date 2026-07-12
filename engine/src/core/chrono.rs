use std::{
    cmp::min,
    hint::unreachable_unchecked,
    time::{Duration, Instant},
};

use crate::{
    core::{color::colors, params::IParams, position::Position, search::limit::UciLimit, turn::Turn},
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
    prediction_time: Option<Instant>,

    /// Factors
    entropy_factor: Option<f32>,
    movestreak_factor: Option<f32>,
}

impl SoftTargets {
    fn combined_soft_factor(&self) -> f32 {
        let mut factor = 1.0;

        if let Some(movestreak_factor) = self.movestreak_factor {
            factor *= movestreak_factor;
        }

        if let Some(entropy_factor) = self.entropy_factor {
            factor *= entropy_factor;
        }

        factor.clamp(0.2, 1.5)
    }
}

#[derive(Debug, Default)]
struct HardLimits {
    time: Option<Instant>,
}

impl HardLimits {
    pub fn reached_time(&self) -> bool { self.time.is_some_and(|x| Instant::now() >= x) }
}

#[derive(Debug)]
pub struct TimeMan<X: IParams> {
    /// Begin of search
    time_start: Option<Instant>,

    /// Hard limits
    limits: HardLimits,

    /// Soft target
    targets: SoftTargets,

    /// Whether to enable soft targets or not.
    enable_soft_targets: bool,

    /// Params
    params: X::Ref,
}

impl<X: IParams> TimeMan<X>
where
    X::Ref: ChronoParams,
{
    pub fn new(params: X::Ref) -> Self {
        TimeMan {
            time_start: None,
            limits: Default::default(),

            enable_soft_targets: false,
            targets: Default::default(),

            params,
        }
    }

    pub fn new_with_limits(limit: &UciLimit, pos: &Position, params: X::Ref) -> Self {
        let mut new = Self::new(params);
        new.init_limits(limit, pos);
        new
    }

    pub fn init_limits(&mut self, limit: &UciLimit, pos: &Position) {
        let time_per_move = Self::time_per_move(limit, pos.get_turn());
        let time_start = Instant::now();
        let time_limit = time_start + time_per_move;

        self.time_start = Some(time_start);
        self.limits = HardLimits { time: Some(time_limit) };

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

        let now = Instant::now();

        if self.targets.prediction_time.is_some_and(|x| now >= x) {
            return true;
        }

        if let Some(hard_duration) = self.duration_limit()
            && let Some(start) = self.time_start()
        {
            let factor = self.targets.combined_soft_factor();
            let soft_duration = hard_duration.mul_f32(0.5).mul_f32(factor);
            let soft_target = start + soft_duration;

            if now >= soft_target {
                return true;
            }
        }

        false
    }

    pub fn hint_entropy_target(&mut self, entropy: NormalizedEntropy) {
        // Map normalized entropy (0.0 = certain, 1.0 = highly uncertain)
        // to a time multiplier range of 0.5x to 1.5x.
        // Low entropy saves time; high entropy invests extra time.
        let factor = 0.5 + (entropy.v() * 1.0);
        self.targets.entropy_factor = Some(factor);
    }

    pub fn hint_movestreak_target(&mut self, movestreak: u32) {
        // scale down allowed thinking time linearly as the move streak stabilizes.
        // capped at a minimum of 35% of our baseline soft time budget.
        let factor = (1.0 - (movestreak as f32 * 0.08)).max(0.35);
        self.targets.movestreak_factor = Some(factor);
    }

    pub fn hint_time_target(&mut self, last_iter_time: Duration) {
        // don't start another iteration if we expect to run completely out of hard time
        // during it.
        self.targets.prediction_time = self.time_limit().map(|limit| limit - last_iter_time);
    }

    pub fn time_start(&self) -> Option<Instant> { self.time_start }
    pub fn time_limit(&self) -> Option<Instant> { self.limits.time }
    pub fn duration_limit(&self) -> Option<Duration> { self.limits.time.and_then(|end| self.time_start.map(|start| end - start)) }
    pub fn elapsed_search_time(&self) -> Option<Duration> { self.time_start.map(|start| Instant::now() - start) }

    pub fn is_soft_targets_enabled(&self) -> bool { self.enable_soft_targets }
    pub fn enable_soft_targets(&mut self, enable: bool) { self.enable_soft_targets = enable }
}
