use std::rc::Rc;

use crate::core::{
    config::Configuration,
    search::mcts::{
        self,
        eval::hce::{PolicyParams, QSearchParams, TaperValue},
        node::VisitCount,
    },
};

#[cfg(feature = "tunable")]
pub type Params = TunableParams;
#[cfg(feature = "tunable")]
pub type ParamsRef = Rc<TunableParams>;
#[cfg(feature = "tunable")]
pub type CreateParamsError = CreateTunableParamsError;

#[cfg(not(feature = "tunable"))]
pub type Params = HceParams;
#[cfg(not(feature = "tunable"))]
pub type ParamsRef = HceParams;
#[cfg(not(feature = "tunable"))]
pub type CreateParamsError = CreateConcreteParamsError;

#[derive(Debug, Clone)]
pub struct TunableParams {
    hce_policy_temp: f32,
    hce_q_futility_margin: i32,
    hce_q_delta_pruning_threshold: TaperValue,
    select_cpuct: f32,
    mcts_proven_loss_visit_threshold: VisitCount,
    mcts_killer_exploitation: f32,
    mcts_tt_best_move: f32,
}

impl TunableParams {
    pub fn new(
        hce_policy_temp: f32,
        hce_q_futility_margin: i32,
        hce_q_delta_pruning_threshold: TaperValue,
        select_cpuct: f32,
        mcts_proven_loss_visit_threshold: VisitCount,
        mcts_killer_exploitation: f32,
        mcts_tt_best_move: f32,
    ) -> Self {
        Self {
            hce_policy_temp,
            hce_q_futility_margin,
            hce_q_delta_pruning_threshold,
            select_cpuct,
            mcts_proven_loss_visit_threshold,
            mcts_killer_exploitation,
            mcts_tt_best_move,
        }
    }

    pub fn select_cpuct(&self) -> f32 {
        self.select_cpuct
    }
}

impl Default for TunableParams {
    fn default() -> Self {
        let config = Configuration::default();
        Self::try_from(&config).expect("default config should contain healthy values")
    }
}

impl mcts::select::puct::PuctParams for Rc<TunableParams> {
    fn select_cpuct(&self) -> f32 {
        self.select_cpuct
    }
}

impl mcts::search::SearchParams for Rc<TunableParams> {
    fn proven_loss_visit_threshold(&self) -> mcts::node::VisitCount {
        self.mcts_proven_loss_visit_threshold
    }

    fn killer_exploitation(&self) -> f32 {
        self.mcts_killer_exploitation
    }

    fn tt_best_move(&self) -> f32 {
        self.mcts_tt_best_move
    }
}

impl QSearchParams for Rc<TunableParams> {
    fn futility_margin(&self) -> i32 {
        self.hce_q_futility_margin
    }
    fn delta_pruning_threshold(&self) -> TaperValue {
        self.hce_q_delta_pruning_threshold
    }
}

impl PolicyParams for Rc<TunableParams> {
    #[inline(always)]
    fn policy_temperature(&self) -> f32 {
        self.hce_policy_temp
    }
}

impl IParams for TunableParams {
    type Ref = Rc<TunableParams>;
    fn shared(self) -> Rc<TunableParams> {
        Rc::new(self)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CreateTunableParamsError {
    #[error("invalid policy temperature: {0}")]
    InvalidPolicyTemperature(String),
    #[error("invalid futility margin: {0}")]
    InvalidFutilityMargin(String),
    #[error("invalid delta pruning threshold: {0}")]
    InvalidDeltaPruningThreshold(String),
}

impl TryFrom<&Configuration> for TunableParams {
    type Error = CreateTunableParamsError;

    fn try_from(config: &Configuration) -> Result<Self, Self::Error> {
        let hce_policy_temp = config.eval_policy_temperature();
        let hce_q_futility_margin = config.eval_futility_margin();
        let hce_q_delta_pruning_threshold = config.eval_delta_pruning_threshold();
        let select_cpuct = config.select_cpuct();
        let mcts_proven_loss_visit_threshold = config.mcts_proven_loss_visit_threshold();
        let mcts_killer_exploitation = config.mcts_killer_exploitation();
        let mcts_tt_best_move = config.mcts_tt_best_move();
        Ok(Self {
            hce_policy_temp,
            hce_q_futility_margin,
            hce_q_delta_pruning_threshold,
            select_cpuct,
            mcts_proven_loss_visit_threshold,
            mcts_killer_exploitation,
            mcts_tt_best_move,
        })
    }
}

// todo:
// move the HceParams into the nephrid binary target.
// these values are tuned for this specific evaluation function and are are
// likely to have different optimal values for e.g. NNParts

#[derive(Debug, Default, Clone)]
pub struct HceParams;

impl mcts::select::puct::PuctParams for HceParams {
    #[inline(always)]
    fn select_cpuct(&self) -> f32 {
        1.12
    }
}

impl mcts::search::SearchParams for HceParams {
    #[inline(always)]
    fn proven_loss_visit_threshold(&self) -> mcts::node::VisitCount {
        VisitCount(4)
    }

    #[inline(always)]
    fn killer_exploitation(&self) -> f32 {
        1.0
    }

    #[inline(always)]
    fn tt_best_move(&self) -> f32 {
        2.0
    }
}

impl QSearchParams for HceParams {
    #[inline(always)]
    fn futility_margin(&self) -> i32 {
        201
    }

    #[inline(always)]
    fn delta_pruning_threshold(&self) -> TaperValue {
        TaperValue::new(16)
    }
}

impl PolicyParams for HceParams {
    #[inline(always)]
    fn policy_temperature(&self) -> f32 {
        21.26
    }
}

impl IParams for HceParams {
    type Ref = Self;

    #[inline(always)]
    fn shared(self) -> Self::Ref {
        self
    }
}

#[allow(clippy::infallible_try_from)]
impl TryFrom<&Configuration> for HceParams {
    type Error = CreateConcreteParamsError;

    #[inline(always)]
    fn try_from(_: &Configuration) -> Result<Self, Self::Error> {
        Ok(Self)
    }
}

pub type CreateConcreteParamsError = std::convert::Infallible;

pub trait IParams {
    type Ref: ?Sized;
    fn shared(self) -> Self::Ref;
}
