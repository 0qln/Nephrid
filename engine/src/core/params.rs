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
pub type Params = ConcreteParams;
#[cfg(not(feature = "tunable"))]
pub type ParamsRef = ConcreteParams;
#[cfg(not(feature = "tunable"))]
pub type CreateParamsError = CreateConcreteParamsError;

#[derive(Debug, Clone)]
pub struct TunableParams {
    hce_policy_temp: f32,
    hce_q_futility_margin: i32,
    hce_q_delta_pruning_threshold: TaperValue,
    mcts_proven_loss_visit_threshold: VisitCount,
}

impl TunableParams {
    pub fn new(
        policy_temp: f32,
        futility_margin: i32,
        delta_pruning_threshold: TaperValue,
        mcts_proven_loss_visit_threshold: VisitCount,
    ) -> Self {
        Self {
            hce_policy_temp: policy_temp,
            hce_q_futility_margin: futility_margin,
            hce_q_delta_pruning_threshold: delta_pruning_threshold,
            mcts_proven_loss_visit_threshold,
        }
    }
}

impl mcts::search::SearchParams for Rc<TunableParams> {
    fn proven_loss_visit_threshold(&self) -> mcts::node::VisitCount {
        self.mcts_proven_loss_visit_threshold
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
        let policy_temp = config.eval_policy_temperature();
        let futility_margin = config.eval_futility_margin();
        let delta_pruning_threshold = config.eval_delta_pruning_threshold();
        let mcts_proven_loss_visit_threshold = config.mcts_proven_loss_visit_threshold();
        Ok(Self {
            hce_policy_temp: policy_temp,
            hce_q_futility_margin: futility_margin,
            hce_q_delta_pruning_threshold: delta_pruning_threshold,
            mcts_proven_loss_visit_threshold
        })
    }
}

#[derive(Debug, Clone)]
pub struct ConcreteParams;

impl mcts::search::SearchParams for ConcreteParams {
    fn proven_loss_visit_threshold(&self) -> mcts::node::VisitCount {
        VisitCount(4)
    }
}

impl QSearchParams for ConcreteParams {
    #[inline(always)]
    fn futility_margin(&self) -> i32 {
        201
    }

    #[inline(always)]
    fn delta_pruning_threshold(&self) -> TaperValue {
        TaperValue::new(16)
    }
}

impl PolicyParams for ConcreteParams {
    #[inline(always)]
    fn policy_temperature(&self) -> f32 {
        21.26
    }
}

impl IParams for ConcreteParams {
    type Ref = Self;
    fn shared(self) -> Self::Ref {
        self
    }
}

#[allow(clippy::infallible_try_from)]
impl TryFrom<&Configuration> for ConcreteParams {
    type Error = CreateConcreteParamsError;
    fn try_from(_: &Configuration) -> Result<Self, Self::Error> {
        Ok(Self)
    }
}

pub type CreateConcreteParamsError = std::convert::Infallible;

pub trait IParams {
    type Ref: ?Sized;
    fn shared(self) -> Self::Ref;
}
