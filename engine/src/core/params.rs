use std::{ops::Deref, rc::Rc};

use crate::{
    core::{
        chrono::ChronoParams,
        config::Configuration,
        eval::hce::TaperValue,
        search::{
            mcts::{
                self, eval::hce::PolicyParams, node::VisitCount, search::MctsParams,
                select::puct::PuctParams,
            },
            quiesce::QSearchParams,
        },
    },
    math::NormalizedEntropy,
};

pub const trait IParams {
    type Ref: ?Sized + Clone + Deref<Target = Self>;
    fn shared(self) -> Self::Ref;
}

// generic tunable

#[derive(Debug, Clone)]
pub struct TunableParams {
    timeman_entropy_target: NormalizedEntropy,
    hce_policy_temp: f32,
    hce_q_futility_margin: i32,
    hce_q_delta_pruning_threshold: TaperValue,
    select_cpuct: f32,
    mcts_proven_loss_visit_threshold: VisitCount,
    mcts_killer_exploitation: f32,
    mcts_tt_best_move: f32,
}

impl mcts::select::puct::PuctParams for Rc<TunableParams> {
    fn select_cpuct(&self) -> f32 {
        self.select_cpuct
    }
}

impl mcts::search::MctsParams for Rc<TunableParams> {
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

impl ChronoParams for Rc<TunableParams> {
    #[inline(always)]
    fn entropy_target(&self) -> NormalizedEntropy {
        self.timeman_entropy_target
    }
}

impl IParams for TunableParams {
    type Ref = Rc<TunableParams>;

    fn shared(self) -> Rc<TunableParams> {
        Rc::new(self)
    }
}

#[rustfmt::skip] #[cfg(feature = "tunable")] pub type CreateParamsError = CreateTunableParamsError;
#[rustfmt::skip] #[cfg(not(feature = "tunable"))] pub type CreateParamsError = std::convert::Infallible;

#[derive(Debug, thiserror::Error)]
pub enum CreateTunableParamsError {
    #[error("invalid policy temperature: {0}")]
    InvalidPolicyTemperature(String),
    #[error("invalid futility margin: {0}")]
    InvalidFutilityMargin(String),
    #[error("invalid delta pruning threshold: {0}")]
    InvalidDeltaPruningThreshold(String),
}

#[cfg(feature = "tunable")]
impl TryFrom<&Configuration> for TunableParams {
    type Error = CreateTunableParamsError;

    fn try_from(config: &Configuration) -> Result<Self, Self::Error> {
        let timeman_entropy_target = config.timeman_entropy_target();
        let hce_policy_temp = config.eval_policy_temperature();
        let hce_q_futility_margin = config.eval_futility_margin();
        let hce_q_delta_pruning_threshold = config.eval_delta_pruning_threshold();
        let select_cpuct = config.select_cpuct();
        let mcts_proven_loss_visit_threshold = config.mcts_proven_loss_visit_threshold();
        let mcts_killer_exploitation = config.mcts_killer_exploitation();
        let mcts_tt_best_move = config.mcts_tt_best_move();
        Ok(Self {
            timeman_entropy_target,
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

// mcts hce

#[allow(non_camel_case_types)]
#[derive(Debug, Default, Clone)]
pub struct C_MctsHceParams;

#[rustfmt::skip]
impl const PuctParams for C_MctsHceParams {
    #[inline(always)] fn select_cpuct(&self) -> f32 { 0.77 }
}

#[rustfmt::skip]
impl const MctsParams for C_MctsHceParams {
    #[inline(always)] fn proven_loss_visit_threshold(&self) -> VisitCount { VisitCount(5) }
    #[inline(always)] fn killer_exploitation(&self) -> f32 { 0.27 }
    #[inline(always)] fn tt_best_move(&self) -> f32 { 1.65 }
}

#[rustfmt::skip]
impl const QSearchParams for C_MctsHceParams {
    #[inline(always)] fn futility_margin(&self) -> i32 { 166 }
    #[inline(always)] fn delta_pruning_threshold(&self) -> TaperValue { TaperValue::new(16) }
}

#[rustfmt::skip]
impl const PolicyParams for C_MctsHceParams {
    #[inline(always)] fn policy_temperature(&self) -> f32 { 24.58 }
}

impl IParams for C_MctsHceParams {
    type Ref = Self;

    #[inline(always)]
    fn shared(self) -> Self::Ref {
        self
    }
}

#[allow(clippy::infallible_try_from)]
impl TryFrom<&Configuration> for C_MctsHceParams {
    type Error = std::convert::Infallible;

    #[inline(always)]
    fn try_from(_: &Configuration) -> Result<Self, Self::Error> {
        Ok(Self)
    }
}

#[rustfmt::skip] #[cfg(feature = "tunable")] pub type MctsHceParams = TunableParams;
#[rustfmt::skip] #[cfg(not(feature = "tunable"))] pub type MctsHceParams = C_MctsHceParams;

// mcts nn

#[allow(non_camel_case_types)]
#[derive(Debug, Default, Clone)]
pub struct C_MctsNNParams;

#[rustfmt::skip]
impl const PuctParams for C_MctsNNParams {
    #[inline(always)] fn select_cpuct(&self) -> f32 { 0.77 }
}

#[rustfmt::skip]
impl const MctsParams for C_MctsNNParams {
    #[inline(always)] fn proven_loss_visit_threshold(&self) -> VisitCount { VisitCount(5) }
    #[inline(always)] fn killer_exploitation(&self) -> f32 { 0.27 }
    #[inline(always)] fn tt_best_move(&self) -> f32 { 1.65 }
}

#[rustfmt::skip]
impl const PolicyParams for C_MctsNNParams {
    #[inline(always)] fn policy_temperature(&self) -> f32 { 24.58 }
}

impl IParams for C_MctsNNParams {
    type Ref = Self;

    #[inline(always)]
    fn shared(self) -> Self::Ref {
        self
    }
}

#[rustfmt::skip] #[cfg(feature = "tunable")] pub type MctsNNParams = TunableParams;
#[rustfmt::skip] #[cfg(not(feature = "tunable"))] pub type MctsNNParams = C_MctsNNParams;

// mcts pure

#[allow(non_camel_case_types)]
#[derive(Debug, Default, Clone)]
pub struct C_MctsPureParams;

impl IParams for C_MctsPureParams {
    type Ref = Self;

    #[inline(always)]
    fn shared(self) -> Self::Ref {
        self
    }
}

// id hce

#[allow(non_camel_case_types)]
#[derive(Debug, Default, Clone)]
pub struct C_IdHceParams;

impl IParams for C_IdHceParams {
    type Ref = Self;

    #[inline(always)]
    fn shared(self) -> Self::Ref {
        self
    }
}

#[rustfmt::skip]
impl ChronoParams for C_IdHceParams {
    #[inline(always)] fn entropy_target(&self) -> NormalizedEntropy { NormalizedEntropy::new_c(0.6) }
}

#[rustfmt::skip]
impl QSearchParams for C_IdHceParams {
    #[inline(always)] fn futility_margin(&self) -> i32 { 166 }
    #[inline(always)] fn delta_pruning_threshold(&self) -> TaperValue { TaperValue::new(16) }
}

#[rustfmt::skip] #[cfg(feature = "tunable")] pub type IdHceParams = TunableParams;
#[rustfmt::skip] #[cfg(not(feature = "tunable"))] pub type IdHceParams = C_IdHceParams;
