use std::{convert::Infallible, fmt, ops::Deref};

use crate::{
    core::{
        chrono::ChronoParams,
        config::{ConfigBuilder, Configuration},
        eval::hce::TaperValue,
        search::{
            mcts::{eval::hce::PolicyParams, node::VisitCount, search::MctsParams, select::puct::PuctParams},
            quiesce::QSearchParams,
        },
    },
    math::NormalizedEntropy,
};

/// Something that wraps parameters used by some part of the engine.
pub const trait IParams {
    type Ref: ?Sized + Clone;

    /// Get a shared reference to the params.
    fn shared(self) -> Self::Ref;

    fn try_from_config<C: Deref<Target = Configuration>>(config: C) -> Result<Self::Ref, impl fmt::Display>;
}

// const generator

macro_rules! const_params {
    ($name:ident) => {
        paste::paste! {
            #[derive(Debug, Default, Clone)]
            #[allow(non_camel_case_types)]
            pub struct [<C_ $name Params>];

            #[allow(non_camel_case_types)]
            pub type [<C_ $name ParamsRef>] = [<C_ $name Params>];

            #[cfg(feature = "tunable")] pub type [<$name Params>] = TunableParams;
            #[cfg(feature = "tunable")] pub type [<$name ParamsRef>] = TunableParamsRef;
            #[cfg(not(feature = "tunable"))] pub type [<$name Params>] = [<C_ $name Params>];
            #[cfg(not(feature = "tunable"))] pub type [<$name ParamsRef>] = [<C_ $name ParamsRef>];

            pub fn [<$name:snake _params_default>]() -> [<$name ParamsRef>] {
                cfg_select! {
                    feature = "tunable" => [<C_ $name Params>]::tunable(&[<C_ $name Params>]).shared(),
                    _ => [<C_ $name Params>]
                }
            }

            impl IParams for [<C_ $name Params>] {
                type Ref = Self;
                fn shared(self) -> Self::Ref { self }
                fn try_from_config<C: Deref<Target = Configuration>>(_: C) -> Result<Self::Ref, Infallible> { Ok(Self {}) }
            }

            impl [<C_ $name Params>] {
                pub fn tunable(&self) -> TunableParams {
                    let builder = Configuration::builder();
                    let config = self.build_config(builder).build();
                    TunableParams::from_config(&config)
                }
            }
        }
    };
}

// generic tunable

pub type TunableParamsRef = std::rc::Rc<TunableParams>;

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

impl<X: Deref<Target = TunableParams>> PuctParams for X {
    fn select_cpuct(&self) -> f32 { self.select_cpuct }
}

impl<X: Deref<Target = TunableParams>> MctsParams for X {
    fn proven_loss_visit_threshold(&self) -> VisitCount { self.mcts_proven_loss_visit_threshold }
    fn killer_exploitation(&self) -> f32 { self.mcts_killer_exploitation }
    fn tt_best_move(&self) -> f32 { self.mcts_tt_best_move }
}

impl<X: Deref<Target = TunableParams>> QSearchParams for X {
    fn futility_margin(&self) -> i32 { self.hce_q_futility_margin }
    fn delta_pruning_threshold(&self) -> TaperValue { self.hce_q_delta_pruning_threshold }
}

impl<X: Deref<Target = TunableParams>> PolicyParams for X {
    fn policy_temperature(&self) -> f32 { self.hce_policy_temp }
}

impl<X: Deref<Target = TunableParams>> ChronoParams for X {
    fn entropy_target(&self) -> NormalizedEntropy { self.timeman_entropy_target }
}

impl TunableParams {
    fn from_config<C: Deref<Target = Configuration>>(config: C) -> Self {
        let config = config.deref();
        let timeman_entropy_target = config.timeman_entropy_target();
        let hce_policy_temp = config.eval_policy_temperature();
        let hce_q_futility_margin = config.eval_futility_margin();
        let hce_q_delta_pruning_threshold = config.eval_delta_pruning_threshold();
        let select_cpuct = config.select_cpuct();
        let mcts_proven_loss_visit_threshold = config.mcts_proven_loss_visit_threshold();
        let mcts_killer_exploitation = config.mcts_killer_exploitation();
        let mcts_tt_best_move = config.mcts_tt_best_move();
        Self {
            timeman_entropy_target,
            hce_policy_temp,
            hce_q_futility_margin,
            hce_q_delta_pruning_threshold,
            select_cpuct,
            mcts_proven_loss_visit_threshold,
            mcts_killer_exploitation,
            mcts_tt_best_move,
        }
    }

    pub fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
        //
        builder.chrono(&self).policy(&self).qsearch(&self).puct(&self).mcts(&self)
    }
}

impl IParams for TunableParams {
    type Ref = TunableParamsRef;

    fn shared(self) -> Self::Ref { std::rc::Rc::new(self) }

    fn try_from_config<C: Deref<Target = Configuration>>(config: C) -> Result<Self::Ref, CreateTunableParamsError> {
        Ok(Self::from_config(config).shared())
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

// mcts hce

const_params!(MctsHce);

impl C_MctsHceParams {
    pub fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
        //
        builder.puct(self).mcts(self).qsearch(self).policy(self)
    }
}

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

// mcts nn

const_params!(MctsNN);

impl C_MctsNNParams {
    pub fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
        //
        builder.puct(self).mcts(self).policy(self)
    }
}

impl const PuctParams for C_MctsNNParams {
    fn select_cpuct(&self) -> f32 { 0.77 }
}

impl const MctsParams for C_MctsNNParams {
    fn proven_loss_visit_threshold(&self) -> VisitCount { VisitCount(5) }
    fn killer_exploitation(&self) -> f32 { 0.27 }
    fn tt_best_move(&self) -> f32 { 1.65 }
}

impl const PolicyParams for C_MctsNNParams {
    fn policy_temperature(&self) -> f32 { 24.58 }
}

// mcts pure

const_params!(MctsPure);

impl C_MctsPureParams {
    pub fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
        //
        builder
    }
}

// id hce

const_params!(IdHce);

impl C_IdHceParams {
    pub fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
        //
        builder.chrono(self).qsearch(self)
    }
}

impl const ChronoParams for C_IdHceParams {
    fn entropy_target(&self) -> NormalizedEntropy { NormalizedEntropy::new_c(0.6) }
}

impl const QSearchParams for C_IdHceParams {
    fn futility_margin(&self) -> i32 { 166 }
    fn delta_pruning_threshold(&self) -> TaperValue { TaperValue::new(16) }
}
