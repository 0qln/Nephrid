use std::{fmt, marker::PhantomData, ops::Deref};

use crate::{
    core::{
        chrono::ChronoParams,
        config::{ConfigBuilder, Configuration},
        depth::Depth,
        eval::hce::TaperValue,
        search::{
            id::IdParams,
            mcts::{eval::hce::PolicyParams, node::VisitCount, search::MctsParams, select::puct::PuctParams},
            quiesce::QSearchParams,
            score::AnyScore,
        },
    },
    math::NormalizedEntropy,
};

pub const trait IConfigBuilder {
    fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder;
}

/// Something that wraps parameters used by some part of the engine.
pub const trait IParams: IConfigBuilder {
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

            #[cfg(feature = "tunable")] pub type [<$name Params>] = TunableParams<[<C_ $name Params>]>;
            #[cfg(feature = "tunable")] pub type [<$name ParamsRef>] = TunableParamsRef<[<C_ $name Params>]>;
            #[cfg(not(feature = "tunable"))] pub type [<$name Params>] = [<C_ $name Params>];
            #[cfg(not(feature = "tunable"))] pub type [<$name ParamsRef>] = [<C_ $name ParamsRef>];

            pub fn [<$name:snake _params_default>]() -> [<$name ParamsRef>] {
                cfg_select! {
                    feature = "tunable" => [<C_ $name Params>]::tunable(&[<C_ $name Params>]).shared(),
                    _ => [<C_ $name Params>]
                }
            }

            impl [<C_ $name Params>] {
                pub fn tunable(&self) -> TunableParams<[<C_ $name Params>]> {
                    let builder = Configuration::builder();
                    let config = self.build_config(builder).build();
                    TunableParams::from_config(&config)
                }
            }

            impl IParams for [<C_ $name Params>] {
                type Ref = Self;
                fn shared(self) -> Self::Ref { self }
                fn try_from_config<C: Deref<Target = Configuration>>(_: C) -> Result<Self::Ref, std::convert::Infallible> { Ok(Self) }
            }
        }
    };
}

// generic tunable

pub type TunableParamsRef<B> = std::rc::Rc<TunableParams<B>>;

#[derive(Debug, Clone)]
pub struct TunableParams<Base> {
    timeman_entropy_target: NormalizedEntropy,
    timeman_movestreak_target: u32,
    hce_policy_temp: f32,
    hce_q_futility_margin: AnyScore,
    hce_q_delta_pruning_threshold: TaperValue,
    select_cpuct: f32,
    mcts_proven_loss_visit_threshold: VisitCount,
    mcts_killer_exploitation: f32,
    mcts_tt_best_move: f32,
    id_nmp_reduction: Depth,
    id_nmp_phase_threshold: TaperValue,
    id_nmp_depth_factor: u8,
    id_nmp_phase_factor: u32,
    id_nmp_margin: AnyScore,
    _base: PhantomData<Base>,
}

impl<B, X: Deref<Target = TunableParams<B>>> PuctParams for X {
    fn select_cpuct(&self) -> f32 { self.select_cpuct }
}

impl<B, X: Deref<Target = TunableParams<B>>> MctsParams for X {
    fn proven_loss_visit_threshold(&self) -> VisitCount { self.mcts_proven_loss_visit_threshold }
    fn killer_exploitation(&self) -> f32 { self.mcts_killer_exploitation }
    fn tt_best_move(&self) -> f32 { self.mcts_tt_best_move }
}

impl<B, X: Deref<Target = TunableParams<B>>> QSearchParams for X {
    fn futility_margin(&self) -> AnyScore { self.hce_q_futility_margin }
    fn delta_pruning_threshold(&self) -> TaperValue { self.hce_q_delta_pruning_threshold }
}

impl<B, X: Deref<Target = TunableParams<B>>> PolicyParams for X {
    fn policy_temperature(&self) -> f32 { self.hce_policy_temp }
}

impl<B, X: Deref<Target = TunableParams<B>>> ChronoParams for X {
    fn entropy_target(&self) -> NormalizedEntropy { self.timeman_entropy_target }
    fn movestreak_target(&self) -> u32 { self.timeman_movestreak_target }
}

impl<B, X: Deref<Target = TunableParams<B>>> IdParams for X {
    fn nmp_reduction(&self) -> Depth { self.id_nmp_reduction }
    fn nmp_phase_threshold(&self) -> TaperValue { self.id_nmp_phase_threshold }
    fn nmp_depth_factor(&self) -> u8 { self.id_nmp_depth_factor }
    fn nmp_phase_factor(&self) -> u32 { self.id_nmp_phase_factor }
    fn nmp_margin(&self) -> AnyScore { self.id_nmp_margin }
}

impl<B> TunableParams<B> {
    fn from_config<C: Deref<Target = Configuration>>(config: C) -> Self {
        let config = config.deref();
        let timeman_entropy_target = config.timeman_entropy_target();
        let timeman_movestreak_target = config.timeman_movestreak_target();
        let hce_policy_temp = config.eval_policy_temperature();
        let hce_q_futility_margin = config.eval_futility_margin();
        let hce_q_delta_pruning_threshold = config.eval_delta_pruning_threshold();
        let select_cpuct = config.select_cpuct();
        let mcts_proven_loss_visit_threshold = config.mcts_proven_loss_visit_threshold();
        let mcts_killer_exploitation = config.mcts_killer_exploitation();
        let mcts_tt_best_move = config.mcts_tt_best_move();
        let id_nmp_reduction = config.id_nmp_reduction();
        let id_nmp_phase_threshold = config.id_nmp_phase_threshold();
        let id_nmp_depth_factor = config.id_nmp_depth_factor();
        let id_nmp_phase_factor = config.id_nmp_phase_factor();
        let id_nmp_margin = config.id_nmp_margin();
        Self {
            timeman_entropy_target,
            timeman_movestreak_target,
            hce_policy_temp,
            hce_q_futility_margin,
            hce_q_delta_pruning_threshold,
            select_cpuct,
            mcts_proven_loss_visit_threshold,
            mcts_killer_exploitation,
            mcts_tt_best_move,
            id_nmp_reduction,
            id_nmp_phase_threshold,
            id_nmp_depth_factor,
            id_nmp_phase_factor,
            id_nmp_margin,
            _base: PhantomData,
        }
    }
}

impl<B> IParams for TunableParams<B> {
    type Ref = TunableParamsRef<B>;

    fn shared(self) -> Self::Ref { std::rc::Rc::new(self) }

    fn try_from_config<C: Deref<Target = Configuration>>(config: C) -> Result<Self::Ref, CreateTunableParamsError> {
        Ok(Self::from_config(config).shared())
    }
}

impl<B> IConfigBuilder for TunableParams<B> {
    fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
        //
        builder.chrono(&self).policy(&self).qsearch(&self).puct(&self).mcts(&self)
    }
}

impl<B: IConfigBuilder + Default> Default for TunableParams<B> {
    fn default() -> Self {
        let base = B::default();
        let builder = Configuration::builder();
        let builder = base.build_config(builder);
        let config = builder.build();
        Self::from_config(&config)
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

impl IConfigBuilder for C_MctsHceParams {
    fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
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
    #[inline(always)] fn futility_margin(&self) -> AnyScore { AnyScore::new(166) }
    #[inline(always)] fn delta_pruning_threshold(&self) -> TaperValue { TaperValue::new(16) }
}

#[rustfmt::skip]
impl const PolicyParams for C_MctsHceParams {
    #[inline(always)] fn policy_temperature(&self) -> f32 { 24.58 }
}

// mcts nn

const_params!(MctsNn);

impl IConfigBuilder for C_MctsNnParams {
    fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
        //
        builder.puct(self).mcts(self).policy(self)
    }
}

impl const PuctParams for C_MctsNnParams {
    fn select_cpuct(&self) -> f32 { 0.77 }
}

impl const MctsParams for C_MctsNnParams {
    fn proven_loss_visit_threshold(&self) -> VisitCount { VisitCount(5) }
    fn killer_exploitation(&self) -> f32 { 0.27 }
    fn tt_best_move(&self) -> f32 { 1.65 }
}

impl const PolicyParams for C_MctsNnParams {
    fn policy_temperature(&self) -> f32 { 24.58 }
}

// mcts pure

const_params!(MctsPure);

impl IConfigBuilder for C_MctsPureParams {
    fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
        //
        builder
    }
}

impl const MctsParams for C_MctsPureParams {
    fn proven_loss_visit_threshold(&self) -> VisitCount { VisitCount(5) }
    fn killer_exploitation(&self) -> f32 { 0.27 }
    fn tt_best_move(&self) -> f32 { 1.65 }
}

// id hce

const_params!(IdHce);

impl IConfigBuilder for C_IdHceParams {
    fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
        //
        builder.chrono(self).qsearch(self).id(self)
    }
}

impl const ChronoParams for C_IdHceParams {
    fn entropy_target(&self) -> NormalizedEntropy { NormalizedEntropy::new_c(0.55) }
    fn movestreak_target(&self) -> u32 { 6 }
}

impl const QSearchParams for C_IdHceParams {
    fn futility_margin(&self) -> AnyScore { AnyScore::new(166) }
    fn delta_pruning_threshold(&self) -> TaperValue { TaperValue::new(16) }
}

impl const IdParams for C_IdHceParams {
    fn nmp_reduction(&self) -> Depth { Depth::new(2) }
    fn nmp_phase_threshold(&self) -> TaperValue { TaperValue::new(12) }
    fn nmp_depth_factor(&self) -> u8 { 3 }
    fn nmp_phase_factor(&self) -> u32 { 7 }
    fn nmp_margin(&self) -> AnyScore { AnyScore::new(48) }
}

// id nnue

const_params!(IdNnue);

impl IConfigBuilder for C_IdNnueParams {
    fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
        //
        builder.chrono(self).qsearch(self).id(self)
    }
}

impl const ChronoParams for C_IdNnueParams {
    fn entropy_target(&self) -> NormalizedEntropy { NormalizedEntropy::new_c(0.55) }
    fn movestreak_target(&self) -> u32 { 60 }
}

impl const QSearchParams for C_IdNnueParams {
    fn futility_margin(&self) -> AnyScore { AnyScore::new(167) }
    fn delta_pruning_threshold(&self) -> TaperValue { TaperValue::new(15) }
}

impl const IdParams for C_IdNnueParams {
    fn nmp_reduction(&self) -> Depth { Depth::new(1) }
    fn nmp_phase_threshold(&self) -> TaperValue { TaperValue::new(9) }
    fn nmp_depth_factor(&self) -> u8 { 3 }
    fn nmp_phase_factor(&self) -> u32 { 7 }
    fn nmp_margin(&self) -> AnyScore { AnyScore::new(50) }
}
