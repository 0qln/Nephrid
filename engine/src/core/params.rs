use std::{fmt, ops::Deref};

use uom::si::{f32, ratio};

use crate::{
    core::{
        chrono::ChronoParams,
        config::{ConfigBuilder, Configuration},
        depth::Depth,
        eval::hce::TaperValue,
        search::{
            id::{IdParams, ScorerParams},
            mcts::{eval::hce::PolicyParams, node::VisitCount, search::MctsParams, select::puct::PuctParams},
            quiesce::QSearchParams,
            score::AnyScore,
        },
    },
    math::LmrParams,
};

pub const trait IConfigBuilder {
    fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder;
}

/// Something that wraps parameters used by some part of the engine.
pub const trait IParams: IConfigBuilder {
    type Ref: ?Sized + Clone + fmt::Debug;

    /// Get a shared reference to the params.
    fn shared(self) -> Self::Ref;

    fn try_from_config<C: Deref<Target = Configuration>>(config: C) -> Result<Self::Ref, impl fmt::Display>;
}

// const generator

macro_rules! const_params {
    (
        $name:ident {
            $(
                $group:ident : $trait_name:ident {
                    $($trait_item:item)*
                }
            ),* $(,)?
        }
    ) => {
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

            impl IConfigBuilder for [<C_ $name Params>] {
                fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
                    builder
                        $( .$group(self) )*
                }
            }

            // trait implementations
            $(
                impl const $trait_name for [<C_ $name Params>] {
                    $(
                        #[inline(always)]
                        $trait_item
                    )*
                }
            )*
        }
    };
}

// tunable generator

macro_rules! tunable_params {
    (
        $(
            $group:ident : $trait_name:ident {
                $(
                    $field:ident : $field_type:ty {
                        uci: $uci_name:expr,
                        unit: $unit_type:ident,
                        default: $default:expr,
                        min: $min:expr,
                        max: $max:expr,
                        getter: $getter:ident,
                        to_raw: $to_raw:expr,
                        from_raw: $from_raw:expr $(,)?
                    }
                ),* $(,)?
            }
        ),* $(,)?
    ) => {
        paste::paste! {
            $(
                // individual group param struct & trait impl
                #[derive(Debug, Clone)]
                pub struct [<$trait_name Group>] {
                    $( pub $field: $field_type, )*
                }

                impl $trait_name for [<$trait_name Group>] {
                    $(
                        fn $getter(&self) -> $field_type { self.$field }
                    )*
                }

                // individual group config struct
                #[derive(Debug, Clone)]
                pub struct [<$trait_name ConfigGroup>] {
                    $( pub $field: $crate::core::config::ConfigOption<$crate::core::config::Spin<$crate::core::config::$unit_type>>, )*
                }

                impl [<$trait_name ConfigGroup>] {
                    pub fn default_builder() -> Self {
                        Self {
                            $(
                                $field: $crate::core::config::ConfigOption::new(
                                    $uci_name,
                                    $crate::core::config::Spin::new(($to_raw)($default), ($to_raw)($min), ($to_raw)($max))
                                ),
                            )*
                        }
                    }

                    pub fn seed_from(&mut self, params: &impl $trait_name) {
                        $( self.$field.seed(($to_raw)(params.$getter())); )*
                    }

                    pub fn extract_params(&self) -> [<$trait_name Group>] {
                        [<$trait_name Group>] {
                            $( $field: ($from_raw)(&self.$field.value), )*
                        }
                    }

                    pub fn set(&mut self, name: &str, value: &str) -> Option<Result<(), Box<dyn std::error::Error>>> {
                        match name {
                            $( $uci_name => Some(self.$field.set(value)), )*
                            _ => None,
                        }
                    }

                    pub fn print_uci(&self) {
                        $( println!("{}", self.$field); )*
                    }
                }
            )*

            // top-level TunableConfiguration containing all groups
            #[derive(Debug, Clone)]
            pub struct TunableConfiguration {
                $( pub $group: [<$trait_name ConfigGroup>], )*
            }

            impl Default for TunableConfiguration {
                fn default() -> Self {
                    Self {
                        $( $group: [<$trait_name ConfigGroup>]::default_builder(), )*
                    }
                }
            }

            impl TunableConfiguration {
                pub fn set(&mut self, name: &str, value: &str) -> Option<Result<(), Box<dyn std::error::Error>>> {
                    $(
                        if let Some(res) = self.$group.set(name, value) {
                            return Some(res);
                        }
                    )*
                    None
                }

                pub fn print_uci(&self) {
                    $( self.$group.print_uci(); )*
                }
            }

            // top-level TunableParams struct
            #[derive(Debug, Clone)]
            pub struct TunableParams<Base> {
                $( pub $group: [<$trait_name Group>], )*
                _base: std::marker::PhantomData<Base>,
            }

            impl<B> TunableParams<B> {
                fn from_config<C: std::ops::Deref<Target = Configuration>>(config: C) -> Self {
                    Self {
                        $( $group: config.tunable.$group.extract_params(), )*
                        _base: std::marker::PhantomData,
                    }
                }
            }

            // forwarding trait impls for TunableParams
            $(
                impl<B> $trait_name for TunableParams<B> {
                    $(
                        fn $getter(&self) -> $field_type { self.$group.$getter() }
                    )*
                }

                impl<B, X: std::ops::Deref<Target = TunableParams<B>>> $trait_name for X {
                    $(
                        fn $getter(&self) -> $field_type { self.$group.$getter() }
                    )*
                }
            )*

            // generate ConfigBuilder seeder methods automatically
            impl ConfigBuilder {
                $(
                    pub fn $group(mut self, params: &impl $trait_name) -> Self {
                        self.config.tunable.$group.seed_from(params);
                        self
                    }
                )*
            }

           impl<B> IConfigBuilder for TunableParams<B> {
                fn build_config(&self, builder: ConfigBuilder) -> ConfigBuilder {
                    builder
                        $( .$group(self) )*
                }
            }
        }
    };
}

// generic tunable

pub type TunableParamsRef<B> = std::rc::Rc<TunableParams<B>>;

fn ratio_to_raw(val: f32) -> f32::Ratio { f32::Ratio::new::<ratio::ratio>(val) }
fn ratio_from_raw(qty: &f32::Ratio) -> f32 { qty.get::<ratio::ratio>() }

tunable_params! {
    qsearch: QSearchParams {
        futility_margin: AnyScore {
            uci: "qs-futility-margin",
            unit: UciInteger,
            default: AnyScore::new(150),
            min: AnyScore::new(100),
            max: AnyScore::new(300),
            getter: futility_margin,
            to_raw: |s: AnyScore| s.v(),
            from_raw: |v: &i32| AnyScore::new(*v),
        },
        delta_pruning_threshold: TaperValue {
            uci: "qs-delta-pruning-threshold",
            unit: UciInteger,
            default: TaperValue::new(16),
            min: TaperValue::new(0),
            max: TaperValue::new(24),
            getter: delta_pruning_threshold,
            to_raw: |t: TaperValue| t.v(),
            from_raw: |v: &i32| TaperValue::new(*v),
        },
    },

    policy: PolicyParams {
        policy_temperature: f32 {
            uci: "eval-policy-temperature",
            unit: UciPercent,
            default: 20.0,
            min: 1.0,
            max: 100.0,
            getter: policy_temperature,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
    },

    puct: PuctParams {
        select_cpuct: f32 {
            uci: "select-cpuct",
            unit: UciPercent,
            default: 1.4,
            min: 0.01,
            max: 50.0,
            getter: select_cpuct,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
    },

    mcts: MctsParams {
        proven_loss_visit_threshold: VisitCount {
            uci: "mcts-proven-loss-visit-threshold",
            unit: UciInteger,
            default: VisitCount(5),
            min: VisitCount(1),
            max: VisitCount(100),
            getter: proven_loss_visit_threshold,
            to_raw: |v: VisitCount| v.0 as i32,
            from_raw: |v: &i32| { VisitCount(*v as u32) },
        },
        killer_exploitation: f32 {
            uci: "mcts-killer-exploitation",
            unit: UciPercent,
            default: 0.27,
            min: 0.0,
            max: 10.0,
            getter: killer_exploitation,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
        tt_best_move: f32 {
            uci: "mcts-tt-best-move",
            unit: UciPercent,
            default: 1.50,
            min: 0.0,
            max: 10.0,
            getter: tt_best_move,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
    },

    chrono: ChronoParams {
        base_soft_mult: f32 {
            uci: "timeman-base-soft-mult",
            unit: UciPercent,
            default: 0.50,
            min: 0.01,
            max: 2.00,
            getter: base_soft_mult,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
        clamp_lower: f32 {
            uci: "timeman-clamp-lower",
            unit: UciPercent,
            default: 0.30,
            min: 0.00,
            max: 1.00,
            getter: clamp_lower,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
        clamp_upper: f32 {
            uci: "timeman-clamp-upper",
            unit: UciPercent,
            default: 1.50,
            min: 0.10,
            max: 3.00,
            getter: clamp_upper,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
        movestreak_base: f32 {
            uci: "timeman-stability-base",
            unit: UciPercent,
            default: 1.00,
            min: 0.00,
            max: 2.00,
            getter: movestreak_base,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
        movestreak_slope: f32 {
            uci: "timeman-stability-slope",
            unit: UciPercent,
            default: 0.08,
            min: 0.00,
            max: 0.50,
            getter: movestreak_slope,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
        movestreak_floor: f32 {
            uci: "timeman-stability-floor",
            unit: UciPercent,
            default: 0.40,
            min: 0.00,
            max: 1.00,
            getter: movestreak_floor,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
        entropy_base: f32 {
            uci: "timeman-entropy-base",
            unit: UciPercent,
            default: 0.50,
            min: 0.00,
            max: 2.00,
            getter: entropy_base,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
        entropy_weight: f32 {
            uci: "timeman-entropy-weight",
            unit: UciPercent,
            default: 1.00,
            min: 0.00,
            max: 2.00,
            getter: entropy_weight,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
    },

    id: IdParams {
        nmp_reduction: Depth {
            uci: "id-nmp-reduction",
            unit: UciInteger,
            default: Depth::new(2),
            min: Depth::new(0),
            max: Depth::new(10),
            getter: nmp_reduction,
            to_raw: |d: Depth| d.v() as i32,
            from_raw: |v: &i32| Depth::new(*v as u8),
        },
        nmp_phase_threshold: TaperValue {
            uci: "id-nmp-phase-threshold",
            unit: UciInteger,
            default: TaperValue::new(8),
            min: TaperValue::new(0),
            max: TaperValue::new(24),
            getter: nmp_phase_threshold,
            to_raw: |t: TaperValue| t.v(),
            from_raw: |v: &i32| TaperValue::new(*v),
        },
        nmp_depth_factor: u8 {
            uci: "id-nmp-depth-factor",
            unit: UciInteger,
            default: 3,
            min: 1,
            max: 20,
            getter: nmp_depth_factor,
            to_raw: |v: u8| v as i32,
            from_raw: |v: &i32| *v as u8,
        },
        nmp_phase_factor: u32 {
            uci: "id-nmp-phase-factor",
            unit: UciInteger,
            default: 7,
            min: 1,
            max: 50,
            getter: nmp_phase_factor,
            to_raw: |v: u32| v as i32,
            from_raw: |v: &i32| *v as u32,
        },
        nmp_margin: AnyScore {
            uci: "id-nmp-margin",
            unit: UciInteger,
            default: AnyScore::new(50),
            min: AnyScore::new(-350),
            max: AnyScore::new(350),
            getter: nmp_margin,
            to_raw: |s: AnyScore| s.v(),
            from_raw: |v: &i32| AnyScore::new(*v),
        },
        nmp_depth_margin: i32 {
            uci: "id-nmp-depth-margin",
            unit: UciInteger,
            default: 15,
            min: 0,
            max: 100,
            getter: nmp_depth_margin,
            to_raw: |v: i32| v,
            from_raw: |v: &i32| *v,
        },
    },

    scorer: ScorerParams {
        hh_weight: i32 {
            uci: "id-scorer-hh-weight",
            unit: UciInteger,
            default: 64,
            min: 0,
            max: 128,
            getter: hh_weight,
            to_raw: |v: i32| v,
            from_raw: |v: &i32| *v,
        },
    },

    lmr: LmrParams {
        offset: f32 {
            uci: "lmr-offset",
            unit: UciPercent,
            default: 0.99,
            min: 0.0,
            max: 2.0,
            getter: offset,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
        scale: f32 {
            uci: "lmr-scale",
            unit: UciPercent,
            default: 3.14,
            min: 0.10,
            max: 10.0,
            getter: scale,
            to_raw: ratio_to_raw,
            from_raw: ratio_from_raw,
        },
    },
}

impl<Base: fmt::Debug> IParams for TunableParams<Base> {
    type Ref = TunableParamsRef<Base>;

    fn shared(self) -> Self::Ref { std::rc::Rc::new(self) }

    fn try_from_config<C: Deref<Target = Configuration>>(config: C) -> Result<Self::Ref, CreateTunableParamsError> {
        Ok(Self::from_config(config).shared())
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

const_params! {
    MctsHce {
        puct: PuctParams {
            fn select_cpuct(&self) -> f32 { 0.77 }
        },
        mcts: MctsParams {
            fn proven_loss_visit_threshold(&self) -> VisitCount { VisitCount(5) }
            fn killer_exploitation(&self) -> f32 { 0.27 }
            fn tt_best_move(&self) -> f32 { 1.65 }
        },
        qsearch: QSearchParams {
            fn futility_margin(&self) -> AnyScore { AnyScore::new(166) }
            fn delta_pruning_threshold(&self) -> TaperValue { TaperValue::new(16) }
        },
        policy: PolicyParams {
            fn policy_temperature(&self) -> f32 { 24.58 }
        },
        chrono: ChronoParams {
            fn base_soft_mult(&self) -> f32 { 0.50 }
            fn clamp_lower(&self) -> f32 { 0.30 }
            fn clamp_upper(&self) -> f32 { 1.50 }
            fn movestreak_base(&self) -> f32 { 1.00 }
            fn movestreak_slope(&self) -> f32 { 0.08 }
            fn movestreak_floor(&self) -> f32 { 0.40 }
            fn entropy_base(&self) -> f32 { 0.50 }
            fn entropy_weight(&self) -> f32 { 1.00 }
        },
    }
}

// mcts nn

const_params! {
    MctsNn {
        puct: PuctParams {
            fn select_cpuct(&self) -> f32 { 0.77 }
        },
        mcts: MctsParams {
            fn proven_loss_visit_threshold(&self) -> VisitCount { VisitCount(5) }
            fn killer_exploitation(&self) -> f32 { 0.27 }
            fn tt_best_move(&self) -> f32 { 1.65 }
        },
        policy: PolicyParams {
            fn policy_temperature(&self) -> f32 { 24.58 }
        },
        chrono: ChronoParams {
            fn base_soft_mult(&self) -> f32 { 0.50 }
            fn clamp_lower(&self) -> f32 { 0.30 }
            fn clamp_upper(&self) -> f32 { 1.50 }
            fn movestreak_base(&self) -> f32 { 1.00 }
            fn movestreak_slope(&self) -> f32 { 0.08 }
            fn movestreak_floor(&self) -> f32 { 0.40 }
            fn entropy_base(&self) -> f32 { 0.50 }
            fn entropy_weight(&self) -> f32 { 1.00 }
        },
    }
}

// mcts pure

const_params! {
    MctsPure {
        mcts: MctsParams {
            fn proven_loss_visit_threshold(&self) -> VisitCount { VisitCount(5) }
            fn killer_exploitation(&self) -> f32 { 0.27 }
            fn tt_best_move(&self) -> f32 { 1.65 }
        },
        chrono: ChronoParams {
            fn base_soft_mult(&self) -> f32 { 0.50 }
            fn clamp_lower(&self) -> f32 { 0.30 }
            fn clamp_upper(&self) -> f32 { 1.50 }
            fn movestreak_base(&self) -> f32 { 1.00 }
            fn movestreak_slope(&self) -> f32 { 0.08 }
            fn movestreak_floor(&self) -> f32 { 0.40 }
            fn entropy_base(&self) -> f32 { 0.50 }
            fn entropy_weight(&self) -> f32 { 1.00 }
        },
    }
}

// id hce

const_params! {
    IdHce {
        chrono: ChronoParams {
            fn base_soft_mult(&self) -> f32 { 0.50 }
            fn clamp_lower(&self) -> f32 { 0.30 }
            fn clamp_upper(&self) -> f32 { 1.50 }
            fn movestreak_base(&self) -> f32 { 1.00 }
            fn movestreak_slope(&self) -> f32 { 0.08 }
            fn movestreak_floor(&self) -> f32 { 0.40 }
            fn entropy_base(&self) -> f32 { 0.50 }
            fn entropy_weight(&self) -> f32 { 1.00 }
        },
        qsearch: QSearchParams {
            fn futility_margin(&self) -> AnyScore { AnyScore::new(166) }
            fn delta_pruning_threshold(&self) -> TaperValue { TaperValue::new(16) }
        },
        id: IdParams {
            fn nmp_reduction(&self) -> Depth { Depth::new(2) }
            fn nmp_phase_threshold(&self) -> TaperValue { TaperValue::new(12) }
            fn nmp_depth_factor(&self) -> u8 { 3 }
            fn nmp_phase_factor(&self) -> u32 { 7 }
            fn nmp_margin(&self) -> AnyScore { AnyScore::new(48) }
            fn nmp_depth_margin(&self) -> i32 { 15 }
        },
        scorer: ScorerParams {
            fn hh_weight(&self) -> i32 { 64 }
        },
        lmr: LmrParams {
            fn offset(&self) -> f32 { 0.99 }
            fn scale(&self) -> f32 { 3.14 }
        },
    }
}

// id nnue

const_params! {
    IdNnue {
        chrono: ChronoParams {
            fn base_soft_mult(&self) -> f32 { 0.48 }
            fn clamp_lower(&self) -> f32 { 0.34 }
            fn clamp_upper(&self) -> f32 { 1.51 }
            fn movestreak_base(&self) -> f32 { 1.00 }
            fn movestreak_slope(&self) -> f32 { 0.08 }
            fn movestreak_floor(&self) -> f32 { 0.40 }
            fn entropy_base(&self) -> f32 { 0.50 }
            fn entropy_weight(&self) -> f32 { 1.00 }
        },
        qsearch: QSearchParams {
            fn futility_margin(&self) -> AnyScore { AnyScore::new(177) }
            fn delta_pruning_threshold(&self) -> TaperValue { TaperValue::new(2) }
        },
        id: IdParams {
            fn nmp_reduction(&self) -> Depth { Depth::new(2) }
            fn nmp_phase_threshold(&self) -> TaperValue { TaperValue::new(11) }
            fn nmp_depth_factor(&self) -> u8 { 4 }
            fn nmp_phase_factor(&self) -> u32 { 7 }
            fn nmp_margin(&self) -> AnyScore { AnyScore::new(50) }
            fn nmp_depth_margin(&self) -> i32 { 12 }
        },
        scorer: ScorerParams {
            fn hh_weight(&self) -> i32 { 100 }
        },
        lmr: LmrParams {
            fn offset(&self) -> f32 { 0.99 }
            fn scale(&self) -> f32 { 3.14 }
        },
    }
}
