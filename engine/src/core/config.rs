use crate::{
    core::{
        chrono::ChronoParams,
        depth::Depth,
        eval::hce::TaperValue,
        search::{
            id::{IdParams, ScorerParams},
            mcts::{eval::hce::PolicyParams, node::VisitCount, search::MctsParams, select::puct::PuctParams},
            quiesce::QSearchParams,
            score::AnyScore,
        },
    },
    misc::{InvalidValueError, ValueOutOfRangeError},
    math::LmrParams,
};
use std::{
    error::Error,
    fmt,
    ops::{Deref, DerefMut},
    str::FromStr,
};
use thiserror::Error;
use uom::si::{
    f32::Ratio,
    information,
    ratio::{percent, ratio},
    time::millisecond,
    u64::{Information, Time},
};

pub trait UciUnit {
    type Quantity;
    type Raw: FromStr + fmt::Display + PartialOrd + Copy;

    fn to_quantity(raw: Self::Raw) -> Self::Quantity;
    fn to_raw(qty: &Self::Quantity) -> Self::Raw;
}

#[derive(Debug, Clone)]
pub struct UciPercent;
impl UciUnit for UciPercent {
    type Quantity = Ratio;
    type Raw = i32;

    fn to_quantity(raw: Self::Raw) -> Self::Quantity { Ratio::new::<percent>(raw as f32) }
    fn to_raw(qty: &Self::Quantity) -> Self::Raw { qty.get::<percent>().round() as i32 }
}

#[derive(Debug, Clone)]
pub struct UciMebibyte;
impl UciUnit for UciMebibyte {
    type Quantity = Information;
    type Raw = u64;

    fn to_quantity(raw: Self::Raw) -> Self::Quantity { Information::new::<information::mebibyte>(raw) }
    fn to_raw(qty: &Self::Quantity) -> Self::Raw { qty.get::<information::mebibyte>() }
}

#[derive(Debug, Clone)]
pub struct UciInteger;
impl UciUnit for UciInteger {
    type Quantity = i32;
    type Raw = i32;

    fn to_quantity(raw: Self::Raw) -> Self::Quantity { raw }
    fn to_raw(qty: &Self::Quantity) -> Self::Raw { *qty }
}

#[derive(Debug, Clone)]
pub struct UciMillis;
impl UciUnit for UciMillis {
    type Quantity = Time;
    type Raw = u64;

    fn to_quantity(raw: Self::Raw) -> Self::Quantity { Time::new::<millisecond>(raw) }
    fn to_raw(qty: &Self::Quantity) -> Self::Raw { qty.get::<millisecond>() }
}

#[derive(Debug, Error)]
#[error("Unknown option: {0}")]
pub struct UnknownOptionError(pub String);

#[derive(Clone, Debug)]
pub struct ConfigOption<T> {
    pub name: String,
    pub inner: T,
}

impl<T> ConfigOption<T> {
    pub fn new(name: &str, inner: T) -> Self { Self { name: name.to_string(), inner } }
}

impl<U: UciUnit> ConfigOption<Spin<U>>
where
    U::Quantity: Copy,
{
    fn seed(&mut self, value: U::Quantity) {
        self.inner.value = value;
        self.inner.default = value;
    }
}

impl<T> Deref for ConfigOption<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target { &self.inner }
}

impl<T> DerefMut for ConfigOption<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
}

impl<T: fmt::Display> fmt::Display for ConfigOption<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "option name {} type {}", self.name, self.inner) }
}

#[derive(Clone, Debug)]
pub struct Spin<U: UciUnit> {
    pub value: U::Quantity,
    pub default: U::Quantity,
    pub min: U::Quantity,
    pub max: U::Quantity,
}

impl<U: UciUnit> Spin<U>
where
    U::Quantity: Copy,
    U::Raw: fmt::Debug + 'static,
{
    pub fn new(default: U::Quantity, min: U::Quantity, max: U::Quantity) -> Self {
        Self {
            value: default,
            default,
            min,
            max,
        }
    }

    pub fn set(&mut self, value_str: &str) -> Result<(), Box<dyn Error>> {
        let val = value_str.parse::<U::Raw>().map_err(|_| InvalidValueError::new(value_str.to_string()))?;

        let min_raw = U::to_raw(&self.min);
        let max_raw = U::to_raw(&self.max);

        if val < min_raw || val > max_raw {
            return Err(Box::new(ValueOutOfRangeError::new(val, min_raw..=max_raw)));
        }
        self.value = U::to_quantity(val);
        Ok(())
    }
}

impl<U: UciUnit> fmt::Display for Spin<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "spin default {} min {} max {}",
            U::to_raw(&self.default),
            U::to_raw(&self.min),
            U::to_raw(&self.max)
        )
    }
}

#[derive(Clone, Debug)]
pub struct Check {
    pub value: bool,
    pub default: bool,
}

impl Check {
    pub fn new(default: bool) -> Self { Self { value: default, default } }

    pub fn set(&mut self, value_str: &str) -> Result<(), Box<dyn Error>> {
        self.value = value_str.parse::<bool>().map_err(|_| InvalidValueError::new(value_str.to_string()))?;
        Ok(())
    }
}

impl fmt::Display for Check {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "check default {}", self.default) }
}

#[derive(Clone, Debug)]
pub struct StringOption {
    pub value: String,
    pub default: String,
}

impl StringOption {
    pub fn new(default: &str) -> Self {
        Self {
            value: default.to_string(),
            default: default.to_string(),
        }
    }

    pub fn set(&mut self, value_str: &str) { self.value = value_str.to_string(); }
}

impl fmt::Display for StringOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "string default {}", self.default) }
}

#[derive(Clone, Debug)]
pub struct Combo {
    pub value: String,
    pub default: String,
    pub options: Vec<String>,
}

impl Combo {
    pub fn new(default: &str, options: Vec<&str>) -> Self {
        Self {
            value: default.to_string(),
            default: default.to_string(),
            options: options.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn set(&mut self, value_str: &str) -> Result<(), Box<dyn Error>> {
        if self.options.iter().any(|opt| opt == value_str) {
            self.value = value_str.to_string();
            Ok(())
        }
        else {
            Err(Box::new(InvalidValueError::new(value_str.to_string())))
        }
    }
}

impl fmt::Display for Combo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "combo default {}", self.default)?;
        for opt in &self.options {
            write!(f, " var {}", opt)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Button {
    pub callback: fn(),
}

impl Button {
    pub fn new(callback: fn()) -> Self { Self { callback } }

    pub fn trigger(&self) { (self.callback)(); }
}

impl fmt::Display for Button {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "button") }
}

impl fmt::Debug for Button {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Button") }
}

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct Configuration {
    /// # [UCI] Hash size.
    /// the value in MB for memory for hash tables can be changed, this should
    /// be answered with the first "setoptions" command at program boot if the
    /// engine has sent the appropriate "option name Hash" command, which should
    /// be supported by all engines! So the engine should use a very small hash
    /// first as default.
    uci_hash: ConfigOption<Spin<UciMebibyte>>,

    /// # [UCI] Path to Nalimov tablebases.
    /// this is the path on the hard disk to the Nalimov compressed format.
    /// Multiple directories can be concatenated with ";"
    uci_nalimov_path: ConfigOption<StringOption>,

    /// # [UCI] Size of Nalimov tablebase cache.
    /// this is the size in MB for the cache for the nalimov table bases
    /// These last two options should also be present in the initial options
    /// exchange dialog when the engine is booted if the engine supports it
    uci_nalimov_cache: ConfigOption<Spin<UciMebibyte>>,

    /// # [UCI] Ponder Indication
    /// this means that the engine is able to ponder.
    /// The GUI will send this whenever pondering is possible or not.
    /// Note: The engine should not start pondering on its own if this is
    /// enabled, this option is only needed because the engine might change
    /// its time management algorithm when pondering is allowed.
    uci_ponder: ConfigOption<Check>,

    /// # [UCI] Own Book Indication
    /// this means that the engine has its own book which is accessed by the
    /// engine itself. if this is set, the engine takes care of the opening
    /// book and the GUI will never execute a move out of its book for the
    /// engine. If this is set to false by the GUI, the engine should not
    /// access its own book.
    uci_ownbook: ConfigOption<Check>,

    /// # [UCI] MultiPV
    /// the engine supports multi best line or k-best mode. the default value is
    /// 1
    uci_multipv: ConfigOption<Spin<UciInteger>>,

    /// # [UCI] Show current line
    /// the engine can show the current line it is calculating. see "info
    /// currline" above.
    uci_show_currline: ConfigOption<Check>,

    /// # [UCI] Show refutations
    /// the engine can show a move and its refutation in a line. see "info
    /// refutations" above.
    uci_show_refutations: ConfigOption<Check>,

    /// # [UCI] Limit strength
    /// The engine is able to limit its strength to a specific Elo number.
    /// Should always be implemented together with "UCI_Elo".
    uci_limit_strength: ConfigOption<Check>,

    /// # [UCI] Elo strength limit
    /// The engine can limit its strength in Elo within this interval.
    /// Only active when UCI_LimitStrength is true. Should always be
    /// implemented together with "UCI_LimitStrength".
    uci_elo: ConfigOption<Spin<UciInteger>>,

    /// # [UCI] Analyse mode
    /// The engine wants to behave differently when analysing or playing a game.
    /// Set to false when playing a game, true when analysing.
    uci_analyse_mode: ConfigOption<Check>,

    /// # [UCI] Opponent info
    /// The GUI can send the name, title, elo and if the engine is playing a
    /// human or computer to the engine.
    /// Format: [GM|IM|FM|WGM|WIM|none] [<elo>|none] [computer|human] <name>
    uci_opponent: ConfigOption<StringOption>,

    /// # [UCI] Engine about
    /// The engine tells the GUI information about itself, e.g. a license text.
    uci_engine_about: ConfigOption<StringOption>,

    /// # [UCI] Shredder bases path
    /// Path to the folder containing the Shredder endgame databases, or the
    /// path and filename of one Shredder endgame database.
    uci_shredder_bases_path: ConfigOption<StringOption>,

    /// # [UCI] Set position value
    /// The GUI can send this to tell the engine to use a certain value in
    /// centipawns from white's point of view for a specific position.
    /// Formats: <value> + <fen> | clear + <fen> | clearall
    uci_set_position_value: ConfigOption<StringOption>,

    /// Num threads.
    threads: ConfigOption<Spin<UciInteger>>,

    /// Dirichlet noise - alpha parameter.
    dirichlet_alpha: ConfigOption<Spin<UciPercent>>,

    /// Dirichlet noise - epsilon parameter.
    dirichlet_epsilon: ConfigOption<Spin<UciPercent>>,

    /// Path to nn weights file.
    weights_path: ConfigOption<StringOption>,

    /// Path to quantized nnue weights file. If empty, uses shipped nnue.
    nnue_path: ConfigOption<StringOption>,

    /// Whether to keep the game tree in between `go`-commands.
    game_tree_caching: ConfigOption<Check>,

    /// Assumed lag between the GUI starting the engine's clock, the engine
    /// receiving the go command, and the engine actually starting the
    /// search.
    gui_lag: ConfigOption<Spin<UciMillis>>,

    /// Whether to enable ponder mode (i.e., keep searching on the opponent's
    /// time until the opponent actually moves).
    ponder: ConfigOption<Check>,

    /// Evaluation policy temperature.
    eval_policy_temperature: ConfigOption<Spin<UciPercent>>,

    /// Margin for futility pruning. In centipawns
    qs_futility_margin: ConfigOption<Spin<UciInteger>>,

    /// Margin for delta pruning. A tapervalue.
    qs_delta_pruning_threshold: ConfigOption<Spin<UciInteger>>,

    /// Cpuct constant for selection.
    select_cpuct: ConfigOption<Spin<UciPercent>>,

    /// Visit threshold for proven loss cutoff in mcts.
    mcts_proven_loss_visit_threshold: ConfigOption<Spin<UciInteger>>,

    /// Killer exploitation factor for mcts.
    mcts_killer_exploitation: ConfigOption<Spin<UciPercent>>,

    /// Bonus for tt best move in mcts.
    mcts_tt_best_move: ConfigOption<Spin<UciPercent>>,

    // Chrono params.
    timeman_base_soft_mult: ConfigOption<Spin<UciPercent>>,
    timeman_clamp_lower: ConfigOption<Spin<UciPercent>>,
    timeman_clamp_upper: ConfigOption<Spin<UciPercent>>,
    timeman_stability_base: ConfigOption<Spin<UciPercent>>,
    timeman_stability_slope: ConfigOption<Spin<UciPercent>>,
    timeman_stability_floor: ConfigOption<Spin<UciPercent>>,
    timeman_entropy_base: ConfigOption<Spin<UciPercent>>,
    timeman_entropy_weight: ConfigOption<Spin<UciPercent>>,

    /// [Iterative Deepening] Null Move Pruning reduction (R).
    id_nmp_reduction: ConfigOption<Spin<UciInteger>>,

    /// [Iterative Deepening] Null Move Pruning phase threshold.
    id_nmp_phase_threshold: ConfigOption<Spin<UciInteger>>,

    /// [Iterative Deepening] Null Move Pruning depth factor.
    id_nmp_depth_factor: ConfigOption<Spin<UciInteger>>,

    /// [Iterative Deepening] Null Move Pruning phase factor.
    id_nmp_phase_factor: ConfigOption<Spin<UciInteger>>,

    /// [Iterative Deepening] Null Move Pruning margin.
    id_nmp_margin: ConfigOption<Spin<UciInteger>>,

    /// [Iterative Deepening] Null Move Pruning depth margin.
    id_nmp_depth_margin: ConfigOption<Spin<UciInteger>>,

    /// [Iterative Deepening] Scorer hyper-heuristic weight.
    id_scorer_hh_weight: ConfigOption<Spin<UciInteger>>,

    /// [Late Move Reductions] Base offset applied to the reduction.
    lmr_offset: ConfigOption<Spin<UciPercent>>,

    /// [Late Move Reductions] Divisor scaling the log-log reduction term.
    lmr_scale: ConfigOption<Spin<UciPercent>>,
}

impl Configuration {
    /// Start building a [`Configuration`].
    ///
    /// The returned [`ConfigBuilder`] is seeded with baseline defaults for
    /// every option. Use the per-trait setters (e.g.
    /// [`ConfigBuilder::qsearch`], [`ConfigBuilder::mcts`]) to override
    /// only the option groups a given search algorithm actually needs, so a
    /// params type only has to implement the param traits for the options
    /// it cares about.
    #[rustfmt::skip]
    #[allow(clippy::unit_arg)]
    pub fn builder() -> ConfigBuilder {
        fn _mebibyte(v: u64) -> Information { Information::new::<information::mebibyte>(v) }
        fn _ratio(v: f32) -> Ratio { Ratio::new::<ratio>(v) }
        fn _millis(v: u64) -> Time { Time::new::<millisecond>(v) }

        ConfigBuilder {
            config: Self {
                uci_hash: ConfigOption::new("Hash", Spin::<UciMebibyte>::new(_mebibyte(16), _mebibyte(1), _mebibyte(64 * 1024 * 1024))),
                uci_nalimov_path: ConfigOption::new("NalimovPath", StringOption::new("")),
                uci_nalimov_cache : ConfigOption::new("NalimovCache", Spin::<UciMebibyte>::new(_mebibyte(16), _mebibyte(1), _mebibyte(64 * 1024 * 1024))),
                uci_ponder: ConfigOption::new("Ponder", Check::new(false)),
                uci_ownbook: ConfigOption::new("OwnBook", Check::new(false)),
                uci_multipv: ConfigOption::new("MultiPV", Spin::<UciInteger>::new(1, 1, 500)),
                uci_show_currline: ConfigOption::new("UCI_ShowCurrLine", Check::new(false)),
                uci_show_refutations: ConfigOption::new("UCI_ShowRefutations", Check::new(false)),
                uci_limit_strength: ConfigOption::new("UCI_LimitStrength", Check::new(false)),
                uci_elo: ConfigOption::new("UCI_Elo", Spin::<UciInteger>::new(1320, 1320, 3190)),
                uci_analyse_mode: ConfigOption::new("UCI_AnalyseMode", Check::new(false)),
                uci_opponent: ConfigOption::new("UCI_Opponent", StringOption::new("")),
                uci_engine_about: ConfigOption::new("UCI_EngineAbout", StringOption::new("")),
                uci_shredder_bases_path: ConfigOption::new("UCI_ShredderbasesPath", StringOption::new("")),
                uci_set_position_value: ConfigOption::new("UCI_SetPositionValue", StringOption::new("")),
                threads: ConfigOption::new("threads", Spin::new(1, 1, 1)),
                dirichlet_alpha: ConfigOption::new("dirichlet-alpha", Spin::<UciPercent>::new(_ratio(0.3), _ratio(0.), _ratio(10.))),
                dirichlet_epsilon: ConfigOption::new("dirichlet-epsilon", Spin::<UciPercent>::new(_ratio(0.25), _ratio(0.), _ratio(1.))),
                weights_path: ConfigOption::new("weights-path", StringOption::new("./weights")),
                nnue_path: ConfigOption::new("nnue-path", StringOption::new("")),
                game_tree_caching: ConfigOption::new("game-tree-caching", Check::new(true)),
                gui_lag: ConfigOption::new("gui-lag", Spin::<UciMillis>::new(_millis(100), _millis(1), _millis(10_000))),
                ponder: ConfigOption::new("ponder", Check::new(true)),
                eval_policy_temperature: ConfigOption::new("eval-policy-temperature", Spin::<UciPercent>::new(_ratio(20.), _ratio(1.), _ratio(100.))),
                qs_futility_margin: ConfigOption::new("qs-futility-margin", Spin::new(150, 100, 300)),
                qs_delta_pruning_threshold: ConfigOption::new("qs-delta-pruning-threshold", Spin::new(16, 0, 24)),
                select_cpuct: ConfigOption::new("select-cpuct", Spin::<UciPercent>::new(_ratio(1.4), _ratio(0.01), _ratio(50.))),
                mcts_proven_loss_visit_threshold: ConfigOption::new("mcts-proven-loss-visit-threshold", Spin::new(5, 1, 100)),
                mcts_killer_exploitation: ConfigOption::new("mcts-killer-exploitation", Spin::<UciPercent>::new(_ratio(0.27), _ratio(0.), _ratio(10.))),
                mcts_tt_best_move: ConfigOption::new("mcts-tt-best-move", Spin::<UciPercent>::new(_ratio(1.50), _ratio(0.), _ratio(10.))),
                timeman_base_soft_mult: ConfigOption::new("timeman-base-soft-mult", Spin::<UciPercent>::new(_ratio(0.50), _ratio(0.01), _ratio(2.))),
                timeman_clamp_lower: ConfigOption::new("timeman-clamp-lower", Spin::<UciPercent>::new(_ratio(0.30), _ratio(0.), _ratio(1.))),
                timeman_clamp_upper: ConfigOption::new("timeman-clamp-upper", Spin::<UciPercent>::new(_ratio(1.50), _ratio(0.10), _ratio(3.))),
                timeman_stability_base: ConfigOption::new("timeman-stability-base", Spin::<UciPercent>::new(_ratio(1.00), _ratio(0.), _ratio(2.))),
                timeman_stability_slope: ConfigOption::new("timeman-stability-slope", Spin::<UciPercent>::new(_ratio(0.08), _ratio(0.), _ratio(0.50))),
                timeman_stability_floor: ConfigOption::new("timeman-stability-floor", Spin::<UciPercent>::new(_ratio(0.40), _ratio(0.), _ratio(1.))),
                timeman_entropy_base: ConfigOption::new("timeman-entropy-base", Spin::<UciPercent>::new(_ratio(0.50), _ratio(0.), _ratio(2.))),
                timeman_entropy_weight: ConfigOption::new("timeman-entropy-weight", Spin::<UciPercent>::new(_ratio(1.00), _ratio(0.), _ratio(2.))),
                id_nmp_reduction: ConfigOption::new("id-nmp-reduction", Spin::new(2, 0, 10)),
                id_nmp_phase_threshold: ConfigOption::new("id-nmp-phase-threshold", Spin::new(8, 0, 24)),
                id_nmp_depth_factor: ConfigOption::new("id-nmp-depth-factor", Spin::new(3, 1, 20)),
                id_nmp_phase_factor: ConfigOption::new("id-nmp-phase-factor", Spin::new(7, 1, 50)),
                id_nmp_margin: ConfigOption::new("id-nmp-margin", Spin::new(50, -350, 350)),
                id_nmp_depth_margin: ConfigOption::new("id-nmp-depth-margin", Spin::new(15, 0, 100)),
                id_scorer_hh_weight: ConfigOption::new("id-scorer-hh-weight", Spin::new(64, 0, 128)),
                lmr_offset: ConfigOption::new("lmr-offset", Spin::<UciPercent>::new(_ratio(0.99), _ratio(0.), _ratio(2.))),
                lmr_scale: ConfigOption::new("lmr-scale", Spin::<UciPercent>::new(_ratio(3.14), _ratio(0.10), _ratio(10.)))
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConfigBuilder {
    config: Configuration,
}

impl ConfigBuilder {
    /// Seed the quiescence-search options from [`QSearchParams`].
    #[rustfmt::skip]
    pub fn qsearch(mut self, params: &impl QSearchParams) -> Self {
        let cfg = &mut self.config;
        cfg.qs_futility_margin.seed(params.futility_margin().v());
        cfg.qs_delta_pruning_threshold.seed(params.delta_pruning_threshold().v());
        self
    }

    /// Seed the policy options from [`PolicyParams`].
    #[rustfmt::skip]
    pub fn policy(mut self, params: &impl PolicyParams) -> Self {
        let cfg = &mut self.config;
        cfg.eval_policy_temperature.seed(Ratio::new::<ratio>(params.policy_temperature()));
        self
    }

    /// Seed the selection options from [`PuctParams`].
    #[rustfmt::skip]
    pub fn puct(mut self, params: &impl PuctParams) -> Self {
        let cfg = &mut self.config;
        cfg.select_cpuct.seed(Ratio::new::<ratio>(params.select_cpuct()));
        self
    }

    /// Seed the mcts options from [`MctsParams`].
    #[rustfmt::skip]
    pub fn mcts(mut self, params: &impl MctsParams) -> Self {
        let cfg = &mut self.config;
        cfg.mcts_proven_loss_visit_threshold.seed(params.proven_loss_visit_threshold().0 as i32);
        cfg.mcts_killer_exploitation.seed(Ratio::new::<ratio>(params.killer_exploitation()));
        cfg.mcts_tt_best_move.seed(Ratio::new::<ratio>(params.tt_best_move()));
        self
    }

    /// Seed the time-management options from [`ChronoParams`].
    #[rustfmt::skip]
    pub fn chrono(mut self, params: &impl ChronoParams) -> Self {
        let cfg = &mut self.config;
        cfg.timeman_base_soft_mult.seed(Ratio::new::<ratio>(params.base_soft_mult()));
        cfg.timeman_clamp_lower.seed(Ratio::new::<ratio>(params.clamp_lower()));
        cfg.timeman_clamp_upper.seed(Ratio::new::<ratio>(params.clamp_upper()));
        cfg.timeman_stability_base.seed(Ratio::new::<ratio>(params.movestreak_base()));
        cfg.timeman_stability_slope.seed(Ratio::new::<ratio>(params.movestreak_slope()));
        cfg.timeman_stability_floor.seed(Ratio::new::<ratio>(params.movestreak_floor()));
        cfg.timeman_entropy_base.seed(Ratio::new::<ratio>(params.entropy_base()));
        cfg.timeman_entropy_weight.seed(Ratio::new::<ratio>(params.entropy_weight()));
        self
    }

    /// Seed the iterative-deepening options from [`IdParams`].
    #[rustfmt::skip]
    pub fn id(mut self, params: &impl IdParams) -> Self {
        let cfg = &mut self.config;
        cfg.id_nmp_reduction.seed(params.nmp_reduction().v() as i32);
        cfg.id_nmp_phase_threshold.seed(params.nmp_phase_threshold().v());
        cfg.id_nmp_depth_factor.seed(params.nmp_depth_factor() as i32);
        cfg.id_nmp_phase_factor.seed(params.nmp_phase_factor() as i32);
        cfg.id_nmp_margin.seed(params.nmp_margin().v());
        cfg.id_nmp_depth_margin.seed(params.nmp_depth_margin());
        self
    }

    /// Seed the iterative-deepening scorer options from [`ScorerParams`].
    #[rustfmt::skip]
    pub fn scorer(mut self, params: &impl ScorerParams) -> Self {
        let cfg = &mut self.config;
        cfg.id_scorer_hh_weight.seed(params.hh_weight());
        self
    }

    /// Seed the late-move-reduction options from [`LmrParams`].
    #[rustfmt::skip]
    pub fn lmr(mut self, params: &impl LmrParams) -> Self {
        let cfg = &mut self.config;
        cfg.lmr_offset.seed(Ratio::new::<ratio>(params.offset()));
        cfg.lmr_scale.seed(Ratio::new::<ratio>(params.scale()));
        self
    }

    /// Finish building the [`Configuration`].
    pub fn build(self) -> Configuration { self.config }
}

// #[cfg(feature = "tunable")]
impl Configuration {
    pub fn eval_policy_temperature(&self) -> f32 { self.eval_policy_temperature.value.get::<ratio>() }
    pub fn eval_futility_margin(&self) -> AnyScore { AnyScore::new(self.qs_futility_margin.value) }
    pub fn eval_delta_pruning_threshold(&self) -> TaperValue { TaperValue::new(self.qs_delta_pruning_threshold.value) }
    pub fn select_cpuct(&self) -> f32 { self.select_cpuct.value.get::<ratio>() }
    pub fn mcts_proven_loss_visit_threshold(&self) -> VisitCount { VisitCount(self.mcts_proven_loss_visit_threshold.value as u32) }
    pub fn mcts_killer_exploitation(&self) -> f32 { self.mcts_killer_exploitation.value.get::<ratio>() }
    pub fn mcts_tt_best_move(&self) -> f32 { self.mcts_tt_best_move.value.get::<ratio>() }
    pub fn timeman_base_soft_mult(&self) -> f32 { self.timeman_base_soft_mult.value.get::<ratio>() }
    pub fn timeman_clamp_lower(&self) -> f32 { self.timeman_clamp_lower.value.get::<ratio>() }
    pub fn timeman_clamp_upper(&self) -> f32 { self.timeman_clamp_upper.value.get::<ratio>() }
    pub fn timeman_stability_base(&self) -> f32 { self.timeman_stability_base.value.get::<ratio>() }
    pub fn timeman_stability_slope(&self) -> f32 { self.timeman_stability_slope.value.get::<ratio>() }
    pub fn timeman_stability_floor(&self) -> f32 { self.timeman_stability_floor.value.get::<ratio>() }
    pub fn timeman_entropy_base(&self) -> f32 { self.timeman_entropy_base.value.get::<ratio>() }
    pub fn timeman_entropy_weight(&self) -> f32 { self.timeman_entropy_weight.value.get::<ratio>() }
    pub fn id_nmp_reduction(&self) -> Depth { Depth::new(self.id_nmp_reduction.value as u8) }
    pub fn id_nmp_phase_threshold(&self) -> TaperValue { TaperValue::new(self.id_nmp_phase_threshold.value) }
    pub fn id_nmp_depth_factor(&self) -> u8 { self.id_nmp_depth_factor.value as u8 }
    pub fn id_nmp_phase_factor(&self) -> u32 { self.id_nmp_phase_factor.value as u32 }
    pub fn id_nmp_margin(&self) -> AnyScore { AnyScore::new(self.id_nmp_margin.value) }
    pub fn id_nmp_depth_margin(&self) -> i32 { self.id_nmp_depth_margin.value }
    pub fn id_scorer_hh_weight(&self) -> i32 { self.id_scorer_hh_weight.value }
    pub fn lmr_offset(&self) -> f32 { self.lmr_offset.value.get::<ratio>() }
    pub fn lmr_scale(&self) -> f32 { self.lmr_scale.value.get::<ratio>() }
}

impl Configuration {
    pub fn uci_hash(&self) -> Information { self.uci_hash.value }
    pub fn uci_nalimov_path(&self) -> &str { &self.uci_nalimov_path.value }
    pub fn uci_nalimov_cache(&self) -> Information { self.uci_nalimov_cache.value }
    pub fn uci_ponder(&self) -> bool { self.uci_ponder.value }
    pub fn uci_ownbook(&self) -> bool { self.uci_ownbook.value }
    pub fn uci_multipv(&self) -> i32 { self.uci_multipv.value }
    pub fn uci_show_currline(&self) -> bool { self.uci_show_currline.value }
    pub fn uci_show_refutations(&self) -> bool { self.uci_show_refutations.value }
    pub fn uci_limit_strength(&self) -> bool { self.uci_limit_strength.value }
    pub fn uci_elo(&self) -> i32 { self.uci_elo.value }
    pub fn uci_analyse_mode(&self) -> bool { self.uci_analyse_mode.value }
    pub fn uci_opponent(&self) -> &str { &self.uci_opponent.value }
    pub fn uci_engine_about(&self) -> &str { &self.uci_engine_about.value }
    pub fn uci_shredder_bases_path(&self) -> &str { &self.uci_shredder_bases_path.value }
    pub fn uci_set_position_value(&self) -> &str { &self.uci_set_position_value.value }
    pub fn threads(&self) -> i32 { self.threads.value }
    pub fn dirichlet_alpha(&self) -> f32 { self.dirichlet_alpha.value.get::<ratio>() }
    pub fn dirichlet_epsilon(&self) -> f32 { self.dirichlet_epsilon.value.get::<ratio>() }
    pub fn weights_path(&self) -> &str { &self.weights_path.value }
    pub fn nnue_path(&self) -> &str { &self.nnue_path.value }
    pub fn game_tree_caching(&self) -> bool { self.game_tree_caching.value }
    pub fn gui_lag(&self) -> u16 { self.gui_lag.value.get::<millisecond>() as u16 }
    pub fn ponder(&self) -> bool { self.ponder.value }

    #[rustfmt::skip]
    #[allow(clippy::unit_arg)]
    pub fn set(&mut self, name: &str, value: &str) -> Result<(), Box<dyn Error>> {
        match name.to_lowercase().as_str() {
            "dirichlet-alpha" => self.dirichlet_alpha.set(value),
            "dirichlet-epsilon" => self.dirichlet_epsilon.set(value),
            "game-tree-caching" => self.game_tree_caching.set(value),
            "gui-lag" => self.gui_lag.set(value),
            "hash" => self.uci_hash.set(value),
            "nalimovpath" => Ok(self.uci_nalimov_path.set(value)),
            "nalimovcache" => self.uci_nalimov_cache.set(value),
            "ownbook" => self.uci_ownbook.set(value),
            "multipv" => self.uci_multipv.set(value),
            "uci_showcurrline" => self.uci_show_currline.set(value),
            "uci_showrefutations" => self.uci_show_refutations.set(value),
            "uci_limitstrength" => self.uci_limit_strength.set(value),
            "uci_elo" => self.uci_elo.set(value),
            "uci_analysemode" => self.uci_analyse_mode.set(value),
            "uci_opponent" => Ok(self.uci_opponent.set(value)),
            "uci_engineabout" => Ok(self.uci_engine_about.set(value)),
            "uci_shredderbasespath" => Ok(self.uci_shredder_bases_path.set(value)),
            "uci_setpositionvalue" => Ok(self.uci_set_position_value.set(value)),
            "ponder" => self.ponder.set(value),
            "threads" => self.threads.set(value),
            "weights-path" => Ok(self.weights_path.set(value)),
            "nnue-path" => Ok(self.nnue_path.set(value)),
            #[cfg(feature = "tunable")] "qs-delta-pruning-threshold" => self.qs_delta_pruning_threshold.set(value),
            #[cfg(feature = "tunable")] "qs-futility-margin" => self.qs_futility_margin.set(value),
            #[cfg(feature = "tunable")] "eval-policy-temperature" => self.eval_policy_temperature.set(value),
            #[cfg(feature = "tunable")] "mcts-killer-exploitation" => self.mcts_killer_exploitation.set(value),
            #[cfg(feature = "tunable")] "mcts-proven-loss-visit-threshold" => self.mcts_proven_loss_visit_threshold.set(value),
            #[cfg(feature = "tunable")] "mcts-tt-best-move" => self.mcts_tt_best_move.set(value),
            #[cfg(feature = "tunable")] "select-cpuct" => self.select_cpuct.set(value),
            #[cfg(feature = "tunable")] "timeman-base-soft-mult" => self.timeman_base_soft_mult.set(value),
            #[cfg(feature = "tunable")] "timeman-clamp-lower" => self.timeman_clamp_lower.set(value),
            #[cfg(feature = "tunable")] "timeman-clamp-upper" => self.timeman_clamp_upper.set(value),
            #[cfg(feature = "tunable")] "timeman-stability-base" => self.timeman_stability_base.set(value),
            #[cfg(feature = "tunable")] "timeman-stability-slope" => self.timeman_stability_slope.set(value),
            #[cfg(feature = "tunable")] "timeman-stability-floor" => self.timeman_stability_floor.set(value),
            #[cfg(feature = "tunable")] "timeman-entropy-base" => self.timeman_entropy_base.set(value),
            #[cfg(feature = "tunable")] "timeman-entropy-weight" => self.timeman_entropy_weight.set(value),
            #[cfg(feature = "tunable")] "id-nmp-reduction" => self.id_nmp_reduction.set(value),
            #[cfg(feature = "tunable")] "id-nmp-phase-threshold" => self.id_nmp_phase_threshold.set(value),
            #[cfg(feature = "tunable")] "id-nmp-depth-factor" => self.id_nmp_depth_factor.set(value),
            #[cfg(feature = "tunable")] "id-nmp-phase-factor" => self.id_nmp_phase_factor.set(value),
            #[cfg(feature = "tunable")] "id-nmp-margin" => self.id_nmp_margin.set(value),
            #[cfg(feature = "tunable")] "id-nmp-depth-margin" => self.id_nmp_depth_margin.set(value),
            #[cfg(feature = "tunable")] "id-scorer-hh-weight" => self.id_scorer_hh_weight.set(value),
            #[cfg(feature = "tunable")] "lmr-offset" => self.lmr_offset.set(value),
            #[cfg(feature = "tunable")] "lmr-scale" => self.lmr_scale.set(value),
            _ => Err(Box::new(UnknownOptionError(name.to_string()))),
        }
    }

    pub fn print_uci(&self) {
        // uci options
        println!("{}", self.uci_analyse_mode);
        println!("{}", self.uci_elo);
        println!("{}", self.uci_engine_about);
        println!("{}", self.uci_hash);
        println!("{}", self.uci_limit_strength);
        println!("{}", self.uci_multipv);
        println!("{}", self.uci_nalimov_cache);
        println!("{}", self.uci_nalimov_path);
        println!("{}", self.uci_opponent);
        println!("{}", self.uci_ownbook);
        println!("{}", self.uci_ponder);
        println!("{}", self.uci_set_position_value);
        println!("{}", self.uci_show_currline);
        println!("{}", self.uci_show_refutations);
        println!("{}", self.uci_shredder_bases_path);

        // custom options
        println!("{}", self.dirichlet_alpha);
        println!("{}", self.dirichlet_epsilon);
        println!("{}", self.game_tree_caching);
        println!("{}", self.gui_lag);
        println!("{}", self.nnue_path);
        println!("{}", self.ponder);
        println!("{}", self.threads);
        println!("{}", self.weights_path);

        // tunable params
        if cfg!(feature = "tunable") {
            println!("{}", self.qs_delta_pruning_threshold);
            println!("{}", self.qs_futility_margin);
            println!("{}", self.eval_policy_temperature);
            println!("{}", self.mcts_killer_exploitation);
            println!("{}", self.mcts_proven_loss_visit_threshold);
            println!("{}", self.mcts_tt_best_move);
            println!("{}", self.select_cpuct);
            println!("{}", self.timeman_base_soft_mult);
            println!("{}", self.timeman_clamp_lower);
            println!("{}", self.timeman_clamp_upper);
            println!("{}", self.timeman_stability_base);
            println!("{}", self.timeman_stability_slope);
            println!("{}", self.timeman_stability_floor);
            println!("{}", self.timeman_entropy_base);
            println!("{}", self.timeman_entropy_weight);
            println!("{}", self.id_nmp_reduction);
            println!("{}", self.id_nmp_phase_threshold);
            println!("{}", self.id_nmp_depth_factor);
            println!("{}", self.id_nmp_phase_factor);
            println!("{}", self.id_nmp_margin);
            println!("{}", self.id_nmp_depth_margin);
            println!("{}", self.id_scorer_hh_weight);
            println!("{}", self.lmr_offset);
            println!("{}", self.lmr_scale);
        }
    }
}
