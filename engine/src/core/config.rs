use crate::{core::{eval::hce::TaperValue, search::mcts::node::VisitCount}, math::{self, NormalizedEntropy}};
use crate::{
    core::{
        chrono::ChronoParams,
        search::{
            mcts::{eval::hce::PolicyParams, search::MctsParams, select::puct::PuctParams},
            quiesce::QSearchParams,
        },
    },
    misc::{InvalidValueError, ValueOutOfRangeError},
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
    type Raw = f32;

    fn to_quantity(raw: Self::Raw) -> Self::Quantity {
        Ratio::new::<percent>(raw)
    }
    fn to_raw(qty: &Self::Quantity) -> Self::Raw {
        qty.get::<percent>()
    }
}

#[derive(Debug, Clone)]
pub struct UciMebibyte;
impl UciUnit for UciMebibyte {
    type Quantity = Information;
    type Raw = u64;

    fn to_quantity(raw: Self::Raw) -> Self::Quantity {
        Information::new::<information::mebibyte>(raw)
    }
    fn to_raw(qty: &Self::Quantity) -> Self::Raw {
        qty.get::<information::mebibyte>()
    }
}

#[derive(Debug, Clone)]
pub struct UciInteger;
impl UciUnit for UciInteger {
    type Quantity = i32;
    type Raw = i32;

    fn to_quantity(raw: Self::Raw) -> Self::Quantity {
        raw
    }
    fn to_raw(qty: &Self::Quantity) -> Self::Raw {
        *qty
    }
}

#[derive(Debug, Clone)]
pub struct UciMillis;
impl UciUnit for UciMillis {
    type Quantity = Time;
    type Raw = u64;

    fn to_quantity(raw: Self::Raw) -> Self::Quantity {
        Time::new::<millisecond>(raw)
    }
    fn to_raw(qty: &Self::Quantity) -> Self::Raw {
        qty.get::<millisecond>()
    }
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
    pub fn new(name: &str, inner: T) -> Self {
        Self { name: name.to_string(), inner }
    }
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
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for ConfigOption<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T: fmt::Display> fmt::Display for ConfigOption<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "option name {} type {}", self.name, self.inner)
    }
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
        let val = value_str
            .parse::<U::Raw>()
            .map_err(|_| InvalidValueError::new(value_str.to_string()))?;

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
    pub fn new(default: bool) -> Self {
        Self { value: default, default }
    }

    pub fn set(&mut self, value_str: &str) -> Result<(), Box<dyn Error>> {
        self.value = value_str
            .parse::<bool>()
            .map_err(|_| InvalidValueError::new(value_str.to_string()))?;
        Ok(())
    }
}

impl fmt::Display for Check {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "check default {}", self.default)
    }
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

    pub fn set(&mut self, value_str: &str) {
        self.value = value_str.to_string();
    }
}

impl fmt::Display for StringOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "string default {}", self.default)
    }
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
    pub fn new(callback: fn()) -> Self {
        Self { callback }
    }

    pub fn trigger(&self) {
        (self.callback)();
    }
}

impl fmt::Display for Button {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "button")
    }
}

impl fmt::Debug for Button {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Button")
    }
}

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct Configuration {
    /// Hash size.
    hash: ConfigOption<Spin<UciMebibyte>>,

    /// Num threads.
    threads: ConfigOption<Spin<UciInteger>>,

    /// Clear hash tables.
    clear_hash: ConfigOption<Button>,

    /// Dirichlet noise - alpha parameter.
    dirichlet_alpha: ConfigOption<Spin<UciPercent>>,

    /// Dirichlet noise - epsilon parameter.
    dirichlet_epsilon: ConfigOption<Spin<UciPercent>>,

    /// Path to nn weights file.
    weights_path: ConfigOption<StringOption>,

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
    eval_futility_margin: ConfigOption<Spin<UciInteger>>,

    /// Margin for delta pruning. A tapervalue.
    eval_delta_pruning_threshold: ConfigOption<Spin<UciInteger>>,

    /// Cpuct constant for selection.
    select_cpuct: ConfigOption<Spin<UciPercent>>,

    /// Visit threshold for proven loss cutoff in mcts.
    mcts_proven_loss_visit_threshold: ConfigOption<Spin<UciInteger>>,

    /// Killer exploitation factor for mcts.
    mcts_killer_exploitation: ConfigOption<Spin<UciPercent>>,

    /// Bonus for tt best move in mcts.
    mcts_tt_best_move: ConfigOption<Spin<UciPercent>>,

    /// Target entropy for time management.
    timeman_entropy_target: ConfigOption<Spin<UciPercent>>,
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
    pub fn builder() -> ConfigBuilder {
        fn _mebibyte(v: u64) -> Information { Information::new::<information::mebibyte>(v) }
        fn _ratio(v: f32) -> Ratio { Ratio::new::<ratio>(v) }
        fn _millis(v: u64) -> Time { Time::new::<millisecond>(v) }

        ConfigBuilder {
            config: Self {
                hash: ConfigOption::new("hash", Spin::<UciMebibyte>::new(_mebibyte(16), _mebibyte(1), _mebibyte(64 * 1024 * 1024))),
                threads: ConfigOption::new("threads", Spin::new(1, 1, 1)),
                clear_hash: ConfigOption::new("clearhash", Button::new(clear_hash_impl)),
                dirichlet_alpha: ConfigOption::new("dirichlet-alpha", Spin::<UciPercent>::new(_ratio(0.3), _ratio(0.), _ratio(10.))),
                dirichlet_epsilon: ConfigOption::new("dirichlet-epsilon", Spin::<UciPercent>::new(_ratio(0.25), _ratio(0.), _ratio(1.))),
                weights_path: ConfigOption::new("weights-path", StringOption::new("./weights")),
                game_tree_caching: ConfigOption::new("game-tree-caching", Check::new(true)),
                gui_lag: ConfigOption::new("gui-lag", Spin::<UciMillis>::new(_millis(100), _millis(1), _millis(10_000))),
                ponder: ConfigOption::new("ponder", Check::new(true)),
                eval_policy_temperature: ConfigOption::new("eval-policy-temperature", Spin::<UciPercent>::new(_ratio(20.), _ratio(1.), _ratio(100.))),
                eval_futility_margin: ConfigOption::new("eval-futility-margin", Spin::new(150, 100, 300)),
                eval_delta_pruning_threshold: ConfigOption::new("eval-delta-pruning-threshold", Spin::new(16, 0, 24)),
                select_cpuct: ConfigOption::new("select-cpuct", Spin::<UciPercent>::new(_ratio(1.4), _ratio(0.01), _ratio(50.))),
                mcts_proven_loss_visit_threshold: ConfigOption::new("mcts-proven-loss-visit-threshold", Spin::new(5, 1, 100)),
                mcts_killer_exploitation: ConfigOption::new("mcts-killer-exploitation", Spin::<UciPercent>::new(_ratio(0.27), _ratio(0.), _ratio(10.))),
                mcts_tt_best_move: ConfigOption::new("mcts-tt-best-move", Spin::<UciPercent>::new(_ratio(1.50), _ratio(0.), _ratio(10.))),
                timeman_entropy_target: ConfigOption::new("timeman-entropy-target", Spin::<UciPercent>::new(_ratio(0.60), _ratio(0.), _ratio(1.))),
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
        cfg.eval_futility_margin.seed(params.futility_margin());
        cfg.eval_delta_pruning_threshold.seed(params.delta_pruning_threshold().v() as i32);
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
        cfg.timeman_entropy_target.seed(Ratio::new::<ratio>(params.entropy_target().v()));
        self
    }

    /// Finish building the [`Configuration`].
    pub fn build(self) -> Configuration {
        self.config
    }
}

impl Configuration {
    pub fn eval_policy_temperature(&self) -> f32 {
        self.eval_policy_temperature.value.get::<ratio>()
    }

    pub fn eval_futility_margin(&self) -> i32 {
        self.eval_futility_margin.value
    }

    pub fn eval_delta_pruning_threshold(&self) -> TaperValue {
        TaperValue::new(self.eval_delta_pruning_threshold.value as u32)
    }

    pub fn select_cpuct(&self) -> f32 {
        self.select_cpuct.value.get::<ratio>()
    }

    pub fn mcts_proven_loss_visit_threshold(&self) -> VisitCount {
        VisitCount(self.mcts_proven_loss_visit_threshold.value as u32)
    }

    pub fn mcts_killer_exploitation(&self) -> f32 {
        self.mcts_killer_exploitation.value.get::<ratio>()
    }

    pub fn mcts_tt_best_move(&self) -> f32 {
        self.mcts_tt_best_move.value.get::<ratio>()
    }

    pub fn timeman_entropy_target(&self) -> math::NormalizedEntropy {
        NormalizedEntropy::new(self.timeman_entropy_target.value.get::<ratio>())
    }
}

impl Configuration {
    pub fn hash(&self) -> Information {
        self.hash.value
    }

    pub fn threads(&self) -> i32 {
        self.threads.value
    }

    pub fn dirichlet_alpha(&self) -> f32 {
        self.dirichlet_alpha.value.get::<ratio>()
    }

    pub fn dirichlet_epsilon(&self) -> f32 {
        self.dirichlet_epsilon.value.get::<ratio>()
    }

    pub fn weights_path(&self) -> &str {
        &self.weights_path.value
    }

    pub fn game_tree_caching(&self) -> bool {
        self.game_tree_caching.value
    }

    pub fn gui_lag(&self) -> u16 {
        self.gui_lag.value.get::<millisecond>() as u16
    }

    pub fn ponder(&self) -> bool {
        self.ponder.value
    }

    #[rustfmt::skip]
    pub fn set(&mut self, name: &str, value: &str) -> Result<(), Box<dyn Error>> {
        match name.to_lowercase().as_str() {
            "clearhash" => Ok(self.clear_hash.trigger()),
            "dirichlet-alpha" => self.dirichlet_alpha.set(value),
            "dirichlet-epsilon" => self.dirichlet_epsilon.set(value),
            "game-tree-caching" => self.game_tree_caching.set(value),
            "gui-lag" => self.gui_lag.set(value),
            "hash" => self.hash.set(value),
            "ponder" => self.ponder.set(value),
            "threads" => self.threads.set(value),
            "weights-path" => Ok(self.weights_path.set(value)),
            #[cfg(feature = "tunable")] "eval-delta-pruning-threshold" => self.eval_delta_pruning_threshold.set(value),
            #[cfg(feature = "tunable")] "eval-futility-margin" => self.eval_futility_margin.set(value),
            #[cfg(feature = "tunable")] "eval-policy-temperature" => self.eval_policy_temperature.set(value),
            #[cfg(feature = "tunable")] "mcts-killer-exploitation" => self.mcts_killer_exploitation.set(value),
            #[cfg(feature = "tunable")] "mcts-proven-loss-visit-threshold" => self.mcts_proven_loss_visit_threshold.set(value),
            #[cfg(feature = "tunable")] "mcts-tt-best-move" => self.mcts_tt_best_move.set(value),
            #[cfg(feature = "tunable")] "select-cpuct" => self.select_cpuct.set(value),
            #[cfg(feature = "tunable")] "timeman-entropy-target" => self.timeman_entropy_target.set(value),
            _ => Err(Box::new(UnknownOptionError(name.to_string()))),
        }
    }

    pub fn print_uci(&self) {
        println!("{}", self.clear_hash);
        println!("{}", self.dirichlet_alpha);
        println!("{}", self.dirichlet_epsilon);
        println!("{}", self.game_tree_caching);
        println!("{}", self.gui_lag);
        println!("{}", self.hash);
        println!("{}", self.ponder);
        println!("{}", self.threads);
        println!("{}", self.weights_path);
        if cfg!(feature = "tunable") {
            println!("{}", self.eval_delta_pruning_threshold);
            println!("{}", self.eval_futility_margin);
            println!("{}", self.eval_policy_temperature);
            println!("{}", self.mcts_killer_exploitation);
            println!("{}", self.mcts_proven_loss_visit_threshold);
            println!("{}", self.mcts_tt_best_move);
            println!("{}", self.select_cpuct);
            println!("{}", self.timeman_entropy_target);
        }
    }
}

pub fn clear_hash_impl() {
    todo!("clear hashing tables (e.g. transposition table)");
}
