use crate::{
    core::{
        eval::hce::TaperValue, params::HceParams, search::{mcts::{
            eval::hce::PolicyParams, node::VisitCount, search::SearchParams,
            select::puct::PuctParams,
        }, quiesce::QSearchParams}
    },
    misc::{InvalidValueError, ValueOutOfRangeError},
};
use std::{
    fmt,
    ops::{Deref, DerefMut},
};
use thiserror::Error;
use uom::si::information::mebibyte;
use uom::si::u64::Information;

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
pub struct Spin {
    pub value: i32,
    pub default: i32,
    pub min: i32,
    pub max: i32,
}

impl Spin {
    pub fn new(default: i32, min: i32, max: i32) -> Self {
        Self {
            value: default,
            default,
            min,
            max,
        }
    }

    pub fn set(&mut self, value_str: &str) -> Result<(), Box<dyn std::error::Error>> {
        let val = value_str
            .parse::<i32>()
            .map_err(|_| InvalidValueError::new(value_str.to_string()))?;

        if val < self.min || val > self.max {
            return Err(Box::new(ValueOutOfRangeError::new(
                val,
                self.min..=self.max,
            )));
        }
        self.value = val;
        Ok(())
    }
}

impl fmt::Display for Spin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "spin default {} min {} max {}",
            self.default, self.min, self.max
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

    pub fn set(&mut self, value_str: &str) -> Result<(), Box<dyn std::error::Error>> {
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

    pub fn set(&mut self, value_str: &str) -> Result<(), Box<dyn std::error::Error>> {
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
    hash: ConfigOption<Spin>,

    /// Num threads.
    threads: ConfigOption<Spin>,

    /// Clear hash tables.
    clear_hash: ConfigOption<Button>,

    /// Dirichlet noise - alpha parameter.
    dirichlet_alpha: ConfigOption<Spin>,

    /// Dirichlet noise - epsilon parameter.
    dirichlet_epsilon: ConfigOption<Spin>,

    /// Path to nn weights file.
    weights_path: ConfigOption<StringOption>,

    /// Whether to keep the game tree in between `go`-commands.
    game_tree_caching: ConfigOption<Check>,

    /// Assumed lag between the GUI starting the engine's clock, the engine
    /// receiving the go command, and the engine actually starting the
    /// search. (in ms)
    gui_lag: ConfigOption<Spin>,

    /// Whether to enable ponder mode (i.e., keep searching on the opponent's
    /// time until the opponent actually moves).
    ponder: ConfigOption<Check>,

    /// Evaluation policy temperature. In percent.
    eval_policy_temperature: ConfigOption<Spin>,

    /// Margin for futility pruning. In centipawns.
    eval_futility_margin: ConfigOption<Spin>,

    /// Margin for delta pruning. A tapervalue.
    eval_delta_pruning_threshold: ConfigOption<Spin>,

    /// Cpuct constant for selection. In percent.
    select_cpuct: ConfigOption<Spin>,

    /// Visit threshold for proven loss cutoff in mcts. In visits.
    mcts_proven_loss_visit_threshold: ConfigOption<Spin>,

    /// Killer exploitation factor for mcts. In percent.
    mcts_killer_exploitation: ConfigOption<Spin>,

    /// Bonus for tt best move in mcts. In percent.
    mcts_tt_best_move: ConfigOption<Spin>,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            hash: ConfigOption::new("hash", Spin::new(16, 1, 64 * 1024 * 1024)),
            threads: ConfigOption::new("threads", Spin::new(1, 1, 1)),
            clear_hash: ConfigOption::new("clearhash", Button::new(clear_hash_impl)),
            dirichlet_alpha: ConfigOption::new("dirichlet-alpha", Spin::new(30, 0, 1000)),
            dirichlet_epsilon: ConfigOption::new("dirichlet-epsilon", Spin::new(25, 0, 100)),
            weights_path: ConfigOption::new("weights-path", StringOption::new("./weights")),
            game_tree_caching: ConfigOption::new("game-tree-caching", Check::new(true)),
            gui_lag: ConfigOption::new("gui-lag", Spin::new(100, 1, 10_000)),
            ponder: ConfigOption::new("ponder", Check::new(true)),
            eval_policy_temperature: ConfigOption::new(
                "eval-policy-temperature",
                Spin::new((HceParams.policy_temperature() * 100.) as i32, 1, 10000),
            ),
            eval_futility_margin: ConfigOption::new(
                "eval-futility-margin",
                Spin::new(HceParams.futility_margin(), 100, 300),
            ),
            eval_delta_pruning_threshold: ConfigOption::new(
                "eval-delta-pruning-threshold",
                Spin::new(HceParams.delta_pruning_threshold().v() as i32, 0, 24),
            ),
            select_cpuct: ConfigOption::new(
                "select-cpuct",
                Spin::new((HceParams.select_cpuct() * 100.) as i32, 1, 5000),
            ),
            mcts_proven_loss_visit_threshold: ConfigOption::new(
                "mcts-proven-loss-visit-threshold",
                Spin::new(HceParams.proven_loss_visit_threshold().0 as i32, 1, 100),
            ),
            mcts_killer_exploitation: ConfigOption::new(
                "mcts-killer-exploitation",
                Spin::new((HceParams.killer_exploitation() * 100.) as i32, 0, 5000),
            ),
            mcts_tt_best_move: ConfigOption::new(
                "mcts-tt-best-move",
                Spin::new((HceParams.tt_best_move() * 100.) as i32, 0, 5000),
            ),
        }
    }
}

impl Configuration {
    // Getters

    pub fn hash(&self) -> Information {
        Information::new::<mebibyte>(self.hash.value as u64)
    }

    pub fn threads(&self) -> i32 {
        self.threads.value
    }

    pub fn dirichlet_alpha(&self) -> f32 {
        self.dirichlet_alpha.value as f32 / 100.
    }

    pub fn dirichlet_epsilon(&self) -> f32 {
        self.dirichlet_epsilon.value as f32 / 100.
    }

    pub fn weights_path(&self) -> &str {
        &self.weights_path.value
    }

    pub fn game_tree_caching(&self) -> bool {
        self.game_tree_caching.value
    }

    pub fn gui_lag(&self) -> u16 {
        self.gui_lag.value as u16
    }

    pub fn ponder(&self) -> bool {
        self.ponder.value
    }

    pub fn eval_policy_temperature(&self) -> f32 {
        self.eval_policy_temperature.value as f32 / 100.
    }

    pub fn eval_futility_margin(&self) -> i32 {
        self.eval_futility_margin.value
    }

    pub fn eval_delta_pruning_threshold(&self) -> TaperValue {
        TaperValue::new(self.eval_delta_pruning_threshold.value as u32)
    }

    pub fn select_cpuct(&self) -> f32 {
        self.select_cpuct.value as f32 / 100.
    }

    pub fn mcts_proven_loss_visit_threshold(&self) -> VisitCount {
        VisitCount(self.mcts_proven_loss_visit_threshold.value as u32)
    }

    pub fn mcts_killer_exploitation(&self) -> f32 {
        self.mcts_killer_exploitation.value as f32 / 100.
    }

    pub fn mcts_tt_best_move(&self) -> f32 {
        self.mcts_tt_best_move.value as f32 / 100.
    }

    // Setter

    pub fn set(&mut self, name: &str, value: &str) -> Result<(), Box<dyn std::error::Error>> {
        match name.to_lowercase().as_str() {
            "hash" => self.hash.set(value),
            "threads" => self.threads.set(value),
            "clearhash" => {
                self.clear_hash.trigger();
                Ok(())
            }
            "dirichlet-alpha" => self.dirichlet_alpha.set(value),
            "dirichlet-epsilon" => self.dirichlet_epsilon.set(value),
            "weights-path" => {
                self.weights_path.set(value);
                Ok(())
            }
            "game-tree-caching" => self.game_tree_caching.set(value),
            "gui-lag" => self.gui_lag.set(value),
            "ponder" => self.ponder.set(value),
            #[cfg(feature = "tunable")]
            "eval-policy-temperature" => self.eval_policy_temperature.set(value),
            #[cfg(feature = "tunable")]
            "eval-futility-margin" => self.eval_futility_margin.set(value),
            #[cfg(feature = "tunable")]
            "eval-delta-pruning-threshold" => self.eval_delta_pruning_threshold.set(value),
            #[cfg(feature = "tunable")]
            "select-cpuct" => self.select_cpuct.set(value),
            #[cfg(feature = "tunable")]
            "mcts-proven-loss-visit-threshold" => self.mcts_proven_loss_visit_threshold.set(value),
            #[cfg(feature = "tunable")]
            "mcts-killer-exploitation" => self.mcts_killer_exploitation.set(value),
            #[cfg(feature = "tunable")]
            "mcts-tt-best-move" => self.mcts_tt_best_move.set(value),
            _ => Err(Box::new(UnknownOptionError(name.to_string()))),
        }
    }

    pub fn print_uci(&self) {
        println!("{}", self.hash);
        println!("{}", self.threads);
        println!("{}", self.clear_hash);
        println!("{}", self.dirichlet_alpha);
        println!("{}", self.dirichlet_epsilon);
        println!("{}", self.weights_path);
        println!("{}", self.game_tree_caching);
        println!("{}", self.gui_lag);
        println!("{}", self.ponder);
        #[cfg(feature = "tunable")]
        println!("{}", self.eval_policy_temperature);
        #[cfg(feature = "tunable")]
        println!("{}", self.eval_futility_margin);
        #[cfg(feature = "tunable")]
        println!("{}", self.eval_delta_pruning_threshold);
        #[cfg(feature = "tunable")]
        println!("{}", self.select_cpuct);
        #[cfg(feature = "tunable")]
        println!("{}", self.mcts_proven_loss_visit_threshold);
        #[cfg(feature = "tunable")]
        println!("{}", self.mcts_killer_exploitation);
        #[cfg(feature = "tunable")]
        println!("{}", self.mcts_tt_best_move);
    }
}

pub fn clear_hash_impl() {
    todo!("clear hashing tables (e.g. transposition table)");
}
