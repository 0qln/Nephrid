use crate::{
    core::params::TunableConfiguration,
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
    pub fn seed(&mut self, value: U::Quantity) {
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

    pub tunable: TunableConfiguration,
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
                tunable: TunableConfiguration::default()
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConfigBuilder {
    pub config: Configuration,
}

impl ConfigBuilder {
    // /// Seed the quiescence-search options from [`QSearchParams`].
    // #[rustfmt::skip]
    // pub fn qsearch(mut self, params: &impl QSearchParams) -> Self {
    //     let cfg = &mut self.config;
    //     cfg.qs_futility_margin.seed(params.futility_margin().v());
    //     cfg.qs_delta_pruning_threshold.seed(params.delta_pruning_threshold().
    // v());     self
    // }

    // /// Seed the policy options from [`PolicyParams`].
    // #[rustfmt::skip]
    // pub fn policy(mut self, params: &impl PolicyParams) -> Self {
    //     let cfg = &mut self.config;
    //     cfg.eval_policy_temperature.seed(Ratio::new::<ratio>(params.
    // policy_temperature()));     self
    // }

    // /// Seed the selection options from [`PuctParams`].
    // #[rustfmt::skip]
    // pub fn puct(mut self, params: &impl PuctParams) -> Self {
    //     let cfg = &mut self.config;
    //     cfg.select_cpuct.seed(Ratio::new::<ratio>(params.select_cpuct()));
    //     self
    // }

    // /// Seed the mcts options from [`MctsParams`].
    // #[rustfmt::skip]
    // pub fn mcts(mut self, params: &impl MctsParams) -> Self {
    //     let cfg = &mut self.config;
    //     cfg.mcts_proven_loss_visit_threshold.seed(params.
    // proven_loss_visit_threshold().0 as i32);
    //     cfg.mcts_killer_exploitation.seed(Ratio::new::<ratio>(params.
    // killer_exploitation()));     cfg.mcts_tt_best_move.
    // seed(Ratio::new::<ratio>(params.tt_best_move()));     self
    // }

    // /// Seed the time-management options from [`ChronoParams`].
    // #[rustfmt::skip]
    // pub fn chrono(mut self, params: &impl ChronoParams) -> Self {
    //     let cfg = &mut self.config;
    //     cfg.timeman_base_soft_mult.seed(Ratio::new::<ratio>(params.
    // base_soft_mult()));     cfg.timeman_clamp_lower.
    // seed(Ratio::new::<ratio>(params.clamp_lower()));
    //     cfg.timeman_clamp_upper.seed(Ratio::new::<ratio>(params.clamp_upper()));
    //     cfg.timeman_stability_base.seed(Ratio::new::<ratio>(params.
    // movestreak_base()));     cfg.timeman_stability_slope.
    // seed(Ratio::new::<ratio>(params.movestreak_slope()));
    //     cfg.timeman_stability_floor.seed(Ratio::new::<ratio>(params.
    // movestreak_floor()));     cfg.timeman_entropy_base.
    // seed(Ratio::new::<ratio>(params.entropy_base()));
    //     cfg.timeman_entropy_weight.seed(Ratio::new::<ratio>(params.
    // entropy_weight()));     self
    // }

    // /// Seed the iterative-deepening options from [`IdParams`].
    // #[rustfmt::skip]
    // pub fn id(mut self, params: &impl IdParams) -> Self {
    //     let cfg = &mut self.config;
    //     cfg.id_nmp_reduction.seed(params.nmp_reduction().v() as i32);
    //     cfg.id_nmp_phase_threshold.seed(params.nmp_phase_threshold().v());
    //     cfg.id_nmp_depth_factor.seed(params.nmp_depth_factor() as i32);
    //     cfg.id_nmp_phase_factor.seed(params.nmp_phase_factor() as i32);
    //     cfg.id_nmp_margin.seed(params.nmp_margin().v());
    //     cfg.id_nmp_depth_margin.seed(params.nmp_depth_margin());
    //     self
    // }

    // /// Seed the iterative-deepening scorer options from [`ScorerParams`].
    // #[rustfmt::skip]
    // pub fn scorer(mut self, params: &impl ScorerParams) -> Self {
    //     let cfg = &mut self.config;
    //     cfg.id_scorer_hh_weight.seed(params.hh_weight());
    //     self
    // }

    // /// Seed the late-move-reduction options from [`LmrParams`].
    // #[rustfmt::skip]
    // pub fn lmr(mut self, params: &impl LmrParams) -> Self {
    //     let cfg = &mut self.config;
    //     cfg.lmr_offset.seed(Ratio::new::<ratio>(params.offset()));
    //     cfg.lmr_scale.seed(Ratio::new::<ratio>(params.scale()));
    //     self
    // }

    /// Finish building the [`Configuration`].
    pub fn build(self) -> Configuration { self.config }
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

    #[allow(clippy::unit_arg)]
    pub fn set(&mut self, name: &str, value: &str) -> Result<(), Box<dyn Error>> {
        let key = name.to_lowercase();

        match key.as_str() {
            "dirichlet-alpha" => return self.dirichlet_alpha.set(value),
            "dirichlet-epsilon" => return self.dirichlet_epsilon.set(value),
            "game-tree-caching" => return self.game_tree_caching.set(value),
            "gui-lag" => return self.gui_lag.set(value),
            "hash" => return self.uci_hash.set(value),
            "nalimovpath" => return Ok(self.uci_nalimov_path.set(value)),
            "nalimovcache" => return self.uci_nalimov_cache.set(value),
            "ownbook" => return self.uci_ownbook.set(value),
            "multipv" => return self.uci_multipv.set(value),
            "uci_showcurrline" => return self.uci_show_currline.set(value),
            "uci_showrefutations" => return self.uci_show_refutations.set(value),
            "uci_limitstrength" => return self.uci_limit_strength.set(value),
            "uci_elo" => return self.uci_elo.set(value),
            "uci_analysemode" => return self.uci_analyse_mode.set(value),
            "uci_opponent" => return Ok(self.uci_opponent.set(value)),
            "uci_engineabout" => return Ok(self.uci_engine_about.set(value)),
            "uci_shredderbasespath" => return Ok(self.uci_shredder_bases_path.set(value)),
            "uci_setpositionvalue" => return Ok(self.uci_set_position_value.set(value)),
            "ponder" => return self.ponder.set(value),
            "threads" => return self.threads.set(value),
            "weights-path" => return Ok(self.weights_path.set(value)),
            "nnue-path" => return Ok(self.nnue_path.set(value)),
            _ => {}
        };

        if cfg!(feature = "tunable")
            && let Some(res) = self.tunable.set(&key, value)
        {
            return res;
        }

        Err(Box::new(UnknownOptionError(name.to_string())))
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

        // tunable options
        if cfg!(feature = "tunable") {
            self.tunable.print_uci();
        }
    }
}
