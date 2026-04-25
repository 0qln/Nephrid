use burn_cuda::Cuda;
use crossbeam_channel::{RecvTimeoutError, Sender, bounded};
use engine::core::{
    config::Configuration,
    r#move::MoveList,
    move_iter::fold_legal_moves,
    position::PgnResultValue,
    search::mcts::{
        self, CreateNNPartsError, MctsParts, NNParts,
        back::MctsSolver,
        eval::{
            self, Evaluation, Evaluator, Policy, Quality, RawLogits, VisitCounts,
            nn::{TraceInfo, get_node_history},
        },
        nn::{self, BoardInputTensor, POLICY_OUTPUTS, StateInputTensor},
        node::{
            self, BranchId, WinRate,
            node_state::{Evaluated, HasBranches},
        },
        noise::DirichletNoiser,
        search::{BatchItem, Selection},
        select::{Score, Selector},
    },
    zobrist,
};
use itertools::Itertools;
use rand::{SeedableRng, rngs::SmallRng};
use rayon::{
    ThreadPool, ThreadPoolBuilder,
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
};
use std::{
    cmp::max,
    error::Error,
    ops::ControlFlow,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant},
};

use burn::{Tensor, config::Config, prelude::Backend};
use engine::{
    core::{
        Game,
        color::Color,
        r#move::Move,
        position::{FenImport, Position},
        search::mcts::{
            SearchState,
            eval::{GameResult, Guess},
            mcts,
            nn::{
                BOARD_INPUT_HISTORY, BoardInputFloats, Model, StateInputFloats, board_input,
                state_input,
            },
            node::Tree,
            strategy::{MctsFindBest, MctsStrategy},
        },
    },
    uci::tokens::Tokenizer,
};

use crate::{
    caching::{Cache, CacheEntry},
    data::{BoardInput, FenItemRaw, StateInput},
};

#[derive(Config, Debug)]
pub struct SelfplayConfig {
    pub concurrency: usize,
    pub allowed_moves: Option<u16>,
    pub unfinished_games: String,
    pub eval_models: usize,
    pub eval_batch_size: usize,
    pub eval_batch_wait_timeout_ms: u64,
}

pub enum UnfinishedGameHandling {
    /// Discard the game and don't include it in the training data at all.
    Discard,

    /// Declare the game a draw.
    Draw,

    /// UnfinishedGameHandling::Approx, where instead of setting the target
    /// value to a draw, set the target value to (value/visits) evaluation of
    /// the last root node or something that way we don't discard it and maybe
    /// get a estimation of the end value that is more accurate than just
    /// declaring it a draw
    Approx,
}

impl TryFrom<&str> for UnfinishedGameHandling {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "discard" => Ok(Self::Discard),
            "draw" => Ok(Self::Draw),
            "approx" => Ok(Self::Approx),
            _ => Err(format!("Invalid unfinished game handling: {value}")),
        }
    }
}

#[derive(Config, Debug)]
pub struct MctsConfig {
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
}

#[derive(Config, Debug)]
pub struct LimitConfig {
    pub max_iterations: Option<u64>,
    pub max_nodes: Option<u64>,
    pub min_nodes: Option<u64>,
    pub max_terminal_nodes: Option<u64>,
}

impl LimitConfig {
    pub fn init(&self) -> SelfPlayLimit {
        SelfPlayLimit {
            iterations: self.max_iterations.unwrap_or(u64::MAX),
            max_nodes: self.max_nodes.unwrap_or(u64::MAX),
            min_nodes: self.min_nodes.unwrap_or(0),
            terminal_nodes: self.max_terminal_nodes.unwrap_or(u64::MAX),
        }
    }
}

enum WorkerResult<P: PlayoutItem> {
    CacheHit {
        playout_items: Vec<P>,
        solved: bool,
    },
    SelfPlay {
        game: Box<Game>,
        playout_items: Vec<P>,
        hash: zobrist::Hash,
        outcome: Outcome,
        best_move: Move,
        solved: bool,
    },
    Aborted {},
}

fn try_cache_hit<P: PlayoutItem>(
    fen: &str,
    i: usize,
    n: usize,
    cache: &Cache,
) -> Option<WorkerResult<P>>
where
    P::Target: Clone,
{
    let pos = match Position::from_fen(fen) {
        Ok(p) => p,
        Err(e) => {
            log::error!(target: "data", "[FEN {i:>3}/{n:<3}] Invalid FEN: {e}");
            return Some(WorkerResult::CacheHit {
                playout_items: vec![],
                solved: false,
            });
        }
    };
    let hash = pos.get_key();
    let moving_color = pos.get_turn();

    if let Some(&CacheEntry { outcome, best_move, .. }) = cache.get(hash) {
        log::debug!(target: "data", "[FEN {i:>3}/{n:<3}] Cache hit for proven game");

        // Build a single decision with one‑hot policy
        let decision = build_cached_decision(&pos, best_move, moving_color);
        let playout_item = P::from((outcome, &[decision]));
        let is_solved = matches!(outcome, Outcome::Discrete(GameResult::Win { .. }));
        Some(WorkerResult::CacheHit {
            playout_items: vec![playout_item],
            solved: is_solved,
        })
    }
    else {
        None
    }
}

fn build_cached_decision<T: Target>(
    pos: &Position,
    best_move: Move,
    moving_color: Color,
) -> Decision<T> {
    let mut moves = MoveList::new();
    _ = fold_legal_moves::<_, _, _>(pos, (), |_, m| {
        moves.push(m);
        ControlFlow::Continue::<(), _>(())
    });

    let visit_counts = moves
        .iter()
        .map(|&mov| (mov, if mov == best_move { 1 } else { 0 }))
        .collect_vec();

    let target = T::from_visit_counts(VisitCounts(visit_counts));

    Decision {
        input: Input {
            board_in: board_input(pos),
            state_in: state_input(pos),
        },
        target,
        state: State { mov: best_move, moving_color },
        stats: Stats::default(),
    }
}

fn run_self_play_for_fen<P: PlayoutItem + Clone>(
    fen: &str,
    i: usize,
    n: usize,
    limit: &SelfPlayLimit,
    config: &SelfplayConfig,
    parts: &MctsTrainParts,
) -> Result<WorkerResult<P>, Box<dyn Error>>
where
    P::Target: Clone,
{
    log::debug!(target: "data", "[FEN {i:>2}/{n:<2}] Starting self-play with fen '{fen}'");

    let game = Game::from_fen(FenImport(&mut Tokenizer::new(fen)))?;
    let root_hash = game.position().get_key();
    let SelfPlay { game, results } = self_play::<P>(i, n, game, limit.clone(), config, parts)?;
    if results.is_empty() {
        return Ok(WorkerResult::Aborted {});
    }
    let playout_items = results.iter().map(|r| r.playout_item.clone()).collect();
    let final_outcome = game
        .position()
        .game_result()
        .map(Outcome::Discrete)
        .unwrap_or(Outcome::Discrete(GameResult::Draw));
    let best_move = results[0].decision.state.mov;
    let is_solved = matches!(final_outcome, Outcome::Discrete(GameResult::Win { .. }));
    Ok(WorkerResult::SelfPlay {
        game: Box::new(game),
        playout_items,
        hash: root_hash,
        outcome: final_outcome,
        best_move,
        solved: is_solved,
    })
}

#[derive(Debug, Clone, Copy)]
pub enum Outcome {
    Discrete(GameResult),
    Continuous {
        quality: Quality,
        relative_to: Color,
    },
}

pub trait PlayoutItem: for<'a> From<(Outcome, &'a [Decision<Self::Target>])> {
    type Target: Target;
}

pub struct BatchStats {
    pub games_total: usize,
    pub games_completed: usize,
    pub games_solved: usize,
}

pub struct BatchGenerator {
    pool: ThreadPool,
}

impl BatchGenerator {
    pub fn new(config: &SelfplayConfig) -> Result<Self, Box<dyn Error>> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(config.concurrency)
            .build()?;
        Ok(Self { pool })
    }

    pub fn generate_batch<B, P>(
        &self,
        model: Model<B>,
        fens: &[FenItemRaw],
        limit: &SelfPlayLimit,
        config: &SelfplayConfig,
        mcts: &MctsConfig,
        cache: &mut Cache,
    ) -> Result<(Vec<P>, BatchStats), Box<dyn Error>>
    where
        B: Backend,
        P: Clone + PlayoutItem + Send,
        P::Target: Clone,
    {
        let device = B::Device::default();
        let worker_tx = spawn_inference_workers(
            model.clone(),
            device,
            config.eval_models,
            config.eval_batch_size,
            Duration::from_millis(config.eval_batch_wait_timeout_ms),
        );
        let parts = MctsTrainParts::new(
            BatchedNNEvaluator::new(worker_tx),
            mcts.dirichlet_alpha,
            mcts.dirichlet_epsilon,
        );

        let num_fens = fens.len();
        let time_start = Instant::now();
        let games_total = num_fens;
        let games_completed = AtomicUsize::new(0);

        let worker_results: Vec<WorkerResult<P>> = self.pool.install(|| {
            fens.iter()
            .enumerate()
            .collect_vec()
            .into_par_iter()
            .with_max_len(1)
            .filter_map(|(i, fen)| {
                let i = i + 1;
                let n = num_fens;
                let fen_str = fen.fen.clone();

                // 1. try cache
                if let Some(cached) = try_cache_hit::<P>(&fen_str, i, n, cache) {
                    games_completed.fetch_add(1, Ordering::Relaxed);
                    return Some(cached);
                }

                // 2. otherwise run self‑play
                let ret = match run_self_play_for_fen::<P>(&fen_str, i, n, limit, config, &parts) {
                    Ok(result) => {
                        if let WorkerResult::SelfPlay { ref game, .. } = result {
                            let pgn = game.to_pgn();
                            log::info!(target: "data", "[FEN {i:>3}/{n:<3}] Generated game:\n{pgn}");
                        }
                        games_completed.fetch_add(1, Ordering::Relaxed);
                        Some(result)
                    },
                    Err(err) => {
                        log::error!(target: "data", "[FEN {i:>3}/{n:<3}] Self‑play error: {err}");
                        None
                    }
                };

                let games_completed = games_completed.load(Ordering::Relaxed);
                let perc_complete = (games_completed as f32 / games_total as f32) * 100.0;
                log::info!(target: "data", "Batch: {perc_complete:>3.2}% complete");

                ret
            })
            .collect()
        });

        let mut all_playout_items = Vec::new();
        let mut games_completed = 0;
        let mut games_solved = 0;

        for res in worker_results {
            match res {
                WorkerResult::CacheHit { playout_items, solved } => {
                    all_playout_items.extend(playout_items);
                    games_completed += 1;
                    games_solved += solved as usize;
                }
                WorkerResult::SelfPlay {
                    playout_items,
                    hash,
                    outcome,
                    best_move,
                    solved,
                    ..
                } => {
                    all_playout_items.extend(playout_items);
                    games_completed += 1;
                    games_solved += solved as usize;
                    cache.insert(hash, outcome, best_move);
                }
                WorkerResult::Aborted {} => {}
            }
        }

        let elapsed = time_start.elapsed().as_secs_f32();
        let games_per_sec = games_completed as f32 / elapsed;
        let games_per_hour = games_per_sec * 3600.0;
        log::info!(target: "data", "Batch generated: {:4>} games ({:.2} games/hour), {:.2}% solved",
        games_completed, games_per_hour, (games_solved as f64 / games_completed as f64) * 100.0);

        Ok((
            all_playout_items,
            BatchStats {
                games_total: num_fens,
                games_completed,
                games_solved,
            },
        ))
    }
}

#[derive(Debug)]
pub struct MctsTrainStrategy {
    infer: MctsFindBest,

    nodes_begin: u64,
    terminal_nodes_begin: u64,
    iterations: u64,
    start_time: Option<Instant>,

    i: usize,
    n: usize,

    limit: SelfPlayLimit,
}

impl MctsTrainStrategy {
    pub fn new(limit: SelfPlayLimit, i: usize, n: usize) -> Self {
        Self {
            limit,
            infer: MctsFindBest::default(),
            i,
            n,
            nodes_begin: 0,
            terminal_nodes_begin: 0,
            iterations: 0,
            start_time: None,
        }
    }
}

impl MctsStrategy for MctsTrainStrategy {
    type Result = (<MctsFindBest as MctsStrategy>::Result,);
    type Step = (<MctsFindBest as MctsStrategy>::Step,);

    fn start(&mut self, tree: &mut Tree, pos: &Position) {
        self.infer.start(tree, pos);
        self.nodes_begin = tree.size() as u64;
        self.terminal_nodes_begin = tree.terminal_nodes() as u64;
        self.iterations = 0;
        self.start_time = Some(Instant::now());
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        self.iterations += 1;
        let step = self.infer.step(tree);
        (step,)
    }

    fn should_stop(&mut self, tree: &Tree) -> bool {
        // check limits
        if self.limit.is_reached(
            tree.size() as u64 - self.nodes_begin,
            tree.terminal_nodes() as u64 - self.terminal_nodes_begin,
            self.iterations,
        ) {
            return true;
        }

        // proven win/loss at root.
        let root = tree.node(tree.root());
        let root_value = root.value();
        if root_value.is_proven_win() || root_value.is_proven_loss() {
            return true;
        }

        false
    }

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        let inference_result = self.infer.result(tree);
        let iters = self.iterations;
        let nodes = tree.size() as u64 - self.nodes_begin;
        let now = Instant::now();
        let elapsed = now.saturating_duration_since(self.start_time.unwrap());
        let millis = elapsed.as_millis();
        let nps = (nodes as u128 * 1_000_000_000) / max(elapsed.as_nanos(), 1);
        let root = tree.node_switch(tree.root()).get::<Evaluated>().unwrap();
        let policy = tree.branches(root).iter().map(|b| b.policy()).collect_vec();
        let entropy = entropy(policy.iter().cloned());
        let seldepth = tree.maxheight();
        let depth = tree.compute_minheight();
        let nps_str = if millis > 0 { &format!(" {nps}nps ") } else { "" };
        log::debug!(target: "data", "[MCTS {:>3}/{:<3}] {} iters  {} nodes  {}ms {nps_str} {}d/{}sd  {:.3} H(π)", self.i, self.n, iters, nodes, millis, depth, seldepth, entropy);
        (inference_result,)
    }
}

// The inputs to the model.
#[derive(Debug, Clone)]
pub struct Input {
    pub board_in: BoardInputFloats,
    pub state_in: StateInputFloats,
}

pub trait Target: for<'a> From<&'a Tree> {
    fn from_visit_counts(visit_counts: VisitCounts) -> Self;
}

// Some state info at the time of the move.
// 0: The move that was made
// 1: The color of the moving player.
#[derive(Debug, Clone)]
pub struct State {
    pub mov: Move,
    pub moving_color: Color,
}

// Some interesting stats about the decision.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub policy_avg: f32,
    pub policy_stddev: f32,
    pub policy_entropy: f32,
}

fn entropy(xs: impl Iterator<Item = f32>) -> f32 {
    -xs.filter(|&x| x > 0.).map(|x| x * x.log2()).sum::<f32>()
}

fn avg(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len() as f32
}

fn variance(xs: &[f32]) -> f32 {
    let avg = avg(xs);
    xs.iter().map(|x| (x - avg).powi(2)).sum::<f32>() / xs.len() as f32
}

fn stddev(xs: &[f32]) -> f32 {
    variance(xs).sqrt()
}

impl Stats {
    pub fn new(guess: Guess) -> Self {
        let policy_values = guess.policy();

        let policy_avg = avg(policy_values.as_slice());
        let policy_stddev = stddev(policy_values.as_slice());
        let policy_entropy = entropy(policy_values.iter());

        Self {
            policy_avg,
            policy_stddev,
            policy_entropy,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Decision<T: Target> {
    pub input: Input,
    pub target: T,
    pub state: State,
    pub stats: Stats,
}

impl<'a, T: Target> From<&'a Decision<T>> for StateInput {
    fn from(decision: &'a Decision<T>) -> Self {
        Self(decision.input.state_in)
    }
}

impl<'a, T: Target> From<&'a [Decision<T>]> for BoardInput {
    fn from(decisions: &'a [Decision<T>]) -> Self {
        Self(
            decisions
                .iter()
                .map(|d| d.input.board_in)
                .collect::<Vec<_>>(),
        )
    }
}

#[derive(Debug)]
pub struct SelfPlayResult<P: PlayoutItem> {
    pub playout_item: P,
    pub decision: Decision<P::Target>,
}

pub struct SelfPlay<P: PlayoutItem> {
    game: Game,
    results: Vec<SelfPlayResult<P>>,
}

pub fn self_play<P: PlayoutItem>(
    i: usize,
    n: usize,
    mut game: Game,
    limit: SelfPlayLimit,
    config: &SelfplayConfig,
    parts: &MctsTrainParts,
) -> Result<SelfPlay<P>, Box<dyn Error>>
where
    P::Target: Clone,
{
    let unfinished_game_handling =
        UnfinishedGameHandling::try_from(config.unfinished_games.as_str())
            .map_err(|e| format!("Invalid config for unfinished games handling: {e}"))?;

    let mut decisions = Vec::<Decision<P::Target>>::new();
    let mut mcts_state = SearchState::default();

    let outcome: Outcome = {
        let game_result;
        let mut completed_moves = 0;
        loop {
            if let Some(result) = game.position().game_result() {
                game_result = Outcome::Discrete(result);
                break;
            }

            if config
                .allowed_moves
                .is_some_and(|max_moves| completed_moves >= max_moves)
            {
                match unfinished_game_handling {
                    UnfinishedGameHandling::Discard => {
                        return Ok(SelfPlay { game, results: vec![] });
                    }
                    UnfinishedGameHandling::Draw => {
                        game_result = Outcome::Discrete(GameResult::Draw);
                        break;
                    }
                    UnfinishedGameHandling::Approx => {
                        let pos = game.position();
                        let tree = &mcts_state.tree;
                        let root_id = tree.root();
                        let root_evaluated = tree.try_node::<Evaluated>(root_id);
                        let win_rate = root_evaluated.map(WinRate::from).unwrap_or_default();
                        let value = eval::Value::from(win_rate);
                        let quality = eval::Quality::from(value);
                        game_result = Outcome::Continuous {
                            quality,
                            relative_to: pos.get_turn(),
                        };
                        break;
                    }
                }
            }

            let strat = MctsTrainStrategy::new(limit.clone(), i, n);
            let result = mcts::<{ MPV }, MctsTrainConfig, _>(
                game.position_mut(),
                parts,
                &mut mcts_state,
                strat,
            );

            let pos = game.position();
            let turn = pos.get_turn();

            let mov = result.0;

            let b_in = board_input(pos);
            let s_in = state_input(pos);

            let mov = mov.expect(
                "if we don't have a gameresult then the search should've yielded a bestmove.",
            );

            let state = State { mov, moving_color: turn };
            let input = Input { board_in: b_in, state_in: s_in };
            let target = P::Target::from(&mcts_state.tree);
            // let stats = Stats::new(guess.clone());
            let stats = Stats::default();
            decisions.push(Decision { input, target, state, stats });

            game.push_move(mov);
            mcts_state.advance_to(mov);
            completed_moves += 1;
        }
        game_result
    };

    if let Outcome::Discrete(result) = outcome {
        log::debug!(target: "data", "[FEN {i:>2}/{n:<2}] Finished self-play with result '{}'", PgnResultValue(Some(result)));
    }
    else {
        // log idk?
    }

    let mut results = Vec::<SelfPlayResult<_>>::new();

    for (index, decision) in decisions.iter().enumerate() {
        let decisions_begin = index.saturating_sub(BOARD_INPUT_HISTORY - 1);
        let decisions_end = index;
        let decisions = &decisions[decisions_begin..=decisions_end];
        let playout_item = P::from((outcome, decisions));

        results.push(SelfPlayResult {
            playout_item,
            decision: decision.clone(),
        });
    }

    Ok(SelfPlay { game, results })
}

pub const MPV: usize = 64;

pub struct MctsTrainConfig;
impl mcts::MctsConfig for MctsTrainConfig {
    type Parts = MctsTrainParts;
    type Strat = MctsTrainStrategy;
}

pub struct MctsTestConfig;
impl mcts::MctsConfig for MctsTestConfig {
    type Parts = NNParts<Cuda<f32>>;
    type Strat = MctsTrainStrategy;
}

#[derive(Debug)]
pub struct MctsTrainParts {
    pub evaluator: BatchedNNEvaluator,
    alpha: f32,
    epsilon: f32,
}

impl MctsParts for MctsTrainParts {
    type Selector = MctsTrainSelector;
    type Backprop = MctsSolver;
    type Evaluator = BatchedNNEvaluator;
    type Noiser = DirichletNoiser;

    fn selector(&self) -> Self::Selector {
        Default::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        self.evaluator.clone()
    }

    fn backprop(&self) -> Self::Backprop {
        Default::default()
    }

    fn noiser(&self) -> Self::Noiser {
        let rng = SmallRng::from_os_rng();
        DirichletNoiser::new(self.alpha, self.epsilon, rng)
    }
}

impl TryFrom<&Configuration> for MctsTrainParts {
    type Error = CreateNNPartsError;

    fn try_from(_config: &Configuration) -> Result<Self, Self::Error> {
        unimplemented!("dont use this here")
    }
}

impl MctsTrainParts {
    pub fn new(evaluator: BatchedNNEvaluator, alpha: f32, epsilon: f32) -> Self {
        Self { evaluator, alpha, epsilon }
    }
}

pub struct MctsTrainSelector {
    c: f32,
    policy_weight: f32,
}

impl MctsTrainSelector {
    pub fn new(c: f32, policy_weight: f32) -> Self {
        Self { c, policy_weight }
    }

    /// Weighted policy, where
    ///     w,p,p_hat \el [0;1]
    ///     w=0 => p_hat=1
    ///     w=1 => p_hat=p
    fn weighted_policy(p: f32, w: f32) -> f32 {
        1.0 - w + p * w
    }
}

impl Default for MctsTrainSelector {
    fn default() -> Self {
        Self {
            c: f32::sqrt(2.0),
            policy_weight: 1.0,
        }
    }
}

impl Selector for MctsTrainSelector {
    fn score(&self, tree: &Tree, branch_id: BranchId, cap_n_i: u32) -> Score {
        let branch = tree.branch(branch_id);
        let node = tree.node(branch.node());

        let n_i = node.visits() as f32;
        let value = node.value();

        assert!(!value.0.is_nan(), "value WAS NAN");
        assert!(!branch.policy().is_nan(), "policy WAS NAN");
        assert!(!n_i.is_nan(), "n_i WAS NAN");

        let policy = Self::weighted_policy(branch.policy(), self.policy_weight);
        let exploitation = if n_i == 0.0 { 0.0 } else { value / n_i };
        let exploration = self.c * policy * (cap_n_i as f32).sqrt() / (1f32 + n_i);
        let result = exploitation + exploration;

        assert!(!result.is_nan(), "score was NAN");

        Score::new(result)
    }

    fn min_score(&self) -> Score {
        Score::new(f32::NEG_INFINITY)
    }
}

/// The request sent from an MCTS thread to the Inference Worker.
pub struct NNRequest {
    pub board_history: Vec<BoardInputFloats>,
    pub state: StateInputFloats,
    /// The channel where the worker will send the evaluation back to the
    /// specific MCTS thread.
    pub responder: Sender<NNResponse>,
}

/// The raw output from the neural network sent back to the MCTS thread.
pub struct NNResponse {
    pub value: f32,
    pub raw_logits: RawLogits,
}

/// Spawns a pool of dedicated inference threads pulling from a single
/// load-balanced queue.
pub fn spawn_inference_workers<B: Backend>(
    model: Model<B>,
    device: B::Device,
    num_workers: usize,
    batch_size_max: usize,
    batch_wait_timeout: Duration,
) -> Sender<NNRequest> {
    // 1. Create a single, bounded global queue.
    // Bounding it to ~4x the total batch capacity provides natural backpressure,
    // preventing the 256 MCTS threads from OOM-crashing your RAM.
    let capacity = batch_size_max * num_workers * 4;
    let (req_tx, req_rx) = bounded::<NNRequest>(capacity);

    // 2. Spawn the requested number of worker threads
    for _ in 0..num_workers {
        let req_rx = req_rx.clone();
        let model = model.clone();
        let device = device.clone();

        thread::spawn(move || {
            let mut batch: Vec<NNRequest> = Vec::with_capacity(batch_size_max);

            loop {
                // Block and wait for the first request to start a new batch
                let Ok(first_req) = req_rx.recv()
                else {
                    break; // Channel closed, gracefully exit thread
                };
                batch.push(first_req);

                // Try to fill the rest of the batch up to `batch_size_max`
                while batch.len() < batch_size_max {
                    // Since your config has eval_batch_wait_timeout_ms: 0,
                    // this will instantly drain available requests and fire.
                    match req_rx.recv_timeout(batch_wait_timeout) {
                        Ok(req) => batch.push(req),
                        Err(RecvTimeoutError::Timeout) => break, // Fire what we have
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }

                // --- TENSOR CONSTRUCTION AND FORWARD PASS ---
                let board_batch = BoardInputTensor::<B>::cat(
                    batch
                        .iter()
                        .map(|req| nn::board_history_input(&req.board_history, &device))
                        .collect_vec(),
                    0,
                );

                let state_batch = StateInputTensor::<B>::cat(
                    batch
                        .iter()
                        .map(|req| Tensor::from_floats([req.state], &device))
                        .collect_vec(),
                    0,
                );

                let (values, raw_policies) = model.forward(board_batch, state_batch);

                let values_data = values.into_data().as_slice::<f32>().unwrap().to_vec();
                let policies_data = raw_policies.into_data().as_slice::<f32>().unwrap().to_vec();

                // --- ROUTE RESPONSES BACK ---
                for (i, req) in batch.drain(..).enumerate() {
                    let value = values_data[i];
                    let mut raw_logits = RawLogits::null();

                    let start_idx = i * POLICY_OUTPUTS;
                    raw_logits
                        .0
                        .copy_from_slice(&policies_data[start_idx..start_idx + POLICY_OUTPUTS]);

                    let _ = req.responder.send(NNResponse { value, raw_logits });
                }
            }
        });
    }

    req_tx
}

#[derive(Clone, Debug)]
pub struct BatchedNNEvaluator {
    /// The global queue to send requests to the Inference Worker.
    worker_tx: Sender<NNRequest>,
}

impl BatchedNNEvaluator {
    pub fn new(worker_tx: Sender<NNRequest>) -> Self {
        Self { worker_tx }
    }
}

impl Evaluator for BatchedNNEvaluator {
    type TraceData = TraceInfo;

    fn trace<S: HasBranches>(
        &self,
        _node: node::NodeId<S>,
        _tree: &Tree,
        pos: &mut Position,
    ) -> Self::TraceData {
        TraceInfo::new(pos)
    }

    fn eval_batch<const X: usize>(
        &mut self,
        tree: &Tree,
        selection: &Selection<X, Self::TraceData>,
        leafs: &[&BatchItem<Self::TraceData>],
    ) -> impl Iterator<Item = Evaluation> {
        let batch_size = leafs.len();
        if batch_size == 0 {
            return vec![].into_iter();
        }

        // Create a one-off channel just for this specific evaluation batch
        let (resp_tx, resp_rx) = bounded::<NNResponse>(batch_size);

        // 1. Submit all leaves to the inference worker
        for &leaf in leafs {
            let history = get_node_history(selection, leaf);
            let state = leaf.data.inputs.state;

            let _ = self.worker_tx.send(NNRequest {
                board_history: history,
                state,
                responder: resp_tx.clone(),
            });
        }

        // 2. Wait for the worker to process them and reply.
        // Because the worker processes requests in the exact order they are received,
        // the responses will arrive in the same order we sent them.
        let mut evaluations = Vec::with_capacity(batch_size);

        for &leaf in leafs {
            // This blocks the MCTS thread until the GPU is done with this specific node.
            let response = resp_rx
                .recv()
                .expect("Inference worker died or disconnected");

            let turn = leaf.turn;
            let moves = tree.move_indices(leaf.node);

            let raw_logits = response.raw_logits;
            let guess = Guess {
                relative_to: turn,
                quality: Quality::new(response.value),
                policy: Policy::from_raw_logits(&raw_logits, moves, 1.0).expect("a policy"),
            };

            evaluations.push(Evaluation::Guess(Box::new(guess)));
        }

        evaluations.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct SelfPlayLimit {
    /// NEVER stop if we haven't found atleast this many nodes.
    pub min_nodes: u64,

    /// ALWAYS stop if we have found atleast this many terminal nodes.
    pub terminal_nodes: u64,

    /// ALWAYS stop if we have found atleast this many nodes.
    /// (equivalent to the UCI nodes limit)
    pub max_nodes: u64,

    pub iterations: u64,
}

impl SelfPlayLimit {
    pub fn is_reached(&self, nodes: u64, terminal_nodes: u64, iterations: u64) -> bool {
        if nodes < self.min_nodes {
            return false;
        }

        nodes >= self.max_nodes
            || terminal_nodes >= self.terminal_nodes
            || iterations >= self.iterations
    }
}

impl Default for SelfPlayLimit {
    fn default() -> Self {
        Self {
            min_nodes: u64::MIN,
            terminal_nodes: u64::MAX,
            max_nodes: u64::MAX,
            iterations: u64::MAX,
        }
    }
}
