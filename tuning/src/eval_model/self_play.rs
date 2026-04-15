use crossbeam_channel::{RecvTimeoutError, Sender, bounded};
use engine::core::{
    config::Configuration,
    position::PgnResultValue,
    search::mcts::{
        CreateNNPartsError, MctsParts,
        back::MctsSolver,
        eval::{
            Evaluation, Evaluator, Policy, Quality, RawLogits,
            nn::{TraceInfo, get_node_history},
        },
        nn::{self, BoardInputTensor, POLICY_OUTPUTS, StateInputTensor},
        node::{
            self, Branch, NodeData,
            node_state::{Evaluated, HasBranches},
        },
        noise::DirichletNoiser,
        search::{BatchItem, Selection},
        select::{Selector, puct},
    },
};
use itertools::Itertools;
use rand::{SeedableRng, rngs::SmallRng};
use rayon::{
    ThreadPoolBuilder,
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
};
use std::{
    cmp::max,
    error::Error,
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
        search::{
            limit::Limit,
            mcts::{
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
    },
    uci::tokens::Tokenizer,
};

use crate::data::{BoardInput, FenItemRaw, StateInput};

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
    Discard,
    Draw,
}

impl TryFrom<&str> for UnfinishedGameHandling {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "discard" => Ok(Self::Discard),
            "draw" => Ok(Self::Draw),
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
    pub fn init(&self) -> Limit {
        Limit {
            iterations: self.max_iterations.unwrap_or(u64::MAX),
            max_nodes: self.max_nodes.unwrap_or(u64::MAX),
            min_nodes: self.min_nodes.unwrap_or(0),
            terminal_nodes: self.max_terminal_nodes.unwrap_or(u64::MAX),
            ..Default::default()
        }
    }
}

pub trait PlayoutItem: for<'a> From<(GameResult, &'a [Decision<Self::Target>])> {
    type Target: Target;
}

pub struct BatchStats {
    pub games_total: usize,
    pub games_completed: usize,
    pub games_solved: usize,
}

pub fn generate_batch<B: Backend, P: PlayoutItem + Send>(
    model: Model<B>,
    fens: &[FenItemRaw],
    limit: &Limit,
    config: &SelfplayConfig,
    mcts: &MctsConfig,
) -> Result<(Vec<P>, BatchStats), Box<dyn Error>>
where
    P: Clone,
    P::Target: Clone,
{
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.concurrency)
        .build()?;

    let device = B::Device::default();

    let worker_tx = spawn_inference_workers(
        model.clone(),
        device.clone(),
        config.eval_models,
        config.eval_batch_size,
        Duration::from_millis(config.eval_batch_wait_timeout_ms),
    );

    let num_fens = fens.len();

    let games_total = fens.len();
    let games_completed = AtomicUsize::new(0);
    let games_solved = AtomicUsize::new(0);

    let time_start = Instant::now();

    let playout_items = pool.install(|| {
        fens.iter()
            .enumerate()
            .collect_vec()
            .into_par_iter()
            .with_max_len(1)
            .flat_map(|(i, fen)| {
                let i = i + 1; // so it starts at 1
                let n = num_fens;

                let fen = fen.fen.clone();

                let parts = MctsTrainParts::new(
                    BatchedNNEvaluator::new(worker_tx.clone()),
                    mcts.dirichlet_alpha,
                    mcts.dirichlet_epsilon,
                );

                match self_play::<P>(i, n, &fen, limit.clone(), config, &parts) {
                    Ok(result) => {
                        let pgn = result.0.to_pgn();
                        log::info!(target: "data", "[FEN {i:>3}/{n:<3}] Generated game:\n{pgn}");

                        games_completed.fetch_add(1, Ordering::Relaxed);
                        let games_completed = games_completed.load(Ordering::Relaxed);
                        let elapsed = time_start.elapsed().as_secs_f32();
                        let games_per_sec = games_completed as f32 / elapsed;
                        let games_per_hour = games_per_sec * 60. * 60.;
                        let perc_complete = (games_completed as f32 / games_total as f32) * 100.0;
                        log::info!(target: "data", "[FEN] {perc_complete:>3.2}% complete  {games_per_hour} games/h  ");

                        if !result.1.is_empty() {
                            games_solved.fetch_add(1, Ordering::Relaxed);
                        }

                        result.1
                            .iter()
                            .map(|x| x.playout_item.clone())
                            .collect::<Vec<_>>()
                    }
                    Err(err) => {
                        log::error!(target: "data", "[FEN {i:>3}/{n:<3}] Error while playing fen '{fen}':\n{err}");
                        vec![]
                    }
                }
            })
            .collect::<Vec<_>>()
    });

    let stats = BatchStats {
        games_total,
        games_completed: games_completed.load(Ordering::Relaxed),
        games_solved: games_solved.load(Ordering::Relaxed),
    };

    Ok((playout_items, stats))
}

#[derive(Default, Debug)]
pub struct MctsTrainStrategy {
    infer: MctsFindBest,

    nodes_begin: u64,
    terminal_nodes_begin: u64,
    iterations: u64,
    time_limit: Option<Instant>,
    start_time: Option<Instant>,

    i: usize,
    n: usize,
}

impl MctsTrainStrategy {
    pub fn new(i: usize, n: usize) -> Self {
        Self {
            infer: MctsFindBest::default(),
            i,
            n,
            ..Default::default()
        }
    }
}

impl MctsStrategy for MctsTrainStrategy {
    type Result = (<MctsFindBest as MctsStrategy>::Result,);
    type Step = (<MctsFindBest as MctsStrategy>::Step,);

    fn start(&mut self, tree: &mut Tree, pos: &Position, limit: &Limit) {
        let time_per_move = limit.time_per_move(pos);
        let now = Instant::now();

        self.infer.start(tree, pos, limit);
        self.nodes_begin = tree.size() as u64;
        self.terminal_nodes_begin = tree.terminal_nodes() as u64;
        self.iterations = 0;
        self.start_time = Some(now);
        self.time_limit = Some(now + time_per_move);
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        self.iterations += 1;
        let step = self.infer.step(tree);
        (step,)
    }

    fn should_stop(&mut self, tree: &Tree, limit: &Limit) -> bool {
        // check limits
        if limit.is_active()
            && limit.is_reached(
                tree.size() as u64 - self.nodes_begin,
                tree.terminal_nodes() as u64 - self.terminal_nodes_begin,
                Instant::now(),
                self.time_limit.unwrap(),
                self.iterations,
            )
        {
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

pub trait Target: for<'a> From<&'a Tree> {}

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

pub fn self_play<P: PlayoutItem>(
    i: usize,
    n: usize,
    fen: &str,
    limit: Limit,
    config: &SelfplayConfig,
    parts: &MctsTrainParts,
) -> Result<(Game, Vec<SelfPlayResult<P>>), Box<dyn Error>>
where
    P::Target: Clone,
{
    let unfinished_game_handling =
        UnfinishedGameHandling::try_from(config.unfinished_games.as_str())
            .map_err(|e| format!("Invalid config for unfinished games handling: {e}"))?;

    let mut game = Game::from_fen(FenImport(&mut Tokenizer::new(fen)))?;

    let mut decisions = Vec::<Decision<P::Target>>::new();
    let mut mcts_state = SearchState::default();

    log::debug!(target: "data", "[FEN {i:>2}/{n:<2}] Starting self-play with fen '{fen}'");

    let eval: GameResult = {
        let game_result;
        let mut completed_moves = 0;
        loop {
            if let Some(result) = game.position().game_result() {
                game_result = result;
                break;
            }

            if config
                .allowed_moves
                .is_some_and(|max_moves| completed_moves >= max_moves)
            {
                match unfinished_game_handling {
                    UnfinishedGameHandling::Discard => {
                        return Ok((game, vec![]));
                    }
                    UnfinishedGameHandling::Draw => {
                        game_result = GameResult::Draw;
                        break;
                    }
                }
            }

            let strat = MctsTrainStrategy::new(i, n);
            let result = mcts(game.position_mut(), parts, &mut mcts_state, &limit, strat);

            let pos = game.position();
            let turn = pos.get_turn();

            let mov = result.0;

            let b_in = board_input(&pos);
            let s_in = state_input(&pos);

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

    log::debug!(target: "data", "[FEN {i:>2}/{n:<2}] Finished self-play with result '{}'", PgnResultValue(Some(eval)));

    let mut result = Vec::<SelfPlayResult<_>>::new();

    for (index, decision) in decisions.iter().enumerate() {
        let decisions_begin = index.saturating_sub(BOARD_INPUT_HISTORY - 1);
        let decisions_end = index;
        let decisions = &decisions[decisions_begin..=decisions_end];
        let playout_item = P::from((eval, decisions));

        result.push(SelfPlayResult {
            playout_item,
            decision: decision.clone(),
        });
    }

    Ok((game, result))
}

#[derive(Debug)]
pub struct MctsTrainParts {
    pub evaluator: BatchedNNEvaluator,
    alpha: f32,
    epsilon: f32,
}

impl<'a> MctsParts for &'a MctsTrainParts {
    type Selector = MctsTrainSelector;
    type Backprop = MctsSolver;
    type Evaluator = BatchedNNEvaluator;
    type Noiser = DirichletNoiser;
    type Instance = MctsTrainParts;

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

// todo: for training try to use a selector that weighs the actualy game results
// higher than the value estimations... maybe that will help idk though
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
    type Score = puct::Score;

    fn score(&self, node: &NodeData, branch: &Branch, cap_n_i: u32) -> Self::Score {
        let n_i = node.visits() as f32;
        let value = node.value();
        let policy = Self::weighted_policy(branch.policy(), self.policy_weight);
        let exploitation = if n_i == 0.0 { 0.0 } else { value / n_i };
        let exploration = self.c * policy * (cap_n_i as f32).sqrt() / (1f32 + n_i);
        puct::Score::new(exploitation + exploration)
    }

    fn min_score(&self) -> Self::Score {
        puct::Score::new(f32::NEG_INFINITY)
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
