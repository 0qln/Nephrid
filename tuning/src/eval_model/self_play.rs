use engine::core::{
    config::Configuration,
    search::mcts::{
        CreateNNPartsError, MctsParts,
        back::DefaultBackuper,
        eval::nn::NNEvaluator,
        node::{Branch, NodeData},
        noise::DirichletNoiser,
        select::{Selector, puct},
    },
};
use rand::{SeedableRng, rngs::SmallRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{error::Error, path::PathBuf, sync::Mutex, time::Instant};

use burn::{module::AutodiffModule, prelude::Backend, tensor::backend::AutodiffBackend};
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

pub trait PlayoutItem: for<'a> From<(GameResult, &'a [Decision<Self::Target>])> {
    type Target: Target;
}

pub fn generate_batch<B: AutodiffBackend, P: PlayoutItem + Send>(
    model: &Model<B>,
    fens: &[FenItemRaw],
    limit: &Limit,
) -> Result<Vec<P>, Box<dyn Error>>
where
    P: Clone,
    P::Target: Clone,
{
    let mutex = Mutex::new(model.to_owned());
    let playout_items = fens
        .par_iter()
        .flat_map(|fen| {
            let fen = fen.fen.clone();
            let model = {
                let lock_result = mutex.lock();
                let model_lock = lock_result.expect("Unable to acquire lock.");
                model_lock.valid()
            };

            let device = B::Device::default();

            match self_play::<_, P>(&fen, limit.clone(), model, device) {
                Ok(result) => {
                    // log pgn information
                    let pgn = result.0.to_pgn();
                    log::debug!(target: "games", "{pgn}");

                    // map each playout_item to a training target.
                    let items = result.1;
                    items
                        .iter()
                        .map(|x| x.playout_item.clone())
                        .collect::<Vec<_>>()
                }
                Err(err) => {
                    log::error!(target: "games", "[Fen {fen}] Error: {err}");
                    vec![]
                }
            }
        })
        .collect::<Vec<_>>();

    Ok(playout_items)
}

#[derive(Default, Debug)]
pub struct MctsTrainStrategy {
    infer: MctsFindBest,
    nodes_begin: u64,
    terminal_nodes_begin: u64,
    iterations: u64,
    time_limit: Option<Instant>,
}

impl MctsTrainStrategy {
    pub fn new() -> Self {
        Self {
            infer: MctsFindBest::default(),
            ..Default::default()
        }
    }
}

impl MctsStrategy for MctsTrainStrategy {
    type Result = (<MctsFindBest as MctsStrategy>::Result,);
    type Step = (<MctsFindBest as MctsStrategy>::Step,);

    fn start(&mut self, tree: &mut Tree, pos: &Position, limit: &Limit) {
        let time_per_move = limit.time_per_move(pos);

        self.infer.start(tree, pos, limit);
        self.nodes_begin = tree.size() as u64;
        self.terminal_nodes_begin = tree.terminal_nodes() as u64;
        self.time_limit = Some(Instant::now() + time_per_move);
        self.iterations = 0;
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        self.iterations += 1;
        let step = self.infer.step(tree);
        (step,)
    }

    fn should_stop(&mut self, tree: &Tree, limit: &Limit) -> bool {
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

        false
    }

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        let inference_result = self.infer.result(tree);
        log::info!(target: "games", "[MCTS] Iterations: {} Nodes: {}", self.iterations, tree.size() as u64 - self.nodes_begin);
        // todo: log::info!(target: "games", "Base Policy: {:#?}",
        // tree.get_root().borrow().iter_branches().map(|b| format!("[{}] p: {}, WDL
        // found: {}", b.mov(), b.policy(), todo!("Gather how many w/d/l we have found
        // in the game tree during mcts"))).collect_vec());
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
    pub policy_variance: f32,
    pub policy_entropy: f32,
}

impl Stats {
    pub fn new(guess: Guess) -> Self {
        let policy_values = guess.policy();

        let policy_avg = policy_values.sum() / policy_values.len() as f32;

        let variance = policy_values
            .iter()
            .map(|p| (p - policy_avg).powi(2))
            .sum::<f32>()
            / policy_values.len() as f32;
        let policy_variance = variance.sqrt();

        let policy_entropy = -policy_values
            .iter()
            .filter(|&p| p > 0.0)
            .map(|p| p * p.log2())
            .sum::<f32>();

        Self {
            policy_avg,
            policy_variance,
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
        Self(decisions.iter().map(|d| d.input.board_in).collect::<Vec<_>>())
    }
}

#[derive(Debug)]
pub struct SelfPlayResult<P: PlayoutItem> {
    pub playout_item: P,
    pub decision: Decision<P::Target>,
}

pub fn self_play<B: Backend, P: PlayoutItem>(
    fen: &str,
    limit: Limit,
    model: Model<B>,
    device: B::Device,
) -> Result<(Game, Vec<SelfPlayResult<P>>), Box<dyn Error>>
where
    P::Target: Clone,
{
    let fen = FenImport(&mut Tokenizer::new(fen));
    let mut game = Game::from_fen(fen)?;

    let mut decisions = Vec::<Decision<P::Target>>::new();
    let nn_state = MctsTrainParts::new(model, device, 0.3, 0.25);
    let mut mcts_state = SearchState::default();

    let eval: GameResult = {
        let game_result;
        loop {
            if let Some(result) = game.position().game_result() {
                game_result = result;
                break;
            }

            let strat = MctsTrainStrategy::new();
            let result = mcts(
                game.position_mut(),
                &nn_state,
                &mut mcts_state,
                &limit,
                strat,
            );

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
        }
        game_result
    };

    let mut result = Vec::<SelfPlayResult<_>>::new();

    for (index, decision) in decisions.iter().enumerate() {
        // Subtract (HISTORY - 1) so that the inclusive range length is exactly HISTORY.
        let decisions_begin = index.saturating_sub(BOARD_INPUT_HISTORY - 1);
        let decisions = &decisions[decisions_begin..=index];
        let playout_item = P::from((eval, decisions));

        result.push(SelfPlayResult {
            playout_item,
            decision: decision.clone(),
        });
    }

    Ok((game, result))
}

#[derive(Debug)]
pub struct MctsTrainParts<B: Backend> {
    pub nn: Box<Model<B>>,
    pub device: B::Device,
    alpha: f32,
    epsilon: f32,
}

impl<'a, B: Backend> MctsParts for &'a MctsTrainParts<B> {
    type Selector = MctsTrainSelector;
    type Backprop = DefaultBackuper;
    type Evaluator = NNEvaluator<'a, 'a, B>;
    type Noiser = DirichletNoiser;
    type Instance = MctsTrainParts<B>;

    fn selector(&self) -> Self::Selector {
        Default::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        NNEvaluator::<_>::new(&self.nn, &self.device)
    }

    fn backprop(&self) -> Self::Backprop {
        Default::default()
    }

    fn noiser(&self) -> Self::Noiser {
        let rng = SmallRng::from_os_rng();
        DirichletNoiser::new(self.alpha, self.epsilon, rng)
    }
}

impl<B: Backend> TryFrom<&Configuration> for MctsTrainParts<B> {
    type Error = CreateNNPartsError;

    fn try_from(config: &Configuration) -> Result<Self, Self::Error> {
        let alpha = config.dirichlet_alpha();
        let epsilon = config.dirichlet_epsilon();
        let weights = PathBuf::from(config.weights_path());
        let device = B::Device::default();
        let nn = Model::try_from((weights, &device))?;
        Ok(Self::new(nn, device, alpha, epsilon))
    }
}

impl<B: Backend> MctsTrainParts<B> {
    pub fn new(nn: Model<B>, device: B::Device, alpha: f32, epsilon: f32) -> Self {
        Self {
            nn: Box::new(nn),
            device,
            alpha,
            epsilon,
        }
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
        puct::Score(exploitation + exploration)
    }

    fn min_score(&self) -> Self::Score {
        puct::Score(f32::NEG_INFINITY)
    }
}
