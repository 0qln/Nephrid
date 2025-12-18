#![feature(assert_matches)]

//todo: derichlet noise

use engine::core::search::mcts::MctsState;
use engine::core::search::mcts::eval::EvalInfoNode;
use engine::core::search::mcts::eval::Evaluation;
use engine::core::search::mcts::eval::Evaluator;
use engine::core::search::mcts::eval::GameResult;
use engine::core::search::mcts::eval::Guess;
use engine::core::search::mcts::eval::NNEvaluator;
use engine::core::search::mcts::mcts;
use engine::core::search::mcts::strategy::MctsFindBest;
use engine::core::search::mcts::strategy::MctsStrategy;
use std::cell::RefCell;

use burn::prelude::Module;
use burn::record::CompactRecorder;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::env::var;
use std::rc::Rc;
use std::sync::Mutex;
use std::{error::Error, fs, marker::PhantomData};

use burn::{
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    optim::{AdamConfig, Optimizer},
    tensor::backend::AutodiffBackend,
};

use burn::{
    backend::Autodiff,
    config::Config,
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, InMemDataset},
    },
    nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction},
    prelude::Backend,
    tensor::{Int, Tensor, TensorData},
    train::{
        ClassificationOutput, RegressionOutput, TrainOutput, TrainStep, ValidStep,
        metric::{Adaptor, LossInput},
    },
};
use burn_cuda::{Cuda, CudaDevice};
use engine::core::search::mcts::node::Tree;
use engine::{
    core::{
        color::Color,
        r#move::Move,
        move_iter::sliding_piece::magics,
        position::Position,
        search::{
            limit::Limit,
            mcts::nn::{
                BOARD_INPUT_TENSOR_DIM, BoardInputFloats, Model, ModelConfig,
                POLICY_TARGET_TENSOR_DIM, STATE_INPUT_TENSOR_DIM, StateInputFloats, VALUE_DRAW,
                VALUE_LOSE, VALUE_OUTPUT_TENSOR_DIM, VALUE_WIN, board_input, state_input,
            },
        },
        zobrist,
    },
    misc::DebugMode,
    uci::{sync::CancellationToken, tokens::Tokenizer},
};
use itertools::Itertools;
use std::path::{Path, PathBuf};

#[cfg(test)]
pub mod test;

fn main() {
    magics::init();
    zobrist::init();
    log4rs::init_file("./tuning/src/eval_model/log4rs.yml", Default::default()).unwrap();

    type Backend = Cuda<f32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = CudaDevice::default();
    log::info!("Device: {:?}", device);

    let train_dir = "tuning/out/eval_model";
    train::<AutodiffBackend>(
        train_dir,
        TrainingConfig::new(
            ModelConfig::new(),
            AdamConfig::new(),
            "UHO_Lichess_4852_v1.epd".to_string(),
        ),
        device,
    );

    // todo: for deployment, we can `include!()` the serialized `Record`.
}

#[derive(Clone, Debug)]
pub struct PlayoutBatch<B: Backend> {
    pub board_inputs: Tensor<B, BOARD_INPUT_TENSOR_DIM>,
    pub state_inputs: Tensor<B, STATE_INPUT_TENSOR_DIM>,
    pub value_targets: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    pub policy_targets: Tensor<B, POLICY_TARGET_TENSOR_DIM, Int>,
}

#[derive(Clone, Default, Debug)]
pub struct FenItemRaw {
    /// The position string that is to be played out
    pub fen: String,
}

impl FenItemRaw {
    pub fn new(s: String) -> Self {
        Self { fen: s }
    }
}

#[derive(Clone, Debug)]
pub struct PlayoutItem {
    /// Board info
    pub board_input: BoardInputFloats,

    /// State info
    pub state_input: StateInputFloats,

    /// Target Value
    pub value_target: f32,

    /// Target Move index
    pub policy_target: usize,
}

pub struct FenDataset {
    dataset: InMemDataset<FenItemRaw>,
}

impl Dataset<FenItemRaw> for FenDataset {
    fn get(&self, index: usize) -> Option<FenItemRaw> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl FenDataset {
    /// Creates a new train dataset.
    pub fn train(path: &str) -> Self {
        Self::new(path, "train")
    }

    /// Creates a new test dataset.
    pub fn test(path: &str) -> Self {
        Self::new(path, "test")
    }

    fn new(path: &str, split: &str) -> Self {
        let root = FenDataset::load_path(path, split);
        let fens = FenDataset::read_edp(&root, split, 0.9, 1_000);

        let items: Vec<_> = fens
            .into_iter()
            .map(|s| FenItemRaw::new(s.to_owned()))
            .collect();

        println!("{:?}", items);

        let dataset = InMemDataset::new(items);

        Self { dataset }
    }

    fn load_path(path: &str, _split: &str) -> PathBuf {
        let mut buf = PathBuf::new();
        buf.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
        buf.push("tuning/resources/datasets/edp");
        buf.push(path);
        buf
    }

    /// num_fens_total: the number of fens in |train + test|
    fn read_edp<P: AsRef<Path>>(
        root: &P,
        split: &str,
        split_ratio: f32,
        num_fens_total: usize,
    ) -> Vec<String> {
        let edp = fs::read_to_string(root).expect("Couldn't read path");
        let lines = edp
            .lines()
            .map(|l| l.to_owned())
            .take(num_fens_total)
            .collect_vec();
        let split_idx = (lines.len() as f32 * split_ratio) as usize;
        match split {
            "train" => lines[..=split_idx].to_vec(),
            "test" => lines[split_idx + 1..].to_vec(),
            _ => panic!("invalid split"),
        }
    }
}

pub struct LossOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    // Value output
    pub value_loss: Tensor<B, 1>,
    // Quality Output
    pub policy_loss: Tensor<B, 1>,
}

impl<B: Backend> LossOutput<B> {
    pub fn new(value_output: RegressionOutput<B>, policy_output: ClassificationOutput<B>) -> Self {
        Self {
            loss: value_output.loss.clone() + policy_output.loss.clone(),
            value_loss: value_output.loss,
            policy_loss: policy_output.loss,
        }
    }
}

// impl<B: Backend> ItemLazy for LossOutput<B> {
//     type ItemSync = LossOutput<NdArray>;

//     fn sync(self) -> Self::ItemSync {
//         LossOutput {
//             value_loss: self.value_loss.sync(),
//             policy_loss: self.policy_loss.sync(),
//         }
//     }
// }

impl<B: Backend> Adaptor<LossInput<B>> for LossOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

type BoardInputTensor<B> = Tensor<B, BOARD_INPUT_TENSOR_DIM>;
type StateInputTensor<B> = Tensor<B, STATE_INPUT_TENSOR_DIM>;

pub fn forward_with_loss<B: Backend>(
    this: &Model<B>,
    board_input: BoardInputTensor<B>,
    state_input: StateInputTensor<B>,
    target_value: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    target_policy: Tensor<B, POLICY_TARGET_TENSOR_DIM, Int>,
) -> LossOutput<B> {
    let (value_output, policy_output) = this.forward(board_input, state_input);

    let value_loss =
        MseLoss::new().forward(value_output.clone(), target_value.clone(), Reduction::Auto);

    let policy_loss = CrossEntropyLossConfig::new()
        .init(&policy_output.device())
        .forward(policy_output.clone(), target_policy.clone());

    LossOutput::new(
        RegressionOutput::new(value_loss, value_output, target_value),
        ClassificationOutput::new(policy_loss, policy_output, target_policy),
    )
}

impl<B: AutodiffBackend> TrainStep<PlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: PlayoutBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<PlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: PlayoutBatch<B>) -> LossOutput<B> {
        forward_with_loss(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        )
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,

    pub optimizer: AdamConfig,

    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 8)]
    pub batch_size: usize,

    #[config(default = 0x_dead_beef)]
    pub seed: u64,

    #[config(default = 1.0e-5)]
    pub learning_rate: f64,

    pub edp_dataset_path: String,
}

fn clean_dir(dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    fs::remove_dir_all(dir).ok();
    fs::create_dir_all(dir).ok();
}

#[derive(Clone, Default)]
pub struct IdentityBatcher<I> {
    item: PhantomData<I>,
}

impl<B: Backend, I: Send + Sync> Batcher<B, I, Vec<I>> for IdentityBatcher<I> {
    fn batch(&self, items: Vec<I>, _device: &B::Device) -> Vec<I> {
        items
    }
}

pub fn train<B: AutodiffBackend>(
    output_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) -> Option<String> {
    let device = Rc::new(device);

    let artifact_dir = &format!("{output_dir}/artifacts");
    clean_dir(artifact_dir);

    config
        .save(format!("{output_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    // Create the dataloaders.
    let dataloader_train =
        DataLoaderBuilder::<B, _, _>::new(IdentityBatcher::<FenItemRaw>::default())
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(0)
            .build(FenDataset::train(&config.edp_dataset_path));

    // let dataloader_test =
    //     DataLoaderBuilder::<B, _, _>::new(IdentityBatcher::<FenItemRaw>::default())
    //         .batch_size(config.batch_size)
    //         .shuffle(config.seed)
    //         .num_workers(0)
    //         .build(FenDataset::test(&config.edp_dataset_path));

    // Create the model and optimizer.
    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimizer.init();

    let mut result_weights: Option<String> = None;

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        {
            for (iteration, fens_batch) in dataloader_train.iter().enumerate() {
                let playouts_batch =
                    generate_batch(&model, &device, &fens_batch).expect("Failed to generate batch");

                let result: _ = TrainStep::step(&model, playouts_batch.clone());

                let msg = format!(
                    "[Train - Epoch {} - Iteration {}] Loss {:.5} (Value: {:.5}, Policy: {:.5})",
                    epoch,
                    iteration,
                    result.item.loss.clone().into_scalar(),
                    result.item.value_loss.clone().into_scalar(),
                    result.item.policy_loss.clone().into_scalar(),
                );
                log::info!(target: "reports::train", "{msg}");

                // Gradients for the current backward pass
                let grads = result.grads;
                // Update the model using the optimizer.
                model = optim.step(config.learning_rate, model, grads);
            }
        }

        // // Get the model without autodiff.
        // let model_valid = model.valid();

        // // Implement our validation loop.
        // for (iteration, batch) in dataloader_test.iter().enumerate() {
        //     let result = forward_with_loss(model_valid, batch);

        //     println!(
        //         "[Valid - Epoch {} - Iteration {}] Loss {}",
        //         epoch,
        //         iteration,
        //         result.loss.clone().into_scalar(),
        //     );
        // }

        let weights_path = format!("{artifact_dir}/model_e-{epoch}");

        model
            .clone()
            .save_file(&weights_path, &CompactRecorder::new())
            .expect("Trained model should be saved successfully");

        result_weights = Some(weights_path);
    }

    result_weights
}

#[derive(Clone, Default)]
pub struct PlayoutBatcher;

impl<B: Backend> Batcher<B, PlayoutItem, PlayoutBatch<B>> for PlayoutBatcher {
    fn batch(&self, items: Vec<PlayoutItem>, device: &B::Device) -> PlayoutBatch<B> {
        let boards = items
            .iter()
            .map(|x| TensorData::from([x.board_input]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let states = items
            .iter()
            .map(|x| TensorData::from([x.state_input]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let values = items
            .iter()
            .map(|x| TensorData::from([[x.value_target]]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let policies = items
            .iter()
            .map(|x| TensorData::from([x.policy_target]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        PlayoutBatch {
            board_inputs: Tensor::cat(boards, 0),
            state_inputs: Tensor::cat(states, 0),
            value_targets: Tensor::cat(values, 0),
            policy_targets: Tensor::cat(policies, 0),
        }
    }
}

fn generate_batch<B: AutodiffBackend>(
    model: &Model<B>,
    device: &B::Device,
    fens: &[FenItemRaw],
) -> Result<PlayoutBatch<B>, Box<dyn Error>> {
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

            match self_play(&fen, model, device) {
                Ok(result) => {
                    // todo: replace this with printing the game in PGN format
                    // log::debug!(target: "games", "[Fen {fen}] {moves}");

                    // log misc information
                    let moves = result
                        .iter()
                        .map(|x| format!("({}, {:?})", x.decision.state.mov, x.decision.stats))
                        .join(" ");
                    log::debug!(target: "games", "[Fen {fen}] {moves}");

                    // map each playout_item to a training target.
                    result.iter().map(|x| x.playout_item.clone()).collect_vec()
                }
                Err(err) => {
                    log::error!(target: "games", "[Fen {fen}] Error: {err}");
                    vec![]
                }
            }
        })
        .collect::<Vec<_>>();

    let batcher = PlayoutBatcher::default();
    Ok(batcher.batch(playout_items, device))
}

#[derive(Default, Debug)]
pub struct MctsTrain {
    infer: MctsFindBest,
}

impl MctsStrategy for MctsTrain {
    type Result = (<MctsFindBest as MctsStrategy>::Result, Tree);
    type Step = (<MctsFindBest as MctsStrategy>::Step,);

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        let inference_result = self.infer.result(tree);
        let tree = tree.to_owned();
        (inference_result, tree)
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        (self.infer.step(tree),)
    }
}

// The inputs to the model.
#[derive(Debug)]
struct Input {
    board_in: BoardInputFloats,
    state_in: StateInputFloats,
}

// Info to help find the training target.
// 0: Most visited move / best_move.
#[derive(Debug)]
struct Target {
    mov: Move,
}

// Some state info at the time of the move.
// 0: The move that was made
// 1: The color of the moving player.
#[derive(Debug)]
struct State {
    mov: Move,
    moving_color: Color,
}

// Some interesting stats about the decision.
#[derive(Debug)]
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

#[derive(Debug)]
struct Decision {
    input: Input,
    target: Target,
    state: State,
    stats: Stats,
}

#[derive(Debug)]
struct SelfPlayResult {
    playout_item: PlayoutItem,
    decision: Decision,
}

fn self_play<B: Backend>(
    pos: &str,
    model: Model<B>,
    device: B::Device,
) -> Result<Vec<SelfPlayResult>, Box<dyn Error>> {
    let limit = Limit {
        is_active: true,
        winc: 10000,
        binc: 10000,
        wtime: 0,
        btime: 0,
        ..Default::default()
    };
    let debug = DebugMode::default();
    let ct = CancellationToken::new();

    let tok = &mut Tokenizer::new(pos);
    let mut pos: Position = tok.try_into()?;

    let mut decisions = Vec::<Decision>::new();
    let mut mcts_state = MctsState::new(Default::default(), model, device);

    let eval: GameResult = {
        let game_result;
        loop {
            let turn = pos.get_turn();
            let result = mcts(
                &pos,
                &mut mcts_state,
                limit.clone(),
                debug.clone(),
                ct.clone(),
                MctsTrain::default(),
            );

            let mov = result.0;
            let tree = result.1;

            if let Some(Evaluation::Terminal(x)) =
                NNEvaluator::<B, 1>::eval_terminal(&tree.get_root().borrow(), &pos)
            {
                game_result = x;
                break;
            }

            let model = &mcts_state.nn;
            let device = &mcts_state.device;
            let mut evaluator = NNEvaluator::<_, 1>::new(model, device);
            evaluator.prepare_node(
                0,
                Rc::new(RefCell::new(EvalInfoNode::new_root(None))),
                tree.get_root(),
                &pos,
            );
            evaluator.eval_guesses();
            let guess = evaluator.get_eval(0);
            let guess = guess.expect("The evaluator should have generated the guess");
            let guess = guess.guess().expect(
                "We specifically told the evaluator to make a guess and not a terminal evaluation.",
            );

            let b_in = board_input(&pos);
            let s_in = state_input(&pos);

            let mov = mov.expect("");
            pos.make_move(mov);
            mcts_state.tree.advance_to(|b| b.mov() == mov);

            let state = State { mov, moving_color: turn };
            let input = Input { board_in: b_in, state_in: s_in };
            let target = Target { mov };
            let stats = Stats::new(guess.clone());
            decisions.push(Decision { input, target, state, stats });
        }
        game_result
    };

    let mut result = Vec::<SelfPlayResult>::new();

    for decision in decisions {
        let value_target = match eval {
            GameResult::Draw => VALUE_DRAW,
            GameResult::Win { relative_to } => {
                if relative_to == decision.state.moving_color {
                    VALUE_WIN
                } else {
                    VALUE_LOSE
                }
            }
        };

        log::debug!(target: "reports::train", "Train item: {:?}, value_target: {:?}", decision, value_target);

        let playout_item = PlayoutItem {
            board_input: decision.input.board_in,
            state_input: decision.input.state_in,
            value_target,
            policy_target: usize::from(decision.target.mov),
        };

        result.push(SelfPlayResult { playout_item, decision });
    }

    Ok(result)
}
