use burn::nn::loss::CrossEntropyLoss;
use burn::prelude::Module;
use burn::record::CompactRecorder;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::env::var;
use std::sync::Mutex;
use std::{error::Error, fs, marker::PhantomData};

use burn::{
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::backend::AutodiffBackend,
};

use burn::{
    backend::{Autodiff, NdArray},
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
        metric::{Adaptor, ItemLazy, LossInput},
    },
};
use burn_cuda::{Cuda, CudaDevice};
use engine::{
    core::{
        color::Color,
        r#move::Move,
        move_iter::sliding_piece::magics,
        position::Position,
        search::{
            self, MctsFindBest, MctsStrategy,
            limit::Limit,
            mcts::{
                self, Evaluation, GameResult,
                eval::model_only_policy::{
                    BOARD_INPUT_TENSOR_DIM, BoardInputFloats, Model, ModelConfig,
                    POLICY_TARGET_TENSOR_DIM, STATE_INPUT_TENSOR_DIM, StateInputFloats,
                    board_input, state_input,
                },
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
    pub policy_loss: ClassificationOutput<B>,
}

impl<B: Backend> LossOutput<B> {
    pub fn loss(&self) -> Tensor<B, 1> {
        // todo: weight decay loss
        let ref policy = self.policy_loss;
        policy.loss.clone()
    }
}

impl<B: Backend> ItemLazy for LossOutput<B> {
    type ItemSync = LossOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        LossOutput {
            policy_loss: self.policy_loss.sync(),
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for LossOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss())
    }
}

type BoardInputTensor<B> = Tensor<B, BOARD_INPUT_TENSOR_DIM>;
type StateInputTensor<B> = Tensor<B, STATE_INPUT_TENSOR_DIM>;

pub fn forward_with_loss<B: Backend>(
    this: &Model<B>,
    board_input: BoardInputTensor<B>,
    state_input: StateInputTensor<B>,
    target_policy: Tensor<B, POLICY_TARGET_TENSOR_DIM, Int>,
) -> LossOutput<B> {
    let policy_output = this.forward(board_input, state_input);

    let policy_loss = CrossEntropyLossConfig::new()
        .init(&policy_output.device())
        .forward(policy_output.clone(), target_policy.clone());

    LossOutput {
        policy_loss: ClassificationOutput::new(policy_loss, policy_output, target_policy),
    }
}

impl<B: AutodiffBackend> TrainStep<PlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: PlayoutBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.policy_targets,
        );

        TrainOutput::new(self, item.loss().backward(), item)
    }
}

impl<B: Backend> ValidStep<PlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: PlayoutBatch<B>) -> LossOutput<B> {
        forward_with_loss(
            self,
            batch.board_inputs,
            batch.state_inputs,
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

                for _ in 0..20 {
                    // let result = TrainStep::step(&model, playouts_batch.clone());
                    let batch = playouts_batch.clone();
                    let output = model.forward(batch.board_inputs, batch.state_inputs);

                    // let policy_loss = CrossEntropyLossConfig::new()
                    //     .init(&policy_output.device())
                    //     .forward(policy_output.clone(), target_policy.clone());

                    let loss = CrossEntropyLoss::new(None, &output.device())
                        .forward(output.clone(), playouts_batch.policy_targets.clone());

                    // let loss = result.item.loss();

                    let msg = format!(
                        "[Train - Epoch {} - Iteration {}] Loss {:.5}",
                        epoch,
                        iteration,
                        loss.clone().into_scalar()
                    );
                    log::info!(target: "reports::train", "{msg}");

                    // Gradients for the current backward pass
                    let grads = loss.backward();
                    // Gradients linked to each parameter of the model.
                    let grads = GradientsParams::from_grads(grads, &model);
                    // Update the model using the optimizer.
                    model = optim.step(config.learning_rate, model, grads);
                }
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

        let policies = items
            .iter()
            .map(|x| TensorData::from([x.policy_target]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        PlayoutBatch {
            board_inputs: Tensor::cat(boards, 0),
            state_inputs: Tensor::cat(states, 0),
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

            match self_play(&fen, &model) {
                Ok(result) => {
                    let moves = result
                        .iter()
                        .map(|x| format!("{}", x.1.state.mov))
                        .join(" ");
                    log::debug!(target: "games", "[Fen {fen}] {moves}");
                    result.iter().map(|x| x.0.clone()).collect_vec()
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
    type Result = (<MctsFindBest as MctsStrategy>::Result, mcts::Tree);
    type Step = (<MctsFindBest as MctsStrategy>::Step,);

    fn result(&mut self, tree: &mut mcts::Tree) -> Self::Result {
        let inference_result = self.infer.result(tree);
        let tree = tree.to_owned();
        (inference_result, tree)
    }

    fn step(&mut self, tree: &mut mcts::Tree) -> Self::Step {
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

#[derive(Debug)]
struct Decision {
    input: Input,
    target: Target,
    state: State,
}

fn self_play<B: Backend>(
    pos: &str,
    model: &Model<B>,
) -> Result<Vec<(PlayoutItem, Decision)>, Box<dyn Error>> {
    let limit = Limit {
        is_active: true,
        winc: 1000,
        binc: 1000,
        wtime: 0,
        btime: 0,
        ..Default::default()
    };
    let debug = DebugMode::default();
    let ct = CancellationToken::new();

    let tok = &mut Tokenizer::new(pos);
    let mut pos: Position = tok.try_into()?;

    let mut decisions = Vec::<Decision>::new();

    let eval: GameResult = {
        let game_result;
        loop {
            let turn = pos.get_turn();
            let result = search::mcts::<MctsTrain, _>(
                pos.clone(),
                model,
                limit.clone(),
                debug.clone(),
                ct.clone(),
            );
            let mov = result.0;
            let tree = result.1;

            match tree.get_root().eval(&pos, model) {
                Evaluation::Guess { .. } => { /* continue with game */ }
                Evaluation::Terminal(result) => {
                    game_result = result;
                    break;
                }
            };

            let b_in = board_input(&pos);
            let s_in = state_input(&pos);

            let mov = mov.expect("if the ");
            pos.make_move(mov);

            let state = State { mov, moving_color: turn };
            let input = Input { board_in: b_in, state_in: s_in };
            let target = Target { mov };
            decisions.push(Decision { input, target, state });
        }
        game_result
    };

    let mut result = Vec::<(PlayoutItem, Decision)>::new();

    for decision in decisions {
        let playout_item = PlayoutItem {
            board_input: decision.input.board_in,
            state_input: decision.input.state_in,
            policy_target: usize::from(decision.target.mov),
        };

        result.push((playout_item, decision));
    }

    Ok(result)
}
