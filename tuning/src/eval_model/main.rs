#![feature(assert_matches)]
use burn::{nn::loss::BinaryCrossEntropyLossConfig, train::MultiLabelClassificationOutput};
use engine::core::{
    config::Configuration,
    search::mcts::{
        CreateNNPartsError, MctsParts, SearchState,
        back::DefaultBackuper,
        eval::{GameResult, Guess, RawPolicy, nn::NNEvaluator, softmax},
        mcts,
        nn::{BOARD_INPUT_HISTORY, POLICY_OUTPUTS, board_history_input},
        node::{Branch, Value},
        select::puct,
        strategy::{MctsFindBest, MctsStrategy},
    },
    turn::Turn,
};
use rand::SeedableRng;
use std::cmp::max;

use burn::{
    prelude::Module,
    record::CompactRecorder,
    tensor::{Tensor, backend::Backend},
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{env::var, error::Error, fs, marker::PhantomData, rc::Rc, sync::Mutex};

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
    tensor::{Float, Int, TensorData},
    train::{
        ClassificationOutput, RegressionOutput, TrainOutput, TrainStep,
        metric::{Adaptor, LossInput},
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
            limit::Limit,
            mcts::{
                nn::{
                    BOARD_INPUT_TENSOR_DIM, BoardInputFloats, Model, ModelConfig,
                    STATE_INPUT_TENSOR_DIM, StateInputFloats, VALUE_DRAW, VALUE_LOSE,
                    VALUE_OUTPUT_TENSOR_DIM, VALUE_WIN, board_input, state_input,
                },
                node::Tree,
                noise::DirichletNoiser,
                select::Selector,
            },
        },
        zobrist,
    },
    misc::DebugMode,
    uci::sync::CancellationToken,
};
use itertools::Itertools;
use rand::rngs::SmallRng;
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
pub struct SLCPlayoutBatch<B: Backend> {
    pub board_inputs: Tensor<B, BOARD_INPUT_TENSOR_DIM>,
    pub state_inputs: Tensor<B, STATE_INPUT_TENSOR_DIM>,
    pub value_targets: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    pub policy_targets: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug)]
pub struct MLCPlayoutBatch<B: Backend> {
    pub board_inputs: Tensor<B, BOARD_INPUT_TENSOR_DIM>,
    pub state_inputs: Tensor<B, STATE_INPUT_TENSOR_DIM>,
    pub value_targets: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    pub policy_targets: Tensor<B, 2, Int>,
}

#[derive(Clone, Debug)]
pub struct ExactLossPlayoutBatch<B: Backend> {
    pub board_inputs: Tensor<B, BOARD_INPUT_TENSOR_DIM>,
    pub state_inputs: Tensor<B, STATE_INPUT_TENSOR_DIM>,
    pub value_targets: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    pub policy_targets: Tensor<B, 2, Float>,
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
pub struct BoardInput(pub Vec<BoardInputFloats>);

impl<'a, T: Target> From<&'a [Decision<T>]> for BoardInput {
    fn from(decisions: &'a [Decision<T>]) -> Self {
        Self(decisions.iter().map(|d| d.input.board_in).collect_vec())
    }
}

#[derive(Clone, Debug)]
pub struct StateInput(pub StateInputFloats);

impl<'a, T: Target> From<&'a Decision<T>> for StateInput {
    fn from(decision: &'a Decision<T>) -> Self {
        Self(decision.input.state_in)
    }
}

#[derive(Clone, Debug)]
pub struct ValueTarget(f32);

impl From<(GameResult, Turn)> for ValueTarget {
    /// value target depending on the game result and the current moving player.
    fn from((result, moving_color): (GameResult, Turn)) -> Self {
        match result {
            GameResult::Draw => Self(VALUE_DRAW),
            GameResult::Win { relative_to } => {
                if relative_to == moving_color {
                    Self(VALUE_WIN)
                }
                else {
                    Self(VALUE_LOSE)
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct SLCPlayoutItem {
    pub board_input: BoardInput,
    pub state_input: StateInput,
    pub value_target: ValueTarget,
    pub policy_target: usize,
}

impl<'a> From<(GameResult, &'a [Decision<SLCTarget>])> for SLCPlayoutItem {
    fn from((result, decisions): (GameResult, &'a [Decision<SLCTarget>])) -> Self {
        let decision = &decisions[decisions.len() - 1];
        let player = decision.state.moving_color;
        Self {
            board_input: BoardInput::from(decisions),
            state_input: StateInput::from(decision),
            value_target: ValueTarget::from((result, player)),
            policy_target: usize::from(decision.target.mov),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MLCPlayoutItem {
    pub board_input: BoardInput,
    pub state_input: StateInput,
    pub value_target: ValueTarget,
    pub policy_target: Vec<usize>,
}

impl<'a> From<(GameResult, &'a [Decision<MLCTarget>])> for MLCPlayoutItem {
    fn from((result, decisions): (GameResult, &'a [Decision<MLCTarget>])) -> Self {
        let decision = &decisions[decisions.len() - 1];
        let player = decision.state.moving_color;
        Self {
            board_input: BoardInput::from(decisions),
            state_input: StateInput::from(decision),
            value_target: ValueTarget::from((result, player)),
            policy_target: decision
                .target
                .moves
                .iter()
                .cloned()
                .map(usize::from)
                .collect_vec(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExactLossPlayoutItem {
    pub board_input: BoardInput,
    pub state_input: StateInput,
    pub value_target: ValueTarget,
    pub policy_target: Vec<f32>,
}

impl<'a> From<(GameResult, &'a [Decision<ExactLossTarget>])> for ExactLossPlayoutItem {
    fn from((result, decisions): (GameResult, &'a [Decision<ExactLossTarget>])) -> Self {
        let decision = &decisions[decisions.len() - 1];
        let player = decision.state.moving_color;
        Self {
            board_input: BoardInput::from(decisions),
            state_input: StateInput::from(decision),
            value_target: ValueTarget::from((result, player)),
            policy_target: decision.target.raw_policy.iter().collect_vec(),
        }
    }
}

impl PlayoutItem for ExactLossPlayoutItem {
    type Target = ExactLossTarget;
}

trait PlayoutItem: for<'a> From<(GameResult, &'a [Decision<Self::Target>])> {
    type Target: Target;
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

/// label loss output
pub struct LossOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    // Value output
    pub value_loss: Tensor<B, 1>,
    // Quality Output
    pub policy_loss: Tensor<B, 1>,
}

impl<B: Backend> LossOutput<B> {
    pub fn new(value_loss: Tensor<B, 1>, policy_loss: Tensor<B, 1>) -> Self {
        Self {
            loss: value_loss.clone() + policy_loss.clone(),
            value_loss,
            policy_loss,
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for LossOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
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

type BoardInputTensor<B> = Tensor<B, BOARD_INPUT_TENSOR_DIM>;
type StateInputTensor<B> = Tensor<B, STATE_INPUT_TENSOR_DIM>;

/// Foward with loss. (Policy loss: single label classification)
pub fn forward_with_loss_slc<B: Backend>(
    this: &Model<B>,
    board_input: BoardInputTensor<B>,
    state_input: StateInputTensor<B>,
    target_value: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    target_policy: Tensor<B, 1, Int>,
) -> LossOutput<B> {
    let (value_output, policy_output) = this.forward(board_input, state_input);

    let value_loss =
        MseLoss::new().forward(value_output.clone(), target_value.clone(), Reduction::Auto);

    let policy_loss = CrossEntropyLossConfig::new()
        .init(&policy_output.device())
        .forward(policy_output.clone(), target_policy.clone());

    LossOutput::new(
        RegressionOutput::new(value_loss, value_output, target_value).loss,
        ClassificationOutput::new(policy_loss, policy_output, target_policy).loss,
    )
}

/// Foward with loss. (Policy loss: multi label classification)
pub fn forward_with_loss_mlc<B: Backend>(
    this: &Model<B>,
    board_input: BoardInputTensor<B>,
    state_input: StateInputTensor<B>,
    target_value: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    target_policy: Tensor<B, 2, Int>,
) -> LossOutput<B> {
    let (value_output, policy_output) = this.forward(board_input, state_input);

    let value_loss =
        MseLoss::new().forward(value_output.clone(), target_value.clone(), Reduction::Auto);

    let policy_loss = BinaryCrossEntropyLossConfig::new()
        .init(&policy_output.device())
        .forward(policy_output.clone(), target_policy.clone());

    LossOutput::new(
        RegressionOutput::new(value_loss, value_output, target_value).loss,
        MultiLabelClassificationOutput::new(policy_loss, policy_output, target_policy).loss,
    )
}

// /// Multi-label classification output adapted for multiple metrics.
// pub struct ExactProbsClassificationOutput<B: Backend> {
//     /// The loss.
//     pub loss: Tensor<B, 1>,

//     /// The output.
//     pub output: Tensor<B, 2>,

//     /// The targets.
//     pub targets: Tensor<B, 2>,
// }

// impl<B: Backend> ExactProbsClassificationOutput<B> {
//     pub fn new
// }
// impl<B: Backend> ItemLazy for MultiLabelClassificationOutput<B> {
//     type ItemSync = MultiLabelClassificationOutput<NdArray>;

//     fn sync(self) -> Self::ItemSync {
//         let [output, loss, targets] = Transaction::default()
//             .register(self.output)
//             .register(self.loss)
//             .register(self.targets)
//             .execute()
//             .try_into()
//             .expect("Correct amount of tensor data");

//         let device = &Default::default();

//         MultiLabelClassificationOutput {
//             output: Tensor::from_data(output, device),
//             loss: Tensor::from_data(loss, device),
//             targets: Tensor::from_data(targets, device),
//         }
//     }
// }

/// Foward with loss. (Policy loss: exact probabilities)
pub fn forward_with_loss_exact_loss<B: Backend>(
    this: &Model<B>,
    board_input: BoardInputTensor<B>,
    state_input: StateInputTensor<B>,
    target_value: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    target_policy: Tensor<B, 2, Float>,
) -> LossOutput<B> {
    let (value_output, policy_output) = this.forward(board_input, state_input);

    let value_loss =
        MseLoss::new().forward(value_output.clone(), target_value.clone(), Reduction::Auto);

    let policy_loss = KLDivergenceLoss::new().forward(policy_output.clone(), target_policy.clone());

    LossOutput::new(
        RegressionOutput::new(value_loss, value_output, target_value).loss,
        // ExactProbsClassificationOutput::new(policy_loss, policy_output, target_policy).loss,
        policy_loss,
    )
}

impl<B: AutodiffBackend> TrainStep<MLCPlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: MLCPlayoutBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss_mlc(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: AutodiffBackend> TrainStep<SLCPlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: SLCPlayoutBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss_slc(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: AutodiffBackend> TrainStep<ExactLossPlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: ExactLossPlayoutBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss_exact_loss(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

// impl<B: Backend> ValidStep<PlayoutBatch<B>, MLCLossOutput<B>> for Model<B> {
//     fn step(&self, batch: PlayoutBatch<B>) -> MLCLossOutput<B> {
//         forward_with_loss_mlc(
//             self,
//             batch.board_inputs,
//             batch.state_inputs,
//             batch.value_targets,
//             batch.policy_targets,
//         )
//     }
// }

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
    //     DataLoaderBuilder::<B, _,
    // _>::new(IdentityBatcher::<FenItemRaw>::default())
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

                let result = TrainStep::step(&model, playouts_batch.clone());

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

/// Train a net to approximate a specific policy while performing mcts.
pub fn learn_batch() {}

#[derive(Clone, Default)]
pub struct PlayoutBatcher;

impl<B: Backend> Batcher<B, SLCPlayoutItem, SLCPlayoutBatch<B>> for PlayoutBatcher {
    fn batch(&self, items: Vec<SLCPlayoutItem>, device: &B::Device) -> SLCPlayoutBatch<B> {
        let boards = items
            .iter()
            .map(|x| board_history_input(&x.board_input.0, device))
            .collect();

        let states = items
            .iter()
            .map(|x| TensorData::from([x.state_input.0]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let values = items
            .iter()
            .map(|x| TensorData::from([[x.value_target.0]]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let policies = items
            .iter()
            .map(|x| TensorData::from([x.policy_target]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        SLCPlayoutBatch {
            board_inputs: Tensor::cat(boards, 0),
            state_inputs: Tensor::cat(states, 0),
            value_targets: Tensor::cat(values, 0),
            policy_targets: Tensor::cat(policies, 0),
        }
    }
}

impl<B: Backend> Batcher<B, MLCPlayoutItem, MLCPlayoutBatch<B>> for PlayoutBatcher {
    fn batch(&self, items: Vec<MLCPlayoutItem>, device: &B::Device) -> MLCPlayoutBatch<B> {
        let boards = items
            .iter()
            .map(|x| board_history_input(&x.board_input.0, device))
            .collect();

        let states = items
            .iter()
            .map(|x| TensorData::from([x.state_input.0]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let values = items
            .iter()
            .map(|x| TensorData::from([[x.value_target.0]]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let policies = items
            .iter()
            .map(|x| TensorData::from(&x.policy_target[..]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        MLCPlayoutBatch {
            board_inputs: Tensor::cat(boards, 0),
            state_inputs: Tensor::cat(states, 0),
            value_targets: Tensor::cat(values, 0),
            policy_targets: Tensor::cat(policies, 0),
        }
    }
}

impl<B: Backend> Batcher<B, ExactLossPlayoutItem, ExactLossPlayoutBatch<B>> for PlayoutBatcher {
    fn batch(
        &self,
        items: Vec<ExactLossPlayoutItem>,
        device: &B::Device,
    ) -> ExactLossPlayoutBatch<B> {
        let boards = items
            .iter()
            .map(|x| board_history_input(&x.board_input.0, device))
            .collect();

        let states = items
            .iter()
            .map(|x| TensorData::from([x.state_input.0]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let values = items
            .iter()
            .map(|x| TensorData::from([[x.value_target.0]]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let policies = items
            .into_iter()
            .map(|x| {
                TensorData::from([
                    TryInto::<[f32; POLICY_OUTPUTS]>::try_into(x.policy_target).expect("")
                ])
            })
            .map(|x| Tensor::from_data(x, device))
            .collect();

        ExactLossPlayoutBatch {
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
) -> Result<ExactLossPlayoutBatch<B>, Box<dyn Error>> {
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

            match self_play::<_, ExactLossPlayoutItem>(&fen, model, device) {
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

    let batcher = PlayoutBatcher;
    Ok(batcher.batch(playout_items, device))
}

#[derive(Default, Debug)]
pub struct MctsTrain {
    infer: MctsFindBest,
    steps: usize,
}

impl MctsStrategy for MctsTrain {
    type Result = (<MctsFindBest as MctsStrategy>::Result, Tree);
    type Step = (<MctsFindBest as MctsStrategy>::Step,);

    fn start(&mut self, _: &mut Tree) {}

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        let inference_result = self.infer.result(tree);
        let tree = tree.to_owned();
        log::info!(target: "games", "Completed MCTS Iterations: {}", self.steps);
        // todo: log::info!(target: "games", "Base Policy: {:#?}",
        // tree.get_root().borrow().iter_branches().map(|b| format!("[{}] p: {}, WDL
        // found: {}", b.mov(), b.policy(), todo!("Gather how many w/d/l we have found
        // in the game tree during mcts"))).collect_vec());
        (inference_result, tree)
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        self.steps += 1;
        (self.infer.step(tree),)
    }
}

// The inputs to the model.
#[derive(Debug, Clone)]
struct Input {
    board_in: BoardInputFloats,
    state_in: StateInputFloats,
}

// Info to help find the training target.
// 0: Most visited move / best_move.
#[derive(Debug, Clone)]
struct SLCTarget {
    mov: Move,
}

impl<'a> From<&'a Tree> for SLCTarget {
    fn from(tree: &'a Tree) -> Self {
        Self {
            mov: tree.best_move().expect("Tree should have a bestmove"),
        }
    }
}

impl Target for SLCTarget {}

#[derive(Debug, Clone)]
struct MLCTarget {
    moves: Vec<Move>,
}

impl<'a> From<&'a Tree> for MLCTarget {
    fn from(tree: &'a Tree) -> Self {
        Self {
            moves: tree.best_moves(Value(0.5)),
        }
    }
}

impl Target for MLCTarget {}

#[derive(Debug, Clone)]
struct ExactLossTarget {
    raw_policy: RawPolicy,
}

impl From<&Tree> for ExactLossTarget {
    fn from(tree: &Tree) -> Self {
        Self {
            raw_policy: {
                let root = tree.get_root();
                let root = root.into_ct();
                let root = root.evaluated().expect("Root should be evaluated");
                let root = root.borrow();

                let branches = root.branches();

                let mut raw_policy = RawPolicy::null();
                for branch in branches.iter() {
                    raw_policy.set(usize::from(branch.mov()), branch.visits() as f32);
                }
                softmax(raw_policy.inner_mut(), 10.);
                raw_policy
            },
        }
    }
}

impl Target for ExactLossTarget {}

trait Target: for<'a> From<&'a Tree> {}

// Some state info at the time of the move.
// 0: The move that was made
// 1: The color of the moving player.
#[derive(Debug, Clone)]
struct State {
    mov: Move,
    moving_color: Color,
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
struct Decision<T: Target> {
    input: Input,
    target: T,
    state: State,
    stats: Stats,
}

#[derive(Debug)]
struct SelfPlayResult<P: PlayoutItem> {
    playout_item: P,
    decision: Decision<P::Target>,
}

fn self_play<B: Backend, P: PlayoutItem>(
    pos: &str,
    model: Model<B>,
    device: B::Device,
) -> Result<Vec<SelfPlayResult<P>>, Box<dyn Error>>
where
    P::Target: Clone,
{
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

    let mut pos = Position::from_fen(pos)?;

    let mut decisions = Vec::<Decision<P::Target>>::new();
    let nn_state = TrainParts::new(model, device, 0.3, 0.25);
    let mut mcts_state = SearchState::default();

    println!("{pos:?}");

    let eval: GameResult = {
        let mut x = 0;
        let game_result;
        loop {
            x += 1;
            if x > 5 {
                game_result = GameResult::Draw;
                break;
            }
            let turn = pos.get_turn();
            let result = mcts(
                &mut pos,
                &nn_state,
                &mut mcts_state,
                limit.clone(),
                debug.clone(),
                ct.clone(),
                None,
                MctsTrain::default(),
            );

            let mov = result.0;
            let tree = result.1;

            if let Some(result) = pos.game_result() {
                game_result = result;
                break;
            }

            // let model = &nn_state.nn;
            // let device = &nn_state.device;
            // let mut evaluator = NNEvaluator::<_, 1>::new(model, device);
            // let eval_node = evaluator.init(tree.get_root(), &pos);
            // let sel_node =
            // evaluator.batch_eval(0, eval_node);
            // evaluator.eval_guesses();
            // let guess = evaluator.get_eval(0);
            // let guess = guess.expect("The evaluator should have generated the guess");
            // let guess = guess.guess().expect(
            //     "We specifically told the evaluator to make a guess and not a terminal
            // evaluation.", );

            let b_in = board_input(&pos);
            let s_in = state_input(&pos);

            let mov = mov.expect("");
            pos.make_move(mov);

            println!("{pos:?}");

            mcts_state.tree.advance_to(|b| b.mov() == mov);

            let state = State { mov, moving_color: turn };
            let input = Input { board_in: b_in, state_in: s_in };
            let target = P::Target::from(&tree);
            // let stats = Stats::new(guess.clone());
            let stats = Stats::default();
            decisions.push(Decision { input, target, state, stats });
        }
        game_result
    };

    let mut result = Vec::<SelfPlayResult<_>>::new();

    for (index, decision) in decisions.iter().enumerate() {
        let decisions_begin = max(0, index as i32 - BOARD_INPUT_HISTORY as i32) as usize;
        let decisions_end = index;
        let decisions = &decisions[decisions_begin..=decisions_end];
        let playout_item = P::from((eval, decisions));

        result.push(SelfPlayResult {
            playout_item,
            decision: decision.clone(),
        });
    }

    Ok(result)
}

#[derive(Debug)]
pub struct TrainParts<B: Backend> {
    pub nn: Box<Model<B>>,
    pub device: B::Device,
    alpha: f32,
    epsilon: f32,
}

impl<'a, B: Backend> MctsParts for &'a TrainParts<B> {
    type Selector = TrainSelector;
    type Backprop = DefaultBackuper;
    type Evaluator = NNEvaluator<'a, 'a, B>;
    type Noiser = DirichletNoiser;
    type Instance = TrainParts<B>;

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

impl<B: Backend> TryFrom<&Configuration> for TrainParts<B> {
    type Error = CreateNNPartsError;

    fn try_from(config: &Configuration) -> Result<Self, Self::Error> {
        let alpha = config.dirichlet_alpha();
        let epsilon = config.dirichlet_epsilon();
        let weights = PathBuf::from(config.weights_path());
        let device = B::Device::default();
        let nn = Model::try_from((weights, &device)).map_err(Self::Error::LoadNNError)?;
        Ok(Self::new(nn, device, alpha, epsilon))
    }
}

impl<B: Backend> TrainParts<B> {
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
pub struct TrainSelector {
    c: f32,
    policy_weight: f32,
}

impl TrainSelector {
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

impl Default for TrainSelector {
    fn default() -> Self {
        Self {
            c: f32::sqrt(2.0),
            policy_weight: 1.0,
        }
    }
}

impl Selector for TrainSelector {
    type Score = puct::Score;

    fn score(&self, branch: &Branch, cap_n_i: u32) -> Self::Score {
        let n_i = branch.visits() as f32;
        let value = branch.node().borrow().value();
        let policy = Self::weighted_policy(branch.policy(), self.policy_weight);
        let exploitation = if n_i == 0.0 { 0.0 } else { value / n_i };
        let exploration = self.c * policy * (cap_n_i as f32).sqrt() / (1f32 + n_i);
        puct::Score(exploitation + exploration)
    }

    fn min_score(&self) -> Self::Score {
        puct::Score(f32::NEG_INFINITY)
    }
}

// #[derive(Default)]
// pub struct TrainBackprop {
//     default: DefaultBackuper,
// }

// impl Backpropagater for TrainBackprop {
//     fn update(_node: &mut Node, _value: f32) {
//         ()
//     }

//     fn backpropagate<T>(&self, leaf: SelectionNodeRef<T>, eval: &Evaluation)
// {         // default backup
//         self.default.backpropagate(leaf.clone(), eval);

//         // update our diagnostics or something ...
//         if let Evaluation::Terminal(_result) = eval {}
//     }
// }

/// Calculate the KL divergence loss from predictions and probabilistic targets.
#[derive(Module, Debug, Clone)]
pub struct KLDivergenceLoss {}

impl Default for KLDivergenceLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl KLDivergenceLoss {
    pub fn new() -> Self {
        Self {}
    }

    /// Compute the KL divergence loss.
    ///
    /// `predictions` is expected to be a probability distribution.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size]` or `[batch_size, num_classes]`
    /// - targets: `[batch_size]` or `[batch_size, num_classes]` (probabilities)
    pub fn forward<B: Backend, const D: usize>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        // For numerical stability, clamp values to avoid log(0)
        let epsilon = 1e-8;
        let predictions_clamped = predictions.clone().clamp(epsilon, 1.0);
        let targets_clamped = targets.clone().clamp(epsilon, 1.0);

        // KL(P||Q) = sum[ P * log(P) - P * log(Q) ]
        // where P = targets, Q = predictions
        let kl_div = targets_clamped.clone() * targets_clamped.clone().log()
            - targets_clamped * predictions_clamped.log();

        // Sum over class dimension if multi-dimensional
        let loss_per_sample = if D > 1 { kl_div.sum_dim(D - 1) } else { kl_div };

        // Average over batch
        loss_per_sample.mean()
    }
}
