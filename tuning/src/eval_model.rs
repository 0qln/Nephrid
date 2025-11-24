use std::{error::Error, fs};

use burn::{
    backend::Autodiff,
    config::Config,
    data::dataloader::batcher::Batcher,
    nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction},
    optim::AdamConfig,
    prelude::Backend,
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
    train::{ClassificationOutput, RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use burn_cuda::{Cuda, CudaDevice};
use engine::{
    core::{
        bitboard,
        move_iter::sliding_piece::magics,
        position::Position,
        search::{
            self,
            limit::Limit,
            mcts::eval::model::{
                BOARD_INPUT_CHANNELS, BOARD_INPUT_TENSOR_DIM, Model, ModelConfig, POLICY_OUTPUTS,
                POLICY_TARGET_TENSOR_DIM, STATE_INPUT_LEN, STATE_INPUT_TENSOR_DIM,
                VALUE_OUTPUT_TENSOR_DIM,
            },
        },
        zobrist,
    },
    misc::DebugMode,
    uci::{sync::CancellationToken, tokens::Tokenizer},
};

fn main() {
    type Backend = Cuda<f32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = CudaDevice::default();
    println!("Device: {:?}", device);

    let model = ModelConfig::new().init::<Backend>(&device);
    println!("Model: {:#?}", model);

    magics::init();
    zobrist::init();

    // let batch = generate_batch(&model);
    let result = self_play(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &model,
    );
    println!("{result:?}");
}

#[derive(Clone, Debug)]
pub struct SelfplayBatch<B: Backend> {
    pub board_inputs: Tensor<B, BOARD_INPUT_TENSOR_DIM>,
    pub state_inputs: Tensor<B, STATE_INPUT_TENSOR_DIM>,
    pub value_targets: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    pub policy_targets: Tensor<B, POLICY_TARGET_TENSOR_DIM, Int>,
}

pub struct SelfplayItem {
    pub board_input: [bitboard::Floats; BOARD_INPUT_CHANNELS],
    pub state_input: [f32; STATE_INPUT_LEN],
    pub value_target: f32,
    pub policy_target: [f32; POLICY_OUTPUTS],
}

pub type LossOutput<B> = (
    // Value output
    RegressionOutput<B>,
    // Quality Output
    ClassificationOutput<B>,
);

pub fn forward_with_loss<B: Backend>(
    this: &Model<B>,
    board_input: Tensor<B, BOARD_INPUT_TENSOR_DIM>,
    state_input: Tensor<B, STATE_INPUT_TENSOR_DIM>,
    target_value: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    target_policy: Tensor<B, POLICY_TARGET_TENSOR_DIM, Int>,
) -> LossOutput<B> {
    let (value_output, policy_output) = this.forward(board_input, state_input);

    let value_loss =
        MseLoss::new().forward(value_output.clone(), target_value.clone(), Reduction::Auto);

    let policy_loss = CrossEntropyLossConfig::new()
        .init(&policy_output.device())
        .forward(policy_output.clone(), target_policy.clone());

    (
        RegressionOutput::new(value_loss, value_output, target_value),
        ClassificationOutput::new(policy_loss, policy_output, target_policy),
    )
}

impl<B: AutodiffBackend> TrainStep<SelfplayBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: SelfplayBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        );

        let (ref value, ref policy) = item;
        let loss = value.loss.clone() + policy.loss.clone(); // todo: weight decay
        TrainOutput::new(self, loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<SelfplayBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: SelfplayBatch<B>) -> LossOutput<B> {
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
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 0x_dead_beef)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    fs::remove_dir_all(artifact_dir).ok();
    fs::create_dir_all(artifact_dir).ok();
}

// pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
//     create_artifact_dir(artifact_dir);
//     config
//         .save(format!("{artifact_dir}/config.json"))
//         .expect("Config should be saved successfully");

//     B::seed(&device, config.seed);

//     let batcher = SelfplayBatcher::default();

//     let dataloader_train = DataLoaderBuilder::new(batcher.clone())
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(MnistDataset::train());

//     let dataloader_test = DataLoaderBuilder::new(batcher)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(MnistDataset::test());

//     let learner = LearnerBuilder::new(artifact_dir)
//         .metric_train_numeric(AccuracyMetric::new())
//         .metric_valid_numeric(AccuracyMetric::new())
//         .metric_train_numeric(LossMetric::new())
//         .metric_valid_numeric(LossMetric::new())
//         .with_file_checkpointer(CompactRecorder::new())
//         .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
//         .num_epochs(config.num_epochs)
//         .summary()
//         .build(
//             config.model.init::<B>(&device),
//             config.optimizer.init(),
//             config.learning_rate,
//         );

//     let result = learner.fit(dataloader_train, dataloader_test);

//     result
//         .model
//         .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
//         .expect("Trained model should be saved successfully");
// }

#[derive(Clone, Default)]
pub struct SelfplayBatcher;

impl<B: Backend> Batcher<B, SelfplayItem, SelfplayBatch<B>> for SelfplayBatcher {
    fn batch(&self, items: Vec<SelfplayItem>, device: &B::Device) -> SelfplayBatch<B> {
        let boards = items
            .iter()
            .map(|x| TensorData::from(x.board_input))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let states = items
            .iter()
            .map(|x| TensorData::from(x.state_input))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let values = items
            .iter()
            .map(|x| TensorData::from([x.value_target]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let policies = items
            .iter()
            .map(|x| TensorData::from(x.policy_target))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        SelfplayBatch {
            board_inputs: Tensor::cat(boards, 0),
            state_inputs: Tensor::cat(states, 0),
            value_targets: Tensor::cat(values, 0),
            policy_targets: Tensor::cat(policies, 0),
        }
    }
}

fn self_play<B: Backend>(pos: &str, model: &Model<B>) -> Result<(), Box<dyn Error>> {
    let limit = Limit {
        is_active: true,
        winc: 100,
        binc: 100,
        wtime: 0,
        btime: 0,
        ..Default::default()
    };
    let debug = DebugMode::default();
    let ct = CancellationToken::new();

    let tok = &mut Tokenizer::new(pos);
    let mut pos: Position = tok
        .try_into()
        .expect(format!("Invalid FEN: {pos}").as_str());

    println!("{pos}");

    loop {
        let mov = search::mcts(pos.clone(), model, limit.clone(), debug.clone(), ct.clone());

        if mov.is_none() || pos.fifty_move_rule() || pos.has_threefold_repetition() {
            break;
        }

        pos.make_move(mov.unwrap());

        println!("{pos}");
    }

    Ok(())
}
