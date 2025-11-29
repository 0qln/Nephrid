use std::{error::Error, fs};

use burn::{
    backend::{Autodiff, NdArray},
    config::Config,
    data::{
        dataloader::batcher::Batcher,
        dataset::{InMemDataset, transform::MapperDataset},
    },
    nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
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
            self, MctsStrategy, MctsUci,
            limit::Limit,
            mcts::{
                self, Evaluation, GameResult,
                eval::model::{
                    BOARD_INPUT_TENSOR_DIM, BoardInputFloats, Model, ModelConfig,
                    POLICY_TARGET_TENSOR_DIM, STATE_INPUT_TENSOR_DIM, StateInputFloats, VALUE_DRAW,
                    VALUE_LOSE, VALUE_OUTPUT_TENSOR_DIM, VALUE_WIN, board_input, state_input,
                },
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

    let artifact_dir = "/tmp/nephrid/eval_model";
    train::<AutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
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

pub struct PlayoutItemRaw {
    /// The position string that is to be played out
    pub fen: String,
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

// struct FenToPlayoutItem;

// impl Mapper<PlayoutItemRaw, PlayoutItem> for FenToPlayoutItem {
//     fn map(&self, item: &PlayoutItemRaw) -> PlayoutItem {
//         todo!("Playout the fen from `item` and write the result into  a PlayoutItem.")
//     }
// }

// type MappedDataset = MapperDataset<InMemDataset<PlayoutItemRaw>, FenToPlayoutItem, PlayoutItemRaw>;

// pub struct PlayoutDataset {
//     dataset: MappedDataset,
// }

// impl Dataset<PlayoutItem> for PlayoutDataset {
//     fn get(&self, index: usize) -> Option<PlayoutItem> {
//         self.dataset.get(index)
//     }

//     fn len(&self) -> usize {
//         self.dataset.len()
//     }
// }

pub struct LossOutput<B: Backend> {
    // Value output
    value_loss: RegressionOutput<B>,
    // Quality Output
    policy_loss: ClassificationOutput<B>,
}

impl<B: Backend> LossOutput<B> {
    pub fn loss(&self) -> Tensor<B, 1> {
        // todo: weight decay loss
        let ref value = self.value_loss;
        let ref policy = self.value_loss;
        value.loss.clone() + policy.loss.clone()
    }
}

impl<B: Backend> ItemLazy for LossOutput<B> {
    type ItemSync = LossOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        LossOutput {
            value_loss: self.value_loss.sync(),
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
    target_value: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    target_policy: Tensor<B, POLICY_TARGET_TENSOR_DIM, Int>,
) -> LossOutput<B> {
    let (value_output, policy_output) = this.forward(board_input, state_input);

    let value_loss =
        MseLoss::new().forward(value_output.clone(), target_value.clone(), Reduction::Auto);

    let policy_loss = CrossEntropyLossConfig::new()
        .init(&policy_output.device())
        .forward(policy_output.clone(), target_policy.clone());

    LossOutput {
        value_loss: RegressionOutput::new(value_loss, value_output, target_value),
        policy_loss: ClassificationOutput::new(policy_loss, policy_output, target_policy),
    }
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

        TrainOutput::new(self, item.loss().backward(), item)
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

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    // let learner = LearnerBuilder::new(artifact_dir)
    //     .metric_train_numeric(LossMetric::new())
    //     .metric_valid_numeric(LossMetric::new())
    //     .with_file_checkpointer(CompactRecorder::new())
    //     .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
    //     .num_epochs(config.num_epochs)
    //     .summary()
    //     .build(
    //         config.model.init::<B>(&device),
    //         config.optimizer.init(),
    //         config.learning_rate,
    //     );

    // let result = learner.fit(dataloader_train, dataloader_test);

    //
    // # Custom training loop variant:
    //

    // Create the model and optimizer.
    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimizer.init();

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        let batcher = PlayoutBatcher::default();

        // let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        //     .batch_size(config.batch_size)
        //     // .shuffle(config.seed)
        //     .num_workers(config.num_workers)
        //     .build();

        // let dataloader_test = DataLoaderBuilder::new(batcher)
        //     .batch_size(config.batch_size)
        //     // .shuffle(config.seed)
        //     .num_workers(config.num_workers)
        //     .build();

        // Implement our training loop.
        for iteration in 0..config.batch_size {
            // for (iteration, batch) in dataloader_train.iter().enumerate() {
            let batch = generate_batch(&model, &device).expect("Failed to generate batch");
            let result = TrainStep::step(&model, batch);
            let loss = result.item.loss();

            println!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            );

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.learning_rate, model, grads);
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
    }

    // result
    //     .model
    //     .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
    //     .expect("Trained model should be saved successfully");
}

#[derive(Clone, Default)]
pub struct PlayoutBatcher;

impl<B: Backend> Batcher<B, PlayoutItem, PlayoutBatch<B>> for PlayoutBatcher {
    fn batch(&self, items: Vec<PlayoutItem>, device: &B::Device) -> PlayoutBatch<B> {
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

fn generate_batch<B: Backend>(
    model: &Model<B>,
    device: &B::Device,
) -> Result<PlayoutBatch<B>, Box<dyn Error>> {
    let playout_items = self_play(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &model,
    )?;

    let batcher = PlayoutBatcher::default();
    Ok(batcher.batch(playout_items, device))
}

#[derive(Default, Debug)]
pub struct MctsTrain {
    infer: MctsUci,
}

impl MctsStrategy for MctsTrain {
    type Result = (<MctsUci as MctsStrategy>::Result, mcts::Tree);

    fn result(&mut self, tree: &mut mcts::Tree) -> Self::Result {
        let inference_result = self.infer.result(tree);
        let tree = tree.to_owned();
        (inference_result, tree)
    }

    fn step(&mut self, tree: &mut mcts::Tree) {
        self.infer.step(tree);
    }
}

fn self_play<B: Backend>(pos: &str, model: &Model<B>) -> Result<Vec<PlayoutItem>, Box<dyn Error>> {
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

    // The inputs to the model.
    type Input = (BoardInputFloats, StateInputFloats);

    // Info to help find the training target.
    // 0: Most visited move / best_move.
    type Target = (Move,);

    // Some state info at the time of the move.
    // 0: The move that was made
    // 1: The color of the moving player.
    type State = (Move, Color);

    let mut decisions = Vec::<(Input, Target, State)>::new();

    let eval: GameResult = {
        let mut game_result = GameResult::Draw;
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

            let mov = mov.expect("we just checked that it is not none");
            pos.make_move(mov);

            let state_info = (mov, turn);
            let inputs = (b_in, s_in);
            let targets = (mov,);
            decisions.push((inputs, targets, state_info));
        }
        game_result
    };

    let mut result = Vec::<PlayoutItem>::new();

    for (input, target, state) in decisions {
        let value_target = match eval {
            GameResult::Draw => VALUE_DRAW,
            GameResult::Win { relative_to } => {
                if relative_to == state.1 {
                    VALUE_WIN
                } else {
                    VALUE_LOSE
                }
            }
            _ => panic!("The only evaluation result has to be on that ends the game."),
        };
        let playout_item = PlayoutItem {
            board_input: input.0,
            state_input: input.1,
            value_target,
            policy_target: usize::from(target.0),
        };
        result.push(playout_item);
    }

    Ok(result)
}
