use crate::{
    data::build_dataloader,
    io::{get_resume_state, load_weights, save_checkpoint, setup_environment},
    loss::PlayoutBatcher,
    self_play::{Decision, Target, generate_batch},
};
use burn::train::TrainOutput;
use rand::{SeedableRng, seq::SliceRandom};

use burn::{
    optim::{AdamConfig, Optimizer},
    tensor::backend::AutodiffBackend,
};

use burn::{
    backend::Autodiff, config::Config, data::dataloader::batcher::Batcher, train::TrainStep,
};
use burn_cuda::{Cuda, CudaDevice};
use engine::core::{
    move_iter::sliding_piece::magics,
    search::{limit::Limit, mcts::nn::ModelConfig},
    zobrist,
};
use rand::rngs::SmallRng;

use crate::{
    io::get_config,
    loss::{LossOutput, el::ExactLossPlayoutItem},
};

#[cfg(test)]
pub mod test;

pub mod data;
pub mod io;
pub mod loss;
pub mod self_play;

fn main() {
    magics::init();
    zobrist::init();
    log4rs::init_file("./tuning/src/eval_model/log4rs.yml", Default::default()).unwrap();

    type Backend = Cuda<f32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = CudaDevice::default();
    log::info!("Device: {:?}", device);

    let train_dir = "tuning/out/eval_model";
    let config = get_config("tuning/src/eval_model/config.json");

    train::<AutodiffBackend>(train_dir, config, device);
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

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,

    pub optimizer: AdamConfig,

    pub limit: LimitConfig,

    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 8)]
    pub batch_size: usize,

    #[config(default = 256)]
    pub mini_batch_size: usize,

    #[config(default = 0x_dead_beef)]
    pub seed: u64,

    #[config(default = 1.0e-5)]
    pub learning_rate: f64,

    pub edp_dataset_path: String,

    #[config(default = 1_000)]
    pub edp_dataset_fens_total: usize,
}

pub fn train<B: AutodiffBackend>(
    output_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) -> Option<String> {
    let artifact_dir = format!("{output_dir}/artifacts");
    setup_environment::<B>(&artifact_dir, output_dir, &config, &device);

    let dataloader_train = build_dataloader::<B>(&config);

    // 1. Discover where we left off
    let (start_epoch, next_iteration, mut latest_checkpoint) = get_resume_state(&artifact_dir);

    // 2. Initialize Model & Optimizer
    let limit = config.limit.init();
    let mut optim = config.optimizer.init();
    let mut model = config.model.init::<B>(&device);
    model = load_weights(model, &device, &latest_checkpoint);

    // 3. Main Training Loop
    let mut rng = SmallRng::seed_from_u64(config.seed);
    for epoch in start_epoch..=config.num_epochs {
        let skip_count = if epoch == start_epoch { next_iteration } else { 0 };

        for (iteration, fens_batch) in dataloader_train.iter().enumerate().skip(skip_count) {
            // Generate and shuffle MCTS playouts
            let mut playout_items =
                generate_batch::<B, ExactLossPlayoutItem>(&model, &fens_batch, &limit)
                    .expect("Failed to generate batch");
            playout_items.shuffle(&mut rng);

            // Process Mini-batches
            let batcher = PlayoutBatcher;
            for (chunk_idx, chunk) in playout_items.chunks(config.mini_batch_size).enumerate() {
                let playouts_batch = batcher.batch(chunk.to_vec(), &device);
                let result = TrainStep::step(&model, playouts_batch);

                log_step(epoch, iteration, chunk_idx, &result);

                model = optim.step(config.learning_rate, model, result.grads);
            }

            // Save the new state
            save_checkpoint(
                &model,
                &artifact_dir,
                epoch,
                iteration,
                &mut latest_checkpoint,
            );
        }
    }

    latest_checkpoint
}

fn log_step<B: AutodiffBackend>(
    epoch: usize,
    iteration: usize,
    chunk_idx: usize,
    result: &TrainOutput<LossOutput<B>>,
) {
    let msg = format!(
        "[Train - Epoch {} - Iteration {}.{}] Loss {:.5} (Value: {:.5}, Policy: {:.5})",
        epoch,
        iteration,
        chunk_idx,
        result.item.loss.clone().into_scalar(),
        result.item.value_loss.clone().into_scalar(),
        result.item.policy_loss.clone().into_scalar(),
    );
    log::info!(target: "reports::train", "{msg}");
}
