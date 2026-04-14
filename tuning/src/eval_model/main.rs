use crate::{
    data::build_dataloader,
    io::{load_weights, save_checkpoint, setup_environment},
    loss::PlayoutBatcher,
    self_play::{Decision, LimitConfig, MctsConfig, SelfplayConfig, Target, generate_batch},
};
use burn::{
    module::AutodiffModule,
    prelude::ToElement,
    train::{TrainOutput, ValidStep},
};
use clap::Parser;
use rand::{SeedableRng, seq::SliceRandom};

use burn::{
    optim::{AdamConfig, Optimizer},
    tensor::backend::AutodiffBackend,
};

use burn::{
    backend::Autodiff, config::Config, data::dataloader::batcher::Batcher, train::TrainStep,
};
use burn_cuda::{Cuda, CudaDevice};
use engine::core::{move_iter::sliding_piece::magics, search::mcts::nn::ModelConfig, zobrist};
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
    let args = Args::parse();

    let base_dir = "tuning/out/eval_model/main";
    let phase_name = format!("phase{}", args.phase);
    let output_dir = format!("{base_dir}/{phase_name}");
    let artifact_dir = format!("{output_dir}/artifacts");
    let config_file = format!("tuning/src/eval_model/{phase_name}-config.json");

    let config = get_config(&config_file);

    let resume_action = io::resolve_checkpoint(base_dir, args.phase);

    train::<AutodiffBackend>(&phase_name, &config, &artifact_dir, resume_action, &device);
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The phase number to train (e.g., 1, 2, 3)
    #[arg(short, long)]
    phase: usize,
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,

    pub optimizer: AdamConfig,

    pub limit: LimitConfig,

    pub self_play: SelfplayConfig,

    pub mcts: MctsConfig,

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

    pub epd_dataset_path: String,

    #[config(default = 1_000)]
    pub epd_dataset_fens_total: usize,
}

pub fn train<B: AutodiffBackend>(
    phase_name: &str,
    config: &TrainingConfig,
    artifact_dir: &str,
    resume_action: io::ResumeAction,
    device: &B::Device,
) -> Option<String> {
    let (start_epoch, next_iteration, mut current_model_path) = match resume_action {
        io::ResumeAction::Resume(e, i, path) => {
            log::info!(target: "train", "Resuming {} from epoch {}, iteration {}", phase_name, e, i);
            (e, i, Some(path))
        }
        io::ResumeAction::Transfer(path) => {
            log::info!(target: "train", "Starting {} fresh, utilizing weights from: {}", phase_name, path);
            (1, 0, Some(path))
        }
        io::ResumeAction::Scratch => {
            log::info!(target: "train", "Starting {} entirely from scratch.", phase_name);
            (1, 0, None)
        }
    };

    log_mdc::insert("p", phase_name);

    setup_environment::<B>(&artifact_dir, &config, &device);

    let train = build_dataloader::<B>(&config, "train");
    let test = build_dataloader::<B>(&config, "test");

    let limit = config.limit.init();
    let self_play = &config.self_play;
    let mcts_cfg = &config.mcts;
    let mut optim = config.optimizer.init();
    let mut model = config.model.init::<B>(&device);

    model = load_weights(model, &device, &current_model_path);

    let mut rng = SmallRng::seed_from_u64(config.seed);
    for epoch in start_epoch..=config.num_epochs {
        let skip_count = if epoch == start_epoch { next_iteration } else { 0 };

        for (iteration, fens_batch) in train.iter().enumerate().skip(skip_count) {
            // Generate and shuffle MCTS playouts
            let mut playout_items = generate_batch::<_, ExactLossPlayoutItem>(
                model.valid(),
                &fens_batch,
                &limit,
                &self_play,
                &mcts_cfg,
            )
            .expect("Failed to generate batch");
            playout_items.shuffle(&mut rng);

            // Process Mini-batches
            let batcher = PlayoutBatcher;
            for (chunk_idx, chunk) in playout_items.chunks(config.mini_batch_size).enumerate() {
                let playouts_batch = batcher.batch(chunk.to_vec(), device);
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
                &mut current_model_path,
            );
        }

        // Test
        {
            let mut val_loss_sum = 0.0;
            let mut val_value_loss_sum = 0.0;
            let mut val_policy_loss_sum = 0.0;
            let mut val_batches = 0;

            let model = model.valid();

            for (_val_iteration, fens_batch) in test.iter().enumerate() {
                // Generate playouts for the validation batch
                let playout_items = generate_batch::<_, ExactLossPlayoutItem>(
                    model.clone(),
                    &fens_batch,
                    &limit,
                    &self_play,
                    &mcts_cfg,
                )
                .expect("Failed to generate validation batch");

                let batcher = PlayoutBatcher;
                for chunk in playout_items.chunks(config.mini_batch_size) {
                    let playouts_batch = batcher.batch(chunk.to_vec(), device);

                    // Use ValidStep to evaluate without computing gradients
                    let result = ValidStep::step(&model, playouts_batch);

                    // Accumulate losses for averaging
                    val_loss_sum += result.loss.into_scalar().to_f64();
                    val_value_loss_sum += result.value_loss.into_scalar().to_f64();
                    val_policy_loss_sum += result.policy_loss.into_scalar().to_f64();
                    val_batches += 1;
                }
            }

            // Log the averaged validation metrics for the epoch
            if val_batches > 0 {
                log::info!(target: "test",
                    "[Test - Epoch {}] Avg Loss {:.5} (Value: {:.5}, Policy: {:.5})",
                    epoch,
                    val_loss_sum / val_batches as f64,
                    val_value_loss_sum / val_batches as f64,
                    val_policy_loss_sum / val_batches as f64,
                );
            }
        }
    }

    current_model_path
}

fn log_step<B: AutodiffBackend>(
    epoch: usize,
    iteration: usize,
    chunk_idx: usize,
    result: &TrainOutput<LossOutput<B>>,
) {
    log::info!(target: "train",
            "[Train - Epoch {} - Iteration {}.{}] Loss {:.5} (Value: {:.5}, Policy: {:.5})",
            epoch,
            iteration,
            chunk_idx,
            result.item.loss.clone().into_scalar(),
            result.item.value_loss.clone().into_scalar(),
            result.item.policy_loss.clone().into_scalar(),
    );
}
