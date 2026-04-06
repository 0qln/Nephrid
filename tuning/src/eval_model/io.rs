use std::fs;

use burn::{
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};
use engine::core::search::mcts::nn::Model;

use crate::TrainingConfig;

pub fn get_config(path: &str) -> TrainingConfig {
    TrainingConfig::load(&path).expect("Couldn't load config.json at {path:?}")
}

/// Scans the artifact directory and returns: (Start Epoch, Next Iteration,
/// Option<Checkpoint Path>)
pub fn get_resume_state(artifact_dir: &str) -> (usize, usize, Option<String>) {
    let latest = fs::read_dir(artifact_dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.path().file_stem()?.to_str()?.to_owned();
            let (e_str, i_str) = name.strip_prefix("model_e-")?.split_once("_i-")?;

            let e = e_str.parse::<usize>().ok()?;
            let i = i_str.parse::<usize>().ok()?;

            Some((e, i, format!("{artifact_dir}/{name}")))
        })
        .max_by_key(|&(e, i, _)| (e, i));

    match latest {
        Some((e, i, path)) => (e, i + 1, Some(path)), // Start on the NEXT iteration
        None => (1, 0, None),                         // Start from scratch
    }
}

pub fn setup_environment<B: AutodiffBackend>(
    artifact_dir: &str,
    _output_dir: &str,
    config: &TrainingConfig,
    device: &B::Device,
) {
    fs::create_dir_all(artifact_dir).ok();
    B::seed(device, config.seed);
}

pub fn load_weights<B: AutodiffBackend>(
    model: Model<B>,
    device: &B::Device,
    checkpoint_path: &Option<String>,
) -> Model<B> {
    if let Some(path) = checkpoint_path {
        log::info!(target: "reports::train", "Resuming training from {}", path);
        let record = CompactRecorder::new()
            .load(path.clone().into(), device)
            .expect("Failed to load checkpoint weights.");
        model.load_record(record)
    }
    else {
        log::info!(target: "reports::train", "Starting training from scratch.");
        model
    }
}

pub fn save_checkpoint<B: AutodiffBackend>(
    model: &Model<B>,
    artifact_dir: &str,
    epoch: usize,
    iteration: usize,
    last_checkpoint: &mut Option<String>,
) {
    let weights_path = format!("{artifact_dir}/model_e-{epoch}_i-{iteration}");

    model
        .clone()
        .save_file(&weights_path, &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    *last_checkpoint = Some(weights_path);
}
