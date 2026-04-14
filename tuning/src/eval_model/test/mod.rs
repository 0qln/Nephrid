use std::{env::var, path::PathBuf};

use crate::{
    data::{FenDataset, FenItemRaw},
    io::{ResumeAction, get_config},
    loss::ValueTarget,
    self_play::MctsTrainStrategy,
    train,
};
use burn::{
    backend::Autodiff,
    data::dataset::Dataset,
    module::Module,
    record::{CompactRecorder, Recorder},
};
use burn_cuda::{Cuda, CudaDevice};
use engine::core::{
    color::colors,
    coordinates::squares::{A5, B5, F3, G1},
    r#move::{Move, move_flags::QUIET},
    move_iter::sliding_piece::magics,
    position::Position,
    search::{
        limit::Limit,
        mcts::{
            NNParts, SearchState,
            eval::{GameResult, RawPolicy},
            mcts,
            nn::ModelConfig,
        },
    },
    zobrist,
};

const OUT_DIR: &str = "out/eval_model/test";
const SRC_DIR: &str = "src/eval_model/test";

fn proj_root() -> String {
    var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable")
}

fn config_file(test_name: &str) -> String {
    let mut buf = PathBuf::new();
    buf.push(&proj_root());
    buf.push("tuning");
    buf.push(SRC_DIR);
    buf.push(format!("{}.json", test_name));
    buf.to_str().unwrap().to_string()
}

fn log_config_file() -> String {
    let mut buf = PathBuf::new();
    buf.push(proj_root());
    buf.push("tuning");
    buf.push(SRC_DIR);
    buf.push("log4rs.yml");
    buf.to_str().unwrap().to_string()
}

fn artifact_dir(test_name: &str) -> String {
    let mut buf = PathBuf::new();
    buf.push(&proj_root());
    buf.push("tuning");
    buf.push(OUT_DIR);
    buf.push(test_name);
    buf.push("artifacts");
    buf.to_str().unwrap().to_string()
}

pub mod logs {
    use super::*;

    pub fn init() {
        let config = log_config_file();

        // Ignore the result so tests don't panic if initialized twice
        match log4rs::init_file(&config, Default::default()) {
            Ok(()) => {
                println!("Loaded logging config from {config}");
            }
            Err(e) => {
                println!(
                    "Failed to load logging config from {config}, using default config. (Error: \
                     {e})"
                );
            }
        }
    }
}

#[ignore]
#[test]
pub fn test_network_can_overfit_hardcoded_target() {
    use crate::{
        PlayoutBatcher,
        data::{BoardInput, StateInput},
        loss::el::{self, ExactLossPlayoutItem},
    };
    use burn::{
        data::dataloader::batcher::Batcher,
        optim::{AdamConfig, Optimizer},
        train::TrainStep,
    };
    use engine::core::{
        position::Position,
        search::mcts::nn::{ModelConfig, POLICY_OUTPUTS, VALUE_WIN, board_input, state_input},
    };

    // Initialize environment
    logs::init();
    magics::init();
    zobrist::init();

    type Backend = burn_cuda::Cuda<f32>;
    type AutodiffBackend = burn::backend::Autodiff<Backend>;
    let device = burn_cuda::CudaDevice::default();

    let pos = Position::start_position();

    let b_in = board_input(&pos);
    let s_in = state_input(&pos);
    let board_history = vec![b_in];

    // Define our hardcoded targets
    let target_move_index = usize::from(Move::new(G1, F3, QUIET));
    let mut target_policy = [0.; POLICY_OUTPUTS];
    target_policy[target_move_index] = 1.0;

    // We also want it to learn this is a winning position
    let target_value = VALUE_WIN;

    // Create the Playout Item & Batch
    let item = ExactLossPlayoutItem {
        board_input: BoardInput(board_history),
        state_input: StateInput(s_in),
        value_target: ValueTarget(target_value),
        policy_target: el::PolicyTarget(RawPolicy::new(target_policy)),
    };

    let batcher = PlayoutBatcher::default();
    let batch = batcher.batch(vec![item], &device);

    // Initialize Model and Optimizer
    let mut model = ModelConfig::new().init::<AutodiffBackend>(&device);
    let mut optim = AdamConfig::new().init::<AutodiffBackend, _>();

    // Train loop (Overfit on this exact batch for 100 epochs)
    for epoch in 1..=100 {
        let step_result = TrainStep::step(&model, batch.clone());

        // Update the model weights
        model = optim.step(0.005, model, step_result.grads);

        if epoch % 10 == 0 {
            let loss = step_result.item.loss.into_scalar();
            let v_loss = step_result.item.value_loss.into_scalar();
            let p_loss = step_result.item.policy_loss.into_scalar();
            println!("[Epoch {epoch:03}] Loss: {loss:.5} (V: {v_loss:.5}, P: {p_loss:.5})");
        }
    }

    // 7. Validation: Check if the network actually learned it!
    // We convert the trained Autodiff module into a valid (inference-only) module
    use burn::module::AutodiffModule;
    let valid_model = model.valid();

    // Run inference on the identical inputs
    let (value_pred, policy_logits) =
        valid_model.forward(batch.board_inputs.valid(), batch.state_inputs.valid());

    // Check Value Output
    let pred_v = value_pred.into_scalar();
    println!("Predicted Value: {pred_v:.5} (Target: {target_value})");
    assert!(
        (pred_v - target_value).abs() < 0.1,
        "Value head failed to overfit!"
    );

    // Apply Softmax to convert raw logits into actual probabilities (0.0 to 1.0)
    use burn::tensor::activation::softmax;
    let policy_probs = softmax(policy_logits, 1);

    // Extract the policy tensor into a standard Rust Vec
    let pred_p = policy_probs.into_data();
    let pred_p = pred_p.as_slice::<f32>().unwrap();

    // Find the move index with the highest probability
    let (best_idx, max_prob) = pred_p
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!(
        "Predicted Best Move Index: {best_idx} with probability {max_prob:.5} (Target Index: \
         {target_move_index})"
    );

    assert_eq!(
        best_idx, target_move_index,
        "Policy head failed to predict the hardcoded move index!"
    );
    assert!(
        *max_prob > 0.90,
        "Policy head found the move but isn't confident enough! (Prob: {max_prob})"
    );

    println!("Overfitting Test Passed!");
}

#[ignore]
#[test]
pub fn learn_mate_in_1() {
    logs::init();
    magics::init();
    zobrist::init();

    type Backend = Cuda<f32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = CudaDevice::default();

    let test_name = "learn_mate_in_1";
    let phase_name = test_name;
    let config_file = config_file(test_name);
    let artifact_dir = artifact_dir(test_name);

    let config = get_config(&config_file);

    let result_weights = train::<AutodiffBackend>(
        phase_name,
        &config,
        &artifact_dir,
        ResumeAction::Scratch,
        &device,
    )
    .unwrap();

    // test
    let num_fens_total = config.epd_dataset_fens_total;
    let fen_path = &config.epd_dataset_path.clone();
    let test_fen = FenDataset::from_epd(fen_path, "train", num_fens_total);
    let fen = Dataset::<FenItemRaw>::get(&test_fen, 0).expect("no fen in dataset");
    let mut pos = Position::from_fen(&fen.fen).expect("Bad fen");

    let (best_move_played, best_move_by_policy) = {
        let record = CompactRecorder::new()
            .load(result_weights.into(), &device)
            .expect("Should be able to load the model weights from the provided file");

        let model = ModelConfig::new()
            .init::<Backend>(&device)
            .load_record(record);

        let mut mcts_state = SearchState::default();
        let parts = NNParts::new(model, device, 0.3, 0.);

        let limit = Limit {
            is_active: true,
            // should be enough to find the mate in 1 with a trained policy and batched MCTS.
            iterations: 3,
            ..Default::default()
        };

        let result = mcts(
            &mut pos,
            &parts,
            &mut mcts_state,
            &limit,
            MctsTrainStrategy::new(1, 1),
        );

        for b in mcts_state.tree.branches_rt(mcts_state.tree.root()) {
            log::debug!(target: "test", "Move: {:?}, Policy: {:.5}", b.mov(), b.policy());
        }

        let best_policy = mcts_state
            .tree
            .branches_rt(mcts_state.tree.root())
            .iter()
            .max_by(|a, b| a.policy().partial_cmp(&b.policy()).unwrap())
            .unwrap()
            .mov();

        let best_move = result.0.unwrap();

        (best_move, best_policy)
    };

    println!("Best move: {best_move_played}");
    println!("Best by policy: {best_move_by_policy}");

    let expected_move = Move::new(B5, A5, QUIET);
    assert_eq!(
        best_move_played, expected_move,
        "MCTS did not find the mate in 1 move!"
    );
    assert_eq!(
        best_move_by_policy, expected_move,
        "Policy head did not assign the highest probability to the mate in 1 move!"
    );
}

#[ignore]
#[test]
pub fn learn_mate_in_2() {
    logs::init();
    magics::init();
    zobrist::init();

    type Backend = Cuda<f32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = CudaDevice::default();
    log::info!("Device: {:?}", device);

    let test_name = "learn_mate_in_2";
    let phase_name = test_name;
    let config_file = config_file(test_name);
    let artifact_dir = artifact_dir(test_name);

    let config = get_config(&config_file);

    let result_weights = train::<AutodiffBackend>(
        phase_name,
        &config,
        &artifact_dir,
        ResumeAction::Scratch,
        &device,
    )
    .unwrap();

    // test
    let num_fens_total = config.epd_dataset_fens_total;
    let fen_path = &config.epd_dataset_path.clone();
    let test_fen = FenDataset::from_epd(fen_path, "train", num_fens_total);
    let fen = Dataset::<FenItemRaw>::get(&test_fen, 0).expect("no fen in dataset");
    let mut pos = Position::from_fen(&fen.fen).expect("Bad fen");

    let result = {
        let record = CompactRecorder::new()
            .load(result_weights.into(), &device)
            .expect("Should be able to load the model weights from the provided file");

        let model = ModelConfig::new()
            .init::<Backend>(&device)
            .load_record(record);

        let limit = Limit {
            is_active: true,
            iterations: 25,
            ..Default::default()
        };

        let mut mcts_state = SearchState::default();
        let nn_state = NNParts::new(model, device, 0.3, 0.);

        for _ in 0..3 {
            let result = mcts(
                &mut pos,
                &nn_state,
                &mut mcts_state,
                &limit,
                MctsTrainStrategy::new(1, 1),
            );

            for b in mcts_state.tree.branches_rt(mcts_state.tree.root()) {
                log::debug!(target: "test", "Move: {:?}, Policy: {:.5}", b.mov(), b.policy());
            }

            let best_policy = mcts_state
                .tree
                .branches_rt(mcts_state.tree.root())
                .iter()
                .max_by(|a, b| a.policy().partial_cmp(&b.policy()).unwrap())
                .unwrap()
                .mov();

            let best_move = result.0.unwrap();
            pos.make_move(best_move);
            mcts_state.advance_to(best_move);

            log::info!(target: "test", "Best move: {best_move}");
            log::info!(target: "test", "Best by policy: {best_policy}");
        }

        pos.game_result()
    };

    assert_eq!(Some(GameResult::Win { relative_to: colors::WHITE }), result)
}
