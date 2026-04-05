use std::{env::var, path::PathBuf};

use crate::{
    data::{FenDataset, FenItemRaw},
    io::get_config,
    self_play::MctsTrainStrategy,
    train,
};
use burn::{
    backend::Autodiff,
    config::Config,
    data::dataset::Dataset,
    module::Module,
    record::{CompactRecorder, Recorder},
};
use burn_cuda::{Cuda, CudaDevice};
use engine::core::{
    coordinates::squares,
    r#move::{Move, move_flags},
    move_iter::sliding_piece::magics,
    position::Position,
    search::{
        limit::Limit,
        mcts::{NNParts, SearchState, mcts, nn::ModelConfig, node::node_state::NodeState},
    },
    zobrist,
};

const OUT_DIR: &str = "tuning/out/eval_model/test";
const DIRICHLET_ALPHA: f32 = 0.3;
const DIRICHLET_EPS: f32 = 0.25;

pub mod logs {
    use super::*;

    pub fn init() {
        let mut buf = PathBuf::new();
        buf.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
        buf.push("tuning/src/eval_model/test/log4rs.yml");
        let config = buf.to_str().unwrap();
        println!("{:?}", config);
        // Ignore the result so tests don't panic if initialized twice
        let _ = log4rs::init_file(config, Default::default());
    }
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
    log::info!("Device: {:?}", device);

    let mut config_path = PathBuf::new();
    config_path.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
    config_path.push("tuning/src/eval_model/test/config.json");

    let config = get_config(config_path.to_str().expect("Invalid config path"));

    config
        .save(format!("{OUT_DIR}/config.json"))
        .expect("Failed to save config.");

    let num_fens_total = config.edp_dataset_fens_total;

    let result_weights = train::<AutodiffBackend>(
        &format!("{OUT_DIR}/learn_mate_in_1"),
        config,
        device.clone(),
    )
    .expect("No epoch was completed");

    // test
    let test_fen = FenDataset::new("mate_in_1.edp", "train", num_fens_total);
    let fen = Dataset::<FenItemRaw>::get(&test_fen, 0).expect("no fen in dataset");
    let mut pos = Position::from_fen(&fen.fen).expect("Bad fen");

    let result = {
        let record = CompactRecorder::new()
            .load(result_weights.into(), &device)
            .expect("Should be able to load the model weights from the provided file");

        let model = ModelConfig::new()
            .init::<Backend>(&device)
            .load_record(record);

        let mut mcts_state = SearchState::default();
        let nn_state = NNParts::new(model, device, DIRICHLET_ALPHA, DIRICHLET_EPS);

        let limit = Limit {
            is_active: true,
            winc: 100,
            binc: 100,
            wtime: 0,
            btime: 0,
            ..Default::default()
        };

        mcts(
            &mut pos,
            &nn_state,
            &mut mcts_state,
            &limit,
            MctsTrainStrategy::new(),
        )
        .0
    };

    println!("{:#?}", result);

    // Note: ensure this target move matches the first FEN in your specific .edp
    // file!
    assert_eq!(
        result,
        Some(Move::new(squares::B5, squares::A5, move_flags::QUIET))
    )
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

    let mut config_path = PathBuf::new();
    config_path.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
    config_path.push("tuning/src/eval_model/test/config.json");

    let config = get_config(config_path.to_str().expect("Invalid config path"));

    config
        .save(format!("{OUT_DIR}/config.json"))
        .expect("Failed to save config.");

    let num_fens_total = config.edp_dataset_fens_total;

    let result_weights = train::<AutodiffBackend>(
        &format!("{OUT_DIR}/learn_mate_in_2"),
        config,
        device.clone(),
    )
    .expect("No epoch was completed");

    // test
    let test_fen = FenDataset::new("mate_in_2.edp", "train", num_fens_total);
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
            winc: 100,
            binc: 100,
            wtime: 0,
            btime: 0,
            ..Default::default()
        };

        let mut mcts_state = SearchState::default();
        let nn_state = NNParts::new(model, device, DIRICHLET_ALPHA, DIRICHLET_EPS);

        // us/mov-1
        let result = mcts(
            &mut pos,
            &nn_state,
            &mut mcts_state,
            &limit,
            MctsTrainStrategy::new(),
        );
        let mov = result.0.expect("Search should have completed by now");
        pos.make_move(mov);
        mcts_state.advance_to(mov);

        // them/mov-1
        let result = mcts(
            &mut pos,
            &nn_state,
            &mut mcts_state,
            &limit,
            MctsTrainStrategy::new(),
        );
        let mov = result.0.expect("Search should have completed by now");
        pos.make_move(mov);
        mcts_state.advance_to(mov);

        // us/mov-2 (This search should find the mate!)
        let result = mcts(
            &mut pos,
            &nn_state,
            &mut mcts_state,
            &limit,
            MctsTrainStrategy::new(),
        );

        // Apply the mate
        if let Some(mov) = result.0 {
            pos.make_move(mov);
            mcts_state.advance_to(mov);
        }

        mcts_state.tree.node(mcts_state.tree.root()).state()
    };

    assert_eq!(NodeState::Terminal, result)
}
