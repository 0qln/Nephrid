use std::{env::var, path::PathBuf};

use crate::{FenDataset, FenItemRaw, MctsTrain, TrainingConfig, train};
use burn::{
    backend::Autodiff,
    config::Config,
    data::dataset::Dataset,
    prelude::Module,
    record::{CompactRecorder, Recorder},
};
use burn_cuda::{Cuda, CudaDevice};
use engine::{
    core::{
        coordinates::squares,
        r#move::{Move, move_flags},
        move_iter::sliding_piece::magics,
        position::Position,
        search::{
            limit::Limit,
            mcts::{NNParts, SearchState, mcts, nn::ModelConfig},
        },
        zobrist,
    },
    misc::DebugMode,
    uci::sync::CancellationToken,
};

const OUT_DIR: &'static str = "tuning/out/eval_model/test";
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
        log4rs::init_file(config, Default::default()).unwrap();
    }
}

// todo
// #[test]
// pub fn can_approximate_policy() {
//     magics::init();
//     zobrist::init();

//     // This is the policy that we want to learn. e.g. if we pretend that this
// policy was the     // result of the selfplay phase, training the enging
// toward this should result in very     // mminimal loss.
//     let target_move = Move::new(squares::B5, squares::A5, move_flags::QUIET);
//     let target_policy = {
//         let mut pol = RawPolicy::null();
//         pol.set(usize::from(target_move), 1.0_f32);
//         pol
//     };

//     type Backend = Cuda<f32>;
//     type AutodiffBackend = Autodiff<Backend>;

//     let device = CudaDevice::default();
//     log::info!("Device: {:?}", device);

//     let result_weights = {
//         let mut config_path = PathBuf::new();
//         config_path.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT
// variable"));         config_path.push("tuning/src/eval_model/test/config.
// json");

//         let config = TrainingConfig::load(&config_path).expect(&format!(
//             "Couldn't load config.json at {:?}",
//             config_path.to_str()
//         ));

//         config
//             .save(&format!("{OUT_DIR}/config.json"))
//             .expect("Failed to save config.");

//         train::<AutodiffBackend>(
//             &format!("{OUT_DIR}/learn_mate_in_1"),
//             config,
//             device.clone(),
//         )
//         .expect("No epoch was completed")
//     };
// }

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

    let result_weights = {
        let mut config_path = PathBuf::new();
        config_path.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
        config_path.push("tuning/src/eval_model/test/config.json");

        let config = TrainingConfig::load(&config_path).expect(&format!(
            "Couldn't load config.json at {:?}",
            config_path.to_str()
        ));

        config
            .save(&format!("{OUT_DIR}/config.json"))
            .expect("Failed to save config.");

        train::<AutodiffBackend>(
            &format!("{OUT_DIR}/learn_mate_in_1"),
            config,
            device.clone(),
        )
        .expect("No epoch was completed")
    };

    // test
    let test_fen = FenDataset::new("mate_in_1.edp", "train");
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
        let debug = DebugMode::default();
        let ct = CancellationToken::new();

        mcts(
            &mut pos,
            &nn_state,
            &mut mcts_state,
            limit.clone(),
            debug.clone(),
            ct.clone(),
            MctsTrain::default(),
        )
    };
    println!("{:#?}", result);

    assert_eq!(
        result.0,
        Some(Move::new(squares::B5, squares::A5, move_flags::QUIET))
    )
}

#[ignore]
#[test]
pub fn learn_mate_in_2() {
    magics::init();
    zobrist::init();

    type Backend = Cuda<f32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = CudaDevice::default();
    log::info!("Device: {:?}", device);

    let result_weights = {
        let mut config_path = PathBuf::new();
        config_path.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
        config_path.push("tuning/src/eval_model/test/config.json");

        let config = TrainingConfig::load(&config_path).expect(&format!(
            "Couldn't load config.json at {:?}",
            config_path.to_str()
        ));

        config
            .save(&format!("{OUT_DIR}/config.json"))
            .expect("Failed to save config.");

        train::<AutodiffBackend>(
            &format!("{OUT_DIR}/learn_mate_in_2"),
            config,
            device.clone(),
        )
        .expect("No epoch was completed")
    };

    // test
    let test_fen = FenDataset::new("mate_in_2.edp", "train");
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
        let debug = DebugMode::default();
        let ct = CancellationToken::new();

        let mut mcts_state = SearchState::default();
        let nn_state = NNParts::new(model, device, DIRICHLET_ALPHA, DIRICHLET_EPS);

        // us/mov-1
        let result = mcts(
            &mut pos,
            &nn_state,
            &mut mcts_state,
            limit.clone(),
            debug.clone(),
            ct.clone(),
            MctsTrain::default(),
        );
        let mov = result.0.expect("Search should have completed by now");
        pos.make_move(mov);
        mcts_state.tree.advance_to(|b| b.mov() == mov);

        // them/mov-1
        let result = mcts(
            &mut pos,
            &nn_state,
            &mut mcts_state,
            limit.clone(),
            debug.clone(),
            ct.clone(),
            MctsTrain::default(),
        );
        let mov = result.0.expect("Search should have completed by now");
        pos.make_move(mov);
        mcts_state.tree.advance_to(|b| b.mov() == mov);

        // us/mov-2
        let result = mcts(
            &mut pos,
            &nn_state,
            &mut mcts_state,
            limit.clone(),
            debug.clone(),
            ct.clone(),
            MctsTrain::default(),
        );
        let mov = result.0.expect("Search should have completed by now");
        pos.make_move(mov);
        mcts_state.tree.advance_to(|b| b.mov() == mov);

        let root = mcts_state.tree.get_root();
        let root = root.into_ct();
        let root = root.evaluated().expect("Root should be evaluated");
        let root = root.borrow();

        let branches = root.branches();
        branches.iter().count()
    };

    // result should be a mating position
    assert_eq!(0, result)
}
