use std::env::var;
use std::path::PathBuf;

use crate::FenDataset;
use crate::FenItemRaw;
use crate::MctsTrain;
use crate::TrainingConfig;
use crate::train;
use burn::backend::Autodiff;
use burn::config::Config;
use burn::data::dataset::Dataset;
use burn::prelude::Module;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use burn_cuda::Cuda;
use burn_cuda::CudaDevice;
use engine::core::coordinates::squares;
use engine::core::r#move::Move;
use engine::core::r#move::move_flags;
use engine::core::move_iter::sliding_piece::magics;
use engine::core::position::Position;
use engine::core::search::limit::Limit;
use engine::core::search::mcts::NNParts;
use engine::core::search::mcts::SearchState;
use engine::core::search::mcts::mcts;
use engine::core::search::mcts::nn::ModelConfig;
use engine::core::zobrist;
use engine::misc::DebugMode;
use engine::uci::sync::CancellationToken;
use engine::uci::tokens::Tokenizer;

const OUT_DIR: &'static str = "tuning/out/eval_model/test";

#[test]
pub fn learn_mate_in_1() {
    magics::init();
    zobrist::init();

    {
        let mut buf = PathBuf::new();
        buf.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
        buf.push("tuning/src/eval_model/test/log4rs.yml");
        let config = buf.to_str().unwrap();
        println!("{:?}", config);
        log4rs::init_file(config, Default::default()).unwrap();
    }

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
    let tok = &mut Tokenizer::new(&fen.fen);
    let pos: Position = tok.try_into().expect("");
    let result = {
        let record = CompactRecorder::new()
            .load(result_weights.into(), &device)
            .expect("Should be able to load the model weights from the provided file");

        let model = ModelConfig::new()
            .init::<Backend>(&device)
            .load_record(record);

        let mut mcts_state = SearchState::default();
        let nn_state = NNParts::new(model, device);

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
            &pos,
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

#[test]
pub fn learn_mate_in_2() {
    magics::init();
    zobrist::init();

    {
        let mut buf = PathBuf::new();
        buf.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
        buf.push("tuning/src/eval_model/test/log4rs.yml");
        let config = buf.to_str().unwrap();
        println!("{:?}", config);
        log4rs::init_file(config, Default::default()).unwrap();
    }

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
    let tok = &mut Tokenizer::new(&fen.fen);
    let mut pos: Position = tok.try_into().expect("");
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
        let nn_state = NNParts::new(model, device);

        // us/mov-1
        let result = mcts(
            &pos,
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
            &pos,
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
            &pos,
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

        result.1.get_root().borrow().iter_branches().count()
    };

    // result should be a mating position
    assert_eq!(0, result)
}
