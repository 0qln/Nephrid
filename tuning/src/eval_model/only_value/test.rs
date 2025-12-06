use std::env::var;
use std::path::PathBuf;

use super::FenDataset;
use super::FenItemRaw;
use super::MctsTrain;
use super::TrainingConfig;
use super::train;
use burn::backend::Autodiff;
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
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
use engine::core::search;
use engine::core::search::limit::Limit;
use engine::core::search::mcts::eval::model_only_value::ModelConfig;
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

    // train
    let result_weights = train::<AutodiffBackend>(
        &format!("{OUT_DIR}/learn_mate_in_1"),
        TrainingConfig::new(
            ModelConfig::new(),
            AdamConfig::new(),
            "mate_in_1.edp".to_string(),
        )
        .with_batch_size(1)
        .with_num_epochs(1000)
        .with_learning_rate(1.0e-4),
        device.clone(),
    )
    .expect("No epoch was completed");

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

        search::mcts::<MctsTrain, _>(
            pos.clone(),
            &model,
            limit.clone(),
            debug.clone(),
            ct.clone(),
        )
    };
    println!("{:#?}", result);

    assert_eq!(
        result.0,
        Some(Move::new(squares::B5, squares::A5, move_flags::QUIET))
    )
}
