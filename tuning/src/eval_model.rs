use std::{error::Error, ops::ControlFlow};

use burn::prelude::Backend;
use burn_cuda::{Cuda, CudaDevice};
use engine::{
    core::{
        move_iter::{fold_legal_moves, sliding_piece::magics},
        position::Position,
        search::{
            self,
            limit::Limit,
            mcts::eval::model::{Model, ModelConfig},
        },
        zobrist,
    },
    misc::DebugMode,
    uci::{sync::CancellationToken, tokens::Tokenizer},
};

fn main() {
    magics::init();
    zobrist::init();

    let device = CudaDevice::default();
    println!("Device: {:?}", device);

    type Backend = Cuda<f32>;
    let model = ModelConfig::new().init::<Backend>(&device);
    println!("Model: {:#?}", model);

    // let batch = generate_batch(&model);
    let result = self_play(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &model,
    );
    println!("{result:?}");
}

// #[derive(Clone, Debug)]
// pub struct PlayoutBatch<B: Backend> {}

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
