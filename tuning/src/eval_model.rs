use std::error::Error;

use burn_cuda::{Cuda, CudaDevice};
use engine::{
    core::{
        Engine, color::colors, execute_uci, move_iter::sliding_piece::magics, position::Position,
        search::mcts::eval::model::ModelConfig, zobrist,
    },
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
}

fn self_play(pos: &str, e: &mut Engine) -> Result<(), Box<dyn Error>> {
    let ct = CancellationToken::new();

    execute_uci(e, "ucinewgame".to_string(), ct.clone())?;
    execute_uci(e, format!("position fen {pos}"), ct.clone())?;

    execute_uci(e, "ucinewgame".to_string(), ct.clone())?;
    execute_uci(e, format!("position fen {pos}"), ct.clone())?;

    let tok = &mut Tokenizer::new(pos);

    let position: Position = tok
        .try_into()
        .expect(format!("Invalid FEN: {pos}").as_str());

    loop {
        let command = match position.get_turn() {
            colors::WHITE => "go wtime 1000",
            colors::BLACK => "go btime 1000",
            _ => panic!("Invalid color"),
        };

        execute_uci(e, command.to_string(), ct.clone())?
            .right()
            .expect("go-command didn't return a join handle.")
            .join()
            .expect("failed to join thread");
    }
}
