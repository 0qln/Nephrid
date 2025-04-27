use burn_cuda::{Cuda, CudaDevice};
use engine::core::{
    color::Color, execute_uci, move_iter::sliding_piece::magics, position::{self, Position}, search::mcts::eval::model::ModelConfig, zobrist, Engine
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

fn self_play(pos: &str, w: &mut Engine, b: &mut Engine) -> Score {
    let ct = CancellationToken::new();

    execute_uci(w, "ucinewgame".to_string(), ct);
    execute_uci(w, format!("position fen {pos}"), ct);

    execute_uci(b, "ucinewgame".to_string(), ct);
    execute_uci(b, format!("position fen {pos}"), ct);

    let position: Position = pos
        .try_into()
        .expect(format!("Invalid FEN: {pos}").as_str());
    
    loop {
        match position.get_turn() {
            Color::WHITE => execute_uci(w, "go wtime 1000".to_string(), ct),
            Color::BLACK => execute_uci(b, "go btime 1000".to_string(), ct)
        }
    }
}
