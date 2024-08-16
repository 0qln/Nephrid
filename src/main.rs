use std::io::stdin;
use uci::CancellationToken;
use engine::{ Engine, position::Position, config::Configuration };

mod engine;
mod uci;

fn main() {
    let input_stream = stdin();
    let mut cancellation_token = CancellationToken::default();
    let mut engine = Engine {
        config: Configuration::default(),
        position: Position::default(),
        cancellation_token: &cancellation_token
    };
    while !cancellation_token.is_cancelled() {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => {
                engine.execute_uci(&input);                    
            }
            Err(err) => uci::out(&format!("Error: {err}")),
        }
    }
}
