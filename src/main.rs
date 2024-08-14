use std::io::stdin;
use uci::CancellationToken;
use engine::{
    execute_uci, 
    position::Position, 
    config::Configuration
};

mod engine;
mod uci;

fn main() {
    let mut configuration = Configuration::default();
    let input_stream = stdin();
    let cancellation_token = CancellationToken::default();
    let position = Some(Position::default());
    while !cancellation_token.is_cancelled() {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => {
                match input.trim().as_ref() {
                    "d" => {
                        
                    }
                }

                execute_uci(
                    &input, 
                    &mut configuration, 
                    &cancellation_token, 
                    &position
                );                    
            }
            Err(err) => uci::out(&format!("Error: {err}")),
        }
    }
}
