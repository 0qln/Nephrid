use nephrid::engine::move_iter::sliding_piece::magics;
use nephrid::engine::{execute_uci, Engine};
use nephrid::uci::{
    sync::{self, CancellationToken},
    tokens::Tokenizer,
};
use std::io::stdin;

fn main() {
    // Safety: this is the start of the program.
    unsafe { magics::init(0xdeadbeef) };
    let input_stream = stdin();
    let mut engine = Engine::default();
    let cmd_cancellation: CancellationToken = CancellationToken::default();
    loop {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => execute_uci(
                &mut engine,
                &mut Tokenizer::new(input.as_str()),
                cmd_cancellation.clone(),
            ),
            Err(err) => sync::out(&format!("Error: {err}")),
        }
    }
}