use nephrid::engine::move_iter::sliding_piece::magics;
use nephrid::engine::{execute_uci, Engine};
use nephrid::uci::{
    sync::{self, CancellationToken},
    tokens::Tokenizer,
};
use std::io::stdin;

fn main() {
    magics::init(0xdeadbeef);

    let input_stream = stdin();
    let mut engine = Engine::default();
    let mut cmd_cancellation = CancellationToken::new();

    execute_uci(
        &mut engine, 
        &mut Tokenizer::new("ucinewgame"), 
        cmd_cancellation.clone());

    // execute_uci(
    //     &mut engine, 
    //     &mut Tokenizer::new("position startpos"), 
    //     cmd_cancellation.clone());

    // execute_uci(
    //     &mut engine, 
    //     &mut Tokenizer::new("go infinite"), 
    //     cmd_cancellation.clone());

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
        // Replace the cancellation token if it's burned.
        if cmd_cancellation.is_cancelled() {
            cmd_cancellation = Default::default();
        }
    }
}