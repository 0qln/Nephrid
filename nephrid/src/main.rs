use engine::core::move_iter::sliding_piece::magics;
use engine::core::{execute_uci, zobrist, Engine};
use engine::uci::sync::{self, CancellationToken};
use std::io::stdin;

fn main() {
    magics::init();
    zobrist::init();

    let input_stream = stdin();
    let mut engine = Engine::default();
    let mut cmd_cancellation = CancellationToken::new();

    execute_uci(
        &mut engine,
        "ucinewgame".to_string(),
        cmd_cancellation.clone(),
    )
    .unwrap();

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
            Ok(_) => {
                if let Err(e) = execute_uci(&mut engine, input, cmd_cancellation.clone()) {
                    sync::out(&format!("{e}"));
                }
            }
            Err(err) => sync::out(&format!("Error: {err}")),
        }
        // Replace the cancellation token if it's burned.
        if cmd_cancellation.is_cancelled() {
            cmd_cancellation = Default::default();
        }
    }
}
