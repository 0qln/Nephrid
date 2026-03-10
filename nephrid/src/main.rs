use engine::{
    core::{Engine, execute_uci, move_iter::sliding_piece::magics, zobrist},
    uci::sync::{self, CancellationToken},
};
use std::io::stdin;

fn main() {
    magics::init();
    zobrist::init();

    let input_stream = stdin();
    let mut engine = Engine::new();
    let mut cmd_cancellation = CancellationToken::new();

    execute_uci(
        &mut engine,
        "ucinewgame".to_string(),
        cmd_cancellation.clone(),
    )
    .unwrap();

    execute_uci(
        &mut engine,
        "position startpos".to_string(),
        cmd_cancellation.clone(),
    )
    .unwrap();

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
