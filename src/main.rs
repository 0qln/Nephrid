use std::{io::stdin, thread};
use uci::CancellationToken;
use engine::Configuration;

mod engine;
mod uci;

fn main() {
    let input_stream = stdin();
    let configuration = Configuration::default();
    let cancellation_sender = CancellationToken::default();
    while !cancellation_sender.is_cancelled() {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => {
                let cancellation_receiver = cancellation_sender.clone();
                thread::spawn(move || { engine::execute_uci(input, configuration, cancellation_receiver) });
            }
            Err(err) => uci::out(&format!("Error: {err}")),
        }
    }
}


