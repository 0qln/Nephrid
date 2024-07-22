use std::{io::stdin, thread};

mod engine;
mod uci;

pub struct CancellationToken;

impl CancellationToken {
    pub fn new() -> Self {
        CancellationToken
    }

    pub fn trigger(&self) {

    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        CancellationToken
    }
}

fn main() {
    let input_stream = stdin();
    loop {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => {
                thread::spawn(move || { engine::execute_uci(input); });
            }
            Err(err) => uci::out(format!("Error: {err}", err)),
        }
    }
}


