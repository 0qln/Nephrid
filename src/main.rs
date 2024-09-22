#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_import_braces)]

#[macro_use] extern crate impl_ops;

use std::io::stdin;
use engine::Engine;
use uci::{
    sync::out,
    tokens::Tokenizer
};

mod engine;
mod uci;
mod macros;

fn main() {
    let input_stream = stdin();
    let mut engine = Engine::default();
    while !engine.cancellation_token.is_cancelled() {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => engine.execute_uci(&mut Tokenizer::new(input.as_str())),                   
            Err(err) => out(&format!("Error: {err}")),
        }
    }
}
