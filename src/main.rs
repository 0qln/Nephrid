#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_import_braces)]
// todo when feature (github.com/rust-lang/rust/issues/67792) is complete:
// - make impls const.
// - remove the ConstFrom trait and use the core::From trait instead.
#![feature(const_trait_impl)]
#![feature(const_for)]
#![feature(const_heap)]
#![feature(unchecked_shifts)]
#[macro_use]
extern crate impl_ops;

use engine::{Engine, execute_uci};
use std::io::stdin;
use uci::{sync::{self, CancellationToken}, tokens::Tokenizer};

mod engine;
mod misc;
mod uci;

fn main() {
    let input_stream = stdin();
    let mut engine = Engine::default();
    let cmd_cancellation: CancellationToken = CancellationToken::default();
    loop {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => execute_uci(
                &mut engine, 
                &mut Tokenizer::new(input.as_str()), 
                cmd_cancellation.clone()),
            Err(err) => sync::out(&format!("Error: {err}")),
        }
    }
}
