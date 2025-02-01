#![warn(clippy::obfuscated_if_else)]
// todo when feature (github.com/rust-lang/rust/issues/67792) is complete:
// - remove the ConstFrom trait and use the core::From trait instead.
#![feature(const_trait_impl)]
#![feature(debug_closure_helpers)]
#![feature(macro_metavar_expr)]
#![feature(non_null_from_ref)]
#![feature(step_trait)]
#![feature(try_trait_v2)]
#![feature(try_blocks)]
#![feature(assert_matches)]

#[macro_use]
extern crate impl_ops;

pub mod engine;
pub mod misc;
pub mod uci;