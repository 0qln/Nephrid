// deny
#![deny(unsafe_op_in_unsafe_fn)]
// warn
#![warn(clippy::obfuscated_if_else)]
// nightly features
#![allow(incomplete_features)]
#![feature(const_trait_impl)]
#![feature(debug_closure_helpers)]
#![feature(macro_metavar_expr)]
#![feature(step_trait)]
#![feature(try_trait_v2)]
#![feature(try_blocks)]
#![feature(assert_matches)]
#![feature(generic_const_exprs)]
#![feature(new_range_api)]
#![feature(range_into_bounds)]
// todo when feature (github.com/rust-lang/rust/issues/67792) is complete:
// - remove the ConstFrom trait and use the core::From trait instead.

#[macro_use]
extern crate impl_ops;

pub mod core;
pub mod misc;
pub mod uci;
