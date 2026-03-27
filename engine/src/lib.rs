// deny
#![deny(unsafe_op_in_unsafe_fn)]
// warn
#![warn(clippy::obfuscated_if_else)]
// allow
#![allow(clippy::just_underscores_and_digits)]
#![allow(clippy::almost_complete_range)]
#![allow(clippy::len_without_is_empty)]
// nightly features
#![allow(incomplete_features)]
#![feature(if_let_guard)]
#![feature(const_trait_impl)]
#![feature(debug_closure_helpers)]
#![feature(macro_metavar_expr)]
#![feature(step_trait)]
#![feature(try_trait_v2)]
#![feature(try_blocks)]
#![feature(assert_matches)]
#![feature(generic_const_exprs)]
#![feature(iter_collect_into)]
#![feature(new_range_api)]
#![feature(range_into_bounds)]
#![feature(control_flow_into_value)]
// todo when feature (https://github.com/rust-lang/rust/issues/67792) is complete:
// - remove the ConstFrom trait and use the core::From trait instead.
#![feature(associated_type_defaults)]

#[macro_use]
extern crate impl_ops;

pub mod core;
pub mod misc;
pub mod uci;
