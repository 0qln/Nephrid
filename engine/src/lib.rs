// deny
#![deny(unsafe_op_in_unsafe_fn)]
// warn
#![warn(clippy::obfuscated_if_else)]
// allow
#![allow(clippy::just_underscores_and_digits)]
#![allow(clippy::almost_complete_range)]
#![allow(clippy::len_without_is_empty)]
#![allow(refining_impl_trait)]
// nightly features
#![allow(incomplete_features)]
#![feature(const_trait_impl)]
#![feature(debug_closure_helpers)]
#![feature(macro_metavar_expr)]
#![feature(step_trait)]
#![feature(try_trait_v2)]
#![feature(try_blocks)]
#![feature(generic_const_exprs)]
#![feature(range_into_bounds)]
#![feature(control_flow_into_value)]
#![feature(strip_circumfix)]
#![feature(const_convert)]
#![feature(iter_collect_into)]
#![feature(const_ops)]
#![feature(const_cmp)]
#![feature(const_default)]
#![feature(const_index)]
#![feature(extend_one)]
#![feature(int_roundings)]
#![feature(derive_const)]
#![feature(const_clone)]
#![feature(const_try)]
#![feature(const_try_residual)]
#![feature(try_trait_v2_residual)]
#![feature(stmt_expr_attributes)]

#[macro_use] extern crate impl_ops;

pub mod core;
pub mod math;
pub mod misc;
pub mod uci;
