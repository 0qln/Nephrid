// deny
#![deny(unsafe_op_in_unsafe_fn)]
// warn
// allow
#![allow(clippy::obfuscated_if_else)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::just_underscores_and_digits)]
#![allow(clippy::almost_complete_range)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::approx_constant)]
#![allow(refining_impl_trait)]
// nightly features
#![allow(incomplete_features)]
#![feature(closure_lifetime_binder)]
#![feature(const_clone)]
#![feature(const_cmp)]
#![feature(const_convert)]
#![feature(const_default)]
#![feature(const_destruct)]
#![feature(const_index)]
#![feature(const_iter)]
#![feature(const_ops)]
#![feature(const_option_ops)]
#![feature(const_trait_impl)]
#![feature(const_try)]
#![feature(const_try_residual)]
#![feature(control_flow_into_value)]
#![feature(debug_closure_helpers)]
#![feature(derive_const)]
#![feature(extend_one)]
#![feature(generic_const_exprs)]
#![feature(impl_trait_in_bindings)]
#![feature(int_roundings)]
#![feature(iter_collect_into)]
#![feature(macro_metavar_expr)]
#![feature(range_into_bounds)]
#![feature(slice_ptr_get)]
#![feature(step_trait)]
#![feature(stmt_expr_attributes)]
#![feature(strip_circumfix)]
#![feature(try_blocks)]
#![feature(try_trait_v2)]
#![feature(try_trait_v2_residual)]

#[macro_use] extern crate impl_ops;

pub mod core;
pub mod math;
pub mod misc;
pub mod uci;
