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
#![feature(debug_closure_helpers)]
#![feature(macro_metavar_expr)]
#![feature(once_cell_get_mut)]
#![feature(linked_list_cursors)]
#![feature(non_null_from_ref)]
#![feature(step_trait)]
// todo: why does this feature not work?
#![feature(gen_blocks)]
#![feature(try_trait_v2)]
#![feature(try_blocks)]

#[macro_use]
extern crate impl_ops;

pub mod engine;
pub mod misc;
pub mod uci;