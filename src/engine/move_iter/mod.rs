use super::{bitboard::Bitboard, r#move::Move, position::Position};
use std::ops;

pub mod pawn;
pub mod knight;
pub mod bishop;
pub mod rook;
pub mod queen;
pub mod king;

pub fn gen_plegals<'a>(pos: &'a Position) -> impl Iterator<Item = Move> + 'a {
    let moves: &[Move] = &[

    ];
}

pub fn legal_moves<'a>(pos: &'a Position) -> impl Iterator<Item = Move> + 'a {
    todo!()
}

fn get_key(relevant_occupancy: Bitboard, magic: MagicData, bits: MagicBits) -> MagicKey {
    ((relevant_occupancy * magic) >> (64 - bits)) as MagicKey    
}

impl_op!(* |a: Bitboard, b: MagicData| -> i64 { a.v as i64 * b } );

// todo: make actual wrappers
pub type MagicBits = usize;
pub type MagicData = i64;
pub type MagicKey = usize;

fn init_sliding_piece() {
}
