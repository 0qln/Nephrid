use crate::engine::{bitboard::Bitboard, coordinates::Square, r#move::Move, piece::SlidingPieceType, position::Position};

use super::{bishop, rook, sliding_piece};

#[inline]
pub fn gen_legals_check_none(pos: &Position) -> impl Iterator<Item = Move> + '_ {
    sliding_piece::gen_legals_check_none(pos, SlidingPieceType::QUEEN, compute_attacks)
}

#[inline]
pub const fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    Bitboard {
        v: rook::compute_attacks(sq, occupancy).v | bishop::compute_attacks(sq, occupancy).v,
    }
}
