use crate::engine::{bitboard::Bitboard, coordinates::Square, r#move::Move, piece::{IPieceType, PieceType, SlidingPieceType}, position::Position};

use super::{bishop::{self, Bishop}, rook::{self, Rook}, sliding_piece::{self, Attacks}};

pub struct Queen;

impl IPieceType for Queen {
    const ID: PieceType = PieceType::QUEEN;
}

impl Attacks for Queen {
    fn compute_attacks_0_occ(sq: Square) -> Bitboard {
        Rook::compute_attacks_0_occ(sq) | Bishop::compute_attacks_0_occ(sq)
    }

    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        Rook::compute_attacks(sq, occupancy) | Bishop::compute_attacks(sq, occupancy)
    }

    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        Rook::lookup_attacks(sq, occupancy) | Bishop::lookup_attacks(sq, occupancy)
    }
}

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
