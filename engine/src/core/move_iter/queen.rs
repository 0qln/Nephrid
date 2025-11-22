use crate::core::{
    bitboard::Bitboard,
    coordinates::Square,
    piece::{piece_type, IPieceType, PieceType},
};

use super::{
    bishop::Bishop,
    rook::Rook,
    sliding_piece::{SlidingAttacks, SlidingPieceType},
};

pub struct Queen;

impl IPieceType for Queen {
    const ID: PieceType = piece_type::QUEEN;
}

impl SlidingAttacks for Queen {
    #[inline]
    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        Rook::compute_attacks(sq, occupancy) | Bishop::compute_attacks(sq, occupancy)
    }

    #[inline]
    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        Rook::lookup_attacks(sq, occupancy) | Bishop::lookup_attacks(sq, occupancy)
    }
}

impl SlidingPieceType for Queen {}
