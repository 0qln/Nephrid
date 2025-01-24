use crate::engine::{
    bitboard::Bitboard,
    coordinates::Square,
    piece::{IPieceType, PieceType},
};

use super::{bishop::Bishop, rook::Rook, sliding_piece::Attacks};

pub struct Queen;

impl IPieceType for Queen {
    const ID: PieceType = PieceType::QUEEN;
}

impl Attacks for Queen {
    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        Rook::compute_attacks(sq, occupancy) | Bishop::compute_attacks(sq, occupancy)
    }

    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        Rook::lookup_attacks(sq, occupancy) | Bishop::lookup_attacks(sq, occupancy)
    }
}
