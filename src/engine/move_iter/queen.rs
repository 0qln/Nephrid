use crate::engine::{bitboard::Bitboard, coordinates::Square};

use super::{bishop, rook};

#[inline]
pub const fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    Bitboard {
        v: rook::compute_attacks(sq, occupancy).v | bishop::compute_attacks(sq, occupancy).v,
    }
}
