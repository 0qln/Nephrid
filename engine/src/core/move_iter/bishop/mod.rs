use crate::{
    core::{
        bitboard::Bitboard, coordinates::{CompassRose, DiagA1H8, DiagA8H1, Square}, move_iter::bishop, piece::{IPieceType, PieceType}
    },
    misc::ConstFrom,
};

use super::sliding_piece::{self, magics::MagicGen, SlidingAttacks, SlidingPieceType};

#[cfg(test)]
mod tests;

pub struct Bishop;

impl Bishop {
    fn compute_attacks_0_occ(sq: Square) -> Bitboard {
        bishop::compute_attacks_0_occ(sq)
    }
}

impl IPieceType for Bishop {
    const ID: PieceType = PieceType::BISHOP;
}

impl MagicGen for Bishop {
    fn relevant_occupancy(sq: Square) -> Bitboard {
        Self::compute_attacks_0_occ(sq).and_not_c(Bitboard::edges())
    }

    fn relevant_occupancy_num_combinations() -> usize {
        // center is generally max
        let max = Self::relevant_occupancy(Square::E4);
        (1 << max.pop_cnt()) as usize
    }
}

impl SlidingAttacks for Bishop {
    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        crate::core::move_iter::bishop::compute_attacks(sq, occupancy)
    }

    #[allow(static_mut_refs)]
    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        // Safety: The caller has to assert, that the table is initialized.
        unsafe { sliding_piece::magics::BISHOP_MAGICS.get(sq).get(occupancy) }
    }
}

impl SlidingPieceType for Bishop {}

/// Computes the attacks of the bishop on the square `sq`.
pub const fn compute_attacks_0_occ(sq: Square) -> Bitboard {
    let a1h8 = Bitboard::from_c(DiagA1H8::from_c(sq));
    let a8h1 = Bitboard::from_c(DiagA8H1::from_c(sq));
    Bitboard {
        v: (a1h8.v | a8h1.v) ^ Bitboard::from_c(sq).v,
    }
}

/// Computes the attacks of the bishop on the square `sq` with the given `occupancy`.
fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    let a1h8 = Bitboard::from_c(DiagA1H8::from_c(sq));
    let a8h1 = Bitboard::from_c(DiagA8H1::from_c(sq));
    let nort = Bitboard::split_north(sq);
    let sout = Bitboard::split_south(sq);

    let mut result = Bitboard::empty();

    // south east
    let ray = a8h1 & sout;
    let occupands = occupancy & ray;
    let nearest = occupands.msb();
    let range = nearest.map_or(Bitboard::full(), Bitboard::split_north);
    let moves = range.shift_c::<{ CompassRose::SOEA.v() }>() & ray;
    result |= moves;

    // south west
    let ray = a1h8 & sout;
    let occupands = occupancy & ray;
    let nearest = occupands.msb();
    let range = nearest.map_or(Bitboard::full(), Bitboard::split_north);
    let moves = range.shift_c::<{ CompassRose::SOWE.v() }>() & ray;
    result |= moves;

    // north west
    let ray = a8h1 & nort;
    let occupands = occupancy & ray;
    let nearest = occupands.lsb();
    let range = nearest.map_or(Bitboard::full(), Bitboard::split_south);
    let moves = range.shift_c::<{ CompassRose::NOWE.v() }>() & ray;
    result |= moves;

    // north east
    let ray = a1h8 & nort;
    let occupands = occupancy & ray;
    let nearest = occupands.lsb();
    let range = nearest.map_or(Bitboard::full(), Bitboard::split_south);
    let moves = range.shift_c::<{ CompassRose::NOEA.v() }>() & ray;
    result |= moves;

    result
}
