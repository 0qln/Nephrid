use crate::{engine::{bitboard::Bitboard, coordinates::{CompassRose, DiagA1H8, DiagA8H1, Square}, r#move::Move, piece::{IPieceType, PieceType, SlidingPieceType}, position::Position}, misc::ConstFrom};

use super::sliding_piece::{self, magics::MagicGen, Attacks};

#[cfg(test)]
mod tests;

pub struct Bishop;

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

impl Attacks for Bishop {
    fn compute_attacks_0_occ(sq: Square) -> Bitboard {
        crate::engine::move_iter::bishop::compute_attacks_0_occ(sq)
    }

    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        crate::engine::move_iter::bishop::compute_attacks(sq, occupancy)
    }

    #[allow(static_mut_refs)]
    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        // Safety: The caller has to assert, that the table is initialized.
        unsafe { sliding_piece::magics::BISHOP_MAGICS.get(sq).get(occupancy) }
    }
}

#[inline]
pub fn gen_legals_check_none(pos: &Position) -> impl Iterator<Item = Move> + '_ {
    sliding_piece::gen_legals_check_none(pos, SlidingPieceType::BISHOP, compute_attacks)
}

/// Computes the attacks of the bishop on the square `sq`.
pub const fn compute_attacks_0_occ(sq: Square) -> Bitboard {
    let a1h8 = Bitboard::from_c(DiagA1H8::from_c(sq));
    let a8h1 = Bitboard::from_c(DiagA8H1::from_c(sq));
    Bitboard { v: (a1h8.v | a8h1.v) ^ Bitboard::from_c(sq).v }
}

/// Computes the attacks of the bishop on the square `sq` with the given `occupancy`.
pub const fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    let a1h8 = Bitboard::from_c(DiagA1H8::from_c(sq));
    let a8h1 = Bitboard::from_c(DiagA8H1::from_c(sq));
    let nort = Bitboard::split_north(sq);
    let sout = Bitboard::split_south(sq);
    
    let mut result = Bitboard::empty();
    
    // south east
    let ray = a8h1.v & sout.v;
    let occupands = Bitboard { v: occupancy.v & ray };
    let nearest = occupands.msb();
    let range = match nearest {
        Some(t) => Bitboard::split_north(t),
        None => Bitboard::full(),
    };
    let moves = range.shift_c::<{CompassRose::SOEA.v()}>().v & ray;
    result.v |= moves;
    
    // south west
    let ray = a1h8.v & sout.v;
    let occupands = Bitboard { v: occupancy.v & ray };
    let nearest = occupands.msb();
    let range = match nearest {
        Some(t) => Bitboard::split_north(t),
        None => Bitboard::full(),
    };
    let moves = range.shift_c::<{CompassRose::SOWE.v()}>().v & ray;
    result.v |= moves;
    
    // north west
    let ray = a8h1.v & nort.v;
    let occupands = Bitboard { v: occupancy.v & ray };
    let nearest = occupands.lsb();
    let range = match nearest {
        Some(t) => Bitboard::split_south(t),
        None => Bitboard::full(),
    };
    let moves = range.shift_c::<{CompassRose::NOWE.v()}>().v & ray;
    result.v |= moves;
    
    // north east
    let ray = a1h8.v & nort.v;
    let occupands = Bitboard { v: occupancy.v & ray };
    let nearest = occupands.lsb();
    let range = match nearest {
        Some(t) => Bitboard::split_south(t),
        None => Bitboard::full(),
    };
    let moves = range.shift_c::<{CompassRose::NOEA.v()}>().v & ray;
    result.v |= moves;

    result
}