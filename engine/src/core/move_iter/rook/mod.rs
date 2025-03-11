use crate::{
    core::{
        bitboard::Bitboard, coordinates::{CompassRose, File, Rank, Square}, move_iter::rook, piece::{IPieceType, PieceType}
    },
    misc::ConstFrom,
};

use super::sliding_piece::{self, magics::MagicGen, SlidingAttacks, SlidingPieceType};

#[cfg(test)]
mod tests;

pub struct Rook;

impl IPieceType for Rook {
    const ID: PieceType = PieceType::ROOK;
}

impl Rook {
    #[inline]
    const fn relevant_file(file: File) -> Bitboard {
        Bitboard::from_c(file)
            .and_not_c(Bitboard::from_c(Rank::_1))
            .and_not_c(Bitboard::from_c(Rank::_8))
    }

    #[inline]
    const fn relevant_rank(rank: Rank) -> Bitboard {
        Bitboard::from_c(rank)
            .and_not_c(Bitboard::from_c(File::A))
            .and_not_c(Bitboard::from_c(File::H))
    }
}

impl MagicGen for Rook {
    fn relevant_occupancy(sq: Square) -> Bitboard {
        !Bitboard::from_c(sq)
            & (Self::relevant_file(File::from_c(sq)) | Self::relevant_rank(Rank::from_c(sq)))
    }

    #[inline]
    fn relevant_occupancy_num_combinations() -> usize {
        // its the same for each corner, but the corners are generally max,
        // because the square itself is not excluded additionally.
        let max = Self::relevant_occupancy(Square::A1);
        (1 << max.pop_cnt()) as usize
    }
}

impl SlidingAttacks for Rook {
    #[inline]
    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        rook::compute_attacks(sq, occupancy)
    }

    #[allow(static_mut_refs)]
    #[inline]
    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        // Safety: The caller has to assert, that the table is initialized.
        sliding_piece::magics::rook_magics().get(sq).get(occupancy)
    }
}

impl SlidingPieceType for Rook {}

/// Computes the attacks of the rook on the square `sq`.
pub const fn compute_attacks_0_occ(sq: Square) -> Bitboard {
    let file_bb = Bitboard::from_c(File::from_c(sq));
    let rank_bb = Bitboard::from_c(Rank::from_c(sq));
    Bitboard {
        v: (file_bb.v | rank_bb.v) ^ Bitboard::from_c(sq).v,
    }
}

/// Computes the attacks of the rook on the square `sq` with the given `occupancy`.
fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    let file = File::from_c(sq);
    let rank = Rank::from_c(sq);
    let file_bb = Bitboard::from_c(file);
    let rank_bb = Bitboard::from_c(rank);
    let nort_bb = Bitboard::split_north(sq);
    let sout_bb = Bitboard::split_south(sq);
    let mut result = Bitboard::empty();

    // south
    let ray = file_bb & sout_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.msb();
    let range = nearest.map_or(Bitboard::full(), Bitboard::split_north);
    let moves = range.shift_c::<{ CompassRose::SOUT.v() }>() & ray;
    result |= moves;

    // north
    let ray = file_bb & nort_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.lsb();
    let range = nearest.map_or(Bitboard::full(), Bitboard::split_south);
    let moves = range.shift_c::<{ CompassRose::NORT.v() }>() & ray;
    result |= moves;

    // west
    let ray = rank_bb & sout_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.msb();
    let range = nearest.map_or(Bitboard::full(), Bitboard::split_north);
    let moves = range.shift_c::<{ CompassRose::WEST.v() }>() & ray;
    result |= moves;

    // east
    let ray = rank_bb & nort_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.lsb();
    let range = nearest.map_or(Bitboard::full(), Bitboard::split_south);
    let moves = range.shift_c::<{ CompassRose::EAST.v() }>() & ray;
    result |= moves;

    result
}
