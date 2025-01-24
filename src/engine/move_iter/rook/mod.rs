use crate::{
    engine::{
        bitboard::Bitboard, coordinates::{CompassRose, File, Rank, Square}, r#move::Move, piece::{IPieceType, PieceType, SlidingPieceType}, position::Position
    },
    misc::ConstFrom,
};

use super::sliding_piece::{self, magics::MagicGen, Attacks};

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

impl Attacks for Rook {
    #[inline]
    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        crate::engine::move_iter::rook::compute_attacks(sq, occupancy)
    }

    #[allow(static_mut_refs)]
    #[inline]
    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
        // Safety: The caller has to assert, that the table is initialized.
        unsafe { sliding_piece::magics::ROOK_MAGICS.get(sq).get(occupancy) }
    }
}

/// Computes the attacks of the rook on the square `sq`.
pub const fn compute_attacks_0_occ(sq: Square) -> Bitboard {
    let file_bb = Bitboard::from_c(File::from_c(sq));
    let rank_bb = Bitboard::from_c(Rank::from_c(sq));
    Bitboard {
        v: (file_bb.v | rank_bb.v) ^ Bitboard::from_c(sq).v,
    }
}

/// Computes the attacks of the rook on the square `sq` with the given `occupancy`.
const fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    let file = File::from_c(sq);
    let rank = Rank::from_c(sq);
    let file_bb = Bitboard::from_c(file);
    let rank_bb = Bitboard::from_c(rank);
    let nort_bb = Bitboard::split_north(sq);
    let sout_bb = Bitboard::split_south(sq);
    let mut result = Bitboard::empty();

    // south
    let ray = file_bb.v & sout_bb.v;
    let occupands = Bitboard {
        v: occupancy.v & ray,
    };
    let nearest = occupands.msb();
    // todo: this is an inlined 'map_or'. When nightly rust provides
    // it, this should be replaced again. The others are the same
    let range = {
        let default = Bitboard::full();
        let f = Bitboard::split_north;
        match nearest {
            Some(t) => f(t),
            None => default,
        }
    };
    let moves = range.shift_c::<{ CompassRose::SOUT.v() }>().v & ray;
    result.v |= moves;

    // north
    let ray = file_bb.v & nort_bb.v;
    let occupands = Bitboard {
        v: occupancy.v & ray,
    };
    let nearest = occupands.lsb();
    let range = match nearest {
        Some(t) => Bitboard::split_south(t),
        None => Bitboard::full(),
    };
    let moves = range.shift_c::<{ CompassRose::NORT.v() }>().v & ray;
    result.v |= moves;

    // west
    let ray = rank_bb.v & sout_bb.v;
    let occupands = Bitboard {
        v: occupancy.v & ray,
    };
    let nearest = occupands.msb();
    let range = match nearest {
        Some(t) => Bitboard::split_north(t),
        None => Bitboard::full(),
    };
    let moves = range.shift_c::<{ CompassRose::WEST.v() }>().v & ray;
    result.v |= moves;

    // east
    let ray = rank_bb.v & nort_bb.v;
    let occupands = Bitboard {
        v: occupancy.v & ray,
    };
    let nearest = occupands.lsb();
    let range = match nearest {
        Some(t) => Bitboard::split_south(t),
        None => Bitboard::full(),
    };
    let moves = range.shift_c::<{ CompassRose::EAST.v() }>().v & ray;
    result.v |= moves;

    result
}
