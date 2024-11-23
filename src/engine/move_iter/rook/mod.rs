use crate::{engine::{bitboard::Bitboard, coordinates::{CompassRose, File, Rank, Square}}, misc::ConstFrom};

#[cfg(test)]
mod tests;

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
    let occupands = Bitboard { v: occupancy.v & ray };
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
    let moves = range.shift_c::<{CompassRose::SOUT.v()}>().v & ray;
    result.v |= moves;

    // north
    let ray = file_bb.v & nort_bb.v;
    let occupands = Bitboard { v: occupancy.v & ray };
    let nearest = occupands.lsb();
    let range = match nearest {
        Some(t) => Bitboard::split_south(t),
        None => Bitboard::full(),
    };
    let moves = range.shift_c::<{CompassRose::NORT.v()}>().v & ray;
    result.v |= moves;

    // west
    let ray = rank_bb.v & sout_bb.v;
    let occupands = Bitboard { v: occupancy.v & ray };
    let nearest = occupands.msb();
    let range = match nearest {
        Some(t) => Bitboard::split_north(t),
        None => Bitboard::full(),
    };
    let moves = range.shift_c::<{CompassRose::WEST.v()}>().v & ray;
    result.v |= moves;

    // east
    let ray = rank_bb.v & nort_bb.v;
    let occupands = Bitboard { v: occupancy.v & ray };
    let nearest = occupands.lsb();
    let range = match nearest {
        Some(t) => Bitboard::split_south(t),
        None => Bitboard::full(),
    };
    let moves = range.shift_c::<{CompassRose::EAST.v()}>().v & ray;
    result.v |= moves;

    result
}