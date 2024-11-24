use crate::{engine::{bitboard::Bitboard, coordinates::{CompassRose, DiagA1H8, DiagA8H1, Square}}, misc::ConstFrom};

#[cfg(test)]
mod tests;

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