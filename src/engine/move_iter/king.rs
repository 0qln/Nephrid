use crate::{
    engine::{
        bitboard::Bitboard,
        coordinates::{File, Rank, Square},
    },
    misc::ConstFrom,
};

// todo: can be computed at compile time
//
pub fn compute_attacks(sq: Square) -> Bitboard {
    let file = File::from_c(sq);
    let rank = Rank::from_c(sq);

    let mut result = Bitboard::empty();

    if file > File::A {
        // Safety: file is in range 1.., so file - 1 is still a valid file.
        let west = unsafe { File::from_v(file.v() - 1) };
        result |= Bitboard::from_c(west);
    }

    if file < File::H {
        // Safety: file is in range 0..7, so file + 1 is still a valid file.
        let east = unsafe { File::from_v(file.v() + 1) };
        result |= Bitboard::from_c(east);
    }
    
    if rank > Rank::_1 {
        // Safety: rank is in range 1.., so rank - 1 is still a valid rank.
        let south = unsafe { Rank::from_v(rank.v() - 1) };
        result |= Bitboard::from_c(south);
    }
    
    if rank < Rank::_8 {
        // Safety: rank is in range 0..7, so rank + 1 is still a valid rank.
        let north = unsafe { Rank::from_v(rank.v() + 1) };
        result |= Bitboard::from_c(north);
    }

    result
}
