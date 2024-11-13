use crate::{engine::{
    bitboard::Bitboard,
    coordinates::File
}, misc::ConstFrom};

use super::coordinates::CompassRose;

// TODO: test

pub fn white_pawn_attacks(pawn: Bitboard) -> Bitboard {
    let mut result = Bitboard::empty(); 
    result |= (pawn & !Bitboard::from_c(File::A)).shift_c::<{CompassRose::NOWE.v()}>();
    result |= (pawn & !Bitboard::from_c(File::H)).shift_c::<{CompassRose::NOEA.v()}>();
    result
}

pub fn black_pawn_attacks(pawn: Bitboard) -> Bitboard {
    let mut result = Bitboard::empty(); 
    result |= (pawn & !Bitboard::from_c(File::A)).shift_c::<{CompassRose::SOWE.v()}>();
    result |= (pawn & !Bitboard::from_c(File::H)).shift_c::<{CompassRose::SOEA.v()}>();
    result
}
