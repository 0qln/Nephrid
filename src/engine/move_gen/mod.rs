use std::marker::PhantomData;
use crate::engine::{
    position::Position, 
    r#move::Move, 
    bitboard::Bitboard,
    coordinates::{Square, File}
};

use super::coordinates::CompassRose;
use super::piece::PieceType;

pub mod piece_type {
    use crate::engine::{
        color::Color,
        coordinates::Rank,
    };
    pub struct Pawns;
    
    impl Pawns {
        pub fn promotion_rank(color: Color) -> Rank {
            match color {
                Color::White => 7.try_into().unwrap(),
                Color::Black => 0.try_into().unwrap()
            }
        } 
    }

    pub struct Knights;
}

pub mod move_type {
    pub struct Legals;
    pub struct PseudoLegals;
    pub struct Attacks;
    pub struct Resolves;
}

pub mod gen_part {
    pub struct One;
    pub struct Two;
}

pub struct MoveGen<'a, PiceType, MoveType, Part> {
    pos: &'a Position,
    current: u8,
    piece_type: PhantomData<PiceType>,
    move_type: PhantomData<MoveType>,
    part: PhantomData<Part>,
}

impl Iterator for MoveGen<'_, piece_type::Pawns, move_type::PseudoLegals, gen_part::One> {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}


pub struct PseudoLegalPawnMoves<'a> {
    pos: &'a Position,
    current: u8,
    pawns: Bitboard,
    enemies: Bitboard,
    pieces: Bitboard,
    blockers: Bitboard,
}

impl PseudoLegalPawnMoves<'_> {
    pub fn new(pos: &Position) -> Self {
        let pawns = pos.get_bitboard(PieceType::Pawn, pos.get_turn());
        let pieces = pos.get_occupancy();
        let enemies = pos.get_color_bb(!pos.get_turn());
        let current = 0;
        let blockers = pieces << CompassRose::Sout;
    }
}

impl Iterator for PseudoLegalPawnMoves<'_> {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

// TODO: test
pub fn white_pawn_attacks(pawn: Bitboard) -> Bitboard {
    let mut result = Bitboard { v: 0 }; 
    result |= (pawn & !Bitboard::from(File::A)) << CompassRose::NoWe;
    result |= (pawn & !Bitboard::from(File::H)) << CompassRose::NoEa;
    result
}

pub fn black_pawn_attacks(pawn: Bitboard) -> Bitboard {
    let mut result = Bitboard { v: 0 }; 
    result |= (pawn & !Bitboard::from(File::A)) << CompassRose::SoWe;
    result |= (pawn & !Bitboard::from(File::H)) << CompassRose::SoEa;
    result
}
