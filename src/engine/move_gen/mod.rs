use std::marker::PhantomData;
use crate::engine::{
    position::Position, 
    r#move::Move, 
    bitboard::Bitboard,
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
        let blockers = Bitboard { v: (pieces << CompassRose::Sout as isize) as u64 };
        Self { .. }
    }
}

impl Iterator for PseudoLegalPawnMoves<'_> {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}


const WHITE_PAWN_ATTACKS: [Bitboard; 64] = [
    Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 },
Bitboard { v:                         0x20000 },
Bitboard { v:                         0x50000 },
Bitboard { v:                         0xa0000 },
Bitboard { v:                         0x140000 },
Bitboard { v:                         0x280000 },
Bitboard { v:                         0x500000 },
Bitboard { v:                         0xa00000 },
Bitboard { v:                         0x400000 },
Bitboard { v:                         0x2000000 },
Bitboard { v:                         0x5000000 },
Bitboard { v:                         0xa000000 },
Bitboard { v:                         0x14000000 },
Bitboard { v:                         0x28000000 },
Bitboard { v:                         0x50000000 },
Bitboard { v:                         0xa0000000 },
Bitboard { v:                         0x40000000 },
Bitboard { v:                         0x200000000 },
Bitboard { v:                         0x500000000 },
Bitboard { v:                         0xa00000000 },
Bitboard { v:                         0x1400000000 },
Bitboard { v:                         0x2800000000 },
Bitboard { v:                         0x5000000000 },
Bitboard { v:                         0xa000000000 },
Bitboard { v:                         0x4000000000 },
Bitboard { v:                         0x20000000000 },
Bitboard { v:                         0x50000000000 },
Bitboard { v:                         0xa0000000000 },
Bitboard { v:                         0x140000000000 },
Bitboard { v:                         0x280000000000 },
Bitboard { v:                         0x500000000000 },
Bitboard { v:                         0xa00000000000 },
Bitboard { v:                         0x400000000000 },
Bitboard { v:                         0x2000000000000 },
Bitboard { v:                         0x5000000000000 },
Bitboard { v:                         0xa000000000000 },
Bitboard { v:                         0x14000000000000 },
Bitboard { v:                         0x28000000000000 },
Bitboard { v:                         0x50000000000000 },
Bitboard { v:                         0xa0000000000000 },
Bitboard { v:                         0x40000000000000 },
Bitboard { v:                         0x200000000000000 },
Bitboard { v:                         0x500000000000000 },
Bitboard { v:                         0xa00000000000000 },
Bitboard { v:                         0x1400000000000000 },
Bitboard { v:                         0x2800000000000000 },
Bitboard { v:                         0x5000000000000000 },
Bitboard { v:                         0xa000000000000000 },
Bitboard { v:                         0x4000000000000000 },
                        Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }, Bitboard { v: 0 }
];


