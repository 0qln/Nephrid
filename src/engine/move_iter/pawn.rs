use crate::engine::{
    bitboard::Bitboard,
    coordinates::{CompassRose, File, Square},
    piece::PieceType,
    color::Color,
    position::Position,
    r#move::{Move, MoveFlag},
    masks,
};
use std::marker::PhantomData;

pub mod move_type {
    pub struct Legals;
    pub struct PseudoLegals;
    pub struct Attacks;
    pub struct Resolves;
}

pub mod gen_part {
    pub struct SingleStep;
    pub struct DoubleStep;
    pub struct Promotions;
}

pub struct PseudoLegalPawnMovesInfo<'a> {
    pos: &'a Position,
    pawns: Bitboard,
    enemies: Bitboard,
    pieces: Bitboard,
}

impl<'a> PseudoLegalPawnMovesInfo<'a> {
    pub fn new(pos: &'a Position, color: Color) -> Self {
        let pawns = pos.get_bitboard(PieceType::Pawn, color);
        let pieces = pos.get_occupancy();
        let enemies = pos.get_color_bb(!color);
        Self {
            pos,
            pawns,
            pieces,
            enemies,
        }
    }
}

pub struct PseudoLegalPawnMoves<'a, 'b, Stage> {
    info: &'a PseudoLegalPawnMovesInfo<'b>,
    blockers: Bitboard,
    from: Bitboard,
    to: Bitboard,
    stage: PhantomData<Stage>,
}

impl<'a, 'b> PseudoLegalPawnMoves<'a, 'b, gen_part::SingleStep> {
    pub fn new(info: &'a mut PseudoLegalPawnMovesInfo<'b>) -> Self {
        let blockers = info.pieces << CompassRose::Sout;
        let non_promo_pawns = info.pawns & !masks::RANKS[6];
        let to = (non_promo_pawns & !blockers) << CompassRose::Nort;
        let from = to << CompassRose::Sout;
        Self {
            info,
            blockers,
            from, to,
            stage: PhantomData::<gen_part::SingleStep>,
        }
    }
}

impl Iterator for PseudoLegalPawnMoves<'_, '_, gen_part::SingleStep> {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        match self.to.pop_lsb() {
            to => Some(Move::from((self.from.pop_lsb(), to, MoveFlag::Quiet))),
        }
    }
}

// TODO: test
pub fn white_pawn_attacks(pawn: Bitboard) -> Bitboard {
    let mut result = Bitboard { v: 0 };
    result |= (pawn & !Bitboard::from(File::A)) << CompassRose::West;
    result |= (pawn & !Bitboard::from(File::H)) << CompassRose::East;
    result
}

pub fn generate_moves(position: &Position) -> Iterator<Move> {

}
