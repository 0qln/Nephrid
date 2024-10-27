use stage::*;

use crate::engine::coordinates::Square;
use crate::engine::{
    bitboard::Bitboard,
    color::Color,
    coordinates::{CompassRose, File, Squares},
    masks,
    piece::PieceType,
    position::Position,
    r#move::{Move, MoveFlag},
};
use std::marker::PhantomData;

pub mod move_type {
    pub struct Legals;
    pub struct PseudoLegals;
    pub struct Attacks;
    pub struct Resolves;
}

pub mod stage {
    use std::marker::PhantomData;

    pub struct SingleStep;
    pub struct DoubleStep;
    pub struct PromotionInfo;
    pub struct Promotion<Type>(PhantomData<Type>);
    pub struct PromotionCapture<Direction, Type>(PhantomData<Direction>, PhantomData<Type>);
    pub struct Capture<Direction>(PhantomData<Direction>);
    pub struct EnPassant<Direction>(PhantomData<Direction>);

    pub struct West;
    pub struct East;
    pub struct Knight;
    pub struct Bishop;
    pub struct Rook;
    pub struct Queen;
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

pub struct PseudoLegalPawnMoves<Part> {
    from: Bitboard,
    to: Bitboard,
    flag: MoveFlag,
    stage: PhantomData<Part>,
}

impl PseudoLegalPawnMoves<SingleStep> {
    pub fn new(info: &PseudoLegalPawnMovesInfo) -> Self {
        let non_promo_pawns = info.pawns & !masks::RANKS[6];
        let single_step_blocker = info.pieces << CompassRose::Sout;
        let to = (non_promo_pawns & !single_step_blocker) << CompassRose::Nort;
        let from = to << CompassRose::Sout;
        Self {
            from,
            to,
            stage: PhantomData,
            flag: MoveFlag::Quiet,
        }
    }
}

impl PseudoLegalPawnMoves<Capture<West>> {
    pub fn new(info: &PseudoLegalPawnMovesInfo) -> Self {
        let non_promo_pawns = info.pawns & !masks::RANKS[6];
        let capture_west_pawns = non_promo_pawns & !masks::FILES[0];
        let to = (capture_west_pawns << CompassRose::NoWe) & info.enemies;
        let from = to << CompassRose::SoEa;
        Self {
            from,
            to,
            stage: PhantomData,
            flag: MoveFlag::Capture,
        }
    }
}

impl PseudoLegalPawnMoves<Capture<East>> {
    pub fn new(info: &PseudoLegalPawnMovesInfo) -> Self {
        let non_promo_pawns = info.pawns & !masks::RANKS[6];
        let capture_east_pawns = non_promo_pawns & !masks::FILES[7];
        let to = (capture_east_pawns << CompassRose::NoEa) & info.enemies;
        let from = to << CompassRose::SoWe;
        Self {
            from,
            to,
            stage: PhantomData,
            flag: MoveFlag::Capture,
        }
    }
}

impl PseudoLegalPawnMoves<Promotion<Knight>> {
    pub fn new(info: &PseudoLegalPawnMovesInfo) -> Self {
        let single_step_blocker = info.pieces << CompassRose::Sout;
        let promo_pawns = info.pawns & masks::RANKS[6];
        let to = (promo_pawns & !single_step_blocker) << CompassRose::Nort;
        let from = to << CompassRose::Sout;
        Self {
            from,
            to,
            stage: PhantomData,
            flag: MoveFlag::PromotionKnight,
        }
    }
}

impl PseudoLegalPawnMoves<Promotion<Bishop>> {
    pub fn new(info: &PseudoLegalPawnMovesInfo) -> Self {
        let single_step_blocker = info.pieces << CompassRose::Sout;
        let promo_pawns = info.pawns & masks::RANKS[6];
        let to = (promo_pawns & !single_step_blocker) << CompassRose::Nort;
        let from = to << CompassRose::Sout;
        Self {
            from,
            to,
            stage: PhantomData,
            flag: MoveFlag::PromotionBishop,
        }
    }
}

impl PseudoLegalPawnMoves<Promotion<Rook>> {
    pub fn new(info: &PseudoLegalPawnMovesInfo) -> Self {
        let single_step_blocker = info.pieces << CompassRose::Sout;
        let promo_pawns = info.pawns & masks::RANKS[6];
        let to = (promo_pawns & !single_step_blocker) << CompassRose::Nort;
        let from = to << CompassRose::Sout;
        Self {
            from,
            to,
            stage: PhantomData,
            flag: MoveFlag::PromotionRook,
        }
    }
}

impl PseudoLegalPawnMoves<Promotion<Queen>> {
    pub fn new(info: &PseudoLegalPawnMovesInfo) -> Self {
        let single_step_blocker = info.pieces << CompassRose::Sout;
        let promo_pawns = info.pawns & masks::RANKS[6];
        let to = (promo_pawns & !single_step_blocker) << CompassRose::Nort;
        let from = to << CompassRose::Sout;
        Self {
            from,
            to,
            stage: PhantomData,
            flag: MoveFlag::PromotionQueen,
        }
    }
}

impl PseudoLegalPawnMoves<DoubleStep> {
    pub fn new(info: &PseudoLegalPawnMovesInfo) -> Self {
        let single_step_blockers = info.pieces << CompassRose::Sout;
        let double_step_blockers = single_step_blockers | info.pieces << (CompassRose::Sout * 2);
        let double_step_pawns = info.pawns & masks::RANKS[1];
        let to = (double_step_pawns & !double_step_blockers) << (CompassRose::Nort * 2);
        let from = to << (CompassRose::Sout * 2);
        Self {
            from,
            to,
            stage: PhantomData,
            flag: MoveFlag::DoublePawnPush,
        }
    }
}

// todo: promotion captures
 
impl PseudoLegalPawnMoves<EnPassant<West>> {
    /// todo: what happens when u64 << 64? (case when ep square is NONE)
    pub fn new(info: &PseudoLegalPawnMovesInfo) -> Self {
        let to = Bitboard::from(info.pos.get_ep_square());
        let capture_west_pawns = info.pawns & !masks::FILES[0];
        let from = ((capture_west_pawns << CompassRose::NoWe) & to) << CompassRose::SoEa;
        Self {
            from,
            to,
            stage: PhantomData,
            flag: MoveFlag::EnPassant,
        }
    }
} 

impl PseudoLegalPawnMoves<EnPassant<East>> {
    /// todo: what happens when u64 << 64? (case when ep square is NONE)
    pub fn new(info: &PseudoLegalPawnMovesInfo) -> Self {
        let to = Bitboard::from(info.pos.get_ep_square());
        let capture_east_pawns = info.pawns & !masks::FILES[7];
        let from = ((capture_east_pawns << CompassRose::NoEa) & to) << CompassRose::SoWe; 
        Self {
            from,
            to,
            stage: PhantomData,
            flag: MoveFlag::EnPassant,
        }
    }
}

impl<T> Iterator for PseudoLegalPawnMoves<T> {
    type Item = Move;
    fn next(&mut self) -> Option<Self::Item> {
        const NONE: u8 = Squares::None as u8;
        match self.to.pop_lsb() {
            Square { v: NONE } => None,
            sq => Some(Move::new(self.from.pop_lsb(), sq, self.flag)),
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

pub fn generate_plegals_white<'a>(position: &'a Position) -> impl Iterator<Item = Move> + 'a {
    let info = PseudoLegalPawnMovesInfo::new(position, Color::White);
    let single_step = PseudoLegalPawnMoves::<SingleStep>::new(&info);
    let double_step = PseudoLegalPawnMoves::<DoubleStep>::new(&info);
    let knight_promo = PseudoLegalPawnMoves::<Promotion<Knight>>::new(&info);
    let bishop_promo = PseudoLegalPawnMoves::<Promotion<Bishop>>::new(&info);
    let rook_promo = PseudoLegalPawnMoves::<Promotion<Rook>>::new(&info);
    let queen_promo = PseudoLegalPawnMoves::<Promotion<Queen>>::new(&info);
    let capture_west = PseudoLegalPawnMoves::<Capture<West>>::new(&info);
    let capture_east = PseudoLegalPawnMoves::<Capture<East>>::new(&info);
    let ep_west = PseudoLegalPawnMoves::<EnPassant<West>>::new(&info);
    let ep_east = PseudoLegalPawnMoves::<EnPassant<East>>::new(&info);

    // todo: tune the ordering
           single_step
    .chain(double_step)
    .chain(queen_promo)
    .chain(knight_promo)
    .chain(capture_west)
    .chain(capture_east)
    .chain(ep_west)
    .chain(ep_east)
    .chain(bishop_promo)
    .chain(rook_promo)
}
