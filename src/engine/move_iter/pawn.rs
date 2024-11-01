use crate::engine::coordinates::{Rank, Square, SQ_NONE};
use crate::engine::{
    bitboard::Bitboard,
    color::Color,
    coordinates::{CompassRose, File, Squares},
    masks,
    piece::PieceType,
    position::Position,
    r#move::{Move, MoveFlag},
};

pub mod move_type {
    pub struct Legals;
    pub struct PseudoLegals;
    pub struct Attacks;
    pub struct Resolves;
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

pub struct PseudoLegalPawnMoves {
    from: Bitboard,
    to: Bitboard,
    flag: MoveFlag,
}

impl PseudoLegalPawnMoves {
    pub fn w_new_single_step<'a>(info: &'a PseudoLegalPawnMovesInfo) -> Self {
        let non_promo_pawns = info.pawns & !masks::RANKS[6];
        let single_step_blocker = info.pieces << CompassRose::Sout;
        let to = (non_promo_pawns & !single_step_blocker) << CompassRose::Nort;
        let from = to << CompassRose::Sout;
        Self {
            from,
            to,
            flag: MoveFlag::QUIET,
        }
    }
    pub fn w_new_capture_west(info: &PseudoLegalPawnMovesInfo) -> Self {
        let non_promo_pawns = info.pawns & !Bitboard::from(Rank::_7);
        let capture_west_pawns = non_promo_pawns & !Bitboard::from(File::A);
        let to = (capture_west_pawns << CompassRose::NoWe) & info.enemies;
        let from = to << CompassRose::SoEa;
        Self {
            from,
            to,
            flag: MoveFlag::CAPTURE,
        }
    }
    pub fn w_new_capture_east(info: &PseudoLegalPawnMovesInfo) -> Self {
        let non_promo_pawns = info.pawns & !Bitboard::from(Rank::_7);
        let capture_east_pawns = non_promo_pawns & !Bitboard::from(File::H);
        let to = (capture_east_pawns << CompassRose::NoEa) & info.enemies;
        let from = to << CompassRose::SoWe;
        Self {
            from,
            to,
            flag: MoveFlag::CAPTURE,
        }
    }
    pub fn w_new_double_step(info: &PseudoLegalPawnMovesInfo) -> Self {
        let single_step_blockers = info.pieces << CompassRose::Sout;
        let double_step_blockers = single_step_blockers | info.pieces << (CompassRose::Sout * 2);
        let double_step_pawns = info.pawns & masks::RANKS[1];
        let to = (double_step_pawns & !double_step_blockers) << (CompassRose::Nort * 2);
        let from = to << (CompassRose::Sout * 2);
        Self {
            from,
            to,
            flag: MoveFlag::DOUBLE_PAWN_PUSH,
        }
    }
    pub fn w_new_promo(info: &PseudoLegalPawnMovesInfo, flag: MoveFlag) -> Self {
        let single_step_blocker = info.pieces << CompassRose::Sout;
        let promo_pawns = info.pawns & masks::RANKS[6];
        let to = (promo_pawns & !single_step_blocker) << CompassRose::Nort;
        let from = to << CompassRose::Sout;
        Self {
            from,
            to,
            flag,
        }
    }
    pub fn w_new_promo_knight(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn w_new_promo_bishop(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn w_new_promo_rook(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn w_new_promo_queen(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo(info, MoveFlag::PROMOTION_QUEEN)
    }
    pub fn w_new_promo_capture_west(info: &PseudoLegalPawnMovesInfo, flag: MoveFlag) -> Self {
        let promo_pawns = info.pawns & masks::RANKS[6];
        let capture_west_pawns = promo_pawns & !masks::FILES[0];
        let to = (capture_west_pawns << CompassRose::NoWe) & info.enemies;
        let from = to << CompassRose::SoEa;
        Self {
            from,
            to,
            flag,
        }
    }
    pub fn w_new_promo_capture_west_knight(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo_capture_west(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn w_new_promo_capture_west_bishop(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo_capture_west(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn w_new_promo_capture_west_rook(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo_capture_west(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn w_new_promo_capture_west_queen(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo_capture_west(info, MoveFlag::PROMOTION_QUEEN)
    }
    pub fn w_new_promo_capture_east(info: &PseudoLegalPawnMovesInfo, flag: MoveFlag) -> Self {
        let promo_pawns = info.pawns & masks::RANKS[6];
        let capture_east_pawns = promo_pawns & !masks::FILES[7];
        let to = (capture_east_pawns << CompassRose::NoEa) & info.enemies;
        let from = to << CompassRose::SoWe;
        Self {
            from,
            to,
            flag,
        }
    }
    pub fn w_new_promo_capture_east_knight(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo_capture_east(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn w_new_promo_capture_east_bishop(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo_capture_east(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn w_new_promo_capture_east_rook(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo_capture_east(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn w_new_promo_capture_east_queen(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::w_new_promo_capture_east(info, MoveFlag::PROMOTION_QUEEN)
    }
    /// todo: what happens when u64 << 64? (case when ep square is NONE)
    pub fn w_new_ep_west(info: &PseudoLegalPawnMovesInfo) -> Self {
        let to = Bitboard::from(info.pos.get_ep_square());
        let capture_west_pawns = info.pawns & !masks::FILES[0];
        let from = ((capture_west_pawns << CompassRose::NoWe) & to) << CompassRose::SoEa;
        Self {
            from,
            to,
            flag: MoveFlag::EN_PASSANT,
        }
    }
    /// todo: what happens when u64 << 64? (case when ep square is NONE)
    pub fn w_new_ep_east(info: &PseudoLegalPawnMovesInfo) -> Self {
        let to = Bitboard::from(info.pos.get_ep_square());
        let capture_east_pawns = info.pawns & !masks::FILES[7];
        let from = ((capture_east_pawns << CompassRose::NoEa) & to) << CompassRose::SoWe;
        Self {
            from,
            to,
            flag: MoveFlag::EN_PASSANT,
        }
    }
}

impl Iterator for PseudoLegalPawnMoves {
    type Item = Move;
    fn next(&mut self) -> Option<Self::Item> {
        match self.to.pop_lsb() {
            Square::NONE => None,
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

pub fn gen_plegals_white<'a, F>(position: &'a Position) -> impl Iterator<Item = Move> + 'a
where
    F: FnMut(Move),
{
    let info = PseudoLegalPawnMovesInfo::new(position, Color::White);

    // todo: tune the ordering
    let moves: [fn(&PseudoLegalPawnMovesInfo) -> PseudoLegalPawnMoves; 18] = [
        PseudoLegalPawnMoves::w_new_single_step,
        PseudoLegalPawnMoves::w_new_double_step,
        PseudoLegalPawnMoves::w_new_promo_knight,
        PseudoLegalPawnMoves::w_new_promo_bishop,
        PseudoLegalPawnMoves::w_new_promo_rook,
        PseudoLegalPawnMoves::w_new_promo_queen,
        PseudoLegalPawnMoves::w_new_capture_west,
        PseudoLegalPawnMoves::w_new_capture_east,
        PseudoLegalPawnMoves::w_new_ep_west,
        PseudoLegalPawnMoves::w_new_ep_east,
        PseudoLegalPawnMoves::w_new_promo_capture_west_knight,
        PseudoLegalPawnMoves::w_new_promo_capture_east_knight,
        PseudoLegalPawnMoves::w_new_promo_capture_west_bishop,
        PseudoLegalPawnMoves::w_new_promo_capture_east_bishop,
        PseudoLegalPawnMoves::w_new_promo_capture_west_rook,
        PseudoLegalPawnMoves::w_new_promo_capture_east_rook,
        PseudoLegalPawnMoves::w_new_promo_capture_west_rook,
        PseudoLegalPawnMoves::w_new_promo_capture_east_rook,
    ];
    
    // todo: not sure this works
    moves.into_iter().map(move |f| f(&info)).flatten()
}