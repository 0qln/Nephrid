use crate::engine::coordinates::{Rank, Square};
use crate::engine::{
    bitboard::Bitboard,
    color::Color,
    coordinates::{CompassRose, File},
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

// true = white
// false = black
pub type TColor = bool;

const fn color(c: TColor) -> Color {
   if c { Color::White } else { Color::Black } 
}

const fn promo_rank(c: TColor) -> Rank {
    if c { Rank::_7 } else { Rank::_2 }
}

const fn start_rank(c: TColor) -> Rank {
    if c { Rank::_2 } else { Rank::_7 }
}

const fn single_step(c: TColor) -> CompassRose {
    if c { CompassRose::NORT } else { CompassRose::SOUT }
}

const fn double_step(c: TColor) -> CompassRose {
    CompassRose { v: single_step(c).v * 2 }
}


type TCompassRose = isize;

const fn compass_rose(dir: TCompassRose) -> CompassRose {
    CompassRose { v: dir }
}

const fn edge_file(dir: TCompassRose) -> File {
    if dir == CompassRose::WEST.v { File::A } else { File::H }
}

pub struct PseudoLegalPawnMovesInfo<'a, const C: TColor> {
    pos: &'a Position,
    pawns: Bitboard,
    enemies: Bitboard,
    pieces: Bitboard,
}

// todo: theres still a lot of duplicate calculations in the 
// iterator initiations. benchmark, wether it's worth to 
// cache the results here or not.
impl<'a, const C: TColor> PseudoLegalPawnMovesInfo<'a, C> {
    pub fn new(pos: &'a Position) -> Self {
        let color = color(C);
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

    pub fn new_single_step<'a, const C: TColor>(info: &'a PseudoLegalPawnMovesInfo<C>) -> Self {
        let non_promo_pawns = info.pawns & !Bitboard::from(promo_rank(C));
        let single_step_blocker = info.pieces >> single_step(C);
        let to = (non_promo_pawns & !single_step_blocker) << single_step(C);
        let from = to >> single_step(C);
        Self {
            from,
            to,
            flag: MoveFlag::QUIET,
        }
    }

    pub fn new_capture_west<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        const WEST: TCompassRose = CompassRose::WEST.v;
        Self::new_capture::<C, WEST>(info)
    }

    pub fn new_capture_east<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        const EAST: TCompassRose = CompassRose::EAST.v;
        Self::new_capture::<C, EAST>(info)
    }

    fn new_capture<const C: TColor, const Dir: TCompassRose>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        let non_promo_pawns = info.pawns & !Bitboard::from(promo_rank(C));
        let capture_dir = Dir + single_step(C);
        let capturing_pawns = non_promo_pawns & !Bitboard::from(edge_file(Dir));
        let to = (capturing_pawns << capture_dir) & info.enemies;
        let from = to >> capture_dir;
        Self {
            from,
            to,
            flag: MoveFlag::CAPTURE,
        }
    }

    pub fn new_double_step<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        let single_step_blockers = info.pieces << single_step(C);
        let double_step_blockers = single_step_blockers | info.pieces << double_step(C);
        let double_step_pawns = info.pawns & Bitboard::from(start_rank(C));
        let to = (double_step_pawns & !double_step_blockers) << double_step(C);
        let from = to >> double_step(C);
        Self {
            from,
            to,
            flag: MoveFlag::DOUBLE_PAWN_PUSH,
        }
    }
    fn new_promo<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>, flag: MoveFlag) -> Self {
        let single_step_blocker = info.pieces << single_step(C);
        let promo_pawns = info.pawns & Bitboard::from(promo_rank(C));
        let to = (promo_pawns & !single_step_blocker) << single_step(C);
        let from = to >> single_step(C);
        Self {
            from,
            to,
            flag,
        }
    }
    pub fn new_promo_knight<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn new_promo_bishop<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn new_promo_rook<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn new_promo_queen<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo(info, MoveFlag::PROMOTION_QUEEN)
    }

    fn new_promo_capture<const C: TColor, const Dir: TCompassRose>(info: &PseudoLegalPawnMovesInfo<C>, flag: MoveFlag) -> Self {
        let promo_pawns = info.pawns & Bitboard::from(promo_rank(C));
        let capture_west_pawns = promo_pawns & !Bitboard::from(edge_file(Dir));
        let capture_dir = Dir + single_step(C);
        let to = (capture_west_pawns << capture_dir) & info.enemies;
        let from = to >> capture_dir;
        Self {
            from,
            to,
            flag,
        }
    }
    fn new_promo_capture_west<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>, flag: MoveFlag) -> Self {
        const WEST: TCompassRose = CompassRose::WEST.v;
        Self::new_promo_capture::<C, WEST>(info, flag)
    }
    fn new_promo_capture_east<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>, flag: MoveFlag) -> Self {
        const EAST: TCompassRose = CompassRose::EAST.v;
        Self::new_promo_capture::<C, EAST>(info, flag)
    }

    pub fn new_promo_capture_west_knight<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_west(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn new_promo_capture_west_bishop<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_west(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn new_promo_capture_west_rook<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_west(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn new_promo_capture_west_queen<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_west(info, MoveFlag::PROMOTION_QUEEN)
    }
    pub fn w_new_promo_capture_east_knight<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_east(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn w_new_promo_capture_east_bishop<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_east(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn w_new_promo_capture_east_rook<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_east(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn w_new_promo_capture_east_queen<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_east(info, MoveFlag::PROMOTION_QUEEN)
    }

    pub fn new_ep_west<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        const WEST: TCompassRose = CompassRose::WEST.v;
        Self::new_ep::<C, WEST>(info)
    }
    pub fn new_ep_east<const C: TColor>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        const EAST: TCompassRose = CompassRose::EAST.v;
        Self::new_ep::<C, EAST>(info)
    }
    // todo: what happens when u64 << 64? (case when ep square is NONE)
    fn new_ep<const C: TColor, const Dir: TCompassRose>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        let to = Bitboard::from(info.pos.get_ep_square());
        let capturing_pawns = info.pawns & !Bitboard::from(edge_file(Dir));
        let capture_dir = Dir + single_step(C);
        let from = ((capturing_pawns << capture_dir) & to) >> capture_dir;
        // todo: 
        // to is not properly cleaned yet. the iterator will yield 
        // using the to-square with no from-square
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
// pub fn white_pawn_attacks(pawn: Bitboard) -> Bitboard {
//     let mut result = Bitboard { v: 0 };
//     result |= (pawn & !Bitboard::from(File::A)) << CompassRose::West;
//     result |= (pawn & !Bitboard::from(File::H)) << CompassRose::East;
//     result
// }

pub fn gen_pseudo_legals<'a, F, const C: TColor>(position: &'a Position) -> impl Iterator<Item = Move> + 'a
where
    F: FnMut(Move),
{
    let info = PseudoLegalPawnMovesInfo::<C>::new(position);

    // todo: tune the ordering
    let moves: [fn(&PseudoLegalPawnMovesInfo<C>) -> PseudoLegalPawnMoves; 18] = [
        PseudoLegalPawnMoves::new_single_step,
        PseudoLegalPawnMoves::new_double_step,
        PseudoLegalPawnMoves::new_promo_knight,
        PseudoLegalPawnMoves::new_promo_bishop,
        PseudoLegalPawnMoves::new_promo_rook,
        PseudoLegalPawnMoves::new_promo_queen,
        PseudoLegalPawnMoves::new_capture_west,
        PseudoLegalPawnMoves::new_capture_east,
        PseudoLegalPawnMoves::new_ep_west,
        PseudoLegalPawnMoves::new_ep_east,
        PseudoLegalPawnMoves::new_promo_capture_west_knight,
        PseudoLegalPawnMoves::w_new_promo_capture_east_knight,
        PseudoLegalPawnMoves::new_promo_capture_west_bishop,
        PseudoLegalPawnMoves::w_new_promo_capture_east_bishop,
        PseudoLegalPawnMoves::new_promo_capture_west_rook,
        PseudoLegalPawnMoves::w_new_promo_capture_east_rook,
        PseudoLegalPawnMoves::new_promo_capture_west_rook,
        PseudoLegalPawnMoves::w_new_promo_capture_east_rook,
    ];
    
    // todo: somehow make sure the chaining is optimized away... if not do it manually.
    // todo: not sure this works
    moves.into_iter().map(move |f| f(&info)).flatten()
}