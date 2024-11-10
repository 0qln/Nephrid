use crate::engine::color::{Color, TColor};
use crate::engine::coordinates::{Rank, TCompassRose};
use crate::engine::{
    bitboard::Bitboard,
    coordinates::{CompassRose, File},
    piece::PieceType,
    position::Position,
    r#move::{Move, MoveFlag},
};
use crate::misc::ConstFrom;

pub mod move_type {
    pub struct Legals;
    pub struct PseudoLegals;
    pub struct Attacks;
    pub struct Resolves;
}

macro_rules! cg_usage { 
    ($expr: expr) => { 
        [(); {$expr} as usize] 
    }; 
}

const fn promo_rank<const C: TColor>() -> Rank {
    if C { Rank::_7 } else { Rank::_2 }
}

const fn start_rank<const C: TColor>() -> Rank {
    if C { Rank::_2 } else { Rank::_7 }
}

const fn single_step<const C: TColor>() -> TCompassRose {
    if C { CompassRose::NORT.v() } else { CompassRose::SOUT.v() }
}

const fn double_step<const C: TColor>() -> TCompassRose {
    single_step::<C>() * 2
}

const fn step_forward<const Dir: TCompassRose>(bb: Bitboard) -> Bitboard {
    bb.shift_c::<Dir>()
}

const fn step_backward<const Dir: TCompassRose>(bb: Bitboard) -> Bitboard
where 
    cg_usage!(-Dir):
{
    bb.shift_c::<{-Dir}>()
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
        let color = Color::new(C);
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

pub struct PseudoLegalPawnMoves<const C: TColor> 
// where 
//     cg_usage!(single_step::<C>()):,
//     cg_usage!(-single_step::<C>()):
{
    from: Bitboard,
    to: Bitboard,
    flag: MoveFlag,
}

impl<const C: TColor> PseudoLegalPawnMoves<C>
where 
    cg_usage!(single_step::<C>()):,
    cg_usage!(-single_step::<C>()):
{

    pub fn new_single_step<'a>(info: &'a PseudoLegalPawnMovesInfo<C>) -> Self 
    {
        let non_promo_pawns = info.pawns & !Bitboard::from_c(promo_rank::<C>());
        let single_step_blocker = step_backward::<{single_step::<C>()}>(info.pieces);
        let single_step_pawns = non_promo_pawns & !single_step_blocker;
        let to = step_forward::<{single_step::<C>()}>(single_step_pawns);
        let from = step_backward::<{single_step::<C>()}>(to);
        Self {
            from,
            to,
            flag: MoveFlag::QUIET,
        }
    }

    pub fn new_capture_west(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        const WEST: TCompassRose = CompassRose::WEST.v();
        Self::new_capture::<WEST>(info)
    }

    pub fn new_capture_east(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        const EAST: TCompassRose = CompassRose::EAST.v();
        Self::new_capture::<EAST>(info)
    }

    fn new_capture<const Dir: TCompassRose>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        let non_promo_pawns = info.pawns & !Bitboard::from_c(promo_rank::<C>());
        let capture_dir = Dir + single_step::<C>();
        let capturing_pawns = non_promo_pawns & !Bitboard::from_c(File::edge::<Dir>());
        let to = (capturing_pawns << capture_dir) & info.enemies;
        let from = to >> capture_dir;
        Self {
            from,
            to,
            flag: MoveFlag::CAPTURE,
        }
    }

    pub fn new_double_step(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        let single_step_blockers = info.pieces << single_step::<C>();
        let double_step_blockers = single_step_blockers | info.pieces << double_step::<C>();
        let double_step_pawns = info.pawns & Bitboard::from_c(start_rank::<C>());
        let to = (double_step_pawns & !double_step_blockers) << double_step::<C>();
        let from = to >> double_step::<C>();
        Self {
            from,
            to,
            flag: MoveFlag::DOUBLE_PAWN_PUSH,
        }
    }
    fn new_promo(info: &PseudoLegalPawnMovesInfo<C>, flag: MoveFlag) -> Self {
        let single_step_blocker = info.pieces << single_step::<C>();
        let promo_pawns = info.pawns & Bitboard::from_c(promo_rank::<C>());
        let to = (promo_pawns & !single_step_blocker) << single_step::<C>();
        let from = to >> single_step::<C>();
        Self {
            from,
            to,
            flag,
        }
    }
    pub fn new_promo_knight(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn new_promo_bishop(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn new_promo_rook(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn new_promo_queen(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo(info, MoveFlag::PROMOTION_QUEEN)
    }

    fn new_promo_capture<const Dir: TCompassRose>(info: &PseudoLegalPawnMovesInfo<C>, flag: MoveFlag) -> Self {
        let promo_pawns = info.pawns & Bitboard::from_c(promo_rank::<C>());
        let capture_west_pawns = promo_pawns & !Bitboard::from_c(File::edge::<Dir>());
        let capture_dir = Dir + single_step::<C>();
        let to = (capture_west_pawns << capture_dir) & info.enemies;
        let from = to >> capture_dir;
        Self {
            from,
            to,
            flag,
        }
    }
    fn new_promo_capture_west(info: &PseudoLegalPawnMovesInfo<C>, flag: MoveFlag) -> Self {
        const WEST: TCompassRose = CompassRose::WEST.v();
        Self::new_promo_capture::<WEST>(info, flag)
    }
    fn new_promo_capture_east(info: &PseudoLegalPawnMovesInfo<C>, flag: MoveFlag) -> Self {
        const EAST: TCompassRose = CompassRose::EAST.v();
        Self::new_promo_capture::<EAST>(info, flag)
    }

    pub fn new_promo_capture_west_knight(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_west(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn new_promo_capture_west_bishop(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_west(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn new_promo_capture_west_rook(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_west(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn new_promo_capture_west_queen(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_west(info, MoveFlag::PROMOTION_QUEEN)
    }
    pub fn w_new_promo_capture_east_knight(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_east(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn w_new_promo_capture_east_bishop(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_east(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn w_new_promo_capture_east_rook(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_east(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn w_new_promo_capture_east_queen(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        Self::new_promo_capture_east(info, MoveFlag::PROMOTION_QUEEN)
    }

    pub fn new_ep_west(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        const WEST: TCompassRose = CompassRose::WEST.v();
        Self::new_ep::<WEST>(info)
    }
    pub fn new_ep_east(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        const EAST: TCompassRose = CompassRose::EAST.v();
        Self::new_ep::<EAST>(info)
    }

    fn new_ep<const Dir: TCompassRose>(info: &PseudoLegalPawnMovesInfo<C>) -> Self {
        let to = Bitboard::from_c(info.pos.get_ep_square());
        let capturing_pawns = info.pawns & !Bitboard::from_c(File::edge::<Dir>());
        let capture_dir = Dir + single_step::<C>();
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

impl<const C: TColor> Iterator for PseudoLegalPawnMoves<C> {
    type Item = Move;
    fn next(&mut self) -> Option<Self::Item> {
        self.to.pop_lsb().map(|sq| unsafe {
            // todo: write unit tests for this.
            // Safety: the 'from' bb is generated by every constructor in such a way, 
            // that there is always atleast one square in the 'from' bb per square 
            // in the 'to' bb
            Move::new(self.from.pop_lsb().unwrap_unchecked(), sq, self.flag)
        })
    }
}

pub fn gen_pseudo_legals<'a, F, const C: TColor>(position: &'a Position) -> impl Iterator<Item = Move> + 'a
where
    cg_usage!(single_step::<C>()):,
    cg_usage!(-single_step::<C>()):,
    F: FnMut(Move), 
{
    let info = PseudoLegalPawnMovesInfo::<C>::new(position);

    // todo: tune the ordering
    let moves: [fn(&PseudoLegalPawnMovesInfo<C>) -> PseudoLegalPawnMoves::<C>; 18] = [
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