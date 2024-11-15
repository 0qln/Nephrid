use crate::engine::color::{Color, TColor};
use crate::engine::coordinates::{Rank, Square, TCompassRose};
use crate::engine::{
    bitboard::Bitboard,
    coordinates::{CompassRose, File},
    piece::PieceType,
    position::Position,
    r#move::{Move, MoveFlag},
};
use crate::misc::ConstFrom;


pub struct PseudoLegalPawnMovesInfo {
    pawns: Bitboard,
    enemies: Bitboard,
    pieces: Bitboard,
    ep_sq: Option<Square>
}

// todo: theres still a lot of duplicate calculations in the
// iterator initiations. benchmark, wether it's worth to
// cache the results here or not.
impl PseudoLegalPawnMovesInfo {
    pub fn new<const C: TColor>(pos: &Position) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let pieces = pos.get_occupancy();
        let enemies = pos.get_color_bb(!color);
        let ep_sq = pos.get_ep_square();
        Self {
            pawns,
            pieces,
            enemies,
            ep_sq,
        }
    }
}

#[inline]
const fn promo_rank(c: Color) -> Rank {
    match c {
        Color::WHITE => Rank::_7,
        Color::BLACK => Rank::_2,
        _ => unreachable!()
    }
}

#[inline]
const fn start_rank(c: Color) -> Rank {
    match c {
        Color::WHITE => Rank::_2,
        Color::BLACK => Rank::_7,
        _ => unreachable!()
    }
}

#[inline]
const fn single_step(c: Color) -> CompassRose {
    match c {
        Color::WHITE => CompassRose::NORT,
        Color::BLACK => CompassRose::SOUT,
        _ => unreachable!()
    }
}

#[inline]
const fn double_step(c: Color) -> CompassRose {
    single_step(c).double()
}

#[inline]
const fn forward(bb: Bitboard, dir: CompassRose) -> Bitboard {
    bb.shift(dir)
}

#[inline]
const fn backward(bb: Bitboard, dir: CompassRose) -> Bitboard {
    bb.shift(dir.neg())
}

#[inline]
const fn capture(c: Color, dir: CompassRose) -> CompassRose {
    CompassRose::new(dir.v() + single_step(c).v())
}

pub struct PseudoLegalPawnMoves {
    from: Bitboard,
    to: Bitboard,
    flag: MoveFlag,
}

impl PseudoLegalPawnMoves {
    pub fn new_single_step<'a, const C: TColor>(info: &'a PseudoLegalPawnMovesInfo) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let non_promo_pawns = info.pawns & !Bitboard::from_c(promo_rank(color));
        let single_step_blocker = backward(info.pieces, single_step(color));
        let from = non_promo_pawns & !single_step_blocker;
        let to = forward(from, single_step(color));
        Self { from, to, flag: MoveFlag::QUIET, }
    }

    pub fn new_capture<const C: TColor, const DIR: TCompassRose>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let dir = CompassRose::new(DIR);
        let capture_dir = capture(color, dir);
        let non_promo_pawns = info.pawns & !Bitboard::from_c(promo_rank(color));
        let capturing_pawns = non_promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capturing_pawns, capture_dir) & info.enemies;
        let from = backward(to, capture_dir);
        Self { from, to, flag: MoveFlag::CAPTURE, }
    }

    pub fn new_double_step<const C: TColor>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let single_step_blockers = forward(info.pieces, single_step(color));
        let double_step_blockers = single_step_blockers | forward(info.pieces, double_step(color));
        let double_step_pawns = info.pawns & Bitboard::from_c(start_rank(color));
        let from = double_step_pawns & !double_step_blockers;
        let to = forward(from, double_step(color));
        Self { from, to, flag: MoveFlag::DOUBLE_PAWN_PUSH, }
    }

    fn new_promo<const C: TColor>(info: &PseudoLegalPawnMovesInfo, flag: MoveFlag) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let single_step_blocker = forward(info.pieces, single_step(color));
        let promo_pawns = info.pawns & Bitboard::from_c(promo_rank(color));
        let from = promo_pawns & !single_step_blocker;
        let to = forward(from, single_step(color));
        Self { from, to, flag }
    }
    
    // todo: when const generics are stabalized with constraints,
    // replace the MoveFlag parameters with a const generic.

    pub fn new_promo_knight<const C: TColor>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::new_promo::<C>(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn new_promo_bishop<const C: TColor>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::new_promo::<C>(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn new_promo_rook<const C: TColor>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::new_promo::<C>(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn new_promo_queen<const C: TColor>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::new_promo::<C>(info, MoveFlag::PROMOTION_QUEEN)
    }

    fn new_promo_capture<const C: TColor, const DIR: TCompassRose>(
        info: &PseudoLegalPawnMovesInfo,
        flag: MoveFlag,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let dir = CompassRose::new(DIR);
        let capture_dir = capture(color, dir);
        let promo_pawns = info.pawns & Bitboard::from_c(promo_rank(color));
        let capture_west_pawns = promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capture_west_pawns, capture_dir) & info.enemies;
        let from = backward(to, capture_dir);
        Self { from, to, flag }
    }
    pub fn new_promo_capture_knight<const C: TColor, const DIR: TCompassRose>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::new_promo_capture::<C, DIR>(info, MoveFlag::PROMOTION_KNIGHT)
    }
    pub fn new_promo_capture_bishop<const C: TColor, const DIR: TCompassRose>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::new_promo_capture::<C, DIR>(info, MoveFlag::PROMOTION_BISHOP)
    }
    pub fn new_promo_capture_rook<const C: TColor, const DIR: TCompassRose>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::new_promo_capture::<C, DIR>(info, MoveFlag::PROMOTION_ROOK)
    }
    pub fn new_promo_capture_queen<const C: TColor, const DIR: TCompassRose>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Self::new_promo_capture::<C, DIR>(info, MoveFlag::PROMOTION_QUEEN)
    }

    fn new_ep<const C: TColor, const DIR: TCompassRose>(info: &PseudoLegalPawnMovesInfo) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let dir = CompassRose::new(DIR);
        let capture_dir = capture(color, dir);
        let mut to = Bitboard::from_c(info.ep_sq);
        let capturing_pawns = info.pawns & !Bitboard::from_c(File::edge::<DIR>());
        let from = match to.is_empty() {
            true => {
                to = Bitboard::empty();
                Bitboard::empty()               
            },
            false => {
                backward(
                    forward(capturing_pawns, capture_dir) & to, 
                    capture_dir
                )
            }
        };
        Self {
            from,
            to,
            flag: MoveFlag::EN_PASSANT,
        }
    }
}

impl Iterator for PseudoLegalPawnMoves {
    type Item = Move;

    #[inline]
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

pub fn gen_pseudo_legals<const C: TColor>(position: &Position) -> impl Iterator<Item = Move> {
    let info = PseudoLegalPawnMovesInfo::new::<C>(position);

    // todo: tune the ordering
    let moves: [fn(&PseudoLegalPawnMovesInfo) -> PseudoLegalPawnMoves; 18] = [
        PseudoLegalPawnMoves::new_single_step::<C>,
        PseudoLegalPawnMoves::new_double_step::<C>,
        PseudoLegalPawnMoves::new_promo_knight::<C>,
        PseudoLegalPawnMoves::new_promo_bishop::<C>,
        PseudoLegalPawnMoves::new_promo_rook::<C>,
        PseudoLegalPawnMoves::new_promo_queen::<C>,
        PseudoLegalPawnMoves::new_capture::<C, {CompassRose::WEST.v()}>,
        PseudoLegalPawnMoves::new_capture::<C, {CompassRose::EAST.v()}>,
        PseudoLegalPawnMoves::new_ep::<C, {CompassRose::WEST.v()}>,
        PseudoLegalPawnMoves::new_ep::<C, {CompassRose::EAST.v()}>,
        PseudoLegalPawnMoves::new_promo_capture_knight::<C, {CompassRose::WEST.v()}>,
        PseudoLegalPawnMoves::new_promo_capture_knight::<C, {CompassRose::EAST.v()}>,
        PseudoLegalPawnMoves::new_promo_capture_bishop::<C, {CompassRose::WEST.v()}>,
        PseudoLegalPawnMoves::new_promo_capture_bishop::<C, {CompassRose::EAST.v()}>,
        PseudoLegalPawnMoves::new_promo_capture_rook::<C, {CompassRose::WEST.v()}>,
        PseudoLegalPawnMoves::new_promo_capture_rook::<C, {CompassRose::EAST.v()}>,
        PseudoLegalPawnMoves::new_promo_capture_queen::<C, {CompassRose::WEST.v()}>,
        PseudoLegalPawnMoves::new_promo_capture_queen::<C, {CompassRose::EAST.v()}>,
    ];

    // todo: somehow make sure the chaining is optimized away... if not do it manually.
    // todo: not sure this works
    moves.into_iter().map(move |f| f(&info)).flatten()
}
