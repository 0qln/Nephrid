use std::ops::Try;

use crate::engine::color::{Color, TColor};
use crate::engine::coordinates::{EpCaptureSquare, EpTargetSquare, Rank, Square, TCompassRose};
use crate::engine::position::CheckState;
use crate::engine::{
    bitboard::Bitboard,
    coordinates::{CompassRose, File},
    piece::PieceType,
    position::Position,
    r#move::{Move, MoveFlag},
};
use crate::misc::ConstFrom;
use const_for::const_for;

use super::bishop::Bishop;
use super::rook::Rook;
use super::sliding_piece::Attacks;

pub struct PawnMovesInfo<'a> {
    pos: &'a Position,
    pieces: Bitboard,
}

// todo: theres still a lot of duplicate calculations in the
// iterator initiations. benchmark, whether it's worth to
// cache the results here or not.
impl<'a> PawnMovesInfo<'a> {
    pub fn new(pos: &'a Position) -> Self {
        let pieces = pos.get_occupancy();
        Self { pos, pieces }
    }

    pub fn get_enemies(&self, color: Color) -> Bitboard {
        self.pos.get_color_bb(!color)
    }

    pub fn get_pawns(&self, color: Color) -> Bitboard {
        self.pos.get_bitboard(PieceType::PAWN, color)
    }

    pub fn get_ep_sq(&self) -> EpCaptureSquare {
        self.pos.get_ep_capture_square()
    }

    pub fn get_pos(&self) -> &Position {
        self.pos
    }
}

#[inline]
const fn promo_rank(c: Color) -> Rank {
    match c {
        Color::WHITE => Rank::_7,
        Color::BLACK => Rank::_2,
        _ => unreachable!(),
    }
}

#[inline]
const fn start_rank(c: Color) -> Rank {
    match c {
        Color::WHITE => Rank::_2,
        Color::BLACK => Rank::_7,
        _ => unreachable!(),
    }
}

#[inline]
const fn single_step(c: Color) -> CompassRose {
    match c {
        Color::WHITE => CompassRose::NORT,
        Color::BLACK => CompassRose::SOUT,
        _ => unreachable!(),
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

trait Legallity {}
trait Legal: Legallity {
    fn get_blockers(&self) -> Bitboard;
    fn get_king(&self) -> Square;
}

#[derive(Debug, Clone, Copy)]
struct PseudoLegal;
impl Legallity for PseudoLegal {}

trait CheckStateInfo {}
trait CheckStateInfoEmpty: CheckStateInfo {}
trait CheckStateInfoSome: CheckStateInfo {}

type IgnoreCheck = PseudoLegal;
impl CheckStateInfoEmpty for IgnoreCheck {}
impl CheckStateInfo for IgnoreCheck {}

#[derive(Debug, Clone, Copy)]
struct NoCheck<'a> {
    pos: &'a Position,
}
impl Legal for NoCheck<'_> {
    fn get_blockers(&self) -> Bitboard {
        self.pos.get_blockers()
    }

    fn get_king(&self) -> Square {
        let color = self.pos.get_turn();
        let king_bb = self.pos.get_bitboard(PieceType::KING, color);
        // Safety: the board has no king, board is in single check, and gen_legal is used??
        // the context is broken anyway.
        unsafe { king_bb.lsb().unwrap_unchecked() }
    }
}
impl Legallity for NoCheck<'_> {}
impl CheckStateInfo for NoCheck<'_> {}

impl<'a> NoCheck<'a> {
    pub fn new(pos: &'a Position) -> Self {
        Self { pos }
    }
}

#[derive(Debug, Clone, Copy)]
struct SingleCheck<'a> {
    pos: &'a Position,
}
impl Legal for SingleCheck<'_> {
    fn get_blockers(&self) -> Bitboard {
        self.pos.get_blockers()
    }

    fn get_king(&self) -> Square {
        let color = self.pos.get_turn();
        let king_bb = self.pos.get_bitboard(PieceType::KING, color);
        // Safety: the board has no king, board is in single check, and gen_legal is used??
        // the context is broken anyway.
        unsafe { king_bb.lsb().unwrap_unchecked() }
    }
}
impl Legallity for SingleCheck<'_> {}
impl CheckStateInfoSome for SingleCheck<'_> {}
impl CheckStateInfo for SingleCheck<'_> {}

impl<'a> SingleCheck<'a> {
    pub fn new(pos: &'a Position, color: Color) -> Self {
        assert_eq!(pos.get_check_state(), CheckState::Single);
        Self { pos }
    }

    fn get_blocks(&self) -> Bitboard {
        let king = self.get_king();
        // Safety: there is a single checker.
        let checker = unsafe { self.pos.get_checkers().lsb().unwrap_unchecked() };
        Bitboard::between(king, checker)
    }
}

pub struct PawnMoves<T> {
    from: Bitboard,
    to: Bitboard,
    flag: MoveFlag,
    t: T,
}

trait IPawnMoves<T>
where
    Self: Sized,
{
    fn new(from: Bitboard, to: Bitboard, flag: MoveFlag, t: T) -> Self;

    fn single_step<const C: TColor>(info: &PawnMovesInfo, t: T) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let non_promo_pawns = info.get_pawns(color) & !Bitboard::from_c(promo_rank(color));
        let single_step_tabus = backward(info.pieces, single_step(color));
        let from = non_promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, MoveFlag::QUIET, t)
    }

    fn double_step<const C: TColor>(info: &PawnMovesInfo, t: T) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let single_step_tabus = backward(info.pieces, single_step(color));
        let double_step_tabus = backward(info.pieces, double_step(color)) | single_step_tabus;
        let double_step_pawns = info.get_pawns(color) & Bitboard::from_c(start_rank(color));
        let from = double_step_pawns & !double_step_tabus;
        let to = forward(from, double_step(color));
        Self::new(from, to, MoveFlag::DOUBLE_PAWN_PUSH, t)
    }

    fn capture<const C: TColor, const DIR: TCompassRose>(info: &PawnMovesInfo, t: T) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let non_promo_pawns = info.get_pawns(color) & !Bitboard::from_c(promo_rank(color));
        let capturing_pawns = non_promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capturing_pawns, capture_dir) & info.get_enemies(color);
        let from = backward(to, capture_dir);
        Self::new(from, to, MoveFlag::CAPTURE, t)
    }

    fn promo<const C: TColor>(info: &PawnMovesInfo, flag: MoveFlag, t: T) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let single_step_tabus = backward(info.pieces, single_step(color));
        let promo_pawns = info.get_pawns(color) & Bitboard::from_c(promo_rank(color));
        let from = promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, flag, t)
    }

    fn promo_knight<const C: TColor>(info: &PawnMovesInfo, t: T) -> Self {
        Self::promo::<C>(info, MoveFlag::PROMOTION_KNIGHT, t)
    }

    fn promo_bishop<const C: TColor>(info: &PawnMovesInfo, t: T) -> Self {
        Self::promo::<C>(info, MoveFlag::PROMOTION_BISHOP, t)
    }

    fn promo_rook<const C: TColor>(info: &PawnMovesInfo, t: T) -> Self {
        Self::promo::<C>(info, MoveFlag::PROMOTION_ROOK, t)
    }

    fn promo_queen<const C: TColor>(info: &PawnMovesInfo, t: T) -> Self {
        Self::promo::<C>(info, MoveFlag::PROMOTION_QUEEN, t)
    }

    fn pl_promo_capture<const C: TColor, const DIR: TCompassRose>(
        info: &PawnMovesInfo,
        flag: MoveFlag,
        t: T,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let promo_pawns = info.get_pawns(color) & Bitboard::from_c(promo_rank(color));
        let capture_west_pawns = promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capture_west_pawns, capture_dir) & info.get_enemies(color);
        let from = backward(to, capture_dir);
        Self::new(from, to, flag, t)
    }

    fn promo_capture_knight<const C: TColor, const DIR: TCompassRose>(
        info: &PawnMovesInfo,
        t: T,
    ) -> Self {
        Self::pl_promo_capture::<C, DIR>(info, MoveFlag::CAPTURE_PROMOTION_KNIGHT, t)
    }

    fn promo_capture_bishop<const C: TColor, const DIR: TCompassRose>(
        info: &PawnMovesInfo,
        t: T,
    ) -> Self {
        Self::pl_promo_capture::<C, DIR>(info, MoveFlag::CAPTURE_PROMOTION_BISHOP, t)
    }

    fn promo_capture_rook<const C: TColor, const DIR: TCompassRose>(
        info: &PawnMovesInfo,
        t: T,
    ) -> Self {
        Self::pl_promo_capture::<C, DIR>(info, MoveFlag::CAPTURE_PROMOTION_ROOK, t)
    }

    fn promo_capture_queen<const C: TColor, const DIR: TCompassRose>(
        info: &PawnMovesInfo,
        t: T,
    ) -> Self {
        Self::pl_promo_capture::<C, DIR>(info, MoveFlag::CAPTURE_PROMOTION_QUEEN, t)
    }

    fn ep<const C: TColor, const DIR: TCompassRose>(info: &PawnMovesInfo, t: T) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let target = EpTargetSquare::from((info.get_ep_sq(), !color));
        let mut to = Bitboard::from_c(target.v());
        let from = if to.is_empty() {
            Bitboard::empty()
        } else {
            let capture_dir = capture(color, CompassRose::new(DIR));
            let capturing_pawns = info.get_pawns(color) & !Bitboard::from_c(File::edge::<DIR>());
            let from = backward(forward(capturing_pawns, capture_dir) & to, capture_dir);
            if from.is_empty() {
                to = Bitboard::empty();
            }
            from
        };
        Self::new(from, to, MoveFlag::EN_PASSANT, t)
    }
}

impl<'a> IPawnMoves<NoCheck<'a>> for PawnMoves<NoCheck<'a>> {
    fn new(from: Bitboard, to: Bitboard, flag: MoveFlag, t: NoCheck<'a>) -> Self {
        assert!(
            from.pop_cnt() >= to.pop_cnt(),
            "From needs to have atleast as many squares as to."
        );
        Self { from, to, flag, t }
    }

    fn ep<const C: TColor, const DIR: TCompassRose>(info: &PawnMovesInfo, t: NoCheck<'a>) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_sq = info.get_ep_sq();
        let target_sq = EpTargetSquare::from((capture_sq, !color));
        let mut to = Bitboard::from_c(target_sq.v());
        let from = if to.is_empty() {
            Bitboard::empty()
        } else {
            let capture_dir = capture(color, CompassRose::new(DIR));
            let capturing_pawns = info.get_pawns(color) & !Bitboard::from_c(File::edge::<DIR>());
            let from = backward(forward(capturing_pawns, capture_dir) & to, capture_dir);
            if from.is_empty() {
                to = Bitboard::empty();
                Bitboard::empty()
            } else {
                // Check that the king is not in check after the capture happens.
                let occupancy = info.get_pos().get_occupancy();
                let king_sq = t.get_king();
                let capt_bb = Bitboard::from_c(capture_sq.v());
                let occupancy_after_capture = (occupancy ^ from ^ capt_bb) | to;
                let rooks = info.get_pos().get_bitboard(PieceType::ROOK, !color);
                let bishops = info.get_pos().get_bitboard(PieceType::BISHOP, !color);
                let queens = info.get_pos().get_bitboard(PieceType::QUEEN, !color);
                let rook_attacks = Rook::lookup_attacks(king_sq, occupancy_after_capture);
                let bishop_attacks = Bishop::lookup_attacks(king_sq, occupancy_after_capture);
                let q_or_r_check = !rook_attacks.and_c(rooks | queens).is_empty();
                let q_or_b_check = !bishop_attacks.and_c(bishops | queens).is_empty();
                let check = q_or_r_check || q_or_b_check;
                if check {
                    to = Bitboard::empty();
                    Bitboard::empty()
                } else {
                    from
                }
            }
        };
        Self::new(from, to, MoveFlag::EN_PASSANT, t)
    }
}

impl<'a> IPawnMoves<SingleCheck<'a>> for PawnMoves<SingleCheck<'a>> {
    fn new(from: Bitboard, to: Bitboard, flag: MoveFlag, t: SingleCheck<'a>) -> Self {
        assert!(
            from.pop_cnt() >= to.pop_cnt(),
            "From needs to have atleast as many squares as to."
        );
        Self { from, to, flag, t }
    }

    fn single_step<const C: TColor>(info: &PawnMovesInfo, t: SingleCheck<'a>) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let non_promo_pawns = info.get_pawns(color) & !Bitboard::from_c(promo_rank(color));
        let tabu_squares = !t.get_blocks() | info.pieces;
        let single_step_tabus = backward(tabu_squares, single_step(color));
        let from = non_promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, MoveFlag::QUIET, t)
    }

    fn double_step<const C: TColor>(info: &PawnMovesInfo, t: SingleCheck<'a>) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let tabu_squares = info.pieces | !t.get_blocks();
        let single_step_tabus = backward(info.pieces, single_step(color));
        let double_step_tabus = backward(tabu_squares, double_step(color)) | single_step_tabus;
        let double_step_pawns = info.get_pawns(color) & Bitboard::from_c(start_rank(color));
        let from = double_step_pawns & !double_step_tabus;
        let to = forward(from, double_step(color));
        Self::new(from, to, MoveFlag::DOUBLE_PAWN_PUSH, t)
    }

    fn capture<const C: TColor, const DIR: TCompassRose>(
        info: &PawnMovesInfo,
        t: SingleCheck<'a>,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let non_promo_pawns = info.get_pawns(color) & !Bitboard::from_c(promo_rank(color));
        let capturing_pawns = non_promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capturing_pawns, capture_dir) & t.pos.get_checkers();
        let from = backward(to, capture_dir);
        Self::new(from, to, MoveFlag::CAPTURE, t)
    }

    fn promo<const C: TColor>(info: &PawnMovesInfo, flag: MoveFlag, t: SingleCheck<'a>) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let tabu_squares = info.pieces | !t.get_blocks();
        let single_step_tabus = backward(tabu_squares, single_step(color));
        let promo_pawns = info.get_pawns(color) & Bitboard::from_c(promo_rank(color));
        let from = promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, flag, t)
    }

    fn pl_promo_capture<const C: TColor, const DIR: TCompassRose>(
        info: &PawnMovesInfo,
        flag: MoveFlag,
        t: SingleCheck<'a>,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let promo_pawns = info.get_pawns(color) & Bitboard::from_c(promo_rank(color));
        let capture_west_pawns = promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capture_west_pawns, capture_dir) & t.pos.get_checkers();
        let from = backward(to, capture_dir);
        Self::new(from, to, flag, t)
    }

    fn ep<const C: TColor, const DIR: TCompassRose>(
        info: &PawnMovesInfo,
        t: SingleCheck<'a>,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_sq = info.get_ep_sq();
        let target = EpTargetSquare::from((capture_sq, !color));
        let mut to = Bitboard::from_c(target.v());
        let from = if to.is_empty() {
            Bitboard::empty()
        } else {
            let capture_dir = capture(color, CompassRose::new(DIR));
            let capturing_pawns = info.get_pawns(color) & !Bitboard::from_c(File::edge::<DIR>());
            let from = backward(forward(capturing_pawns, capture_dir) & to, capture_dir);
            if from.is_empty() {
                to = Bitboard::empty();
                Bitboard::empty()
            } else {
                // Check that the king is not in check after the capture happens.
                let occupancy = info.get_pos().get_occupancy();
                let king_sq = t.get_king();
                let capt_bb = Bitboard::from_c(capture_sq.v());
                let occupancy_after_capture = (occupancy ^ from ^ capt_bb) | to;
                let rooks = info.get_pos().get_bitboard(PieceType::ROOK, !color);
                let bishops = info.get_pos().get_bitboard(PieceType::BISHOP, !color);
                let queens = info.get_pos().get_bitboard(PieceType::QUEEN, !color);
                let rook_attacks = Rook::lookup_attacks(king_sq, occupancy_after_capture);
                let bishop_attacks = Bishop::lookup_attacks(king_sq, occupancy_after_capture);
                let q_or_r_check = !rook_attacks.and_c(rooks | queens).is_empty();
                let q_or_b_check = !bishop_attacks.and_c(bishops | queens).is_empty();
                let check = q_or_r_check || q_or_b_check;
                if check {
                    to = Bitboard::empty();
                    Bitboard::empty()
                } else {
                    from
                }
            }
        };
        Self::new(from, to, MoveFlag::EN_PASSANT, t)
    }
}

impl<T: CheckStateInfoEmpty> IPawnMoves<T> for PawnMoves<T> {
    fn new(from: Bitboard, to: Bitboard, flag: MoveFlag, t: T) -> Self {
        assert!(
            from.pop_cnt() >= to.pop_cnt(),
            "From needs to have atleast as many squares as to."
        );
        Self { from, to, flag, t }
    }
}

impl Iterator for PawnMoves<PseudoLegal> {
    type Item = Move;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.to.pop_lsb().map(|to| unsafe {
            // todo: write unit tests for this.
            // Safety: the 'from' bb is generated by every constructor in such a way,
            // that there is always atleast one square in the 'from' bb per square
            // in the 'to' bb.
            Move::new(self.from.pop_lsb().unwrap_unchecked(), to, self.flag)
        })
    }
}

impl<T: Legal> Iterator for PawnMoves<T> {
    type Item = Move;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(to) = self.to.pop_lsb() {
            // todo: write unit tests for this.
            // Safety: the 'from' bb is generated by every constructor in such a way,
            // that there is always atleast one square in the 'from' bb per square
            // in the 'to' bb.
            let from = unsafe { self.from.pop_lsb().unwrap_unchecked() };
            let from_bb = Bitboard::from_c(from);

            let blockers = self.t.get_blockers();
            let is_blocker = !(blockers & from_bb).is_empty();
            if is_blocker {
                let pin_mask = Bitboard::ray(from, self.t.get_king());
                let to_bb = Bitboard::from_c(to);
                let is_legal = !(pin_mask & to_bb).is_empty();
                if !is_legal {
                    return self.next();
                }
            }
            Some(Move::new(from, to, self.flag))
        } else {
            None
        }
    }
}

type Contructor<P, T> = fn(&PawnMovesInfo, T) -> P;

#[inline]
fn get_moves<const C: TColor, P, T>() -> [Contructor<P, T>; 18]
where
    P: IPawnMoves<T>,
{
    // todo: tune the ordering
    [
        P::single_step::<C>,
        P::double_step::<C>,
        P::promo_knight::<C>,
        P::promo_bishop::<C>,
        P::promo_rook::<C>,
        P::promo_queen::<C>,
        P::capture::<C, { CompassRose::WEST_C }>,
        P::capture::<C, { CompassRose::EAST_C }>,
        P::ep::<C, { CompassRose::WEST_C }>,
        P::ep::<C, { CompassRose::EAST_C }>,
        P::promo_capture_knight::<C, { CompassRose::WEST_C }>,
        P::promo_capture_knight::<C, { CompassRose::EAST_C }>,
        P::promo_capture_bishop::<C, { CompassRose::WEST_C }>,
        P::promo_capture_bishop::<C, { CompassRose::EAST_C }>,
        P::promo_capture_rook::<C, { CompassRose::WEST_C }>,
        P::promo_capture_rook::<C, { CompassRose::EAST_C }>,
        P::promo_capture_queen::<C, { CompassRose::WEST_C }>,
        P::promo_capture_queen::<C, { CompassRose::EAST_C }>,
    ]
}

#[inline]
fn fold_moves<const C: TColor, P, T, B, F, R>(
    info: PawnMovesInfo,
    legal: T,
    init: B,
    mut f: F,
) -> R
where
    T: Copy,
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
    P: IPawnMoves<T> + Iterator<Item = Move>,
{
    macro_rules! apply {
        ($init:expr, $($constructor:expr),+) => {
            {
                let mut acc = $init;
                $(
                    acc = $constructor(&info, legal).try_fold(acc, &mut f)?;
                )+
                Try::from_output(acc)
            }
        };
    }

    apply!(
        init,
        P::single_step::<C>,
        P::double_step::<C>,
        P::promo_knight::<C>,
        P::promo_bishop::<C>,
        P::promo_rook::<C>,
        P::promo_queen::<C>,
        P::capture::<C, { CompassRose::WEST_C }>,
        P::capture::<C, { CompassRose::EAST_C }>,
        P::ep::<C, { CompassRose::WEST_C }>,
        P::ep::<C, { CompassRose::EAST_C }>,
        P::promo_capture_knight::<C, { CompassRose::WEST_C }>,
        P::promo_capture_knight::<C, { CompassRose::EAST_C }>,
        P::promo_capture_bishop::<C, { CompassRose::WEST_C }>,
        P::promo_capture_bishop::<C, { CompassRose::EAST_C }>,
        P::promo_capture_rook::<C, { CompassRose::WEST_C }>,
        P::promo_capture_rook::<C, { CompassRose::EAST_C }>,
        P::promo_capture_queen::<C, { CompassRose::WEST_C }>,
        P::promo_capture_queen::<C, { CompassRose::EAST_C }>
    )
}

pub fn gen_pseudo_legals(pos: &Position) -> impl Iterator<Item = Move> + '_ {
    let plegal = IgnoreCheck {};
    let color = pos.get_turn();
    let info = PawnMovesInfo::new(pos);
    let moves = match color {
        Color::WHITE => get_moves::<{ Color::WHITE_C }, PawnMoves<IgnoreCheck>, _>(),
        Color::BLACK => get_moves::<{ Color::BLACK_C }, PawnMoves<IgnoreCheck>, _>(),
        _ => unreachable!(),
    };

    moves.into_iter().flat_map(move |f| f(&info, plegal))
}

pub fn gen_legals_check_none(pos: &Position) -> impl Iterator<Item = Move> + '_ {
    let legal = NoCheck::new(pos);
    let color = pos.get_turn();
    let info = PawnMovesInfo::new(pos);
    let moves = match color {
        Color::WHITE => get_moves::<{ Color::WHITE_C }, PawnMoves<NoCheck>, _>(),
        Color::BLACK => get_moves::<{ Color::BLACK_C }, PawnMoves<NoCheck>, _>(),
        _ => unreachable!(),
    };

    moves.into_iter().flat_map(move |f| f(&info, legal))
}

#[inline]
pub fn fold_legals_check_none<B, F, R>(
    pos: &Position,
    init: B,
    f: F,
) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let legal = NoCheck::new(pos);
    let color = pos.get_turn();
    let info = PawnMovesInfo::new(pos);
    match color {
        Color::WHITE => {
            fold_moves::<{ Color::WHITE_C }, PawnMoves<NoCheck>, _, _, _, _>(info, legal, init, f)
        }
        Color::BLACK => {
            fold_moves::<{ Color::BLACK_C }, PawnMoves<NoCheck>, _, _, _, _>(info, legal, init, f)
        }
        _ => unreachable!(),
    }
}

pub fn gen_legals_check_single(pos: &Position) -> impl Iterator<Item = Move> + '_ {
    let color = pos.get_turn();
    let resolve = SingleCheck::new(pos, color);
    let info = PawnMovesInfo::new(pos);
    let moves = match color {
        Color::WHITE => get_moves::<{ Color::WHITE_C }, PawnMoves<SingleCheck>, _>(),
        Color::BLACK => get_moves::<{ Color::BLACK_C }, PawnMoves<SingleCheck>, _>(),
        _ => unreachable!(),
    };

    moves.into_iter().flat_map(move |f| f(&info, resolve))
}

#[inline]
pub fn fold_legals_check_single<B, F, R>(
    pos: &Position,
    init: B,
    f: F,
) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let color = pos.get_turn();
    let resolve = SingleCheck::new(pos, color);
    let info = PawnMovesInfo::new(pos);
    match color {
        Color::WHITE => {
            fold_moves::<{ Color::WHITE_C }, PawnMoves<SingleCheck>, _, _, _, _>(info, resolve, init, f)
        }
        Color::BLACK => {
            fold_moves::<{ Color::BLACK_C }, PawnMoves<SingleCheck>, _, _, _, _>(info, resolve, init, f)
        }
        _ => unreachable!(),
    }
}

pub const fn generic_compute_attacks<const C: TColor>(pawns: Bitboard) -> Bitboard {
    Color::assert_variant(C); // Safety
    let color = unsafe { Color::from_v(C) };
    Bitboard {
        v: {
            let attacks_west = pawns
                .and_not_c(Bitboard::from_c(File::A))
                .shift(capture(color, CompassRose::WEST));
            let attacks_east = pawns
                .and_not_c(Bitboard::from_c(File::H))
                .shift(capture(color, CompassRose::EAST));
            attacks_west.v | attacks_east.v
        },
    }
}

pub const fn compute_attacks(pawns: Bitboard, color: Color) -> Bitboard {
    match color {
        Color::WHITE => generic_compute_attacks::<{ Color::WHITE_C }>(pawns),
        Color::BLACK => generic_compute_attacks::<{ Color::BLACK_C }>(pawns),
        _ => unreachable!(),
    }
}

pub const fn lookup_attacks(sq: Square, color: Color) -> Bitboard {
    match color {
        Color::WHITE => lookup_attacks_white(sq),
        Color::BLACK => lookup_attacks_black(sq),
        _ => unreachable!(),
    }
}

pub const fn lookup_attacks_white(sq: Square) -> Bitboard {
    const ATTACKS: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in Square::A1_C..(Square::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from_c(sq);
            result[sq.v() as usize] = generic_compute_attacks::<{ Color::WHITE_C }>(pawn);
        });
        result
    };
    ATTACKS[sq.v() as usize]
}

pub const fn lookup_attacks_black(sq: Square) -> Bitboard {
    const ATTACKS: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in Square::A1_C..(Square::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from_c(sq);
            result[sq.v() as usize] = generic_compute_attacks::<{ Color::BLACK_C }>(pawn);
        });
        result
    };
    ATTACKS[sq.v() as usize]
}
