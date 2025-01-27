use std::ops::{ControlFlow, Try};

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
    #[inline(always)]
    fn get_blockers(&self) -> Bitboard {
        self.pos.get_blockers()
    }

    #[inline(always)]
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
    #[inline(always)]
    pub fn new(pos: &'a Position) -> Self {
        Self { pos }
    }
}

#[derive(Debug, Clone, Copy)]
struct SingleCheck<'a> {
    pos: &'a Position,
}
impl Legal for SingleCheck<'_> {
    #[inline(always)]
    fn get_blockers(&self) -> Bitboard {
        self.pos.get_blockers()
    }

    #[inline(always)]
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
    #[inline(always)]
    pub fn new(pos: &'a Position) -> Self {
        assert_eq!(pos.get_check_state(), CheckState::Single);
        Self { pos }
    }

    #[inline(always)]
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

    #[inline(always)]
    fn single_step<const C: TColor>(pos: &Position, t: T) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pieces = pos.get_occupancy();
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let non_promo_pawns = pawns & !Bitboard::from_c(promo_rank(color));
        let single_step_tabus = backward(pieces, single_step(color));
        let from = non_promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, MoveFlag::QUIET, t)
    }

    #[inline(always)]
    fn double_step<const C: TColor>(pos: &Position, t: T) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pieces = pos.get_occupancy();
        let single_step_tabus = backward(pieces, single_step(color));
        let double_step_tabus = backward(pieces, double_step(color)) | single_step_tabus;
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let double_step_pawns = pawns & Bitboard::from_c(start_rank(color));
        let from = double_step_pawns & !double_step_tabus;
        let to = forward(from, double_step(color));
        Self::new(from, to, MoveFlag::DOUBLE_PAWN_PUSH, t)
    }

    #[inline(always)]
    fn capture<const C: TColor, const DIR: TCompassRose>(pos: &Position, t: T) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let non_promo_pawns = pawns & !Bitboard::from_c(promo_rank(color));
        let capturing_pawns = non_promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capturing_pawns, capture_dir) & pos.get_color_bb(!color);
        let from = backward(to, capture_dir);
        Self::new(from, to, MoveFlag::CAPTURE, t)
    }

    #[inline(always)]
    fn promo<const C: TColor>(pos: &Position, flag: MoveFlag, t: T) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pieces = pos.get_occupancy();
        let single_step_tabus = backward(pieces, single_step(color));
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let promo_pawns = pawns & Bitboard::from_c(promo_rank(color));
        let from = promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, flag, t)
    }

    #[inline(always)]
    fn promo_knight<const C: TColor>(pos: &Position, t: T) -> Self {
        Self::promo::<C>(pos, MoveFlag::PROMOTION_KNIGHT, t)
    }

    #[inline(always)]
    fn promo_bishop<const C: TColor>(pos: &Position, t: T) -> Self {
        Self::promo::<C>(pos, MoveFlag::PROMOTION_BISHOP, t)
    }

    #[inline(always)]
    fn promo_rook<const C: TColor>(pos: &Position, t: T) -> Self {
        Self::promo::<C>(pos, MoveFlag::PROMOTION_ROOK, t)
    }

    #[inline(always)]
    fn promo_queen<const C: TColor>(pos: &Position, t: T) -> Self {
        Self::promo::<C>(pos, MoveFlag::PROMOTION_QUEEN, t)
    }

    #[inline(always)]
    fn pl_promo_capture<const C: TColor, const DIR: TCompassRose>(
        pos: &Position,
        flag: MoveFlag,
        t: T,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let promo_pawns = pawns & Bitboard::from_c(promo_rank(color));
        let capture_west_pawns = promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capture_west_pawns, capture_dir) & pos.get_color_bb(!color);
        let from = backward(to, capture_dir);
        Self::new(from, to, flag, t)
    }

    #[inline(always)]
    fn promo_capture_knight<const C: TColor, const DIR: TCompassRose>(
        pos: &Position,
        t: T,
    ) -> Self {
        Self::pl_promo_capture::<C, DIR>(pos, MoveFlag::CAPTURE_PROMOTION_KNIGHT, t)
    }

    #[inline(always)]
    fn promo_capture_bishop<const C: TColor, const DIR: TCompassRose>(
        pos: &Position,
        t: T,
    ) -> Self {
        Self::pl_promo_capture::<C, DIR>(pos, MoveFlag::CAPTURE_PROMOTION_BISHOP, t)
    }

    #[inline(always)]
    fn promo_capture_rook<const C: TColor, const DIR: TCompassRose>(pos: &Position, t: T) -> Self {
        Self::pl_promo_capture::<C, DIR>(pos, MoveFlag::CAPTURE_PROMOTION_ROOK, t)
    }

    #[inline(always)]
    fn promo_capture_queen<const C: TColor, const DIR: TCompassRose>(pos: &Position, t: T) -> Self {
        Self::pl_promo_capture::<C, DIR>(pos, MoveFlag::CAPTURE_PROMOTION_QUEEN, t)
    }

    #[inline(always)]
    fn ep<const C: TColor, const DIR: TCompassRose>(pos: &Position, t: T) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let target = EpTargetSquare::from((pos.get_ep_capture_square(), !color));
        let mut to = Bitboard::from_c(target.v());
        let from = if to.is_empty() {
            Bitboard::empty()
        } else {
            let capture_dir = capture(color, CompassRose::new(DIR));
            let pawns = pos.get_bitboard(PieceType::PAWN, color);
            let capturing_pawns = pawns & !Bitboard::from_c(File::edge::<DIR>());
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
    #[inline(always)]
    fn new(from: Bitboard, to: Bitboard, flag: MoveFlag, t: NoCheck<'a>) -> Self {
        debug_assert!(
            from.pop_cnt() >= to.pop_cnt(),
            "From needs to have atleast as many squares as to."
        );
        Self { from, to, flag, t }
    }

    #[inline(always)]
    fn ep<const C: TColor, const DIR: TCompassRose>(pos: &Position, t: NoCheck<'a>) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_sq = pos.get_ep_capture_square();
        let target_sq = EpTargetSquare::from((capture_sq, !color));
        let mut to = Bitboard::from_c(target_sq.v());
        let from = if to.is_empty() {
            Bitboard::empty()
        } else {
            let capture_dir = capture(color, CompassRose::new(DIR));
            let pawns = pos.get_bitboard(PieceType::PAWN, color);
            let capturing_pawns = pawns & !Bitboard::from_c(File::edge::<DIR>());
            let from = backward(forward(capturing_pawns, capture_dir) & to, capture_dir);
            if from.is_empty() {
                to = Bitboard::empty();
                Bitboard::empty()
            } else {
                // Check that the king is not in check after the capture happens.
                let occupancy = pos.get_occupancy();
                let king_sq = t.get_king();
                let capt_bb = Bitboard::from_c(capture_sq.v());
                let occupancy_after_capture = (occupancy ^ from ^ capt_bb) | to;
                let rooks = pos.get_bitboard(PieceType::ROOK, !color);
                let bishops = pos.get_bitboard(PieceType::BISHOP, !color);
                let queens = pos.get_bitboard(PieceType::QUEEN, !color);
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
    #[inline(always)]
    fn new(from: Bitboard, to: Bitboard, flag: MoveFlag, t: SingleCheck<'a>) -> Self {
        debug_assert!(
            from.pop_cnt() >= to.pop_cnt(),
            "From needs to have atleast as many squares as to."
        );
        Self { from, to, flag, t }
    }

    #[inline(always)]
    fn single_step<const C: TColor>(pos: &Position, t: SingleCheck<'a>) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let non_promo_pawns = pawns & !Bitboard::from_c(promo_rank(color));
        let pieces = pos.get_occupancy();
        let tabu_squares = !t.get_blocks() | pieces;
        let single_step_tabus = backward(tabu_squares, single_step(color));
        let from = non_promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, MoveFlag::QUIET, t)
    }

    #[inline(always)]
    fn double_step<const C: TColor>(pos: &Position, t: SingleCheck<'a>) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pieces = pos.get_occupancy();
        let tabu_squares = pieces | !t.get_blocks();
        let single_step_tabus = backward(pieces, single_step(color));
        let double_step_tabus = backward(tabu_squares, double_step(color)) | single_step_tabus;
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let double_step_pawns = pawns & Bitboard::from_c(start_rank(color));
        let from = double_step_pawns & !double_step_tabus;
        let to = forward(from, double_step(color));
        Self::new(from, to, MoveFlag::DOUBLE_PAWN_PUSH, t)
    }

    #[inline(always)]
    fn capture<const C: TColor, const DIR: TCompassRose>(
        pos: &Position,
        t: SingleCheck<'a>,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let non_promo_pawns = pawns & !Bitboard::from_c(promo_rank(color));
        let capturing_pawns = non_promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capturing_pawns, capture_dir) & t.pos.get_checkers();
        let from = backward(to, capture_dir);
        Self::new(from, to, MoveFlag::CAPTURE, t)
    }

    #[inline(always)]
    fn promo<const C: TColor>(pos: &Position, flag: MoveFlag, t: SingleCheck<'a>) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pieces = pos.get_occupancy();
        let tabu_squares = pieces | !t.get_blocks();
        let single_step_tabus = backward(tabu_squares, single_step(color));
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let promo_pawns = pawns & Bitboard::from_c(promo_rank(color));
        let from = promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, flag, t)
    }

    #[inline(always)]
    fn pl_promo_capture<const C: TColor, const DIR: TCompassRose>(
        pos: &Position,
        flag: MoveFlag,
        t: SingleCheck<'a>,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let pawns = pos.get_bitboard(PieceType::PAWN, color);
        let promo_pawns = pawns & Bitboard::from_c(promo_rank(color));
        let capture_west_pawns = promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capture_west_pawns, capture_dir) & t.pos.get_checkers();
        let from = backward(to, capture_dir);
        Self::new(from, to, flag, t)
    }

    #[inline(always)]
    fn ep<const C: TColor, const DIR: TCompassRose>(pos: &Position, t: SingleCheck<'a>) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_sq = pos.get_ep_capture_square();
        let target = EpTargetSquare::from((capture_sq, !color));
        let mut to = Bitboard::from_c(target.v());
        let from = if to.is_empty() {
            Bitboard::empty()
        } else {
            let capture_dir = capture(color, CompassRose::new(DIR));
            let pawns = pos.get_bitboard(PieceType::PAWN, color);
            let capturing_pawns = pawns & !Bitboard::from_c(File::edge::<DIR>());
            let from = backward(forward(capturing_pawns, capture_dir) & to, capture_dir);
            if from.is_empty() {
                to = Bitboard::empty();
                Bitboard::empty()
            } else {
                // Check that the king is not in check after the capture happens.
                let occupancy = pos.get_occupancy();
                let king_sq = t.get_king();
                let capt_bb = Bitboard::from_c(capture_sq.v());
                let occupancy_after_capture = (occupancy ^ from ^ capt_bb) | to;
                let rooks = pos.get_bitboard(PieceType::ROOK, !color);
                let bishops = pos.get_bitboard(PieceType::BISHOP, !color);
                let queens = pos.get_bitboard(PieceType::QUEEN, !color);
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
    #[inline(always)]
    fn new(from: Bitboard, to: Bitboard, flag: MoveFlag, t: T) -> Self {
        debug_assert!(
            from.pop_cnt() >= to.pop_cnt(),
            "From needs to have atleast as many squares as to."
        );
        Self { from, to, flag, t }
    }
}

impl Iterator for PawnMoves<PseudoLegal> {
    type Item = Move;

    #[inline(always)]
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

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(to) = self.to.pop_lsb() {
            // Safety: The 'from' bitboard is constructed to have at least one square
            // per 'to' square, so unwrap_unchecked is safe.
            let from = unsafe { self.from.pop_lsb().unwrap_unchecked() };
            let from_bb = Bitboard::from_c(from);

            let blockers = self.t.get_blockers();
            let is_blocker = !(blockers & from_bb).is_empty();

            if is_blocker {
                let pin_mask = Bitboard::ray(from, self.t.get_king());
                let to_bb = Bitboard::from_c(to);
                let is_legal = !(pin_mask & to_bb).is_empty();

                if !is_legal {
                    continue;
                }
            }

            return Some(Move::new(from, to, self.flag));
        }

        None
    }
}

#[inline]
fn fold_moves<const C: TColor, P, T, B, F, R>(pos: &Position, legal: T, init: B, mut f: F) -> R
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
                    acc = $constructor(pos, legal).try_fold(acc, &mut f)?;
                )+
                try { acc }
            }
        };
    }

    // todo: tune the ordering
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

#[inline]
pub fn fold_legals_check_none<B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let legal = NoCheck::new(pos);
    match pos.get_turn() {
        Color::WHITE => {
            fold_moves::<{ Color::WHITE_C }, PawnMoves<NoCheck>, _, _, _, _>(pos, legal, init, f)
        }
        Color::BLACK => {
            fold_moves::<{ Color::BLACK_C }, PawnMoves<NoCheck>, _, _, _, _>(pos, legal, init, f)
        }
        _ => unreachable!(),
    }
}

#[inline]
pub fn fold_legals_check_single<B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let resolve = SingleCheck::new(pos);
    match pos.get_turn() {
        Color::WHITE => fold_moves::<{ Color::WHITE_C }, PawnMoves<SingleCheck>, _, _, _, _>(
            pos, resolve, init, f,
        ),
        Color::BLACK => fold_moves::<{ Color::BLACK_C }, PawnMoves<SingleCheck>, _, _, _, _>(
            pos, resolve, init, f,
        ),
        _ => unreachable!(),
    }
}

const fn compute_attacks<const C: TColor>(pawns: Bitboard) -> Bitboard {
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

#[inline(always)]
pub fn lookup_attacks(sq: Square, color: Color) -> Bitboard {
    static ATTACKS_W: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in Square::A1_C..(Square::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from_c(sq);
            result[sq.v() as usize] = compute_attacks::<{ Color::WHITE_C }>(pawn);
        });
        result
    };
    static ATTACKS_B: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in Square::A1_C..(Square::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from_c(sq);
            result[sq.v() as usize] = compute_attacks::<{ Color::BLACK_C }>(pawn);
        });
        result
    };
    static ATTACKS: [[Bitboard; 64]; 2] = [ATTACKS_W, ATTACKS_B];
    unsafe {
        // Safety: sq is in range 0..64 and color is in range 0..2
        *ATTACKS
            .get_unchecked(color.v() as usize)
            .get_unchecked(sq.v() as usize)
    }
}
