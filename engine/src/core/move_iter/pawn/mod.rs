use super::{
    bishop::Bishop, king::King, pin_mask, rook::Rook, sliding_piece::SlidingAttacks, FoldMoves,
    NoDoubleCheck,
};
use crate::{
    core::{
        bitboard::Bitboard,
        color::{colors, Color, TColor},
        coordinates::{
            compass_rose, files, squares, CompassRose, EpTargetSquare, File, Square, TCompassRose,
        },
        piece::{piece_type, IPieceType},
        position::Position,
        r#move::{move_flags, Move, MoveFlag},
    },
    misc::ConstFrom,
};
use const_for::const_for;
use std::ops::Try;

use helpers::*;

mod helpers;

pub struct Pawn;

pub struct PawnMoves<'a> {
    from: Bitboard,
    to: Bitboard,
    flag: MoveFlag,
    pos: &'a Position,
}

impl<'a> PawnMoves<'a> {
    #[inline(always)]
    fn new(from: Bitboard, to: Bitboard, flag: MoveFlag, pos: &'a Position) -> Self {
        debug_assert!(
            from.pop_cnt() >= to.pop_cnt(),
            "From needs to have atleast as many squares as to."
        );
        Self { from, to, flag, pos }
    }

    #[inline(always)]
    fn single_step<const C: TColor, T: NoDoubleCheck>(pos: &'a Position) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pawns = pos.get_bitboard(piece_type::PAWN, color);
        let non_promo_pawns = pawns & !Bitboard::from_c(promo_rank(color));
        let pieces = pos.get_occupancy();
        let tabu_squares = !T::quiets_mask(pos, color) | pieces;
        let single_step_tabus = backward(tabu_squares, single_step(color));
        let from = non_promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, move_flags::QUIET, pos)
    }

    #[inline(always)]
    fn double_step<const C: TColor, T: NoDoubleCheck>(pos: &'a Position) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pieces = pos.get_occupancy();
        let tabu_squares = !T::quiets_mask(pos, color) | pieces;
        let single_step_tabus = backward(pieces, single_step(color));
        let double_step_tabus = backward(tabu_squares, double_step(color)) | single_step_tabus;
        let pawns = pos.get_bitboard(piece_type::PAWN, color);
        let double_step_pawns = pawns & Bitboard::from_c(start_rank(color));
        let from = double_step_pawns & !double_step_tabus;
        let to = forward(from, double_step(color));
        Self::new(from, to, move_flags::DOUBLE_PAWN_PUSH, pos)
    }

    #[inline(always)]
    fn capture<const C: TColor, const DIR: TCompassRose, T: NoDoubleCheck>(
        pos: &'a Position,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let pawns = pos.get_bitboard(piece_type::PAWN, color);
        let non_promo_pawns = pawns & !Bitboard::from_c(promo_rank(color));
        let capturing_pawns = non_promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capturing_pawns, capture_dir) & T::captures_mask(pos, color);
        let from = backward(to, capture_dir);
        Self::new(from, to, move_flags::CAPTURE, pos)
    }

    #[inline(always)]
    fn promo_knight<const C: TColor, T: NoDoubleCheck>(pos: &'a Position) -> Self {
        Self::promo::<C, T>(pos, move_flags::PROMOTION_KNIGHT)
    }

    #[inline(always)]
    fn promo_bishop<const C: TColor, T: NoDoubleCheck>(pos: &'a Position) -> Self {
        Self::promo::<C, T>(pos, move_flags::PROMOTION_BISHOP)
    }

    #[inline(always)]
    fn promo_rook<const C: TColor, T: NoDoubleCheck>(pos: &'a Position) -> Self {
        Self::promo::<C, T>(pos, move_flags::PROMOTION_ROOK)
    }

    #[inline(always)]
    fn promo_queen<const C: TColor, T: NoDoubleCheck>(pos: &'a Position) -> Self {
        Self::promo::<C, T>(pos, move_flags::PROMOTION_QUEEN)
    }

    #[inline(always)]
    fn promo<const C: TColor, T: NoDoubleCheck>(pos: &'a Position, flag: MoveFlag) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pieces = pos.get_occupancy();
        let tabu_squares = !T::quiets_mask(pos, color) | pieces;
        let single_step_tabus = backward(tabu_squares, single_step(color));
        let pawns = pos.get_bitboard(piece_type::PAWN, color);
        let promo_pawns = pawns & Bitboard::from_c(promo_rank(color));
        let from = promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, flag, pos)
    }

    #[inline(always)]
    fn promo_capture_knight<const C: TColor, const DIR: TCompassRose, T: NoDoubleCheck>(
        pos: &'a Position,
    ) -> Self {
        Self::pl_promo_capture::<C, DIR, T>(pos, move_flags::CAPTURE_PROMOTION_KNIGHT)
    }

    #[inline(always)]
    fn promo_capture_bishop<const C: TColor, const DIR: TCompassRose, T: NoDoubleCheck>(
        pos: &'a Position,
    ) -> Self {
        Self::pl_promo_capture::<C, DIR, T>(pos, move_flags::CAPTURE_PROMOTION_BISHOP)
    }

    #[inline(always)]
    fn promo_capture_rook<const C: TColor, const DIR: TCompassRose, T: NoDoubleCheck>(
        pos: &'a Position,
    ) -> Self {
        Self::pl_promo_capture::<C, DIR, T>(pos, move_flags::CAPTURE_PROMOTION_ROOK)
    }

    #[inline(always)]
    fn promo_capture_queen<const C: TColor, const DIR: TCompassRose, T: NoDoubleCheck>(
        pos: &'a Position,
    ) -> Self {
        Self::pl_promo_capture::<C, DIR, T>(pos, move_flags::CAPTURE_PROMOTION_QUEEN)
    }

    #[inline(always)]
    fn pl_promo_capture<const C: TColor, const DIR: TCompassRose, T: NoDoubleCheck>(
        pos: &'a Position,
        flag: MoveFlag,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let pawns = pos.get_bitboard(piece_type::PAWN, color);
        let promo_pawns = pawns & Bitboard::from_c(promo_rank(color));
        let capture_west_pawns = promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capture_west_pawns, capture_dir) & T::captures_mask(pos, color);
        let from = backward(to, capture_dir);
        Self::new(from, to, flag, pos)
    }

    #[inline(always)]
    fn ep<const C: TColor, const DIR: TCompassRose, T: NoDoubleCheck>(pos: &'a Position) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_sq = pos.get_ep_capture_square();
        let target = EpTargetSquare::from((capture_sq, !color));
        let mut to = Bitboard::from_c(target.v());
        let from = if to.is_empty() {
            Bitboard::empty()
        }
        else {
            let capture_dir = capture(color, CompassRose::new(DIR));
            let pawns = pos.get_bitboard(piece_type::PAWN, color);
            let capturing_pawns = pawns & !Bitboard::from_c(File::edge::<DIR>());
            let from = backward(forward(capturing_pawns, capture_dir) & to, capture_dir);
            if from.is_empty() {
                to = Bitboard::empty();
                Bitboard::empty()
            }
            else {
                // Safety: king the board has no king, but gen_legal is used,
                // the context is broken anyway.
                let king_bb = pos.get_bitboard(King::ID, color);
                let king_sq = unsafe { king_bb.lsb().unwrap_unchecked() };
                // Check that the king is not in check after the capture happens.
                let occupancy = pos.get_occupancy();
                let capt_bb = Bitboard::from_c(capture_sq.v());
                let occupancy_after_capture = (occupancy ^ from ^ capt_bb) | to;
                let rooks = pos.get_bitboard(piece_type::ROOK, !color);
                let bishops = pos.get_bitboard(piece_type::BISHOP, !color);
                let queens = pos.get_bitboard(piece_type::QUEEN, !color);
                let rook_attacks = Rook::lookup_attacks(king_sq, occupancy_after_capture);
                let bishop_attacks = Bishop::lookup_attacks(king_sq, occupancy_after_capture);
                let q_or_r_check = !rook_attacks.and_c(rooks | queens).is_empty();
                let q_or_b_check = !bishop_attacks.and_c(bishops | queens).is_empty();
                let check = q_or_r_check || q_or_b_check;
                if check {
                    to = Bitboard::empty();
                    Bitboard::empty()
                }
                else {
                    from
                }
            }
        };
        Self::new(from, to, move_flags::EN_PASSANT, pos)
    }
}

impl Iterator for PawnMoves<'_> {
    type Item = Move;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(to) = self.to.pop_lsb() {
            // Safety: The 'from' bitboard is constructed to have at least one square
            // per 'to' square, so unwrap_unchecked is safe.
            let from = unsafe { self.from.pop_lsb().unwrap_unchecked() };

            // todo: the result of this branch is the same for some of the moves,
            // e.g: single and double step, so it can be cached.
            // maybe we can remove this inner loop aswell.

            // Check if the pawn is pinned and the move is valid.
            let pin_mask = pin_mask(self.pos, from);
            if (pin_mask & Bitboard::from_c(to)).is_empty() {
                continue;
            }

            return Some(Move::new(from, to, self.flag));
        }

        None
    }
}

#[inline]
fn fold_moves<const C: TColor, T: NoDoubleCheck, B, F, R>(pos: &'_ Position, init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    macro_rules! apply {
        ($init:expr, $($constructor:expr),+) => {
            {
                let mut acc = $init;
                $(
                    acc = $constructor(pos).try_fold(acc, &mut f)?;
                )+
                try { acc }
            }
        };
    }

    type P<'a> = PawnMoves<'a>;

    // todo: tune the ordering
    apply!(
        init,
        P::single_step::<C, T>,
        P::double_step::<C, T>,
        P::promo_knight::<C, T>,
        P::promo_bishop::<C, T>,
        P::promo_rook::<C, T>,
        P::promo_queen::<C, T>,
        P::capture::<C, { compass_rose::WEST_C }, T>,
        P::capture::<C, { compass_rose::EAST_C }, T>,
        P::ep::<C, { compass_rose::WEST_C }, T>,
        P::ep::<C, { compass_rose::EAST_C }, T>,
        P::promo_capture_knight::<C, { compass_rose::WEST_C }, T>,
        P::promo_capture_knight::<C, { compass_rose::EAST_C }, T>,
        P::promo_capture_bishop::<C, { compass_rose::WEST_C }, T>,
        P::promo_capture_bishop::<C, { compass_rose::EAST_C }, T>,
        P::promo_capture_rook::<C, { compass_rose::WEST_C }, T>,
        P::promo_capture_rook::<C, { compass_rose::EAST_C }, T>,
        P::promo_capture_queen::<C, { compass_rose::WEST_C }, T>,
        P::promo_capture_queen::<C, { compass_rose::EAST_C }, T>
    )
}

impl<C: NoDoubleCheck> FoldMoves<C> for Pawn {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        match pos.get_turn() {
            colors::WHITE => fold_moves::<{ colors::WHITE_C }, C, _, _, _>(pos, init, f),
            colors::BLACK => fold_moves::<{ colors::BLACK_C }, C, _, _, _>(pos, init, f),
            _ => unreachable!(),
        }
    }
}

const fn compute_attacks<const C: TColor>(pawns: Bitboard) -> Bitboard {
    Color::assert_variant(C); // Safety
    let color = unsafe { Color::from_v(C) };
    Bitboard {
        v: {
            let attacks_west = pawns
                .and_not_c(Bitboard::from_c(files::A))
                .shift(capture(color, compass_rose::WEST));
            let attacks_east = pawns
                .and_not_c(Bitboard::from_c(files::H))
                .shift(capture(color, compass_rose::EAST));
            attacks_west.v | attacks_east.v
        },
    }
}

#[inline(always)]
pub fn lookup_attacks(sq: Square, color: Color) -> Bitboard {
    static ATTACKS_W: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in squares::A1_C..(squares::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from_c(sq);
            result[sq.v() as usize] = compute_attacks::<{ colors::WHITE_C }>(pawn);
        });
        result
    };
    static ATTACKS_B: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in squares::A1_C..(squares::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from_c(sq);
            result[sq.v() as usize] = compute_attacks::<{ colors::BLACK_C }>(pawn);
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
