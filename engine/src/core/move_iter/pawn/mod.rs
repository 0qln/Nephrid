use super::{
    FoldMoves, NoDoubleCheck, bishop::Bishop, king::King, pin_mask, rook::Rook,
    sliding_piece::SlidingAttacks,
};
use crate::{
    core::{
        bitboard::Bitboard,
        color::{Color, TColor, colors},
        coordinates::{
            CompassRose, EpTargetSquare, File, Square, TCompassRose, compass_rose, files, squares,
        },
        r#move::{Move, MoveFlag, TMoveFlag, move_flags},
        piece::{IPieceType, piece_type},
        position::Position,
    },
    misc::ConstFrom,
};
use const_for::const_for;
use std::ops::Try;

use helpers::*;

mod helpers;

pub struct Pawn;

trait Variant {}

mod variants {
    use super::*;

    pub struct Pinned<'a> {
        pub pos: &'a Position,
    }
    impl Variant for Pinned<'_> {}

    pub struct Unpinned;
    impl Variant for Unpinned {}
}

struct PawnMoves<V: Variant> {
    from: Bitboard,
    to: Bitboard,
    flag: MoveFlag,
    v_data: V,
}

impl<V: Variant> PawnMoves<V> {
    #[inline(always)]
    fn new(from: Bitboard, to: Bitboard, flag: MoveFlag, v_data: V) -> Self {
        debug_assert!(
            from.pop_cnt() >= to.pop_cnt(),
            "From needs to have atleast as many squares as to."
        );
        Self { from, to, flag, v_data }
    }

    #[inline(always)]
    fn single_step<const Q: bool, const C: TColor, T: NoDoubleCheck>(
        pos: &Position,
        pawns: Bitboard,
        v_data: V,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let non_promo_pawns = pawns & !Bitboard::from_c(promo_rank(color));
        let pieces = pos.get_occupancy();
        let tabu_squares = !T::quiets_mask::<Q>(pos, color) | pieces;
        let single_step_tabus = backward(tabu_squares, single_step(color));
        let from = non_promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, move_flags::QUIET, v_data)
    }

    #[inline(always)]
    fn double_step<const Q: bool, const C: TColor, T: NoDoubleCheck>(
        pos: &Position,
        pawns: Bitboard,
        v_data: V,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let pieces = pos.get_occupancy();
        let tabu_squares = !T::quiets_mask::<Q>(pos, color) | pieces;
        let single_step_tabus = backward(pieces, single_step(color));
        let double_step_tabus = backward(tabu_squares, double_step(color)) | single_step_tabus;
        let double_step_pawns = pawns & Bitboard::from_c(start_rank(color));
        let from = double_step_pawns & !double_step_tabus;
        let to = forward(from, double_step(color));
        Self::new(from, to, move_flags::DOUBLE_PAWN_PUSH, v_data)
    }

    #[inline(always)]
    fn capture<const C: TColor, const DIR: TCompassRose, T: NoDoubleCheck>(
        pos: &Position,
        pawns: Bitboard,
        v_data: V,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let non_promo_pawns = pawns & !Bitboard::from_c(promo_rank(color));
        let capturing_pawns = non_promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capturing_pawns, capture_dir) & T::captures_mask(pos, color);
        let from = backward(to, capture_dir);
        Self::new(from, to, move_flags::CAPTURE, v_data)
    }

    #[inline(always)]
    fn promo<const Q: bool, const C: TColor, T: NoDoubleCheck, const F: TMoveFlag>(
        pos: &Position,
        pawns: Bitboard,
        v_data: V,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        MoveFlag::assert_variant(F); // Safety
        let flag = unsafe { MoveFlag::from_v(F) };
        let pieces = pos.get_occupancy();
        let tabu_squares = !T::quiets_mask::<Q>(pos, color) | pieces;
        let single_step_tabus = backward(tabu_squares, single_step(color));
        let promo_pawns = pawns & Bitboard::from_c(promo_rank(color));
        let from = promo_pawns & !single_step_tabus;
        let to = forward(from, single_step(color));
        Self::new(from, to, flag, v_data)
    }

    #[inline(always)]
    fn promo_capture<
        const C: TColor,
        const DIR: TCompassRose,
        T: NoDoubleCheck,
        const F: TMoveFlag,
    >(
        pos: &Position,
        pawns: Bitboard,
        v_data: V,
    ) -> Self {
        Color::assert_variant(C); // Safety
        let color = unsafe { Color::from_v(C) };
        MoveFlag::assert_variant(F); // Safety
        let flag = unsafe { MoveFlag::from_v(F) };
        let capture_dir = capture(color, CompassRose::new(DIR));
        let promo_pawns = pawns & Bitboard::from_c(promo_rank(color));
        let capture_west_pawns = promo_pawns & !Bitboard::from_c(File::edge::<DIR>());
        let to = forward(capture_west_pawns, capture_dir) & T::captures_mask(pos, color);
        let from = backward(to, capture_dir);
        Self::new(from, to, flag, v_data)
    }

    #[inline(always)]
    fn ep<const C: TColor, const DIR: TCompassRose, T: NoDoubleCheck>(
        pos: &Position,
        pawns: Bitboard,
        v_data: V,
    ) -> Self {
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
        Self::new(from, to, move_flags::EN_PASSANT, v_data)
    }
}

impl Iterator for PawnMoves<variants::Pinned<'_>> {
    type Item = Move;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(to) = self.to.pop_lsb() {
            // Safety: The 'from' bitboard is constructed to have at least one square
            // per 'to' square, so unwrap_unchecked is safe.
            let from = unsafe { self.from.pop_lsb().unwrap_unchecked() };

            // Check if the pawn is pinned and the move is valid.
            let pin_mask = pin_mask(self.v_data.pos, from);
            if (pin_mask & Bitboard::from_c(to)).is_empty() {
                continue;
            }

            return Some(Move::new(from, to, self.flag));
        }

        None
    }
}

impl Iterator for PawnMoves<variants::Unpinned> {
    type Item = Move;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let to = self.to.pop_lsb()?;
        let from = unsafe { self.from.pop_lsb().unwrap_unchecked() };
        Some(Move::new(from, to, self.flag))
    }
}

#[inline]
fn fold_moves<const Q: bool, const C: TColor, T: NoDoubleCheck, B, F, R>(
    pos: &'_ Position,
    init: B,
    mut f: F,
) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    macro_rules! apply {
        ($init:expr, $pawns:expr, $v_data:expr, $($constructor:expr),+) => {
            {
                let mut __acc = $init;
                $(
                    __acc = $constructor(pos, $pawns, $v_data).try_fold(__acc, &mut f)?;
                )+
                __acc
            }
        };
    }

    // todo: tune the ordering

    let color = unsafe { Color::from_v(C) };
    let all_pawns = pos.get_bitboard(piece_type::PAWN, color);
    let pinned_bb = pos.get_blockers();
    let safe_pawns = Bitboard { v: all_pawns.v & !pinned_bb.v };
    let pinned_pawns = Bitboard { v: all_pawns.v & pinned_bb.v };

    let mut acc = init;

    // 3. THE FAST PATH (Branchless, executes 99% of the time)
    if !safe_pawns.is_empty() {
        type P = PawnMoves<variants::Unpinned>;
        use move_flags as flags;

        acc = apply!(
            acc,
            safe_pawns,
            variants::Unpinned,
            P::single_step::<Q, C, T>,
            P::double_step::<Q, C, T>,
            P::promo::<Q, C, T, { flags::PROMOTION_KNIGHT_C }>,
            P::promo::<Q, C, T, { flags::PROMOTION_BISHOP_C }>,
            P::promo::<Q, C, T, { flags::PROMOTION_ROOK_C }>,
            P::promo::<Q, C, T, { flags::PROMOTION_QUEEN_C }>,
            P::capture::<C, { compass_rose::WEST_C }, T>,
            P::capture::<C, { compass_rose::EAST_C }, T>,
            P::ep::<C, { compass_rose::WEST_C }, T>,
            P::ep::<C, { compass_rose::EAST_C }, T>,
            P::promo_capture::<C, { compass_rose::WEST_C }, T, { flags::CAPTURE_PROMOTION_KNIGHT_C }>,
            P::promo_capture::<C, { compass_rose::EAST_C }, T, { flags::CAPTURE_PROMOTION_KNIGHT_C }>,
            P::promo_capture::<C, { compass_rose::WEST_C }, T, { flags::CAPTURE_PROMOTION_BISHOP_C }>,
            P::promo_capture::<C, { compass_rose::EAST_C }, T, { flags::CAPTURE_PROMOTION_BISHOP_C }>,
            P::promo_capture::<C, { compass_rose::WEST_C }, T, { flags::CAPTURE_PROMOTION_ROOK_C }>,
            P::promo_capture::<C, { compass_rose::EAST_C }, T, { flags::CAPTURE_PROMOTION_ROOK_C }>,
            P::promo_capture::<C, { compass_rose::WEST_C }, T, { flags::CAPTURE_PROMOTION_QUEEN_C }>,
            P::promo_capture::<C, { compass_rose::EAST_C }, T, { flags::CAPTURE_PROMOTION_QUEEN_C }>
        );
    }

    if !pinned_pawns.is_empty() {
        type P<'a> = PawnMoves<variants::Pinned<'a>>;
        use move_flags as flags;

        acc = apply!(
            acc,
            pinned_pawns,
            variants::Pinned { pos },
            P::single_step::<Q, C, T>,
            P::double_step::<Q, C, T>,
            P::promo::<Q, C, T, { move_flags::PROMOTION_KNIGHT_C }>,
            P::promo::<Q, C, T, { move_flags::PROMOTION_BISHOP_C }>,
            P::promo::<Q, C, T, { move_flags::PROMOTION_ROOK_C }>,
            P::promo::<Q, C, T, { move_flags::PROMOTION_QUEEN_C }>,
            P::capture::<C, { compass_rose::WEST_C }, T>,
            P::capture::<C, { compass_rose::EAST_C }, T>,
            P::ep::<C, { compass_rose::WEST_C }, T>,
            P::ep::<C, { compass_rose::EAST_C }, T>,
            P::promo_capture::<C, { compass_rose::WEST_C }, T, { flags::CAPTURE_PROMOTION_KNIGHT_C }>,
            P::promo_capture::<C, { compass_rose::EAST_C }, T, { flags::CAPTURE_PROMOTION_KNIGHT_C }>,
            P::promo_capture::<C, { compass_rose::WEST_C }, T, { flags::CAPTURE_PROMOTION_BISHOP_C }>,
            P::promo_capture::<C, { compass_rose::EAST_C }, T, { flags::CAPTURE_PROMOTION_BISHOP_C }>,
            P::promo_capture::<C, { compass_rose::WEST_C }, T, { flags::CAPTURE_PROMOTION_ROOK_C }>,
            P::promo_capture::<C, { compass_rose::EAST_C }, T, { flags::CAPTURE_PROMOTION_ROOK_C }>,
            P::promo_capture::<C, { compass_rose::WEST_C }, T, { flags::CAPTURE_PROMOTION_QUEEN_C }>,
            P::promo_capture::<C, { compass_rose::EAST_C }, T, { flags::CAPTURE_PROMOTION_QUEEN_C }>
        );
    }

    try { acc }
}

impl<const Q: bool, C: NoDoubleCheck> FoldMoves<C, Q> for Pawn {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        match pos.get_turn() {
            colors::WHITE => fold_moves::<Q, { colors::WHITE_C }, C, _, _, _>(pos, init, f),
            colors::BLACK => fold_moves::<Q, { colors::BLACK_C }, C, _, _, _>(pos, init, f),
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

const fn compute_moves<const C: TColor>(pawns: Bitboard) -> Bitboard {
    Color::assert_variant(C); // Safety
    let color = unsafe { Color::from_v(C) };

    // double step
    let dpp_pawns = Bitboard {
        v: Bitboard::from_c(dpp_rank(color)).v & pawns.v,
    };
    let dpp_moves = forward(dpp_pawns, double_step(color));

    // single step
    let moves = forward(pawns, single_step(color));

    Bitboard { v: dpp_moves.v | moves.v }
}

#[inline(always)]
pub fn lookup_moves(sq: Square, color: Color) -> Bitboard {
    static MOVES_W: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in squares::A1_C..(squares::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from_c(sq);
            result[sq.v() as usize] = compute_moves::<{ colors::WHITE_C }>(pawn);
        });
        result
    };
    static MOVES_B: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in squares::A1_C..(squares::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from_c(sq);
            result[sq.v() as usize] = compute_moves::<{ colors::BLACK_C }>(pawn);
        });
        result
    };
    static MOVES: [[Bitboard; 64]; 2] = [MOVES_W, MOVES_B];
    unsafe {
        // Safety: sq is in range 0..64 and color is in range 0..2
        *MOVES
            .get_unchecked(color.v() as usize)
            .get_unchecked(sq.v() as usize)
    }
}
