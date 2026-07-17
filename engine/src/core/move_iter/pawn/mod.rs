use super::{FoldMoves, NoDoubleCheck, bishop::Bishop, king::King, pin_mask, rook::Rook, sliding_piece::SlidingAttacks};
use crate::core::{
    bitboard::Bitboard,
    color::{Color, Perspective, TColor, colors},
    coordinates::{CompassRose, EpTargetSquare, File, Square, TCompassRose, compass_rose, files, pawn_utils::*, squares},
    r#move::{Move, MoveFlag, move_flags},
    move_iter::{Options, captures_targets, quiets_targets},
    piece::{IPieceType, piece_type},
    position::Position,
};
use const_for::const_for;
use std::ops::Try;

use variants::Variant;

pub struct Pawn;

mod variants {
    use std::marker::PhantomData;

    use super::*;

    pub trait Variant {
        type Data;
    }
    pub trait Promo: Variant {}

    pub struct Pinned<'a> {
        pub _pos: PhantomData<&'a ()>,
    }
    impl<'a> Variant for Pinned<'a> {
        type Data = &'a Position;
    }

    pub struct PromoPinned<'a> {
        pub _pos: PhantomData<&'a ()>,
    }
    impl<'a> Promo for PromoPinned<'a> {}
    impl<'a> Variant for PromoPinned<'a> {
        type Data = &'a Position;
    }

    pub struct Unpinned;
    impl Variant for Unpinned {
        type Data = ();
    }

    pub struct PromoUnpinned;
    impl Promo for PromoUnpinned {}
    impl Variant for PromoUnpinned {
        type Data = ();
    }
}

struct PawnMoves<V: Variant> {
    from: Bitboard,
    to: Bitboard,
    flag: MoveFlag,
    v_data: V::Data,
}

macro_rules! constructor {
    () => {
        impl FnOnce(&Position, Bitboard, V::Data) -> Self
    };
}

impl<V: Variant> PawnMoves<V> {
    #[inline(always)]
    fn new(from: Bitboard, to: Bitboard, flag: MoveFlag, v_data: V::Data) -> Self {
        debug_assert!(from.pop_cnt() >= to.pop_cnt(), "From needs to have atleast as many squares as to.");
        Self { from, to, flag, v_data }
    }

    #[inline(always)]
    fn single_step<P: Perspective, T: const NoDoubleCheck>(non_promo_pawns: Bitboard, single_step_tabus: Bitboard) -> constructor!() {
        move |_pos, _pawns, v_data| {
            let from = non_promo_pawns & !single_step_tabus;
            let to = forward(from, single_step(P::COLOR));
            Self::new(from, to, move_flags::QUIET, v_data)
        }
    }

    #[inline(always)]
    fn double_step<P: Perspective, T: const NoDoubleCheck>(tabu_squares: Bitboard, single_step_tabus: Bitboard) -> constructor!() {
        move |_pos, pawns, v_data| {
            let double_step_tabus = backward(tabu_squares, double_step(P::COLOR)) | single_step_tabus;
            let double_step_pawns = pawns & Bitboard::from(start_rank(P::COLOR));
            let from = double_step_pawns & !double_step_tabus;
            let to = forward(from, double_step(P::COLOR));
            Self::new(from, to, move_flags::DOUBLE_PAWN_PUSH, v_data)
        }
    }

    #[inline(always)]
    fn capture<P: Perspective, const DIR: TCompassRose, T: NoDoubleCheck>(non_promo_pawns: Bitboard, capture_targets: Bitboard) -> constructor!() {
        move |_pos, _pawns, v_data| {
            let capture_dir = capture(P::COLOR, CompassRose::new(DIR));
            let capturing_pawns = non_promo_pawns & !Bitboard::from(File::edge::<DIR>());
            let to = forward(capturing_pawns, capture_dir) & capture_targets;
            let from = backward(to, capture_dir);
            Self::new(from, to, move_flags::CAPTURE, v_data)
        }
    }

    #[inline(always)]
    fn ep<O: Options, P: Perspective, const DIR: TCompassRose>(occ: Bitboard) -> constructor!() {
        move |pos, pawns, v_data| {
            let color = P::COLOR;
            let capture_sq = pos.get_ep_capture_square();
            let target = EpTargetSquare::from((capture_sq, !color));
            let to = Bitboard::from(target.v());
            let (from, to) = if to.is_empty() {
                (Bitboard::empty(), to)
            }
            else {
                let capture_dir = capture(color, CompassRose::new(DIR));
                let capturing_pawns = pawns & !Bitboard::from(File::edge::<DIR>());
                let from = backward(forward(capturing_pawns, capture_dir) & to, capture_dir);
                if from.is_empty() {
                    (Bitboard::empty(), Bitboard::empty())
                }
                else {
                    // Safety: the board has no king, but gen_legal is used,
                    // the context is broken anyway.
                    let king_bb = pos.get_bitboard(King::ID, color);
                    let king_sq = unsafe { king_bb.lsb().unwrap_unchecked() };

                    // Check that the king is not in check after the capture happens.
                    let capt_bb = Bitboard::from(capture_sq.v());
                    let occupancy_after_capture = (occ ^ from ^ capt_bb) | to;

                    let bishops = || pos.get_bitboard(piece_type::BISHOP, !color);
                    let queens = pos.get_bitboard(piece_type::QUEEN, !color);
                    let rooks = || pos.get_bitboard(piece_type::ROOK, !color);

                    let rook_attacks = || Rook::lookup_attacks(king_sq, occupancy_after_capture);
                    let bishop_attacks = || Bishop::lookup_attacks(king_sq, occupancy_after_capture);

                    let check = {
                        if !rook_attacks().and_c(rooks() | queens).is_empty() {
                            true
                        }
                        else if !bishop_attacks().and_c(bishops() | queens).is_empty() {
                            true
                        }
                        else {
                            false
                        }
                    };

                    if check {
                        (Bitboard::empty(), Bitboard::empty())
                    }
                    else {
                        (from, to)
                    }
                }
            };

            Self::new(from, to, move_flags::EN_PASSANT, v_data)
        }
    }
}

impl<V: variants::Promo> PawnMoves<V> {
    #[inline(always)]
    fn promo<P: Perspective, T: const NoDoubleCheck>(promo_pawns: Bitboard, single_step_tabus: Bitboard) -> constructor!() {
        move |_pos, _pawns, v_data| {
            let from = promo_pawns & !single_step_tabus;
            let to = forward(from, single_step(P::COLOR));
            Self::new(from, to, move_flags::PROMOTION_KNIGHT, v_data)
        }
    }

    #[inline(always)]
    fn promo_capture<P: Perspective, const DIR: TCompassRose, T: NoDoubleCheck>(promo_pawns: Bitboard, capture_targets: Bitboard) -> constructor!() {
        move |_pos, _pawns, v_data| {
            let capture_dir = capture(P::COLOR, CompassRose::new(DIR));
            let capture_pawns = promo_pawns & !Bitboard::from(File::edge::<DIR>());
            let to = forward(capture_pawns, capture_dir) & capture_targets;
            let from = backward(to, capture_dir);
            Self::new(from, to, move_flags::CAPTURE_PROMOTION_KNIGHT, v_data)
        }
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
            let blockers = self.v_data.get_blockers();
            let pin_mask = self
                .v_data
                .get_bitboard(piece_type::KING, self.v_data.get_turn())
                .lsb()
                .map(|our_king| pin_mask(from, blockers, our_king))
                .unwrap_or_default();
            if (pin_mask & Bitboard::from(to)).is_empty() {
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

impl PawnMoves<variants::PromoUnpinned> {
    #[inline(always)]
    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        let mut acc = init;
        while let Some(to) = self.to.pop_lsb() {
            let from = unsafe { self.from.pop_lsb().unwrap_unchecked() };
            let flag_base = self.flag;
            for flag_offset in 0..4 {
                let flag = flag_base.v() + flag_offset;
                let flag = unsafe { MoveFlag::try_from(flag).unwrap_unchecked() };
                acc = f(acc, Move::new(from, to, flag))?;
            }
        }
        try { acc }
    }
}

impl PawnMoves<variants::PromoPinned<'_>> {
    #[inline(always)]
    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        let mut acc = init;
        while let Some(to) = self.to.pop_lsb() {
            let from = unsafe { self.from.pop_lsb().unwrap_unchecked() };

            // verify the pin mask for pinned promotions
            let blockers = self.v_data.get_blockers();
            let pin_mask = self
                .v_data
                .get_bitboard(piece_type::KING, self.v_data.get_turn())
                .lsb()
                .map(|our_king| pin_mask(from, blockers, our_king))
                .unwrap_or_default();
            if (pin_mask & Bitboard::from(to)).is_empty() {
                continue;
            }

            let flag_base = self.flag;
            for flag_offset in 0..4 {
                let flag = flag_base.v() + flag_offset;
                let flag = unsafe { MoveFlag::try_from(flag).unwrap_unchecked() };
                acc = f(acc, Move::new(from, to, flag))?;
            }
        }
        try { acc }
    }
}

#[inline]
fn fold_moves_for<P: Perspective, O: Options, T: const NoDoubleCheck, B, F, R>(pos: &'_ Position, init: B, mut f: F) -> R
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

    let occ = pos.get_occupancy();
    let all_pawns = pos.get_bitboard(piece_type::PAWN, P::COLOR);
    let (safe_pawns, pinned_pawns) = if O::legal() {
        let pinned_bb = pos.get_blockers();
        (all_pawns & !pinned_bb, all_pawns & pinned_bb)
    }
    else {
        (all_pawns, Bitboard::empty())
    };

    let mut acc = init;

    if !safe_pawns.is_empty() {
        type Moves = PawnMoves<variants::Unpinned>;
        type Promo = PawnMoves<variants::PromoUnpinned>;

        let pawns = safe_pawns;
        let promo_pawns = pawns & Bitboard::from(promo_rank(P::COLOR));
        let capture_targets = captures_targets::<T>(pos, P::COLOR);
        if O::gen_captures() || O::gen_promos() {
            acc = apply!(
                acc,
                pawns,
                (),
                Promo::promo_capture::<P, { compass_rose::WEST_C }, T>(promo_pawns, capture_targets),
                Promo::promo_capture::<P, { compass_rose::EAST_C }, T>(promo_pawns, capture_targets)
            );
        }

        let tabu_squares = !quiets_targets::<T>(pos, P::COLOR) | occ;
        let single_step_dest_tabus = backward(tabu_squares, single_step(P::COLOR));
        if O::gen_promos() {
            acc = apply!(acc, safe_pawns, (), Promo::promo::<P, T>(promo_pawns, single_step_dest_tabus));
        }

        let non_promo_pawns = pawns & !Bitboard::from(promo_rank(P::COLOR));
        if O::gen_captures() {
            acc = apply!(
                acc,
                safe_pawns,
                (),
                Moves::capture::<P, { compass_rose::WEST_C }, T>(non_promo_pawns, capture_targets),
                Moves::capture::<P, { compass_rose::EAST_C }, T>(non_promo_pawns, capture_targets),
                Moves::ep::<O, P, { compass_rose::WEST_C }>(occ),
                Moves::ep::<O, P, { compass_rose::EAST_C }>(occ)
            );
        }

        let single_step_occ_tabus = backward(occ, single_step(P::COLOR));
        if O::gen_quiets() {
            acc = apply!(
                acc,
                safe_pawns,
                (),
                Moves::single_step::<P, T>(non_promo_pawns, single_step_dest_tabus),
                Moves::double_step::<P, T>(tabu_squares, single_step_occ_tabus)
            );
        }
    }

    if O::legal() && !pinned_pawns.is_empty() {
        type Moves<'a> = PawnMoves<variants::Pinned<'a>>;
        type Promo<'a> = PawnMoves<variants::PromoPinned<'a>>;

        let pawns = pinned_pawns;
        let promo_pawns = pawns & Bitboard::from(promo_rank(P::COLOR));
        let capture_targets = captures_targets::<T>(pos, P::COLOR);
        if O::gen_captures() || O::gen_promos() {
            acc = apply!(
                acc,
                pinned_pawns,
                pos,
                Promo::promo_capture::<P, { compass_rose::WEST_C }, T>(promo_pawns, capture_targets),
                Promo::promo_capture::<P, { compass_rose::EAST_C }, T>(promo_pawns, capture_targets)
            );
        }

        if O::gen_promos() {
            // not generating, because pinned pawns cannot make a quiet
            // promoting move
        }

        let non_promo_pawns = pawns & !Bitboard::from(promo_rank(P::COLOR));
        if O::gen_captures() {
            acc = apply!(
                acc,
                pinned_pawns,
                pos,
                Moves::capture::<P, { compass_rose::WEST_C }, T>(non_promo_pawns, capture_targets),
                Moves::capture::<P, { compass_rose::EAST_C }, T>(non_promo_pawns, capture_targets),
                Moves::ep::<O, P, { compass_rose::WEST_C }>(occ),
                Moves::ep::<O, P, { compass_rose::EAST_C }>(occ)
            );
        }

        let tabu_squares = !quiets_targets::<T>(pos, P::COLOR) | occ;
        let single_step_dest_tabus = backward(tabu_squares, single_step(P::COLOR));
        let single_step_occ_tabus = backward(occ, single_step(P::COLOR));
        if O::gen_quiets() {
            acc = apply!(
                acc,
                pinned_pawns,
                pos,
                Moves::single_step::<P, T>(non_promo_pawns, single_step_dest_tabus),
                Moves::double_step::<P, T>(tabu_squares, single_step_occ_tabus)
            );
        }
    }

    try { acc }
}

impl<P: Perspective, O: Options, C: const NoDoubleCheck> FoldMoves<P, C, O> for Pawn {
    #[inline(always)]
    fn fold_moves_for<B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        fold_moves_for::<P, O, C, _, _, _>(pos, init, f)
    }
}

pub const fn compute_attacks_<const C: TColor>(pawns: Bitboard) -> Bitboard {
    Color::assert_variant(C); // Safety
    let color = unsafe { Color::from_v(C) };
    Bitboard {
        v: {
            let attacks_west = pawns.and_not_c(Bitboard::from(files::A)).shift(capture(color, compass_rose::WEST));
            let attacks_east = pawns.and_not_c(Bitboard::from(files::H)).shift(capture(color, compass_rose::EAST));
            attacks_west.v | attacks_east.v
        },
    }
}

pub const fn compute_attacks(pawns: Bitboard, color: Color) -> Bitboard {
    Bitboard {
        v: {
            let attacks_west = pawns.and_not_c(Bitboard::from(files::A)).shift(capture(color, compass_rose::WEST));
            let attacks_east = pawns.and_not_c(Bitboard::from(files::H)).shift(capture(color, compass_rose::EAST));
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
            let pawn = Bitboard::from(sq);
            result[sq.v() as usize] = compute_attacks_::<{ colors::WHITE_C }>(pawn);
        });
        result
    };
    static ATTACKS_B: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in squares::A1_C..(squares::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from(sq);
            result[sq.v() as usize] = compute_attacks_::<{ colors::BLACK_C }>(pawn);
        });
        result
    };
    static ATTACKS: [[Bitboard; 64]; 2] = [ATTACKS_W, ATTACKS_B];
    unsafe {
        // Safety: sq is in range 0..64 and color is in range 0..2
        *ATTACKS.get_unchecked(color.v() as usize).get_unchecked(sq.v() as usize)
    }
}

const fn compute_moves<const C: TColor>(pawns: Bitboard) -> Bitboard {
    Color::assert_variant(C); // Safety
    let color = unsafe { Color::from_v(C) };

    // double step
    let dpp_pawns = Bitboard {
        v: Bitboard::from(dpp_rank(color)).v & pawns.v,
    };
    let dpp_moves = forward(dpp_pawns, double_step(color));

    // single step
    let moves = forward(pawns, single_step(color));

    Bitboard { v: dpp_moves.v | moves.v }
}

#[inline(always)]
pub const fn lookup_moves(sq: Square, color: Color) -> Bitboard {
    static MOVES_W: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in squares::A1_C..(squares::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from(sq);
            result[sq.v() as usize] = compute_moves::<{ colors::WHITE_C }>(pawn);
        });
        result
    };
    static MOVES_B: [Bitboard; 64] = {
        let mut result = [Bitboard::empty(); 64];
        const_for!(sq in squares::A1_C..(squares::H8_C+1) => {
            let sq = unsafe { Square::from_v(sq) };
            let pawn = Bitboard::from(sq);
            result[sq.v() as usize] = compute_moves::<{ colors::BLACK_C }>(pawn);
        });
        result
    };
    static MOVES: [[Bitboard; 64]; 2] = [MOVES_W, MOVES_B];
    unsafe {
        // Safety: sq is in range 0..64 and color is in range 0..2
        *MOVES.get_unchecked(color.v() as usize).get_unchecked(sq.v() as usize)
    }
}
