use std::{hint::unreachable_unchecked, ops::Try};

use bishop::Bishop;
use king::King;
use rook::Rook;

use crate::core::{
    color::{
        Perspective, colors,
        perspectives::{Black, White},
    },
    r#move::move_flags,
    move_iter::sliding_piece::SlidingAttacks,
    piece::{IPieceType, piece_type},
    position,
};

use super::{bitboard::Bitboard, color::Color, coordinates::Square, r#move::Move, position::Position};

pub mod bishop;
pub mod king;
pub mod knight;
pub mod pawn;
pub mod queen;
pub mod rook;
pub mod sliding_piece;

#[cfg(test)] mod test;

/// To squares for quiet moves
#[inline(always)]
pub const fn quiets_targets<C: const NoDoubleCheck>(pos: &Position, color: Color) -> Bitboard {
    match C::check_state() {
        RtCheckState::None => !pos.get_occupancy(),
        RtCheckState::Single => {
            let king_bb = pos.get_bitboard(King::ID, color);
            Bitboard::between(
                // Safety: there is a check, so there has to be a king.
                unsafe { king_bb.lsb().unwrap_unchecked() },
                // Safety: there is a single checker.
                unsafe { pos.get_checkers().lsb().unwrap_unchecked() },
            )
        }
        // Safety: there is no double check, so this case is unreachable.
        RtCheckState::Double => unsafe { unreachable_unchecked() },
    }
}

/// To squares for captures
#[inline(always)]
pub fn captures_targets<C: NoDoubleCheck>(pos: &Position, color: Color) -> Bitboard {
    match C::check_state() {
        RtCheckState::None => pos.get_color_bb(!color),
        RtCheckState::Single => pos.get_checkers(),
        // Safety: there is no double check, so this case is unreachable.
        RtCheckState::Double => unsafe { unreachable_unchecked() },
    }
}

// todo: this flag does not currently include discovered checks. you could
// implement this via also tracking 'allies that block for the enemy king'
// in the position state (just like the normal 'blockers' field).

/// the generated movesets for each gen_{level}_* flags are disjoint within
/// their {level}.
pub const trait Options {
    fn quiet_checks() -> bool { false }
    fn quiet_nochecks() -> bool { false }

    fn capture_checks() -> bool { false }
    fn capture_nochecks() -> bool { false }

    fn promo_checks() -> bool { false }
    fn promo_nochecks() -> bool { false }

    fn gen_quiets() -> bool { Self::quiet_checks() || Self::quiet_nochecks() }
    fn gen_captures() -> bool { Self::capture_checks() || Self::capture_nochecks() }
    fn gen_promos() -> bool { Self::promo_checks() || Self::promo_nochecks() }

    fn gen_checks() -> bool { Self::quiet_checks() || Self::capture_checks() || Self::promo_checks() }
    fn gen_only_checks() -> bool { Self::gen_checks() && !Self::quiet_nochecks() && !Self::capture_nochecks() && !Self::promo_nochecks() }

    /// Whether to generated moves have to be legal. If false, also generates
    /// pseudo legal moves, which's check-rules are not checked.
    fn legal() -> bool { true }
}

pub trait FoldMoves<P: Perspective, Check, O: Options, Input> {
    fn fold_moves_for<B, F, R>(i: &Input, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>;
}

use position::CheckState as RtCheckState;

pub const trait CheckState {
    fn check_state() -> RtCheckState;
}
pub const trait SomeCheck {}
pub const trait NoDoubleCheck: const CheckState {}

pub struct NoCheck;
const impl CheckState for NoCheck {
    #[inline(always)]
    fn check_state() -> RtCheckState { RtCheckState::None }
}
const impl NoDoubleCheck for NoCheck {}

pub struct SingleCheck;
const impl CheckState for SingleCheck {
    #[inline(always)]
    fn check_state() -> RtCheckState { RtCheckState::Single }
}
const impl SomeCheck for SingleCheck {}
const impl NoDoubleCheck for SingleCheck {}

struct DoubleCheck;
const impl CheckState for DoubleCheck {
    #[inline(always)]
    fn check_state() -> RtCheckState { RtCheckState::Double }
}
impl SomeCheck for DoubleCheck {}

#[inline(always)]
fn fold_all_moves_for<P: Perspective, O: Options, C: const NoDoubleCheck, B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let quiets = if O::gen_quiets() {
        quiets_targets::<C>(pos, P::COLOR)
    }
    else {
        Bitboard::empty()
    };

    let captures = if O::gen_captures() {
        captures_targets::<C>(pos, P::COLOR)
    }
    else {
        Bitboard::empty()
    };

    let promos = if O::gen_promos() {
        quiets_targets::<C>(pos, P::COLOR)
    }
    else {
        Bitboard::empty()
    };

    let occ = pos.get_occupancy();
    let blockers = pos.get_blockers();
    let enemies = pos.get_color_bb(P::Opponent::COLOR);
    let kings = pos.get_bitboard(King::ID, P::COLOR);
    let king = kings.lsb();
    let their_k = pos.get_bitboard(King::ID, P::Opponent::COLOR).lsb();

    let queens = pos.get_bitboard(piece_type::QUEEN, P::COLOR);

    let rooks = pos.get_bitboard(piece_type::ROOK, P::COLOR);
    let rook_mask = if (!O::capture_nochecks() || !O::quiet_nochecks())
        && let Some(k) = their_k
    {
        Rook::lookup_attacks(k, occ)
    }
    else {
        Bitboard::full()
    };
    let rook_quiets = {
        let mut t = quiets;
        if !O::quiet_nochecks() {
            t &= rook_mask;
        }
        t
    };
    let rook_captures = {
        let mut t = captures;
        if !O::capture_nochecks() {
            t &= rook_mask;
        }
        t
    };
    init = sliding_piece::fold_moves_for::<_, _, _, P, O, Rook>(queens | rooks, blockers, occ, king, rook_captures, rook_quiets, init, &mut f)?;

    let bishops = pos.get_bitboard(piece_type::BISHOP, P::COLOR);
    let bishop_mask = if (!O::capture_nochecks() || !O::quiet_nochecks())
        && let Some(k) = their_k
    {
        Bishop::lookup_attacks(k, occ)
    }
    else {
        Bitboard::full()
    };
    let bishop_quiets = {
        let mut t = quiets;
        if !O::quiet_nochecks() {
            t &= bishop_mask;
        }
        t
    };
    let bishop_captures = {
        let mut t = captures;
        if !O::capture_nochecks() {
            t &= bishop_mask;
        }
        t
    };
    init =
        sliding_piece::fold_moves_for::<_, _, _, P, O, Bishop>(queens | bishops, blockers, occ, king, bishop_captures, bishop_quiets, init, &mut f)?;

    init = {
        let king_quiets = if O::gen_quiets() {
            let mut t = !occ;

            if !O::quiet_nochecks() {
                t &= Bitboard::empty();
            }

            t
        }
        else {
            Bitboard::empty()
        };

        king::fold_moves_for::<_, _, _, P, O, C>(king, kings, pos, occ, enemies, captures, king_quiets, init, &mut f)?
    };

    let knights = pos.get_bitboard(piece_type::KNIGHT, P::COLOR);
    let knight_mask = if (!O::capture_nochecks() || !O::quiet_nochecks())
        && let Some(k) = their_k
    {
        knight::lookup_attacks(k)
    }
    else {
        Bitboard::full()
    };
    let knight_quiets = {
        let mut t = quiets;
        if !O::quiet_nochecks() {
            t &= knight_mask;
        }
        t
    };
    let knight_captures = {
        let mut t = captures;
        if !O::capture_nochecks() {
            t &= knight_mask;
        }
        t
    };
    init = knight::fold_moves_for::<_, _, _, P, O>(knights, blockers, knight_quiets, knight_captures, init, &mut f)?;

    let pawns = pos.get_bitboard(piece_type::PAWN, P::COLOR);
    let pawn_mask = if (!O::capture_nochecks() || !O::quiet_nochecks())
        && let Some(k) = their_k
    {
        pawn::lookup_attacks(k, P::Opponent::COLOR)
    }
    else {
        Bitboard::full()
    };
    let pawn_quiets = {
        let mut t = quiets;
        if !O::quiet_nochecks() {
            t &= pawn_mask;
        }
        t
    };
    let pawn_captures = {
        let mut t = captures;
        if !O::capture_nochecks() {
            t &= pawn_mask;
        }
        t
    };
    init = pawn::fold_moves_for::<P, O, C, _, _, _>(pawns, occ, blockers, king, pawn_captures, pawn_quiets, promos, pos, init, &mut f)?;

    try { init }
}

impl DoubleCheck {
    #[inline(always)]
    fn fold_moves_for<P: Perspective, O: Options, B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        let kings = pos.get_bitboard(King::ID, P::COLOR);
        let king = kings.lsb();
        let occ = pos.get_occupancy();
        let enemies = pos.get_color_bb(P::Opponent::COLOR);

        king::fold_moves_for_somecheck::<_, _, _, P, O, Self>(pos, kings, king, enemies, occ, !occ, init, f)
    }
}

#[inline]
pub fn fold_moves_for<P: Perspective, O: Options, B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    match pos.get_check_state() {
        RtCheckState::None => fold_all_moves_for::<P, O, NoCheck, _, _, _>(pos, init, f),
        RtCheckState::Single => fold_all_moves_for::<P, O, SingleCheck, _, _, _>(pos, init, f),
        RtCheckState::Double => DoubleCheck::fold_moves_for::<P, O, _, _, _>(pos, init, f),
    }
}

pub mod opt {
    use super::Options;

    pub struct AllLegal;
    const impl Options for AllLegal {
        fn quiet_checks() -> bool { true }
        fn quiet_nochecks() -> bool { true }
        fn capture_checks() -> bool { true }
        fn capture_nochecks() -> bool { true }
        fn promo_checks() -> bool { true }
        fn promo_nochecks() -> bool { true }
    }

    pub struct AllPseudoLegal;
    const impl Options for AllPseudoLegal {
        fn quiet_checks() -> bool { true }
        fn quiet_nochecks() -> bool { true }
        fn capture_checks() -> bool { true }
        fn capture_nochecks() -> bool { true }
        fn promo_checks() -> bool { true }
        fn promo_nochecks() -> bool { true }

        fn legal() -> bool { false }
    }

    pub struct Captures;
    const impl Options for Captures {
        fn quiet_checks() -> bool { false }
        fn quiet_nochecks() -> bool { false }
        fn capture_checks() -> bool { true }
        fn capture_nochecks() -> bool { true }
        fn promo_checks() -> bool { false }
        fn promo_nochecks() -> bool { false }
    }

    pub struct Threats;
    impl Options for Threats {
        fn quiet_checks() -> bool { true }
        fn quiet_nochecks() -> bool { false }

        fn capture_checks() -> bool { true }
        fn capture_nochecks() -> bool { true }

        fn promo_checks() -> bool { true }
        fn promo_nochecks() -> bool { false }

        fn legal() -> bool { true }
    }
}

#[inline]
pub fn fold_moves<O: Options, B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    match pos.get_turn() {
        colors::WHITE => fold_moves_for::<White, O, B, F, R>(pos, init, f),
        colors::BLACK => fold_moves_for::<Black, O, B, F, R>(pos, init, f),
        // Safety: pos.get_turn() is guaranteed to return a valid color.
        _ => unsafe { unreachable_unchecked() },
    }
}

#[inline]
pub fn fold_legal_moves_for<P: Perspective, B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    fold_moves_for::<P, opt::AllLegal, B, F, R>(pos, init, f)
}

#[inline]
pub fn fold_legal_captures_for<P: Perspective, B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    fold_moves_for::<P, opt::Captures, B, F, R>(pos, init, f)
}

#[inline]
pub fn fold_pseudo_legal_moves_for<P: Perspective, B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    fold_moves_for::<P, opt::AllPseudoLegal, B, F, R>(pos, init, f)
}

#[inline]
pub const fn is_blocker(blockers: Bitboard, piece_bb: Bitboard) -> bool { !(blockers & piece_bb).is_empty() }

#[inline]
pub fn pin_mask(piece: Square, blockers: Bitboard, our_king: Square) -> Bitboard {
    if is_blocker(blockers, Bitboard::from(piece)) {
        Bitboard::ray(piece, our_king)
    }
    else {
        Bitboard::full()
    }
}

#[inline(always)]
pub fn map_captures(targets: Bitboard, piece: Square) -> impl Iterator<Item = Move> {
    targets.map(move |target| Move::new(piece, target, move_flags::CAPTURE))
}

#[inline(always)]
pub fn map_quiets(targets: Bitboard, piece: Square) -> impl Iterator<Item = Move> {
    targets.map(move |target| Move::new(piece, target, move_flags::QUIET))
}

// todo:
// read and optimize:
// https://www.chessprogramming.org/Traversing_Subsets_of_a_Set

/// Maps the specified bits into allowed bits (defined by mask).
/// If the mask does not specify atleast the number of bits in
/// needed for a complete mapping, the remaining bits are cut off.
fn map_bits(mut bits: usize, mask: Bitboard) -> Bitboard {
    mask.fold(Bitboard::empty(), |acc, pos| {
        let val = bits & 1;
        bits >>= 1;
        acc | (val << pos)
    })
}

/// convenience wrapper
#[deprecated(since = "0.0.0", note = "use fold_moves instead")]
#[allow(dead_code)]
pub fn dbg_fold_moves<T, C: const CheckState, O: Options, B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
    T: FoldMoves<White, C, O, Position>,
    T: FoldMoves<Black, C, O, Position>,
{
    match pos.get_turn() {
        colors::WHITE => <T as FoldMoves<White, C, O, Position>>::fold_moves_for(pos, init, f),
        colors::BLACK => <T as FoldMoves<Black, C, O, Position>>::fold_moves_for(pos, init, f),
        _ => unreachable!(),
    }
}
