use std::{hint::unreachable_unchecked, ops::Try};

use bishop::Bishop;
use king::King;
use knight::Knight;
use pawn::Pawn;
use queen::Queen;
use rook::Rook;

use crate::core::{r#move::move_flags, piece::IPieceType, position};

use super::{
    bitboard::Bitboard, color::Color, coordinates::Square, r#move::Move, position::Position,
};

pub mod bishop;
pub mod king;
pub mod knight;
pub mod pawn;
pub mod queen;
pub mod rook;
pub mod sliding_piece;

#[cfg(test)]
mod test;

/// To squares for quiet moves
#[inline(always)]
pub fn quiets_targets<C: NoDoubleCheck>(pos: &Position, color: Color) -> Bitboard {
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

pub const trait Options {
    fn gen_quiets() -> bool;
    fn gen_promos() -> bool;

    /// Whether to generated moves have to be legal. If false, also generates
    /// pseudo legal moves, which's check-rules are not checked.
    fn legal() -> bool {
        true
    }
}

pub trait FoldMoves<Check, O: Options> {
    fn fold_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>;
}

use position::CheckState as RtCheckState;

pub const trait CheckState {
    fn check_state() -> RtCheckState;
}
pub trait SomeCheck {}
pub trait NoDoubleCheck: CheckState {}

pub struct NoCheck;
impl CheckState for NoCheck {
    #[inline(always)]
    fn check_state() -> RtCheckState {
        RtCheckState::None
    }
}
impl NoDoubleCheck for NoCheck {}

pub struct SingleCheck;
impl CheckState for SingleCheck {
    #[inline(always)]
    fn check_state() -> RtCheckState {
        RtCheckState::Single
    }
}
impl SomeCheck for SingleCheck {}
impl NoDoubleCheck for SingleCheck {}

struct DoubleCheck;
impl CheckState for DoubleCheck {
    #[inline(always)]
    fn check_state() -> RtCheckState {
        RtCheckState::Double
    }
}
impl SomeCheck for DoubleCheck {}

impl NoCheck {
    #[inline(always)]
    fn fold_moves<O: Options, B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        init = <Rook as FoldMoves<NoCheck, O>>::fold_moves(pos, init, &mut f)?;
        init = <Bishop as FoldMoves<NoCheck, O>>::fold_moves(pos, init, &mut f)?;
        init = <Queen as FoldMoves<NoCheck, O>>::fold_moves(pos, init, &mut f)?;
        init = <King as FoldMoves<NoCheck, O>>::fold_moves(pos, init, &mut f)?;
        init = <Knight as FoldMoves<NoCheck, O>>::fold_moves(pos, init, &mut f)?;
        init = <Pawn as FoldMoves<NoCheck, O>>::fold_moves(pos, init, &mut f)?;
        try { init }
    }
}

impl SingleCheck {
    #[inline(always)]
    fn fold_moves<O: Options, B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        init = <Rook as FoldMoves<SingleCheck, O>>::fold_moves(pos, init, &mut f)?;
        init = <Bishop as FoldMoves<SingleCheck, O>>::fold_moves(pos, init, &mut f)?;
        init = <Queen as FoldMoves<SingleCheck, O>>::fold_moves(pos, init, &mut f)?;
        init = <King as FoldMoves<SingleCheck, O>>::fold_moves(pos, init, &mut f)?;
        init = <Knight as FoldMoves<SingleCheck, O>>::fold_moves(pos, init, &mut f)?;
        init = <Pawn as FoldMoves<SingleCheck, O>>::fold_moves(pos, init, &mut f)?;
        try { init }
    }
}

impl DoubleCheck {
    #[inline(always)]
    fn fold_moves<O: Options, B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        <King as FoldMoves<DoubleCheck, O>>::fold_moves(pos, init, f)
    }
}

#[inline]
pub fn fold_moves<O: Options, B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    match pos.get_check_state() {
        RtCheckState::None => NoCheck::fold_moves::<O, _, _, _>(pos, init, f),
        RtCheckState::Single => SingleCheck::fold_moves::<O, _, _, _>(pos, init, f),
        RtCheckState::Double => DoubleCheck::fold_moves::<O, _, _, _>(pos, init, f),
    }
}

pub mod opt {
    use super::Options;

    pub struct All;
    impl Options for All {
        #[inline(always)]
        fn gen_quiets() -> bool {
            true
        }

        #[inline(always)]
        fn gen_promos() -> bool {
            true
        }
    }

    pub struct AllPseudoLegal;
    impl Options for AllPseudoLegal {
        #[inline(always)]
        fn gen_quiets() -> bool {
            true
        }

        #[inline(always)]
        fn gen_promos() -> bool {
            true
        }

        #[inline(always)]
        fn legal() -> bool {
            false
        }
    }

    pub struct Captures;
    impl Options for Captures {
        #[inline(always)]
        fn gen_quiets() -> bool {
            false
        }

        #[inline(always)]
        fn gen_promos() -> bool {
            false
        }
    }
}

#[inline]
pub fn fold_legal_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    fold_moves::<opt::All, B, F, R>(pos, init, f)
}

#[inline]
pub fn fold_legal_captures<B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    fold_moves::<opt::Captures, B, F, R>(pos, init, f)
}

#[inline]
pub fn fold_pseudo_legal_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    fold_moves::<opt::AllPseudoLegal, B, F, R>(pos, init, f)
}

#[inline]
pub fn is_blocker(blockers: Bitboard, piece: Square) -> bool {
    let piece_bb = Bitboard::from(piece);
    !(blockers & piece_bb).is_empty()
}

#[inline]
pub fn pin_mask(piece: Square, blockers: Bitboard, our_king: Square) -> Bitboard {
    if is_blocker(blockers, piece) {
        Bitboard::ray(piece, our_king)
    }
    else {
        Bitboard::full()
    }
}

#[inline]
pub fn map_captures(targets: Bitboard, piece: Square) -> impl Iterator<Item = Move> {
    targets.map(move |target| Move::new(piece, target, move_flags::CAPTURE))
}

#[inline]
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
