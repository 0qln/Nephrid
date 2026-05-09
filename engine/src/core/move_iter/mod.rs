use std::ops::Try;

use bishop::Bishop;
use king::King;
use knight::Knight;
use pawn::Pawn;
use queen::Queen;
use rook::Rook;

use crate::{core::r#move::move_flags};

use super::{
    bitboard::Bitboard,
    color::Color,
    coordinates::Square,
    r#move::Move,
    piece::IPieceType,
    position::{CheckState, Position},
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

pub trait FoldMoves<Check, const GEN_QUIETS: bool> {
    fn fold_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>;
}

pub trait FoldMovesHelper<Check, const GEN_QUIETS: bool> {
    fn fold_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>;
}

trait SomeCheck {}

trait NoDoubleCheck {
    fn raw_quiets_mask(pos: &Position, color: Color) -> Bitboard;
    fn captures_mask(pos: &Position, color: Color) -> Bitboard;

    #[inline(always)]
    fn quiets_mask<const GEN_QUIETS: bool>(pos: &Position, color: Color) -> Bitboard {
        if !GEN_QUIETS {
            Bitboard::empty()
        }
        else {
            Self::raw_quiets_mask(pos, color)
        }
    }
}

pub struct NoCheck;
impl NoDoubleCheck for NoCheck {
    fn raw_quiets_mask(pos: &Position, _: Color) -> Bitboard {
        !pos.get_occupancy()
    }

    fn captures_mask(pos: &Position, color: Color) -> Bitboard {
        pos.get_color_bb(!color)
    }
}

pub struct SingleCheck;
impl SomeCheck for SingleCheck {}
impl NoDoubleCheck for SingleCheck {
    fn raw_quiets_mask(pos: &Position, color: Color) -> Bitboard {
        let king_bb = pos.get_bitboard(King::ID, color);
        Bitboard::between(
            // Safety: there is a check, so there has to be a king.
            unsafe { king_bb.lsb().unwrap_unchecked() },
            // Safety: there is a single checker.
            unsafe { pos.get_checkers().lsb().unwrap_unchecked() },
        )
    }

    fn captures_mask(pos: &Position, _: Color) -> Bitboard {
        pos.get_checkers()
    }
}

struct DoubleCheck;
impl SomeCheck for DoubleCheck {}

impl<C, const Q: bool> FoldMovesHelper<C, { Q }> for C
where
    C: NoDoubleCheck,
    King: FoldMoves<C, { Q }>,
{
    fn fold_moves<B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        init = <Rook as FoldMoves<C, Q>>::fold_moves(pos, init, &mut f)?;
        init = <Bishop as FoldMoves<C, Q>>::fold_moves(pos, init, &mut f)?;
        init = <Queen as FoldMoves<C, Q>>::fold_moves(pos, init, &mut f)?;
        init = <King as FoldMoves<C, Q>>::fold_moves(pos, init, &mut f)?;
        init = <Knight as FoldMoves<C, Q>>::fold_moves(pos, init, &mut f)?;
        init = <Pawn as FoldMoves<C, Q>>::fold_moves(pos, init, &mut f)?;
        try { init }
    }
}

impl<const Q: bool> FoldMoves<Self, { Q }> for SingleCheck {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        <Self as FoldMovesHelper<SingleCheck, Q>>::fold_moves(pos, init, &mut f)
    }
}

impl<const Q: bool> FoldMoves<Self, { Q }> for NoCheck {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        if Q {
            init = king::fold_legal_castling(pos, init, &mut f)?;
        }
        <Self as FoldMovesHelper<NoCheck, Q>>::fold_moves(pos, init, &mut f)
    }
}

impl<const Q: bool> FoldMoves<Self, { Q }> for DoubleCheck {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        <King as FoldMoves<DoubleCheck, { Q }>>::fold_moves(pos, init, f)
    }
}

#[inline]
pub fn fold_legals<const Q: bool, B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    use CheckState::*;
    match pos.get_check_state() {
        None => <NoCheck as FoldMoves<_, { Q }>>::fold_moves(pos, init, f),
        Single => <SingleCheck as FoldMoves<_, { Q }>>::fold_moves(pos, init, f),
        Double => <DoubleCheck as FoldMoves<_, { Q }>>::fold_moves(pos, init, f),
    }
}

#[inline]
pub fn fold_legal_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    fold_legals::<true, B, F, R>(pos, init, f)
}

#[inline]
pub fn fold_legal_captures<B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    fold_legals::<false, B, F, R>(pos, init, f)
}

#[inline]
pub fn is_blocker(pos: &Position, piece: Square) -> bool {
    let piece_bb = Bitboard::from(piece);
    let blockers = pos.get_blockers();
    !(blockers & piece_bb).is_empty()
}

#[inline]
pub fn pin_mask(pos: &Position, piece: Square) -> Bitboard {
    if is_blocker(pos, piece) {
        let color = pos.get_turn();

        // todo: safely remove branching
        let bb = pos.get_bitboard(King::ID, color)?;

        // Safety: We check whether the bb is empty.
        let king = unsafe { bb.lsb().unwrap_unchecked() };
        Bitboard::ray(piece, king)
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
