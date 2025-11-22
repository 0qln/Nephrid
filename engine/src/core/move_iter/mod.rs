use std::ops::Try;

use bishop::Bishop;
use king::King;
use knight::Knight;
use pawn::Pawn;
use queen::Queen;
use rook::Rook;

use crate::core::r#move::move_flags;
use crate::misc::ConstFrom;

use super::bitboard::Bitboard;
use super::color::Color;
use super::coordinates::Square;
use super::piece::IPieceType;
use super::position::{CheckState, Position};
use super::r#move::Move;

pub mod bishop;
pub mod king;
pub mod knight;
pub mod pawn;
pub mod queen;
pub mod rook;
pub mod sliding_piece;

#[cfg(test)]
mod test;

pub trait FoldMoves<Check> {
    fn fold_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>;
}

pub trait FoldMovesHelper<Check> {
    fn fold_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>;
}

trait SomeCheck {}

trait NoDoubleCheck {
    // todo: these are only interesting for non-king pieces.
    fn quiets_mask(pos: &Position, color: Color) -> Bitboard;
    fn captures_mask(pos: &Position, color: Color) -> Bitboard;
}

pub struct NoCheck;
impl NoDoubleCheck for NoCheck {
    fn quiets_mask(pos: &Position, _: Color) -> Bitboard {
        !pos.get_occupancy()
    }

    fn captures_mask(pos: &Position, color: Color) -> Bitboard {
        pos.get_color_bb(!color)
    }
}

pub struct SingleCheck;
impl SomeCheck for SingleCheck {}
impl NoDoubleCheck for SingleCheck {
    fn quiets_mask(pos: &Position, color: Color) -> Bitboard {
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

impl<C> FoldMovesHelper<C> for C
where
    C: NoDoubleCheck,
    King: FoldMoves<C>,
{
    fn fold_moves<B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        init = <Rook as FoldMoves<C>>::fold_moves(pos, init, &mut f)?;
        init = <Bishop as FoldMoves<C>>::fold_moves(pos, init, &mut f)?;
        init = <Queen as FoldMoves<C>>::fold_moves(pos, init, &mut f)?;
        init = <King as FoldMoves<C>>::fold_moves(pos, init, &mut f)?;
        init = <Knight as FoldMoves<C>>::fold_moves(pos, init, &mut f)?;
        init = <Pawn as FoldMoves<C>>::fold_moves(pos, init, &mut f)?;
        try { init }
    }
}

impl FoldMoves<Self> for SingleCheck {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        <Self as FoldMovesHelper<_>>::fold_moves(pos, init, &mut f)
    }
}

impl FoldMoves<Self> for NoCheck {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        init = king::fold_legal_castling(pos, init, &mut f)?;
        <Self as FoldMovesHelper<_>>::fold_moves(pos, init, &mut f)
    }
}

impl FoldMoves<Self> for DoubleCheck {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        <King as FoldMoves<DoubleCheck>>::fold_moves(pos, init, f)
    }
}

#[inline]
pub fn fold_legal_moves<B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    match pos.get_check_state() {
        CheckState::None => <NoCheck as FoldMoves<_>>::fold_moves(pos, init, f),
        CheckState::Single => <SingleCheck as FoldMoves<_>>::fold_moves(pos, init, f),
        CheckState::Double => <DoubleCheck as FoldMoves<_>>::fold_moves(pos, init, f),
    }
}

#[inline]
pub fn is_blocker(pos: &Position, piece: Square) -> bool {
    let piece_bb = Bitboard::from_c(piece);
    let blockers = pos.get_blockers();
    !(blockers & piece_bb).is_empty()
}

#[inline]
pub fn pin_mask(pos: &Position, piece: Square) -> Bitboard {
    is_blocker(pos, piece)
        .then(|| {
            // Safety: We check if the bb is empty of not.
            let king = unsafe {
                let color = pos.get_turn();
                // todo: safely remove branching
                let bb = pos.get_bitboard(King::ID, color)?;
                bb.lsb().unwrap_unchecked()
            };
            Bitboard::ray(piece, king)
        })
        .unwrap_or(Bitboard::full())
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
