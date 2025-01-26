use std::ops::Try;

use bishop::Bishop;
use queen::Queen;
use rook::Rook;

use super::bitboard::Bitboard;
use super::coordinates::Square;
use super::piece::JumpingPieceType;
use super::position::{CheckState, Position};
use super::r#move::{Move, MoveFlag};

pub mod bishop;
pub mod king;
pub mod knight;
pub mod pawn;
pub mod queen;
pub mod rook;
pub mod sliding_piece;

#[cfg(test)]
mod test;

#[inline(always)]
fn try_fold_multiple<B, F, G, R, const N: usize>(
    iters: [G; N],
    init: B,
    mut f: F,
) -> R
where
    F: FnMut(B, Move) -> R,
    G: FnOnce(B, &mut F) -> R,
    R: Try<Output = B>,
{
    iters.into_iter().try_fold(init, |acc, iter| {
        iter(acc, &mut f)
    })
}

#[inline]
fn fold_moves_check_none<const CAPTURES_ONLY: bool, B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    init = sliding_piece::fold_legals_check_none::<_, _, _, Rook>(pos, init, &mut f)?;
    init = sliding_piece::fold_legals_check_none::<_, _, _, Bishop>(pos, init, &mut f)?;
    init = sliding_piece::fold_legals_check_none::<_, _, _, Queen>(pos, init, &mut f)?;
    init = pawn::fold_legals_check_none(pos, init, &mut f)?;
    init = knight::fold_legals_check_none(pos, init, &mut f)?;
    init = king::fold_legals_check_none(pos, init, &mut f)?;
    king::fold_legal_castling(pos, init, f)
}

#[inline]
fn fold_moves_check_single<const CAPTURES_ONLY: bool, B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    init = sliding_piece::fold_legals_check_single::<_, _, _, Rook>(pos, init, &mut f)?;
    init = sliding_piece::fold_legals_check_single::<_, _, _, Bishop>(pos, init, &mut f)?;
    init = sliding_piece::fold_legals_check_single::<_, _, _, Queen>(pos, init, &mut f)?;
    init = king::fold_legals_check_some(pos, init, &mut f)?;
    init = pawn::fold_legals_check_single(pos, init, &mut f)?;
    knight::fold_legals_check_single(pos, init, &mut f)
}

#[inline]
fn fold_moves_check_double<const CAPTURES_ONLY: bool, B, F, R>(pos: &Position, init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    king::fold_legals_check_some(pos, init, &mut f)
}

#[inline]
pub fn foreach_legal_move<const CAPTURES_ONLY: bool, F, R>(pos: &Position, f: F) -> R
where
    F: FnMut(Move) -> R,
    R: Try<Output = ()>,
{
    #[inline]
    fn call<T, R>(mut f: impl FnMut(T) -> R) -> impl FnMut((), T) -> R {
        move |(), x| f(x)
    }

    fold_legal_move::<CAPTURES_ONLY, _, _, _>(pos, (), call(f))
}

#[inline]
pub fn fold_legal_move<const CAPTURES_ONLY: bool, B, F, R>(pos: &Position, init: B, f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    match pos.get_check_state() {
        CheckState::None => fold_moves_check_none::<CAPTURES_ONLY, _, _, _>(pos, init, f),
        CheckState::Single => fold_moves_check_single::<CAPTURES_ONLY, _, _, _>(pos, init, f),
        CheckState::Double => fold_moves_check_double::<CAPTURES_ONLY, _, _, _>(pos, init, f),
    }
}

#[inline]
pub fn gen_captures(
    attacks: Bitboard,
    enemies: Bitboard,
    piece: Square,
) -> impl Iterator<Item = Move> {
    let targets = attacks & enemies;
    targets.map(move |target| Move::new(piece, target, MoveFlag::CAPTURE))
}

#[inline]
pub fn gen_quiets(
    attacks: Bitboard,
    enemies: Bitboard,
    allies: Bitboard,
    piece: Square,
) -> impl Iterator<Item = Move> {
    let targets = attacks & !allies & !enemies;
    targets.map(move |target| Move::new(piece, target, MoveFlag::QUIET))
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