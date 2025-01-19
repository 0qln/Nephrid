use std::ops::Try;

use super::bitboard::Bitboard;
use super::coordinates::Square;
use super::piece::{JumpingPieceType, SlidingPieceType};
use super::position::{CheckState, Position};
use super::r#move::{Move, MoveFlag};

pub mod bishop;
pub mod jumping_piece;
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
fn fold_legal_moves_check_none<const CAPTURES_ONLY: bool, B, F, R>(pos: &Position, init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let init = sliding_piece::gen_legals_check_none(pos, SlidingPieceType::ROOK, rook::compute_attacks).try_fold(init, &mut f)?;
    let init = sliding_piece::gen_legals_check_none(pos, SlidingPieceType::BISHOP, bishop::compute_attacks).try_fold(init, &mut f)?;
    let init = sliding_piece::gen_legals_check_none(pos, SlidingPieceType::QUEEN, queen::compute_attacks).try_fold(init, &mut f)?;
    let init = pawn::fold_legals_check_none::<CAPTURES_ONLY, _, _, _>(pos, init, &mut f)?;
    // let init = pawn::gen_legals_check_none(pos).try_fold(init, &mut f)?;
    let init = jumping_piece::gen_legals_check_none(pos, JumpingPieceType::KNIGHT, knight::compute_attacks).try_fold(init, &mut f)?;
    let init = king::gen_legals_check_none(pos).try_fold(init, &mut f)?;
    king::gen_legal_castling(pos, pos.get_turn()).try_fold(init, f)
}

fn legal_moves_check_none<const CAPTURES_ONLY: bool>(
    pos: &Position,
) -> impl Iterator<Item = Move> + '_ {
    [
        sliding_piece::gen_legals_check_none(pos, SlidingPieceType::ROOK, rook::compute_attacks),
        sliding_piece::gen_legals_check_none(pos, SlidingPieceType::BISHOP, bishop::compute_attacks),
        sliding_piece::gen_legals_check_none(pos, SlidingPieceType::QUEEN, queen::compute_attacks),
    ].into_iter()
    .flatten()
    .chain(pawn::gen_legals_check_none(pos))
    .chain(jumping_piece::gen_legals_check_none(pos, JumpingPieceType::KNIGHT, knight::compute_attacks))
    .chain(king::gen_legals_check_none(pos))
    .chain(king::gen_legal_castling(pos, pos.get_turn()))
}

fn legal_moves_check_single<const CAPTURES_ONLY: bool>(
    pos: &Position,
) -> impl Iterator<Item = Move> + '_ {
    [
        sliding_piece::gen_legals_check_single(pos, SlidingPieceType::ROOK, rook::compute_attacks),
        sliding_piece::gen_legals_check_single(pos, SlidingPieceType::BISHOP, bishop::compute_attacks),
        sliding_piece::gen_legals_check_single(pos, SlidingPieceType::QUEEN, queen::compute_attacks),
    ].into_iter()
    .flatten()
    .chain(king::gen_legals_check_some(pos))
    .chain(pawn::gen_legals_check_single(pos))
    .chain(jumping_piece::gen_legals_check_single(pos, JumpingPieceType::KNIGHT, knight::compute_attacks))
}

fn legal_moves_check_double<const CAPTURES_ONLY: bool>(
    pos: &Position,
) -> impl Iterator<Item = Move> {
    king::gen_legals_check_some(pos)
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
        CheckState::None => fold_legal_moves_check_none::<CAPTURES_ONLY, _, _, _>(pos, init, f),
        CheckState::Single => legal_moves_check_single::<CAPTURES_ONLY>(pos).try_fold(init, f),
        CheckState::Double => legal_moves_check_double::<CAPTURES_ONLY>(pos).try_fold(init, f),
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

// todo: remove when iterator is const
/// Const version of map_bits
const fn map_bits_c(mut bits: usize, mut mask: Bitboard) -> Bitboard {
    let mut acc = Bitboard::empty();
    while let Some(pos) = mask.next() {
        let val = bits & 1;
        bits >>= 1;
        acc.v |= (val << pos.v()) as u64
    }
    acc
}
