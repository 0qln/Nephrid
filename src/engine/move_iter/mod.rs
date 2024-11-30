use super::bitboard::Bitboard;
use super::color::Color;
use super::coordinates::Square;
use super::piece::SlidingPieceType;
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

pub fn legal_moves_check_none<const CAPTURES_ONLY: bool>(pos: &Position) -> impl Iterator<Item = Move> {
    // generate pseudo legal moves
    [
        sliding_piece::gen_legal_check_none(pos, SlidingPieceType::ROOK, rook::compute_attacks),
        sliding_piece::gen_legal_check_none(pos, SlidingPieceType::BISHOP, bishop::compute_attacks),
        sliding_piece::gen_legal_check_none(pos, SlidingPieceType::QUEEN, queen::compute_attacks),
    ].into_iter().flatten()
}

pub fn legal_moves_check_single<const CAPTURES_ONLY: bool>(pos: &Position) -> impl Iterator<Item = Move> {
    // only generate legal check resolves
    [
        pawn::gen_legal_check_single(pos),
        king::gen_legal_check_some(pos),
    ].into_iter().flatten()
}

pub fn legal_moves_check_double<const CAPTURES_ONLY: bool>(pos: &Position) -> impl Iterator<Item = Move> {
    // only generate legal check resolves by king
    king::gen_legal_check_some(pos)
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
