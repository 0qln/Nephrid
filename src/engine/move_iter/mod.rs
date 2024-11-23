use super::bitboard::Bitboard;
use super::color::Color;
use super::position::Position;
use super::r#move::Move;

pub mod bishop;
pub mod king;
pub mod knight;
pub mod pawn;
pub mod queen;
pub mod rook;

pub fn legal_moves(pos: &Position) -> impl Iterator<Item = Move>
{
    // todo: filter legal moves
    //
    Iterator::chain(
        pawn::gen_pseudo_legals::<{ Color::WHITE.v() }>(pos),
        knight::gen_pseudo_legals::<{ Color::WHITE.v() }>(pos),
    )
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
