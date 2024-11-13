use super::bitboard::Bitboard;
use super::color::Color;
use super::r#move::Move;
use super::position::Position;

pub mod bishop;
pub mod king;
pub mod knight;
pub mod pawn;
pub mod queen;
pub mod rook;

pub fn legal_moves<'a>(pos: &'a Position) -> impl Iterator<Item = Move> + 'a {
    pawn::gen_pseudo_legals::<{Color::WHITE.v}>(pos)
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
const fn map_bits_c(mut bits: usize, mut mask: Bitboard) -> Bitboard {
    let mut acc = Bitboard::empty();
    while let Some(pos) = mask.next() {
        let val = bits & 1;
        bits >>= 1;
        acc.v |= (val << pos.v()) as u64
    }
    acc
}
