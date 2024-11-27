use super::bitboard::Bitboard;
use super::color::Color;
use super::coordinates::Square;
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

// todo: split up each case and only match in the when needed (see king legal moves)
// todo: find a better way of returning the iterator
pub fn legal_moves<const CAPTURES_ONLY: bool>(pos: &Position, color: Color) -> Box<dyn Iterator<Item = Move>> 
{
    let candidates: Box<dyn Iterator<Item = Move>> = match pos.get_check_state() {
        CheckState::Single => Box::new(check_resolves::<{ CAPTURES_ONLY }>(pos, color)),
        CheckState::Double => Box::new(check_resolves_by_king::<{ CAPTURES_ONLY }>(pos, color)),
        CheckState::None => Box::new(pseudo_legal_moves::<{ CAPTURES_ONLY }>(pos, color))
    };
    Box::new(candidates.filter(|m| is_legal_move(pos, m)))
}

fn is_legal_move(pos: &Position, m: &Move) -> bool {
    todo!()
}

fn pseudo_legal_moves<const CAPTURES_ONLY: bool>(pos: &Position, color: Color) -> impl Iterator<Item = Move> {
    todo!()
}

fn check_resolves_by_king<const CAPTURES_ONLY: bool>(pos: &Position, color: Color) -> impl Iterator<Item = Move> {
    todo!()
}

fn check_resolves<const CAPTURES_ONLY: bool>(pos: &Position, color: Color) -> impl Iterator<Item = Move> {
    todo!()
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
