use crunchy::unroll;

use super::{bitboard::Bitboard, coordinates::Square};
use std::ops;

pub mod bishop;
pub mod king;
pub mod knight;
pub mod pawn;
pub mod queen;
pub mod rook;

// pub fn gen_plegals<'a>(pos: &'a Position) -> impl Iterator<Item = Move> + 'a {
//     let moves: &[Move] = &[];
// }
//
// pub fn legal_moves<'a>(pos: &'a Position) -> impl Iterator<Item = Move> + 'a {
//     todo!()
// }

const fn get_key(relevant_occupancy: Bitboard, magic: MagicData, bits: MagicBits) -> MagicKey {
    ((((relevant_occupancy.v as i64).wrapping_mul(magic)) >> (64 - bits)) & 0xFFFFFFFF) as MagicKey
}

// todo: make actual wrappers
pub type MagicBits = usize;
pub type MagicData = i64;
pub type MagicKey = usize;

// todo: can be removed when const fn pointers are released.
#[const_trait]
pub trait SlidingPiece {
    fn relevant_occupancy(sq: Square) -> Bitboard;
    fn compute_attacks(sq: Square, occupied: Bitboard) -> Bitboard;
    fn get_magic(sq: Square) -> MagicData;
    fn get_bits(sq: Square) -> MagicBits;
}

const fn initialize_attacks<T: const SlidingPiece, const N: usize>(
    mut buffer: [[Bitboard; N]; 64]
) -> [[Bitboard; N]; 64] {
    // todo: clean up when const iterators are released.
    unroll! {
        for sq in 0..64 {
            // Safety: the square is valid
            const SQ: Square = unsafe { Square::from_v(sq as u8) };
            let max_blockers = T::relevant_occupancy(SQ);
            let num_blocker_compositions = 1 << max_blockers.v.count_ones();
            let mut i = 0;
            while i < num_blocker_compositions {
                {
                    let occupied = map_bits_c(i, max_blockers);
                    let attacks = T::compute_attacks(SQ, occupied);
                    let key = get_key(occupied, T::get_magic(SQ), T::get_bits(SQ));
                    buffer[sq][key] = attacks;
                }
                i += 1;
            }
        }
    }
    buffer
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
