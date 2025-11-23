use std::mem;

use rand::{rngs::SmallRng, RngCore, SeedableRng};

use crate::core::move_iter::{bishop::Bishop, rook::Rook, sliding_piece::SlidingAttacks};

use super::{find_magics, AttackTable, MagicGen};

#[test]
pub fn find_rook_seeds() {
    find_seeds::<Rook>();
}

#[test]
pub fn find_bishop_seeds() {
    find_seeds::<Bishop>();
}

pub fn find_seeds<T: MagicGen + SlidingAttacks>() {
    let mut seed = 0xdead_beef;
    let mut min = u32::MAX;
    loop {
        let mut rng = SmallRng::seed_from_u64(seed);
        let table: &mut AttackTable = unsafe { &mut mem::zeroed() };
        let magics = find_magics::<T>(table, &mut rng, None);
        let cost = magics.iter().map(|m| m.init_cost).sum::<u32>();
        if cost < min {
            min = cost;
            println!("cost: {}, seed: {}", cost, seed);
        }
        seed = SmallRng::seed_from_u64(seed).next_u64();
    }
}
