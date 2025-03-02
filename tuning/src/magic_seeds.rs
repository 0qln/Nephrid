#![feature(new_zeroed_alloc)]

use std::mem::MaybeUninit;

use rand::{RngCore, SeedableRng, rngs::SmallRng};

use engine::core::move_iter::{
    bishop::Bishop,
    rook::Rook,
    sliding_piece::{
        SlidingAttacks,
        magics::{AttackTable, MagicGen, find_magics},
    },
};

pub fn find_seeds<T: MagicGen + SlidingAttacks>() {
    let mut table: Box<MaybeUninit<AttackTable>> = Box::new_zeroed();
    let table = unsafe { table.assume_init_mut() };
    let mut seed = 0xdead_beef_u64;
    let mut min = u32::MAX;
    loop {
        let mut rng = SmallRng::seed_from_u64(seed);
        let magics = find_magics::<T>(table, &mut rng, None);
        let cost = magics.iter().map(|m| m.cost()).sum::<u32>();
        if cost < min {
            min = cost;
            println!("cost: {}, seed: {}", cost, seed);
        }
        seed = SmallRng::seed_from_u64(seed).next_u64();
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("rook") => find_seeds::<Rook>(),
        Some("bishop") => find_seeds::<Bishop>(),
        _ => panic!("Specify either 'rook' or 'bishop'"),
    };
}
