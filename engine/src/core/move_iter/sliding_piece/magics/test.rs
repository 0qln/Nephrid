use std::iter::once;

use crate::core::{
    coordinates::Square,
    move_iter::{bishop::Bishop, map_bits, rook::Rook, sliding_piece::SlidingAttacks},
};

use super::MagicGen;

#[test]
fn rook() {
    test::<Rook>()
}

#[test]
fn bishop() {
    test::<Bishop>()
}

fn test<T: MagicGen + SlidingAttacks>() {
    super::init();

    for sq in Square::A1..=Square::H8 {
        let full_blockers = T::relevant_occupancy(sq);
        let max_size = T::relevant_occupancy_num_combinations();
        let blockers = (0..max_size)
            .map(|x| map_bits(x, full_blockers))
            .take_while(|&x| x != full_blockers)
            .chain(once(full_blockers));

        for blockers in blockers {
            let compute = T::compute_attacks(sq, blockers);
            let lookup = T::lookup_attacks(sq, blockers);
            assert_eq!(compute, lookup, "sq: {}, blockers: {:?}", sq, blockers);
        }
    }
}
