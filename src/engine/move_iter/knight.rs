use crate::{
    engine::{
        bitboard::Bitboard,
        coordinates::{CompassRose, Square, TCompassRose},
    },
    misc::ConstFrom,
};

// todo: the attacks can be precomputed.

#[inline]
pub /* const */ fn compute_attacks(sq: Square) -> Bitboard {
    let knight = Bitboard::from_c(sq);
    compute_attacks_multiple(knight)
}

/* const  */fn compute_attacks_multiple(knights: Bitboard) -> Bitboard {
    let mut result = Bitboard::empty();
    compute_atttack::<{ CompassRose::NONOWE_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::NONOEA_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::NOWEWE_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::NOEAEA_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::SOSOWE_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::SOSOEA_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::SOWEWE_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::SOEAEA_C }>(knights, &mut result);
    result
}

#[inline]
/* const  */fn compute_atttack<const DIR: TCompassRose>(knight: Bitboard, attacks: &mut Bitboard) {
    let attack_sqrs = Bitboard::from_c(CompassRose::new(-DIR));
    attacks.v |= knight.and_c(attack_sqrs).shift(CompassRose::new(DIR)).v;
    // println!("{:?}", knight);
    // println!("{:?}", attacks);
}
