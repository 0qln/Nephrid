use crate::{engine::{bitboard::Bitboard, coordinates::{CompassRose, Square, TCompassRose}}, misc::ConstFrom};

const fn compute_attacks(sq: Square) -> Bitboard {
    let mut result = Bitboard::empty();
    let knight = Bitboard::from_c(sq);
    
    compute_atttack::<{CompassRose::NONOWE_C}>(knight, &mut result);
    compute_atttack::<{CompassRose::NONOEA_C}>(knight, &mut result);
    compute_atttack::<{CompassRose::NOWEWE_C}>(knight, &mut result);
    compute_atttack::<{CompassRose::NOEAEA_C}>(knight, &mut result);
    compute_atttack::<{CompassRose::SOSOWE_C}>(knight, &mut result);
    compute_atttack::<{CompassRose::SOSOEA_C}>(knight, &mut result);
    compute_atttack::<{CompassRose::SOWEWE_C}>(knight, &mut result);
    compute_atttack::<{CompassRose::SOEAEA_C}>(knight, &mut result);
    
    return result;
}

#[inline]
const fn compute_atttack<const DIR: TCompassRose>(knight: Bitboard, attacks: &mut Bitboard) {
    let attack_sqrs = Bitboard::full().shift(CompassRose::new(-DIR));
    attacks.v |= knight.and_c(attack_sqrs).shift(CompassRose::new(DIR)).v;
}