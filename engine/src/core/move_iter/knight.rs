use std::ops::Try;

use crate::core::{
    bitboard::{Bitboard, BitboardIteratorExt},
    color::Perspective,
    coordinates::{CompassRose, Square, TCompassRose, compass_rose, squares},
    r#move::Move,
    move_iter::{map_captures, map_quiets},
};

use const_for::const_for;

use super::Options;

pub struct Knight;

#[inline(always)]
pub fn fold_moves_for<B, F, R, P: Perspective, O: Options>(
    knights: Bitboard,
    from_mask_discover_check: Bitboard,
    blockers: Bitboard,
    to_mask_captures: Bitboard,
    to_mask_quiets: Bitboard,
    to_mask_checks: Bitboard,
    init: B,
    mut f: F,
) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let knights = {
        if O::legal() {
            // only unpinned knights if legal check required
            knights & !blockers
        }
        else {
            knights
        }
    };

    let only_check_mask_quiet = (!O::quiet_nochecks()).then_some(to_mask_quiets & to_mask_checks).unwrap_or_default();

    let only_check_mask_capt = (!O::capture_nochecks()).then_some(to_mask_captures & to_mask_checks).unwrap_or_default();

    let mut acc = init;

    // todo: if the user only wants checks and nothing else, we can optimize by
    // lookup up the attacks from the enemy king. // or can we ? maybe not cause
    // of discovered checks dklfjsdf this was an old comment grrr technical debt
    // i luv uuu uwu :3
    acc = (knights & from_mask_discover_check).try_fold(acc, |mut acc, piece| -> R {
        let attacks = lookup_attacks(piece);

        // todo: two loops and generate ALL capture first ?

        if O::gen_captures() {
            let target_mask = to_mask_captures;
            acc = map_captures(attacks & target_mask, piece).try_fold(acc, &mut f)?;
        };

        if O::gen_quiets() {
            let target_mask = to_mask_quiets;
            acc = map_quiets(attacks & target_mask, piece).try_fold(acc, &mut f)?;
        }

        try { acc }
    })?;

    acc = (knights & !from_mask_discover_check).try_fold(acc, move |mut acc, piece| -> R {
        let attacks = lookup_attacks(piece);

        if O::gen_captures() {
            let target_mask = if O::capture_nochecks() {
                to_mask_captures
            }
            else {
                only_check_mask_capt
            };
            acc = map_captures(attacks & target_mask, piece).try_fold(acc, &mut f)?;
        };

        if O::gen_quiets() {
            let target_mask = if O::quiet_nochecks() {
                to_mask_quiets
            }
            else {
                only_check_mask_quiet
            };
            acc = map_quiets(attacks & target_mask, piece).try_fold(acc, &mut f)?;
        }

        try { acc }
    })?;

    try { acc }
}

#[inline]
pub const fn lookup_attacks(sq: Square) -> Bitboard {
    static ATTACKS: [Bitboard; 64] = {
        let mut attacks = [Bitboard::empty(); 64];
        const_for!(sq in squares::A1_C..(squares::H8_C+1) => {
            // Safety: we are only iterating over valid squares.
            let sq = unsafe { Square::from_v(sq) };
            attacks[sq.v() as usize] = compute_attacks(sq);
        });
        attacks
    };
    // Safety: sq is in range 0..64
    unsafe { *ATTACKS.get_unchecked(sq.v() as usize) }
}

#[inline]
pub fn lookup_attacks_multiple(knights: Bitboard) -> Bitboard { knights.map(lookup_attacks).aggregate() }

#[inline]
pub const fn compute_attacks(sq: Square) -> Bitboard {
    let knight = Bitboard::from(sq);
    compute_attacks_multiple(knight)
}

pub const fn compute_attacks_multiple(knights: Bitboard) -> Bitboard {
    let mut result = Bitboard::empty();
    compute_atttack::<{ compass_rose::NONOWE_C }>(knights, &mut result);
    compute_atttack::<{ compass_rose::NONOEA_C }>(knights, &mut result);
    compute_atttack::<{ compass_rose::NOWEWE_C }>(knights, &mut result);
    compute_atttack::<{ compass_rose::NOEAEA_C }>(knights, &mut result);
    compute_atttack::<{ compass_rose::SOSOWE_C }>(knights, &mut result);
    compute_atttack::<{ compass_rose::SOSOEA_C }>(knights, &mut result);
    compute_atttack::<{ compass_rose::SOWEWE_C }>(knights, &mut result);
    compute_atttack::<{ compass_rose::SOEAEA_C }>(knights, &mut result);
    result
}

#[inline]
const fn compute_atttack<const DIR: TCompassRose>(knight: Bitboard, attacks: &mut Bitboard) {
    let attack_sqrs = Bitboard::from(CompassRose::new(-DIR));
    attacks.v |= knight.and_c(attack_sqrs).shift(CompassRose::new(DIR)).v;
}
