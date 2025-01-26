use std::ops::Try;

use crate::{
    engine::{
        bitboard::Bitboard,
        coordinates::{CompassRose, Square, TCompassRose},
        move_iter::{gen_captures, gen_quiets},
        piece::PieceType,
        position::{CheckState, Position},
        r#move::Move,
    },
    misc::ConstFrom,
};

use const_for::const_for;

pub fn fold_legals_check_none<B, F, R>(pos: &Position, init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    debug_assert_eq!(pos.get_check_state(), CheckState::None);

    let color = pos.get_turn();
    let enemies = pos.get_color_bb(!color);
    let allies = pos.get_color_bb(color);
    let blockers = pos.get_blockers();

    // Safety: the board has no king, but gen_legal is used, the context is broken anyway.
    let king_bb = pos.get_bitboard(PieceType::KING, color);
    let king = unsafe { king_bb.lsb().unwrap_unchecked() };

    pos.get_bitboard(PieceType::KNIGHT, color)
        .filter_map(|from| {
            let from_bb = Bitboard::from_c(from);
            let is_not_blocker = (blockers & from_bb).is_empty();
            is_not_blocker.then(|| (lookup_attacks(from), from))
        })
        .try_fold(init, |mut acc, (attacks, from)| {
            acc = gen_captures(attacks, enemies, from).try_fold(acc, &mut f)?;
            gen_quiets(attacks, enemies, allies, from).try_fold(acc, &mut f)
        })
}

#[inline]
pub fn lookup_attacks(sq: Square) -> Bitboard {
    static ATTACKS: [Bitboard; 64] = {
        let mut attacks = [Bitboard::empty(); 64];
        const_for!(sq in Square::A1_C..(Square::H8_C+1) => {
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
pub const fn compute_attacks(sq: Square) -> Bitboard {
    let knight = Bitboard::from_c(sq);
    compute_attacks_multiple(knight)
}

const fn compute_attacks_multiple(knights: Bitboard) -> Bitboard {
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
const fn compute_atttack<const DIR: TCompassRose>(knight: Bitboard, attacks: &mut Bitboard) {
    let attack_sqrs = Bitboard::from_c(CompassRose::new(-DIR));
    attacks.v |= knight.and_c(attack_sqrs).shift(CompassRose::new(DIR)).v;
}
