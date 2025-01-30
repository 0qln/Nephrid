use std::ops::Try;

use crate::{
    engine::{
        bitboard::Bitboard,
        coordinates::{CompassRose, Square, TCompassRose},
        move_iter::{map_captures, map_quiets},
        piece::PieceType,
        position::Position,
        r#move::Move,
    },
    misc::ConstFrom,
};

use const_for::const_for;

use super::{is_blocker, FoldMoves, NoDoubleCheck};

pub struct Knight;

impl<C: NoDoubleCheck> FoldMoves<C> for Knight {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B> 
    {
        let color = pos.get_turn();

        pos.get_bitboard(PieceType::KNIGHT, color)
            .filter(|&piece| (!is_blocker(pos, piece)))
            .map(|piece| {
                let legal_attacks = lookup_attacks(piece);
                let legal_quiets = legal_attacks & C::quiets_mask(pos, color);
                let legal_captures = legal_attacks & C::captures_mask(pos, color);
                (legal_captures, legal_quiets, piece)
            })
            .try_fold(init, |mut acc, (captures, quiets, from)| {
                acc = map_captures(captures, from).try_fold(acc, &mut f)?;
                map_quiets(quiets, from).try_fold(acc, &mut f)
            })
    }
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
