use std::ops::Try;

use crate::core::{bitboard::Bitboard, color::Perspective, coordinates::Square, r#move::Move, move_iter::Options, piece::IPieceType};

use super::{map_captures, map_quiets, pin_mask};

pub mod magics;

pub trait SlidingAttacks {
    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard;
    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard;
}

pub trait SlidingPieceType: SlidingAttacks + IPieceType {}

#[inline(always)]
pub fn fold_moves_for<B, F, R, P: Perspective, O: Options, T: SlidingPieceType>(
    pieces: Bitboard,
    blockers: Bitboard,
    occupancy: Bitboard,
    king: Option<Square>,
    capture_targets: Bitboard,
    quiet_targets: Bitboard,
    init: B,
    mut f: F,
) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let mut pieces = pieces;

    pieces.try_fold(init, move |mut acc, piece| {
        let attacks = {
            let attacks = T::lookup_attacks(piece, occupancy);

            if O::legal() {
                let pin_mask = king.map(|k| pin_mask(piece, blockers, k)).unwrap_or(Bitboard::full());
                attacks & pin_mask
            }
            else {
                attacks
            }
        };

        if O::gen_captures() {
            acc = map_captures(attacks & capture_targets, piece).try_fold(acc, &mut f)?;
        };

        if O::gen_quiets() {
            acc = map_quiets(attacks & quiet_targets, piece).try_fold(acc, &mut f)?;
        }

        try { acc }
    })
}
