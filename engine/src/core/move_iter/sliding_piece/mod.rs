use std::ops::Try;

use crate::core::{
    bitboard::{Bitboard, BitboardIteratorExt},
    color::Perspective,
    coordinates::Square,
    r#move::Move,
    move_iter::Options,
    piece::IPieceType,
};

use super::{map_captures, map_quiets, pin_mask};

pub mod magics;

pub trait SlidingAttacks {
    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard;
    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard;
    fn lookup_attacks_multiple(pieces: Bitboard, occupancy: Bitboard) -> Bitboard { pieces.map(|p| Self::lookup_attacks(p, occupancy)).aggregate() }
}

pub trait SlidingPieceType: SlidingAttacks + IPieceType {}

#[inline(always)]
pub fn fold_moves_for<B, F, R, P: Perspective, O: Options, T: SlidingPieceType>(
    pieces: Bitboard,
    from_mask_quiets: Bitboard,
    from_mask_captures: Bitboard,
    blockers: Bitboard,
    occupancy: Bitboard,
    king: Option<Square>,
    to_mask_captures: Bitboard,
    to_mask_quiets: Bitboard,
    to_mask_capture_checks: Bitboard,
    to_mask_quiet_checks: Bitboard,
    init: B,
    mut f: F,
) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let mut pieces = pieces;

    pieces.try_fold(init, move |mut acc, piece| {
        let piece_bb = Bitboard::from(piece);

        let attacks = {
            let attacks = T::lookup_attacks(piece, occupancy);

            if O::legal() {
                let pin_mask = king.map(|k| pin_mask(piece, piece_bb, blockers, k)).unwrap_or(Bitboard::full());
                attacks & pin_mask
            }
            else {
                attacks
            }
        };

        if O::gen_captures() {
            let target_mask = if O::capture_nochecks() || from_mask_captures.contains(piece_bb) {
                to_mask_captures
            }
            else {
                to_mask_captures & to_mask_capture_checks
            };
            acc = map_captures(attacks & target_mask, piece).try_fold(acc, &mut f)?;
        };

        if O::gen_quiets() {
            let target_mask = if O::quiet_nochecks() || from_mask_quiets.contains(piece_bb) {
                to_mask_quiets
            }
            else {
                to_mask_quiets & to_mask_quiet_checks
            };
            acc = map_quiets(attacks & target_mask, piece).try_fold(acc, &mut f)?;
        }

        try { acc }
    })
}
