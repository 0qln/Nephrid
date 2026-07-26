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
    from_mask_discover_check: Bitboard,
    blockers: Bitboard,
    occupancy: Bitboard,
    king: Option<Square>,
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
    let only_check_mask_quiet = (!O::quiet_nochecks()).then_some(to_mask_quiets & to_mask_checks).unwrap_or_default();

    let only_check_mask_capt = (!O::capture_nochecks()).then_some(to_mask_captures & to_mask_checks).unwrap_or_default();

    let attacks = |piece| {
        let piece_bb = Bitboard::from(piece);
        let attacks = T::lookup_attacks(piece, occupancy);

        if O::legal() {
            let pin_mask = king.map(|k| pin_mask(piece, piece_bb, blockers, k)).unwrap_or(Bitboard::full());
            attacks & pin_mask
        }
        else {
            attacks
        }
    };

    let mut acc = init;

    acc = (pieces & from_mask_discover_check).try_fold(acc, |mut acc, piece| -> R {
        let attacks = attacks(piece);

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

    (pieces & !from_mask_discover_check).try_fold(acc, move |mut acc, piece| {
        let attacks = attacks(piece);

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
    })
}
