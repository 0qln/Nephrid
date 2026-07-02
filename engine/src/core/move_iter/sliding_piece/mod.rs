use std::ops::Try;

use crate::core::{
    bitboard::Bitboard, color::Perspective, coordinates::Square, r#move::Move, move_iter::{Options, captures_targets, quiets_targets}, piece::{IPieceType, piece_type}, position::Position
};

use super::{FoldMoves, NoDoubleCheck, map_captures, map_quiets, pin_mask};

pub mod magics;

pub trait SlidingAttacks {
    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard;
    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard;
}

pub trait SlidingPieceType: SlidingAttacks + IPieceType {}

impl<P: Perspective, O: Options, C, T> FoldMoves<P, C, O> for T
where
    C: const NoDoubleCheck,
    T: SlidingPieceType,
{
    #[inline(always)]
    fn fold_moves_for<B, F, R>(pos: &Position, init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        let our_king = pos.get_bitboard(piece_type::KING, P::COLOR).lsb();

        pos.get_bitboard(T::ID, P::COLOR).try_fold(init, move |mut acc, piece| {
            let attacks = {
                let occupancy = pos.get_occupancy();
                let attacks = T::lookup_attacks(piece, occupancy);
                if O::legal() {
                    let blockers = pos.get_blockers();
                    let pin_mask = our_king.map(|k| pin_mask(piece, blockers, k)).unwrap_or(Bitboard::full());
                    attacks & pin_mask
                }
                else {
                    attacks
                }
            };

            if O::gen_captures() {
                let legal_captures = attacks & captures_targets::<C>(pos, P::COLOR);
                acc = map_captures(legal_captures, piece).try_fold(acc, &mut f)?;
            }

            if O::gen_quiets() {
                let legal_quiets = attacks & quiets_targets::<C>(pos, P::COLOR);
                acc = map_quiets(legal_quiets, piece).try_fold(acc, &mut f)?;
            }

            try { acc }
        })
    }
}
