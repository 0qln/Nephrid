use std::ops::Try;

use crate::core::{
    bitboard::Bitboard,
    coordinates::Square,
    r#move::Move,
    move_iter::{Options, captures_targets, quiets_targets},
    piece::{IPieceType, piece_type},
    position::Position,
};

use super::{FoldMoves, NoDoubleCheck, map_captures, map_quiets, pin_mask};

pub mod magics;

pub trait SlidingAttacks {
    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard;
    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard;
}

pub trait SlidingPieceType: SlidingAttacks + IPieceType {}

impl<O: Options, C, T> FoldMoves<C, O> for T
where
    C: NoDoubleCheck,
    T: SlidingPieceType,
{
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        let color = pos.get_turn();
        let our_king = pos.get_bitboard(piece_type::KING, color).lsb();

        pos.get_bitboard(T::ID, color)
            .try_fold(init, move |mut acc, piece| {
                let attacks = {
                    let occupancy = pos.get_occupancy();
                    let attacks = T::lookup_attacks(piece, occupancy);
                    if O::legal() {
                        let blockers = pos.get_blockers();
                        let pin_mask = our_king
                            .map(|k| pin_mask(piece, blockers, k))
                            .unwrap_or(Bitboard::full());
                        attacks & pin_mask
                    }
                    else {
                        attacks
                    }
                };

                if O::gen_captures() {
                    let legal_captures = attacks & captures_targets::<C>(pos, color);
                    acc = map_captures(legal_captures, piece).try_fold(acc, &mut f)?;
                }

                if O::gen_quiets() {
                    let legal_quiets = attacks & quiets_targets::<C>(pos, color);
                    acc = map_quiets(legal_quiets, piece).try_fold(acc, &mut f)?;
                }

                try { acc }
            })
    }
}
