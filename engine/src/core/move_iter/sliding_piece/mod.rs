use std::ops::Try;

use crate::core::{
    bitboard::Bitboard, coordinates::Square, piece::IPieceType, position::Position, r#move::Move,
};

use super::{map_captures, map_quiets, pin_mask, FoldMoves, NoDoubleCheck};

pub mod magics;

pub trait SlidingAttacks {
    fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard;
    fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard;
}

pub trait SlidingPieceType: SlidingAttacks + IPieceType {}

impl<C, T> FoldMoves<C> for T
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

        // Safety: there is a single checker.
        pos.get_bitboard(T::ID, color)
            .try_fold(init, move |mut acc, piece| {
                let occupancy = pos.get_occupancy();
                let attacks = T::lookup_attacks(piece, occupancy);
                let legal_attacks = attacks & pin_mask(pos, piece);

                let legal_quiets = legal_attacks & C::quiets_mask(pos, color);
                let legal_captures = legal_attacks & C::captures_mask(pos, color);

                acc = map_captures(legal_captures, piece).try_fold(acc, &mut f)?;
                map_quiets(legal_quiets, piece).try_fold(acc, &mut f)
            })
    }
}
