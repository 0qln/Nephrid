use std::{hint, ops::Try};

use crate::core::{
    bitboard::{Bitboard, BitboardIteratorExt},
    castling::castling_sides,
    coordinates::{File, Rank, Square, files, ranks, squares},
    r#move::{Move, move_flags},
    move_iter::{captures_targets, king, knight, pawn, quiets_targets},
    piece::{IPieceType, PieceType, piece_type},
    position::Position,
};

use const_for::const_for;

use super::{
    FoldMoves, NoCheck, Options, SomeCheck, bishop::Bishop, map_captures, map_quiets, rook::Rook,
    sliding_piece::SlidingAttacks,
};

pub struct King;

impl IPieceType for King {
    const ID: PieceType = piece_type::KING;
}

impl<O: Options> FoldMoves<NoCheck, O> for King {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        let color = pos.get_turn();

        let king_bb = pos.get_bitboard(King::ID, color);
        if let Some(king) = king_bb.lsb() {
            if O::gen_quiets() {
                init = king::fold_legal_castling(pos, init, &mut f)?;
            }

            let enemy_attacks = nstm_attacks(pos, pos.get_occupancy());

            let attacks = lookup_attacks(king);
            let legal_attacks = attacks & !enemy_attacks;

            let legal_captures = legal_attacks & captures_targets::<NoCheck>(pos, color);
            let legal_quiets = legal_attacks & quiets_targets::<NoCheck>(pos, color);

            init = map_captures(legal_captures, king).try_fold(init, &mut f)?;

            if O::gen_quiets() {
                init = map_quiets(legal_quiets, king).try_fold(init, &mut f)?;
            }

            try { init }
        }
        else {
            // legal positions should have a king...
            hint::cold_path();
            try { init }
        }
    }
}

impl<O: Options, C: SomeCheck> FoldMoves<C, O> for King {
    #[inline(always)]
    fn fold_moves<B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Move) -> R,
        R: Try<Output = B>,
    {
        let color = pos.get_turn();

        let king_bb = pos.get_bitboard(King::ID, color);
        if let Some(king) = king_bb.lsb() {
            // If the to square covers anything, it doesn't matter, because the king will be
            // in check. (=> we don't need to add the to square to occupancy)
            let occupancy_after_king_move = pos.get_occupancy() ^ king_bb;
            let enemy_attacks = nstm_attacks(pos, occupancy_after_king_move);
            let legal_attacks = lookup_attacks(king) & !enemy_attacks;

            let legal_captures = legal_attacks & captures_targets::<NoCheck>(pos, color);
            let legal_quiets = legal_attacks & quiets_targets::<NoCheck>(pos, color);

            init = map_captures(legal_captures, king).try_fold(init, &mut f)?;

            if O::gen_quiets() {
                init = map_quiets(legal_quiets, king).try_fold(init, &mut f)?;
            }

            try { init }
        }
        else {
            // legal positions should have a king...
            hint::cold_path();
            try { init }
        }
    }
}

fn nstm_attacks(pos: &Position, occupancy: Bitboard) -> Bitboard {
    let stm = pos.get_turn();
    let nstm = !stm;

    let pawns = pos.get_bitboard(piece_type::PAWN, !pos.get_turn());
    let knights = pos.get_bitboard(piece_type::KNIGHT, !pos.get_turn());
    let bishops = pos.get_bitboard(piece_type::BISHOP, !pos.get_turn());
    let rooks = pos.get_bitboard(piece_type::ROOK, !pos.get_turn());
    let queens = pos.get_bitboard(piece_type::QUEEN, !pos.get_turn());
    let b_n_q = bishops | queens;
    let r_n_q = rooks | queens;
    let king = pos.get_bitboard(piece_type::KING, !pos.get_turn());

    let bishop_attacks = |sq| Bishop::lookup_attacks(sq, occupancy);
    let rook_attacks = |sq| Rook::lookup_attacks(sq, occupancy);

    pawn::compute_attacks(pawns, nstm)
        | knights.into_iter().map(knight::lookup_attacks).aggregate()
        | b_n_q.into_iter().map(bishop_attacks).aggregate()
        | r_n_q.into_iter().map(rook_attacks).aggregate()
        | king.lsb().map(self::lookup_attacks).unwrap_or_default()
}

pub fn fold_legal_castling<B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let color = pos.get_turn();
    let color_v = color.v() as usize;
    let rank = color * ranks::_8;
    let from = Square::from((files::E, rank));
    let castling = pos.get_castling();

    // todo: this is computed twice, fix this
    let king_bb = pos.get_bitboard(King::ID, color);
    let occupancy_after_king_move = pos.get_occupancy() ^ king_bb;
    let nstm_attacks = nstm_attacks(pos, occupancy_after_king_move);

    if castling.is_true(castling_sides::KING_SIDE, color) {
        const TABU_MASK: [Bitboard; 2] = [Bitboard { v: 0x60_u64 }, Bitboard { v: 0x60_u64 << 56 }];
        let tabus = nstm_attacks | pos.get_occupancy();
        // Safety: Color is in range [0..2]
        let tabu_mask = unsafe { TABU_MASK.get_unchecked(color_v) };
        if (tabus & *tabu_mask).is_empty() {
            // Safety: [e1|e8] + 2 < 63
            let to = unsafe { Square::from_v(from.v() + 2) };
            init = f(init, Move::new(from, to, move_flags::KING_CASTLE))?;
        }
    }

    if castling.is_true(castling_sides::QUEEN_SIDE, color) {
        const BLOCK_MASK: [Bitboard; 2] = [Bitboard { v: 0xE_u64 }, Bitboard { v: 0xE_u64 << 56 }];
        const CHECK_MASK: [Bitboard; 2] = [Bitboard { v: 0xC_u64 }, Bitboard { v: 0xC_u64 << 56 }];
        // Safety: Color is in range [0..2]
        let block_mask = unsafe { BLOCK_MASK.get_unchecked(color_v) };
        let check_mask = unsafe { CHECK_MASK.get_unchecked(color_v) };
        let blockers = pos.get_occupancy();
        let blocked = *block_mask & blockers;
        let checked = *check_mask & nstm_attacks;
        if (blocked | checked).is_empty() {
            // Safety: [e1|e8] - 2 > 0
            let to = unsafe { Square::from_v(from.v() - 2) };
            return f(init, Move::new(from, to, move_flags::QUEEN_CASTLE));
        }
    }

    try { init }
}

pub fn lookup_attacks(sq: Square) -> Bitboard {
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

pub const fn compute_attacks(sq: Square) -> Bitboard {
    let file = File::from(sq);
    let rank = Rank::from(sq);
    let king = Bitboard::from(sq);

    let mut files = Bitboard::from(file);
    if file.v() > files::A_C {
        // Safety: file is in range 1.., so file - 1 is still a valid file.
        let west = unsafe { File::from_v(file.v() - 1) };
        files.v |= Bitboard::from(west).v;
    }

    if file.v() < files::H_C {
        // Safety: file is in range 0..7, so file + 1 is still a valid file.
        let east = unsafe { File::from_v(file.v() + 1) };
        files.v |= Bitboard::from(east).v;
    }

    let mut ranks = Bitboard::from(rank);
    if rank.v() > ranks::_1_C {
        // Safety: rank is in range 1.., so rank - 1 is still a valid rank.
        let south = unsafe { Rank::from_v(rank.v() - 1) };
        ranks.v |= Bitboard::from(south).v;
    }

    if rank.v() < ranks::_8_C {
        // Safety: rank is in range 0..7, so rank + 1 is still a valid rank.
        let north = unsafe { Rank::from_v(rank.v() + 1) };
        ranks.v |= Bitboard::from(north).v;
    }

    files.and_c(ranks).and_not_c(king)
}
