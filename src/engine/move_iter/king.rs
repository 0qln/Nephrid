use std::ops::Try;

use crate::{
    engine::{
        bitboard::Bitboard,
        castling::CastlingSide,
        coordinates::{File, Rank, Square},
        piece::PieceType,
        position::Position,
        r#move::{Move, MoveFlag},
    },
    misc::ConstFrom,
};

use const_for::const_for;

use super::{bishop::Bishop, rook::Rook, sliding_piece::Attacks};

pub fn fold_legals_check_some<B, F, R>(pos: &Position, init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let color = pos.get_turn();
    let king_bb = pos.get_bitboard(PieceType::KING, color);
    let occupancy = pos.get_occupancy();
    let occupancy_after_king_move = occupancy ^ king_bb;
    let rooks = pos.get_bitboard(PieceType::ROOK, !color);
    let bishops = pos.get_bitboard(PieceType::BISHOP, !color);
    let queens = pos.get_bitboard(PieceType::QUEEN, !color);
    let rooks_queens = rooks | queens;
    let bishops_queens = bishops | queens;

    fold_legals_check_none(pos, init, |acc, m| {
        // Make sure that we move the king out of check
        let is_legal = {
            let to_sq = m.get_to();

            // When the king has moved and a sliding piece was a checker, the attacks of
            // that sliding piece will have changed.
            // Note that, in no case does a king move cause an enemy attack to get covered
            // without the king being in check after he moved, which is why we can just
            // append the new attacks of the sliding piece to the existing attacks.
            // The new 'nstm_attacks' are not really nstm_attacks, but only reflect nstm_attacks
            // which are relevant to checking whether the our king is in check.

            // If the to square covers anything, it doesn't matter, because the king will be in check.
            // (=> we don't need to add the to square to occupancy)
            let rook_attacks = Rook::lookup_attacks(to_sq, occupancy_after_king_move);
            let bishop_attacks = Bishop::lookup_attacks(to_sq, occupancy_after_king_move);
            let q_or_r_check = !rook_attacks.and_c(rooks_queens).is_empty();
            let q_or_b_check = !bishop_attacks.and_c(bishops_queens).is_empty();
            let new_check = q_or_r_check || q_or_b_check;
            !new_check
        };

        if is_legal {
            f(acc, m)
        } else {
            try { acc }
        }
    })
}

pub fn fold_legals_check_none<B, F, R>(pos: &Position, init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let color = pos.get_turn();
    let nstm_attacks = pos.get_nstm_attacks();
    let occupancy = pos.get_occupancy();
    let enemies = pos.get_color_bb(!color);
    let king_bb = pos.get_bitboard(PieceType::KING, color);
    let king = king_bb.lsb().unwrap();
    let attacks = lookup_attacks(king);
    let targets = attacks & !nstm_attacks;

    let mut quiets = {
        let targets = targets & !occupancy;
        targets.map(move |target| Move::new(king, target, MoveFlag::QUIET))
    };

    let mut captures = {
        let targets = targets & enemies;
        targets.map(move |target| Move::new(king, target, MoveFlag::CAPTURE))
    };

    let acc = quiets.try_fold(init, &mut f)?;
    captures.try_fold(acc, &mut f)
}

pub fn fold_legal_castling<B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let color = pos.get_turn();
    let color_v = color.v() as usize;
    let rank = color * Rank::_8;
    let from = Square::from_c((File::E, rank));
    let castling = pos.get_castling();
    let nstm_attacks = pos.get_nstm_attacks();

    if castling.is_true(CastlingSide::KING_SIDE, color) {
        const TABU_MASK: [Bitboard; 2] = [Bitboard { v: 0x60_u64 }, Bitboard { v: 0x60_u64 << 56 }];
        let tabus = nstm_attacks | pos.get_occupancy();
        // Safety: Color is in range [0..2]
        let tabu_mask = unsafe { TABU_MASK.get_unchecked(color_v) };
        if (tabus & *tabu_mask).is_empty() {
            // Safety: [e1|e8] + 2 < 63
            let to = unsafe { Square::from_v(from.v() + 2) };
            init = f(init, Move::new(from, to, MoveFlag::KING_CASTLE))?;
        }
    }

    if castling.is_true(CastlingSide::QUEEN_SIDE, color) {
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
            return f(init, Move::new(from, to, MoveFlag::QUEEN_CASTLE));
        }
    }

    try { init }
}

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

pub const fn compute_attacks(sq: Square) -> Bitboard {
    let file = File::from_c(sq);
    let rank = Rank::from_c(sq);
    let king = Bitboard::from_c(sq);

    let mut files = Bitboard::from_c(file);
    if file.v() > File::A_C {
        // Safety: file is in range 1.., so file - 1 is still a valid file.
        let west = unsafe { File::from_v(file.v() - 1) };
        files.v |= Bitboard::from_c(west).v;
    }

    if file.v() < File::H_C {
        // Safety: file is in range 0..7, so file + 1 is still a valid file.
        let east = unsafe { File::from_v(file.v() + 1) };
        files.v |= Bitboard::from_c(east).v;
    }

    let mut ranks = Bitboard::from_c(rank);
    if rank.v() > Rank::_1_C {
        // Safety: rank is in range 1.., so rank - 1 is still a valid rank.
        let south = unsafe { Rank::from_v(rank.v() - 1) };
        ranks.v |= Bitboard::from_c(south).v;
    }

    if rank.v() < Rank::_8_C {
        // Safety: rank is in range 0..7, so rank + 1 is still a valid rank.
        let north = unsafe { Rank::from_v(rank.v() + 1) };
        ranks.v |= Bitboard::from_c(north).v;
    }

    files.and_c(ranks).and_not_c(king)
}
