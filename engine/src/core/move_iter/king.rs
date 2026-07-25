use std::{hint, ops::Try};

use crate::core::{
    bitboard::{Bitboard, BitboardIteratorExt},
    castling::castling_sides,
    color::Perspective,
    coordinates::{File, Rank, Square, files, ranks, squares},
    r#move::{Move, move_flags},
    move_iter::{CheckState, DoubleCheck, RtCheckState, SingleCheck, king, knight, pawn},
    piece::{IPieceType, PieceType, piece_type},
    position::Position,
};

use const_for::const_for;

use super::{Options, SomeCheck, bishop::Bishop, map_captures, map_quiets, rook::Rook, sliding_piece::SlidingAttacks};

pub struct King;

impl IPieceType for King {
    const ID: PieceType = piece_type::KING;
}

#[inline(always)]
pub fn fold_moves_for<B, F, R, P: Perspective, O: Options, C: CheckState>(
    king: Option<Square>,
    king_bb: Bitboard,
    pos: &Position,
    occ: Bitboard,
    enemies: Bitboard,
    captures: Bitboard,
    quiets: Bitboard,
    init: B,
    f: F,
) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    match C::check_state() {
        RtCheckState::None => fold_moves_for_nocheck::<B, F, R, P, O>(king, pos, occ, captures, quiets, init, f),
        RtCheckState::Single => fold_moves_for_somecheck::<B, F, R, P, O, SingleCheck>(pos, king_bb, king, enemies, occ, quiets, init, f),
        RtCheckState::Double => fold_moves_for_somecheck::<B, F, R, P, O, DoubleCheck>(pos, king_bb, king, enemies, occ, quiets, init, f),
    }
}

#[inline(always)]
pub fn fold_moves_for_nocheck<B, F, R, P: Perspective, O: Options>(
    king: Option<Square>,
    pos: &Position,
    occupancy: Bitboard,
    capture_targets: Bitboard,
    quiet_targets: Bitboard,
    mut init: B,
    mut f: F,
) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    // todo
    // the king cannot check.
    // if O::gen_only_checks() {
    //     return try { init };
    // }

    if let Some(king) = king {
        let (attacks, enemy_attacks) = if O::legal() {
            let nstm_attacks = nstm_attacks_for::<P>(pos, occupancy);
            (lookup_attacks(king) & !nstm_attacks, nstm_attacks)
        }
        else {
            (lookup_attacks(king), Bitboard::empty())
        };

        if O::gen_captures() {
            init = map_captures(attacks & capture_targets, king).try_fold(init, &mut f)?;
        }

        if O::quiet_nochecks() {
            if O::legal() {
                init = king::fold_legal_castling::<P, _, _, _>(pos, init, &mut f, enemy_attacks)?;
            }
            else {
                init = king::fold_pseudo_legal_castling::<P, _, _, _>(pos, init, &mut f)?;
            }

            init = map_quiets(attacks & quiet_targets, king).try_fold(init, &mut f)?;
        }

        try { init }
    }
    else {
        // legal positions should have a king...
        hint::cold_path();
        try { init }
    }
}

#[inline(always)]
pub fn fold_moves_for_somecheck<B, F, R, P: Perspective, O: Options, C: SomeCheck>(
    pos: &Position,
    king_bb: Bitboard,
    king: Option<Square>,
    enemies: Bitboard,
    occ: Bitboard,
    quiets: Bitboard,
    mut init: B,
    mut f: F,
) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    // Safety: we are in some kind of check, so the king has to exist.
    let king = unsafe { king.unwrap_unchecked() };

    let attacks = if O::legal() {
        // If the to square covers anything, it doesn't matter, because the king will be
        // in check. (=> we don't need to add the to square to occupancy)
        let occupancy_after_king_move = occ ^ king_bb;
        let enemy_attacks = nstm_attacks_for::<P>(pos, occupancy_after_king_move);
        lookup_attacks(king) & !enemy_attacks
    }
    else {
        lookup_attacks(king)
    };

    if O::gen_captures() {
        init = map_captures(attacks & enemies, king).try_fold(init, &mut f)?;
    }

    if O::gen_quiets() {
        init = map_quiets(attacks & quiets, king).try_fold(init, &mut f)?;
    }

    try { init }
}

#[inline(always)]
pub fn nstm_attacks_for<P: Perspective>(pos: &Position, occupancy: Bitboard) -> Bitboard {
    let nstm = P::Opponent::COLOR;

    let pawns = pos.get_bitboard(piece_type::PAWN, nstm);
    let knights = pos.get_bitboard(piece_type::KNIGHT, nstm);
    let bishops = pos.get_bitboard(piece_type::BISHOP, nstm);
    let rooks = pos.get_bitboard(piece_type::ROOK, nstm);
    let queens = pos.get_bitboard(piece_type::QUEEN, nstm);
    let b_n_q = bishops | queens;
    let r_n_q = rooks | queens;
    let king = pos.get_bitboard(piece_type::KING, nstm);

    let bishop_attacks = |sq| Bishop::lookup_attacks(sq, occupancy);
    let rook_attacks = |sq| Rook::lookup_attacks(sq, occupancy);

    pawn::compute_attacks(pawns, nstm)
        | knights.into_iter().map(knight::lookup_attacks).aggregate()
        | b_n_q.into_iter().map(bishop_attacks).aggregate()
        | r_n_q.into_iter().map(rook_attacks).aggregate()
        | king.lsb().map(self::lookup_attacks).unwrap_or_default()
}

pub fn fold_pseudo_legal_castling<P: Perspective, B, F, R>(pos: &Position, mut init: B, mut f: F) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let rank = P::COLOR * ranks::_8;
    let from = Square::from((files::E, rank));
    let castling = pos.get_castling();
    let occ = pos.get_occupancy();

    if castling.is_true(castling_sides::KING_SIDE, P::COLOR) && (occ & tabu_mask_ks::<P>()).is_empty() {
        // Safety: [e1|e8] + 2 < 63
        let to = unsafe { Square::from_v(from.v() + 2) };
        init = f(init, Move::new(from, to, move_flags::KING_CASTLE))?;
    }

    if castling.is_true(castling_sides::QUEEN_SIDE, P::COLOR) && (occ & block_mask_qs::<P>()).is_empty() {
        // Safety: [e1|e8] - 2 > 0
        let to = unsafe { Square::from_v(from.v() - 2) };
        return f(init, Move::new(from, to, move_flags::QUEEN_CASTLE));
    }

    try { init }
}

pub fn fold_legal_castling<P: Perspective, B, F, R>(pos: &Position, mut init: B, mut f: F, enemy_attacks: Bitboard) -> R
where
    F: FnMut(B, Move) -> R,
    R: Try<Output = B>,
{
    let rank = P::COLOR * ranks::_8;
    let from = Square::from((files::E, rank));
    let castling = pos.get_castling();

    if castling.is_true(castling_sides::KING_SIDE, P::COLOR) {
        let tabus = enemy_attacks | pos.get_occupancy();
        if (tabus & tabu_mask_ks::<P>()).is_empty() {
            // Safety: [e1|e8] + 2 < 63
            let to = unsafe { Square::from_v(from.v() + 2) };
            init = f(init, Move::new(from, to, move_flags::KING_CASTLE))?;
        }
    }

    if castling.is_true(castling_sides::QUEEN_SIDE, P::COLOR) {
        let blocked = block_mask_qs::<P>() & pos.get_occupancy();
        let checked = check_mask_qs::<P>() & enemy_attacks;
        if (blocked | checked).is_empty() {
            // Safety: [e1|e8] - 2 > 0
            let to = unsafe { Square::from_v(from.v() - 2) };
            return f(init, Move::new(from, to, move_flags::QUEEN_CASTLE));
        }
    }

    try { init }
}

pub const fn check_mask_qs<P: Perspective>() -> Bitboard {
    static CHECK_MASK: [Bitboard; 2] = [Bitboard { v: 0xC_u64 }, Bitboard { v: 0xC_u64 << 56 }];
    unsafe { *CHECK_MASK.get_unchecked(P::COLOR.index()) }
}

pub const fn block_mask_qs<P: Perspective>() -> Bitboard {
    static BLOCK_MASK: [Bitboard; 2] = [Bitboard { v: 0xE_u64 }, Bitboard { v: 0xE_u64 << 56 }];
    unsafe { *BLOCK_MASK.get_unchecked(P::COLOR.index()) }
}

pub const fn tabu_mask_ks<P: Perspective>() -> Bitboard {
    static TABU_MASK: [Bitboard; 2] = [Bitboard { v: 0x60_u64 }, Bitboard { v: 0x60_u64 << 56 }];
    unsafe { *TABU_MASK.get_unchecked(P::COLOR.index()) }
}

pub const fn lookup_attacks(sq: Square) -> Bitboard {
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
