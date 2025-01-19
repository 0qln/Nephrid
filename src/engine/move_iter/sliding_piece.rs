use crate::{engine::{
    bitboard::Bitboard, coordinates::Square, r#move::Move, piece::{PieceType, SlidingPieceType}, position::{CheckState, Position}
}, misc::ConstFrom};

use super::{gen_captures, gen_quiets};

pub fn gen_legals_check_single(
    pos: &Position,
    piece_type: SlidingPieceType,
    compute_attacks: fn(Square, Bitboard) -> Bitboard,
) -> impl Iterator<Item = Move> + '_ {
    assert_eq!(pos.get_check_state(), CheckState::Single);
    let color = pos.get_turn();
    let allies = pos.get_color_bb(color);
    let enemies = pos.get_color_bb(!color);
    let occupancy = allies | enemies;
    let blockers = pos.get_blockers();
    let king_bb = pos.get_bitboard(PieceType::KING, color);
    // Safety: king the board has no king, but gen_legal is used,
    // the context is broken anyway. 
    let king = unsafe { king_bb.lsb().unwrap_unchecked() };
    // Safety: there is a single checker.
    let checker = unsafe { pos.get_checkers().lsb().unwrap_unchecked() };
    let resolves = Bitboard::between(king, checker);
    pos.get_bitboard(piece_type.into(), color)
        .flat_map(move |piece| {
            let piece_bb = Bitboard::from_c(piece);
            let is_blocker = !(blockers & piece_bb).is_empty(); 
            let pin_mask = is_blocker
                .then(|| Bitboard::ray(piece, king))
                .unwrap_or(Bitboard::full());
            let attacks = compute_attacks(piece, occupancy);
            let legal_attacks = attacks & pin_mask;
            let legal_resolves = resolves & legal_attacks;
            let legal_captures = legal_attacks & pos.get_checkers();
            let captures = gen_captures(legal_captures, enemies, piece);
            let quiets = gen_quiets(legal_resolves, enemies, allies, piece);
            captures.chain(quiets)
        })
}

pub fn gen_legal_captures_check_single(
    pos: &Position,
    piece_type: SlidingPieceType,
    compute_attacks: fn(Square, Bitboard) -> Bitboard,
) -> impl Iterator<Item = Move> + '_ {
    assert_eq!(pos.get_check_state(), CheckState::Single);
    let color = pos.get_turn();
    let allies = pos.get_color_bb(color);
    let enemies = pos.get_color_bb(!color);
    let occupancy = allies | enemies;
    let blockers = pos.get_blockers();
    let king_bb = pos.get_bitboard(PieceType::KING, color);
    // Safety: there is a single checker.
    let checker = unsafe { pos.get_checkers().lsb().unwrap_unchecked() };
    // Safety: king the board has no king, but gen_legal is used,
    // the context is broken anyway. 
    let king = unsafe { king_bb.lsb().unwrap_unchecked() };
    let resolves = Bitboard::between(king, checker);
    pos.get_bitboard(piece_type.into(), color)
        .flat_map(move |piece| {
            let piece_bb = Bitboard::from_c(piece);
            let is_blocker = !(blockers & piece_bb).is_empty(); 
            let pin_mask = is_blocker
                .then(|| Bitboard::ray(piece, king))
                .unwrap_or(Bitboard::full());
            let attacks = compute_attacks(piece, occupancy);
            let legal_attacks = attacks & pin_mask;
            let legal_captures = legal_attacks & pos.get_checkers();
            gen_captures(legal_captures, enemies, piece)
        })
}

pub fn gen_legals_check_none(
    pos: &Position,
    piece_type: SlidingPieceType,
    compute_attacks: fn(Square, Bitboard) -> Bitboard,
) -> impl Iterator<Item = Move> {
    assert_eq!(pos.get_check_state(), CheckState::None);
    let color = pos.get_turn();
    let allies = pos.get_color_bb(color);
    let enemies = pos.get_color_bb(!color);
    let occupancy = allies | enemies;
    let blockers = pos.get_blockers();
    let king_bb = pos.get_bitboard(PieceType::KING, color);
    // Safety: king the board has no king, but gen_legal is used,
    // the context is broken anyway. 
    let king = unsafe { king_bb.lsb().unwrap_unchecked() };
    pos.get_bitboard(piece_type.into(), color)
        .flat_map(move |piece| {
            let piece_bb = Bitboard::from_c(piece);
            let is_blocker = !(blockers & piece_bb).is_empty(); 
            let pin_mask = is_blocker
                .then(|| Bitboard::ray(piece, king))
                .unwrap_or(Bitboard::full());
            let attacks = compute_attacks(piece, occupancy);
            let legal_attacks = attacks & pin_mask;
            let captures = gen_captures(legal_attacks, enemies, piece);
            let quiets = gen_quiets(legal_attacks, enemies, allies, piece);
            captures.chain(quiets)
        })
}

pub fn gen_legal_captures_check_none(
    pos: &Position,
    piece_type: SlidingPieceType,
    compute_attacks: fn(Square, Bitboard) -> Bitboard,
) -> impl Iterator<Item = Move> {
    assert_eq!(pos.get_check_state(), CheckState::None);
    let color = pos.get_turn();
    let allies = pos.get_color_bb(color);
    let enemies = pos.get_color_bb(!color);
    let occupancy = allies | enemies;
    let blockers = pos.get_blockers();
    let king_bb = pos.get_bitboard(PieceType::KING, color);
    // Safety: king the board has no king, but gen_legal is used,
    // the context is broken anyway. 
    let king = unsafe { king_bb.lsb().unwrap_unchecked() };
    pos.get_bitboard(piece_type.into(), color)
        .flat_map(move |piece| {
            let piece_bb = Bitboard::from_c(piece);
            let is_blocker = !(blockers & piece_bb).is_empty(); 
            let pin_mask = is_blocker
                .then(|| Bitboard::ray(piece, king))
                .unwrap_or(Bitboard::full());
            let attacks = compute_attacks(piece, occupancy);
            let legal_attacks = attacks & pin_mask;
            gen_captures(legal_attacks, enemies, piece)
        })
}

pub fn gen_psuedo_legal_captures(
    pos: &Position,
    piece_type: SlidingPieceType,
    compute_attacks: fn(Square, Bitboard) -> Bitboard,
) -> impl Iterator<Item = Move> {
    let color = pos.get_turn();
    let allies = pos.get_color_bb(color);
    let enemies = pos.get_color_bb(!color);
    pos.get_bitboard(piece_type.into(), color)
        .flat_map(move |piece| {
            let attacks = compute_attacks(piece, allies | enemies);
            gen_captures(attacks, enemies, piece)
        })
}

pub fn gen_psuedo_legals(
    pos: &Position,
    piece_type: SlidingPieceType,
    compute_attacks: fn(Square, Bitboard) -> Bitboard,
) -> impl Iterator<Item = Move> {
    let color = pos.get_turn();
    let allies = pos.get_color_bb(color);
    let enemies = pos.get_color_bb(!color);
    pos.get_bitboard(piece_type.into(), color)
        .flat_map(move |piece| {
            let attacks = compute_attacks(piece, allies | enemies);
            let captures = gen_captures(attacks, enemies, piece);
            let quiets = gen_quiets(attacks, enemies, allies, piece);
            captures.chain(quiets)
        })
}
