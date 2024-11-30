use crate::{engine::{
    bitboard::Bitboard, color::{self, Color}, coordinates::Square, r#move::{Move, MoveFlag}, piece::{PieceType, SlidingPieceType}, position::Position
}, misc::ConstFrom};

use super::{gen_captures, gen_quiets};

pub fn gen_legal_check_single(
    pos: &Position,
    piece_type: SlidingPieceType,
    compute_attacks: fn(Square, Bitboard) -> Bitboard,
) -> impl Iterator<Item = Move> {
    let color = pos.get_turn();
    let allies = pos.get_color_bb(color);
    let enemies = pos.get_color_bb(!color);
}

pub fn gen_legal_check_none(
    pos: &Position,
    piece_type: SlidingPieceType,
    compute_attacks: fn(Square, Bitboard) -> Bitboard,
) -> impl Iterator<Item = Move> {
    let color = pos.get_turn();
    let allies = pos.get_color_bb(color);
    let enemies = pos.get_color_bb(!color);
    let blockers = pos.get_blockers();
    let king_bb = pos.get_bitboard(PieceType::KING, color);
    pos.get_bitboard(piece_type.into(), color)
        .flat_map(move |piece| {
            let attacks = compute_attacks(piece, allies | enemies);
            let capture_targets = attacks & enemies;
            let quiet_targets = attacks & !allies & !enemies;
            let piece_bb = Bitboard::from_c(piece);
            let is_blocker = !(blockers & piece_bb).is_empty(); 
            // Restrict the movement of pinned pieces. 
            capture_targets.filter_map(move |target| {
                let is_illegal = is_blocker && (Bitboard::ray(piece, target) & king_bb).is_empty();
                match is_illegal {
                    false => Some(Move::new(piece, target, MoveFlag::CAPTURE)),
                    true => None
                }
            }).chain(
                quiet_targets.filter_map(move |target| {
                    let is_illegal = is_blocker && (Bitboard::ray(piece, target) & king_bb).is_empty();
                    match is_illegal {
                        false => Some(Move::new(piece, target, MoveFlag::QUIET)),
                        true => None
                    }
                })
            )
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
    pos
        .get_bitboard(piece_type.into(), color)
        .map(move |piece| {
            let attacks = compute_attacks(piece, allies | enemies);
            gen_captures(attacks, enemies, piece)
        })
        .flatten()
}

pub fn gen_psuedo_legal_quiets(
    pos: &Position,
    piece_type: SlidingPieceType,
    compute_attacks: fn(Square, Bitboard) -> Bitboard,
) -> impl Iterator<Item = Move> {
    let color = pos.get_turn();
    let allies = pos.get_color_bb(color);
    let enemies = pos.get_color_bb(!color);
    pos
        .get_bitboard(piece_type.into(), color)
        .map(move |piece| {
            let attacks = compute_attacks(piece, allies | enemies);
            gen_quiets(attacks, enemies, allies, piece)
        })
        .flatten()
}
