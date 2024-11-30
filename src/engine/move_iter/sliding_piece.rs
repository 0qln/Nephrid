use crate::{engine::{
    bitboard::Bitboard, color::{self, Color}, coordinates::Square, r#move::{Move, MoveFlag}, piece::{PieceType, SlidingPieceType}, position::{CheckState, Position}
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
            let pin_mask = if is_blocker { Bitboard::ray(piece, king) } else { Bitboard::full() };
            let attacks = compute_attacks(piece, occupancy);
            let legal_attacks = attacks & pin_mask;
            let captures = {
                let targets = legal_attacks & enemies;
                targets.map(move |target| Move::new(piece, target, MoveFlag::CAPTURE))               
            };
            let quiets = {
                let targets = legal_attacks & !allies & !enemies;
                targets.map(move |target| Move::new(piece, target, MoveFlag::QUIET))
            };
            captures.chain(quiets)
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
