use crate::engine::{
    bitboard::Bitboard, color::Color, coordinates::Square, piece::SlidingPieceType,
    position::Position, r#move::Move,
};

use super::{gen_captures, gen_quiets};

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
