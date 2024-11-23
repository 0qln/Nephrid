use crate::engine::{
    bitboard::Bitboard, color::Color, coordinates::Square, piece::JumpingPieceType,
    position::Position, r#move::Move,
};

use super::{gen_captures, gen_quiets};

pub fn gen_psuedo_legal_captures(
    position: &Position,
    color: Color,
    piece_type: JumpingPieceType,
    compute_attacks: fn(Square) -> Bitboard,
) -> impl Iterator<Item = Move> {
    let allies = position.get_color_bb(color);
    let enemies = position.get_color_bb(!color);
    position
        .get_bitboard(piece_type.into(), color)
        .map(move |piece| {
            let attacks = compute_attacks(piece);
            gen_captures(attacks, enemies, piece)
        })
        .flatten()
}

pub fn gen_psuedo_legal_quiets(
    position: &Position,
    color: Color,
    piece_type: JumpingPieceType,
    compute_attacks: fn(Square) -> Bitboard,
) -> impl Iterator<Item = Move> {
    let allies = position.get_color_bb(color);
    let enemies = position.get_color_bb(!color);
    position
        .get_bitboard(piece_type.into(), color)
        .map(move |piece| {
            let attacks = compute_attacks(piece);
            gen_quiets(attacks, enemies, allies, piece)
        })
        .flatten()
}
