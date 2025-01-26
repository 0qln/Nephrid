use crate::{
    engine::{
        bitboard::Bitboard,
        coordinates::Square,
        piece::{JumpingPieceType, PieceType},
        position::{CheckState, Position},
        r#move::Move,
    },
    misc::ConstFrom,
};

use super::{gen_captures, gen_quiets};

pub fn gen_legals_check_single(
    pos: &Position,
    piece_type: JumpingPieceType,
    compute_attacks: fn(Square) -> Bitboard,
) -> impl Iterator<Item = Move> + '_ {
    assert_eq!(pos.get_check_state(), CheckState::Single);
    let color = pos.get_turn();
    pos.get_bitboard(piece_type.into(), color)
        .flat_map(move |piece| {
            let enemies = pos.get_color_bb(!color);
            let allies = pos.get_color_bb(color);
            let blockers = pos.get_blockers();
            let king_bb = pos.get_bitboard(PieceType::KING, color);
            // Safety: king the board has no king, but gen_legal is used,
            // the context is broken anyway.
            let king = unsafe { king_bb.lsb().unwrap_unchecked() };
            // Safety: there is a single checker.
            let checker = unsafe { pos.get_checkers().lsb().unwrap_unchecked() };
            let resolves = Bitboard::between(king, checker);
            let piece_bb = Bitboard::from_c(piece);
            let is_not_blocker = (blockers & piece_bb).is_empty();
            let legal_attacks = is_not_blocker
                .then(|| compute_attacks(piece))
                .unwrap_or_default();
            let legal_resolves = resolves & legal_attacks;
            let legal_captures = legal_attacks & pos.get_checkers();
            let captures = gen_captures(legal_captures, enemies, piece);
            let quiets = gen_quiets(legal_resolves, enemies, allies, piece);
            captures.chain(quiets)
        })
}
