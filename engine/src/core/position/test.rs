use super::*;
use crate::core::{
    bitboard::Bitboard, color::colors, coordinates::ranks, move_iter::sliding_piece::magics,
    piece::piece_type, ply::Ply, zobrist,
};
use crate::misc::ConstFrom;

#[test]
fn cloning() {
    let fens = ["r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"];

    zobrist::init();
    magics::init();

    for fen in fens {
        let pos = Position::from_fen(fen).unwrap();
        let cloned = pos.clone();
        assert_eq!(pos, cloned);
    }
}

#[test]
fn fen_decoding() {
    zobrist::init();
    magics::init();

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let pos = Position::from_fen(fen).expect("Should not fail.");

    assert_eq!(pos.get_turn(), colors::WHITE);
    assert_eq!(pos.plys_50(), Ply { v: 0 });
    assert_eq!(pos.ply(), Ply { v: 2 });
    assert_eq!(
        pos.get_bitboard(piece_type::PAWN, colors::WHITE),
        Bitboard::from_c(ranks::_2)
    );
}

// relies on fen_decoding working
fn test_fen_encoding(expected_fen: &str, fen: &str, moves: Vec<Move>) {
    zobrist::init();
    magics::init();

    let mut pos = Position::from_fen(fen).expect("Should not fail.");
    for mov in moves.into_iter() {
        pos.make_move(mov);
    }

    assert_eq!(format!("{}", FenExport(&pos)), expected_fen);
}

#[test]
fn fen_encoding_0() {
    test_fen_encoding(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        vec![],
    );
}

#[test]
fn fen_encoding_1() {
    use move_flags::*;
    use squares::*;
    test_fen_encoding(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        vec![Move::new(E2, E4, DOUBLE_PAWN_PUSH)],
    );
}

#[test]
fn fen_encoding_2() {
    use move_flags::*;
    use squares::*;
    test_fen_encoding(
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        vec![
            Move::new(E2, E4, DOUBLE_PAWN_PUSH),
            Move::new(C7, C5, DOUBLE_PAWN_PUSH),
        ],
    );
}

#[test]
fn fen_encoding_3() {
    use move_flags::*;
    use squares::*;
    test_fen_encoding(
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        vec![
            Move::new(E2, E4, DOUBLE_PAWN_PUSH),
            Move::new(C7, C5, DOUBLE_PAWN_PUSH),
            Move::new(G1, F3, QUIET),
        ],
    );
}
