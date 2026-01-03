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

fn test_pgn_encoding_move_section(fen: &str, moves: &[Move], expected: &str) {
    zobrist::init();
    magics::init();

    let mut expected = Tokenizer::new(expected);

    let mut pos = Position::from_fen(fen).unwrap();
    let pgn = PgnExport::from_initial_pos(&mut pos, &moves);
    let moves_sec = format!("{}", pgn.1);
    let mut actual = Tokenizer::new(&moves_sec);

    while let Some(expected) = expected.next_token()
        && let Some(actual) = actual.next_token()
    {
        assert_eq!(expected, actual);
    }
}

#[test]
fn pgn_encoding_san_disambiguation() {
    use move_flags::*;
    use squares::*;

    test_pgn_encoding_move_section(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &[
            Move::new(E2, E4, DOUBLE_PAWN_PUSH),
            Move::new(C7, C5, DOUBLE_PAWN_PUSH),
            Move::new(E4, E5, QUIET),
            Move::new(D7, D5, DOUBLE_PAWN_PUSH),
            Move::new(E5, D6, EN_PASSANT),
            Move::new(G8, H6, QUIET),
            Move::new(G2, G4, DOUBLE_PAWN_PUSH),
            Move::new(B8, D7, QUIET),
            Move::new(H2, H4, DOUBLE_PAWN_PUSH),
            Move::new(H6, G4, CAPTURE),
            Move::new(H1, H3, QUIET),
            Move::new(G4, E5, QUIET),
            Move::new(A2, A4, DOUBLE_PAWN_PUSH),
            Move::new(C5, C4, QUIET),
            Move::new(A1, A3, QUIET),
            Move::new(C4, C3, QUIET),
            Move::new(F1, G2, QUIET),
            Move::new(C3, B2, CAPTURE),
            Move::new(D2, D3, QUIET),
            Move::new(B2, C1, CAPTURE_PROMOTION_KNIGHT),
            Move::new(E1, F1, QUIET),
            Move::new(C1, D3, CAPTURE),
            Move::new(D1, E1, QUIET),
            Move::new(E5, C4, QUIET),
            Move::new(E1, D1, QUIET),
            Move::new(C4, A5, QUIET),
            Move::new(D1, E1, QUIET),
            Move::new(A5, B3, QUIET),
            Move::new(E1, D1, QUIET),
            Move::new(D3, C5, QUIET),
        ],
        "1. e4 c5 2. e5 d5 3. exd6 Nh6 4. g4 Nd7 5. h4 Nxg4 6. Rh3 Nge5 7. a4 c4 8. Raa3 c3 9. Bg2 cxb2 10. d3 bxc1=N 11. Kf1 Ncxd3 12. Qe1 Nc4 13. Qd1 Na5 14. Qe1 Nb3 15. Qd1 Nd3c5",
    );
}

#[test]
fn pgn_encoding_san_castling() {
    use move_flags::*;
    use squares::*;

    test_pgn_encoding_move_section(
        "r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1",
        &[
            Move::new(E1, G1, KING_CASTLE),
            Move::new(E8, C8, QUEEN_CASTLE),
        ],
        "1. O-O O-O-O",
    );
}
