use super::*;
use crate::{
    core::{
        bitboard::Bitboard, color::colors, coordinates::ranks, move_iter::sliding_piece::magics,
        piece::piece_type, ply::Ply, zobrist,
    },
};

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
        Bitboard::from(ranks::_2)
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
    let pgn = ReducedPgn::from_initial_pos(&mut pos, moves);
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
        "1. e4 c5 2. e5 d5 3. exd6 Nh6 4. g4 Nd7 5. h4 Nxg4 6. Rh3 Nge5 7. a4 c4 8. Raa3 c3 9. \
         Bg2 cxb2 10. d3 bxc1=N 11. Kf1 Ncxd3 12. Qe1 Nc4 13. Qd1 Na5 14. Qe1 Nb3 15. Qd1 Nd3c5",
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

/// Helper to construct a position that has just reached a simple 2-fold
/// repetition.
fn build_twofold_repetition_position() -> Position {
    use move_flags::*;
    use squares::*;

    // Start from the default starting position.
    let mut pos = Position::start_position();

    // Construct moves that move knights out and back, returning to the same
    // position.
    //
    // Sequence (from the start position):
    //   1. Ng1f3   1... Ng8f6
    //   2. Nf3g1   2... Nf6g8
    //
    // After these 4 plies, the board and side-to-move are the same as at the start,
    // so the initial position has occurred twice (a 2-fold repetition).
    let moves = [
        Move::new(G1, F3, QUIET),
        Move::new(G8, F6, QUIET),
        Move::new(F3, G1, QUIET),
        Move::new(F6, G8, QUIET),
    ];
    for mv in moves {
        pos.make_move(mv);
    }
    pos
}

#[test]
fn twofold_repetition_is_detected() {
    // Sanity: a fresh starting position should not have a recorded twofold
    // repetition.
    let fresh = Position::start_position();
    assert!(
        !fresh.has_twofold_repetition(),
        "fresh position must not have twofold repetition"
    );

    // After the back-and-forth sequence, the same position has occurred twice, so
    // the 2-fold repetition detector should report true.
    let pos = build_twofold_repetition_position();
    assert!(
        pos.has_twofold_repetition(),
        "position after repetition sequence should report twofold repetition"
    );
}

#[test]
fn twofold_heuristic_applies_only_beyond_root() {
    let pos = build_twofold_repetition_position();

    // `has_moves` here is explicitly passed as `true` because in this constructed
    // position there are still plenty of legal moves available; we only care about
    // how repetition and depth affect the result.
    let has_moves = true;

    // At the root depth, the 2-fold heuristic must *not* treat the position as a
    // draw.
    let result_at_root = pos.search_result_with(has_moves, Depth::ROOT);
    assert!(
        result_at_root.is_none(),
        "2-fold repetition should not be scored as draw at ROOT depth"
    );

    // For a depth strictly greater than ROOT, the 2-fold heuristic should kick in
    // and treat the position as a draw, assuming no 50-move or
    // insufficient-material draw.
    let deeper_depth = Depth::ROOT + 1;
    let result_deeper = pos.search_result_with(has_moves, deeper_depth);
    assert_eq!(
        result_deeper,
        Some(GameResult::Draw),
        "2-fold repetition should be scored as draw only for search_depth > ROOT"
    );
}

/// Helper to parse an EPD line and return (Position, Vec<EpdOp>)
fn parse_epd_line(line: &str) -> Result<(Position, Vec<EpdOp>), EpdLineParseError> {
    let mut tok = Tokenizer::new(line);
    EpdLineImport(&mut tok).try_into()
}

#[test]
fn epd_import_fen_only() {
    zobrist::init();
    magics::init();

    let fen_only = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let (pos, ops) = parse_epd_line(fen_only).expect("Should parse FEN-only EPD line");
    assert_eq!(ops.len(), 0);
    assert_eq!(format!("{}", FenExport(&pos)), fen_only);
}

#[test]
fn epd_import_single_op() {
    zobrist::init();
    magics::init();

    let line = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - bm e2e4;";
    let (pos, ops) = parse_epd_line(line).expect("Should parse EPD with one operation");
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].0, "bm");
    assert_eq!(ops[0].1, "e2e4");
    // Verify FEN part is still correct
    assert_eq!(pos.get_turn(), colors::WHITE);
}

#[test]
fn epd_import_multiple_ops() {
    zobrist::init();
    magics::init();

    let line = "r1bqk2r/p1pp1ppp/2p2n2/8/1b2P3/2N5/PPP2PPP/R1BQKB1R w KQkq - bm Bd3; id \"Crafty \
                Test Pos.28\"; c0 \"DB/GK Philadelphia 1996, Game 5, move 7W (Bd3)\";";
    let (pos, ops) = parse_epd_line(line).expect("Should parse EPD with multiple operations");
    assert_eq!(ops.len(), 3);
    assert_eq!(ops[0].0, "bm");
    assert_eq!(ops[0].1, "Bd3");
    assert_eq!(ops[1].0, "id");
    assert_eq!(ops[1].1, "Crafty Test Pos.28");
    assert_eq!(ops[2].0, "c0");
    assert_eq!(ops[2].1, "DB/GK Philadelphia 1996, Game 5, move 7W (Bd3)");
    // Quick sanity on the position
    assert_eq!(pos.get_turn(), colors::WHITE);
    assert!(
        pos.get_piece(crate::core::coordinates::squares::E4)
            .piece_type()
            == piece_type::PAWN
    );
}

#[test]
fn epd_import_argument_with_spaces() {
    zobrist::init();
    magics::init();

    let line = "8/3r4/pr1Pk1p1/8/7P/6P1/3R3K/5R2 w - - bm Re2+; id \"arasan21.16\"; c0 \"Aldiga \
                (Brainfish 091016)-Knight-king (Komodo 10 64-bit), playchess.com 2016\";";
    let (_pos, ops) =
        parse_epd_line(line).expect("Should parse EPD with argument containing spaces");
    assert_eq!(ops.len(), 3);
    assert_eq!(ops[0].0, "bm");
    assert_eq!(ops[0].1, "Re2+");
    assert_eq!(ops[1].0, "id");
    assert_eq!(ops[1].1, "arasan21.16");
    assert_eq!(ops[2].0, "c0");
    assert!(ops[2].1.contains("Brainfish"));
}

#[test]
fn epd_import_no_whitespace_between_ops() {
    zobrist::init();
    magics::init();

    // Operations can be adjacent without spaces (though usually there are spaces)
    let line = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - bm e2e4;id test;";
    let (_pos, ops) = parse_epd_line(line).expect("Should parse EPD with adjacent ops");
    assert_eq!(ops.len(), 2);
    assert_eq!(ops[0].0, "bm");
    assert_eq!(ops[0].1, "e2e4");
    assert_eq!(ops[1].0, "id");
    assert_eq!(ops[1].1, "test");
}

#[test]
fn epd_import_missing_semicolon() {
    let line = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - bm e2e4";
    let result = parse_epd_line(line);
    assert!(result.is_err());
    match result.unwrap_err() {
        EpdLineParseError::EpdOperationsError(EpdOpParseError::MissingArgumentOrSemicolon) => (),
        e => panic!("Expected MissingArgumentOrSemicolon, got {e}"),
    }
}

#[test]
fn epd_import_missing_opcode() {
    let line = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -  e2e4;";
    let result = parse_epd_line(line);
    assert!(result.is_err());
    match result.unwrap_err() {
        EpdLineParseError::EpdOperationsError(_) => (),
        e => panic!("Expected MissingCode, got {e}"),
    }
}

#[test]
fn epd_import_invalid_fen() {
    let line = "invalid fen part - bm e2e4;";
    let result = parse_epd_line(line);
    assert!(result.is_err());
    match result.unwrap_err() {
        EpdLineParseError::FenError(_) => (),
        e => panic!("Expected FenError, got {e}"),
    }
}

#[test]
fn epd_import_real_world_example() {
    zobrist::init();
    magics::init();

    // Example from Crafty test suite
    let line = "3r1rk1/1p3pnp/p3pBp1/1qPpP3/1P1P2R1/P2Q3R/6PP/6K1 w - - bm Rxh7; c0 \"Mate in 7 \
                moves\"; id \"BT2630-14\";";
    let (pos, ops) = parse_epd_line(line).expect("Should parse real-world EPD");
    assert_eq!(ops.len(), 3);
    assert_eq!(ops[0].0, "bm");
    assert_eq!(ops[0].1, "Rxh7");
    assert_eq!(ops[1].0, "c0");
    assert_eq!(ops[1].1, "Mate in 7 moves");
    assert_eq!(ops[2].0, "id");
    assert_eq!(ops[2].1, "BT2630-14");

    // Verify a few details of the position
    assert_eq!(pos.get_turn(), colors::WHITE);
    assert_eq!(
        pos.get_piece(crate::core::coordinates::squares::H3)
            .piece_type(),
        piece_type::ROOK
    );
    assert_eq!(
        pos.get_piece(crate::core::coordinates::squares::F6)
            .piece_type(),
        piece_type::BISHOP
    );
}
