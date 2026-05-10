use crate::core::{
    color::{Color, colors},
    coordinates::squares,
    r#move::{Move, move_flags},
    move_iter::sliding_piece::magics,
    piece::piece_type,
    position::Position,
    search::mcts::{
        eval::{
            self, GameResult,
            hce::{piece_score, see},
        },
        node::{
            Tree,
            node_state::{Leaf, Terminal},
        },
    },
    zobrist,
};

fn test(pos: Position, expected_result: Option<GameResult>) {
    let mut tree = Tree::new();

    let node = tree.expand_node(
        tree.node_switch(tree.root()).get::<Leaf>().unwrap(),
        &pos,
        pos.ply().into(),
    );

    assert_eq!(
        tree.node_switch(tree.root()).get::<Terminal>().is_some(),
        expected_result.is_some(),
        "if we expect a game result \n\n{expected_result:?}\n\n in position \n\n{pos:?}\n\n, the \
         state transition should be Terminal \n\n{node:?}\n\n, and visa versa"
    );

    if let Some(expected_result) = expected_result {
        let node = tree
            .node_switch(tree.root())
            .get::<Terminal>()
            .expect("if we have a result, the node should be terminal");
        let result = eval::eval_terminal(node, &tree, pos.ply().into(), &pos);
        assert_eq!(result, expected_result);
    }
}

fn test_fen(fen: &str, expected_result: Option<GameResult>) {
    magics::init();
    zobrist::init();

    test(Position::from_fen(fen).unwrap(), expected_result)
}

#[test]
fn normal() {
    test_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        None,
    );
}

#[test]
fn stalemate() {
    test_fen("K7/3r4/2k5/1r6/8/8/8/8 w - - 0 1", Some(GameResult::Draw));
}

#[test]
fn fifty_move_rule() {
    test_fen("8/8/k7/3r4/8/5K2/8/8 b - - 50 54", None);
    test_fen("8/8/k7/3r4/8/5K2/8/8 b - - 100 54", Some(GameResult::Draw));
}

#[test]
fn checkmate_for_black() {
    test_fen(
        "K2r4/2r5/2k5/8/8/8/8/8 w - - 0 1",
        Some(GameResult::Win { relative_to: colors::BLACK }),
    );
}

#[test]
fn checkmate_for_white() {
    test_fen(
        "2k2R2/4R3/K7/8/8/8/8/8 b - - 0 1",
        Some(GameResult::Win { relative_to: colors::WHITE }),
    );
}

#[test]
fn three_fold_repetition() {
    magics::init();
    zobrist::init();

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let mut pos = Position::from_fen(fen).unwrap();

    let mov_w0 = Move::new(squares::G1, squares::F3, move_flags::QUIET);
    let mov_w1 = Move::new(squares::F3, squares::G1, move_flags::QUIET);

    let mov_b0 = Move::new(squares::G8, squares::F6, move_flags::QUIET);
    let mov_b1 = Move::new(squares::F6, squares::G8, move_flags::QUIET);

    for _ in 0..2 {
        // make some moves
        pos.make_move(mov_w0);
        pos.make_move(mov_b0);

        // get back into the position
        pos.make_move(mov_w1);
        pos.make_move(mov_b1);
    }

    test(pos, Some(GameResult::Draw));
}

#[test]
fn insufficent_material() {
    test_fen("8/3k4/8/8/3K4/8/8/8 w - - 0 1", Some(GameResult::Draw));
}

fn run_see_test(fen: &str, mov: Move, us: Color, expected: i32) {
    magics::init();
    zobrist::init();

    let pos = Position::from_fen(fen).unwrap();

    let actual_score = see(&pos.piece_info(), mov, us);

    assert_eq!(
        actual_score, expected,
        "SEE failed for move {:?} in FEN {}. Expected {}, got {}",
        mov, fen, expected, actual_score
    );
}

#[test]
fn see_quiet_move() {
    // e4 move, no captures
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let mov = Move::new(squares::E2, squares::E4, move_flags::QUIET);
    run_see_test(fen, mov, colors::WHITE, 0);
}

#[test]
fn see_undefended_capture() {
    // White knight takes undefended black pawn on d5
    let fen = "8/8/8/3p4/4N3/8/8/8 w - - 0 1";
    let mov = Move::new(squares::E4, squares::D5, move_flags::CAPTURE);

    // Expected: Gains the pawn
    run_see_test(fen, mov, colors::WHITE, piece_score(piece_type::PAWN));
}

#[test]
fn see_equal_trade() {
    // White pawn takes black pawn on d5, black recaptures
    let fen = "8/8/4p3/3p4/4P3/8/8/8 w - - 0 1";
    let mov = Move::new(squares::E4, squares::D5, move_flags::CAPTURE);

    // Expected: +100 for PxP, but opponent recaptures for -100. Net is 0.
    run_see_test(
        fen,
        mov,
        colors::WHITE,
        piece_score(piece_type::PAWN) - piece_score(piece_type::PAWN),
    );
}

#[test]
fn see_losing_capture() {
    // White queen takes defended black pawn on d5
    let fen = "8/8/4p3/3p4/4Q3/8/8/8 w - - 0 1";
    let mov = Move::new(squares::E4, squares::D5, move_flags::CAPTURE);

    // Expected: QxP (+100). Black plays PxQ (-800). Net: -700.
    // (If your Negamax propagation is wrong, this test will fail!)
    run_see_test(
        fen,
        mov,
        colors::WHITE,
        -piece_score(piece_type::QUEEN) + piece_score(piece_type::PAWN),
    );
}

#[test]
fn see_complex_xray_dogpile() {
    // Classic SEE test: White has Rooks on d1, d3. Black has Rook d8, Bishop d6.
    // White initiates: Rd3xd6.
    let fen = "1k1r4/1p5p/3b4/8/8/3R4/1PP4P/1K1R4 w - - 0 1";
    let mov = Move::new(squares::D3, squares::D6, move_flags::CAPTURE);

    // 1. White RxB (+300)
    // 2. Black RxR (+500) -> Net so far: -200
    // 3. White RxR (revealed by x-ray!) (+500) -> Net: +300
    run_see_test(
        fen,
        mov,
        colors::WHITE,
        piece_score(piece_type::BISHOP) - piece_score(piece_type::ROOK)
            + piece_score(piece_type::ROOK),
    );
}

#[test]
fn see_en_passant() {
    // White pawn on e5 captures d5 pawn en passant
    let fen = "8/8/8/3pP3/8/8/8/8 w - d6 0 1";
    // Ensure you pass the EN_PASSANT flag so your code knows it's an EP move!
    let mov = Move::new(squares::E5, squares::D6, move_flags::EN_PASSANT);

    // Expected: +100 for the pawn.
    run_see_test(fen, mov, colors::WHITE, piece_score(piece_type::PAWN));
}

#[test]
fn see_capture_promotion() {
    // White pawn on e7 captures Black Rook on d8 and promotes to Queen.
    // Black has a Rook on c8 ready to recapture the new Queen.
    let fen = "2rr4/4P3/8/8/8/8/8/8 w - - 0 1";
    // Pass the promotion-capture flag (e.g., PROMO_QUEEN_CAPTURE)
    let mov = Move::new(
        squares::E7,
        squares::D8,
        move_flags::CAPTURE_PROMOTION_QUEEN,
    );

    // 1. White captures Rook (+500) and promotes (+800) - loses pawn (-100).
    //    Initial gain: +1300.
    // 2. Black recaptures the newly promoted Queen (+800).
    run_see_test(
        fen,
        mov,
        colors::WHITE,
        piece_score(piece_type::ROOK) + piece_score(piece_type::QUEEN)
            - piece_score(piece_type::PAWN)
            - piece_score(piece_type::QUEEN),
    );
}
