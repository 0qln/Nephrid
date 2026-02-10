use crate::core::{
    color::colors,
    coordinates::squares,
    r#move::{Move, move_flags},
    move_iter::sliding_piece::magics,
    position::Position,
    search::mcts::{
        eval::{Evaluation, Evaluator, GameResult},
        node::Node,
        test::DummyEvaluator,
    },
    zobrist,
};

fn test(fen: &str, expected_result: Option<GameResult>) {
    magics::init();
    zobrist::init();

    let pos = Position::from_fen(fen).unwrap();

    let mut node = Node::leaf();
    node.expand(&pos);

    let result = DummyEvaluator::eval_terminal(&node, &pos);

    // asserts when we expect a result.
    if expected_result.is_some() {
    }
    // asserts when we don't expect a result.
    else {
        assert!(
            node.has_branches(),
            "If the game is ongoing there have to be branches"
        );
    }

    // generic asserts
    assert_eq!(result, expected_result.map(|x| Evaluation::Terminal(x)));
}

#[test]
fn normal() {
    test(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        None,
    );
}

#[test]
fn stalemate() {
    test("K7/3r4/2k5/1r6/8/8/8/8 w - - 0 1", Some(GameResult::Draw));
}

#[test]
fn fifty_move_rule() {
    test("8/8/k7/3r4/8/5K2/8/8 b - - 50 54", None);
    test("8/8/k7/3r4/8/5K2/8/8 b - - 100 54", Some(GameResult::Draw));
}

#[test]
fn checkmate_for_black() {
    test(
        "K2r4/2r5/2k5/8/8/8/8/8 w - - 0 1",
        Some(GameResult::Win { relative_to: colors::BLACK }),
    );
}

#[test]
fn checkmate_for_white() {
    test(
        "2k2R2/4R3/K7/8/8/8/8/8 b - - 0 1",
        Some(GameResult::Win { relative_to: colors::WHITE }),
    );
}

#[test]
fn three_fold_repetition() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let expected_result = Some(GameResult::Draw);

    magics::init();
    zobrist::init();

    let mut pos = Position::from_fen(fen).unwrap();

    let mov_w0 = Move::new(squares::G1, squares::F3, move_flags::QUIET);
    let mov_w1 = Move::new(squares::F3, squares::G1, move_flags::QUIET);

    let mov_b0 = Move::new(squares::G8, squares::F6, move_flags::QUIET);
    let mov_b1 = Move::new(squares::F6, squares::G8, move_flags::QUIET);

    for _ in 0..3 {
        // make some moves
        pos.make_move(mov_w0);
        pos.make_move(mov_b0);

        // get back into the position
        pos.make_move(mov_w1);
        pos.make_move(mov_b1);
    }

    let mut node = Node::leaf();
    node.expand(&pos);

    let result = DummyEvaluator::eval_terminal(&node, &pos);

    // asserts when we expect a result.
    if expected_result.is_some() {
    }
    // asserts when we don't expect a result.
    else {
        assert!(
            node.has_branches(),
            "If the game is ongoing there have to be branches"
        );
    }

    // generic asserts
    assert_eq!(result, expected_result.map(|x| Evaluation::Terminal(x)));
}

#[test]
fn insufficent_material() {
    test("8/3k4/8/8/3K4/8/8/8 w - - 0 1", Some(GameResult::Draw));
}
