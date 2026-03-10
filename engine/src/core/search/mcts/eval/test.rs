use crate::core::{
    color::colors,
    coordinates::squares,
    depth::Depth,
    r#move::{Move, move_flags},
    move_iter::sliding_piece::magics,
    position::Position,
    search::mcts::{
        eval::{Evaluation, Evaluator, GameResult},
        node::{CtNodeRef, Node, RtNodeRef, Tree},
        test::DummyEvaluator,
    },
    zobrist,
};

fn test(pos: Position, expected_result: Option<GameResult>) {
    let node = CtNodeRef::new(Node::new_leaf());
    let mut tree = Tree::new(RtNodeRef::from_ct(node.clone()));
    let node = tree.expand_node(node.clone(), &pos, Depth::ROOT);

    assert_eq!(
        node.terminal().is_some(),
        expected_result.is_some(),
        "if we expect a game result \n\n{expected_result:?}\n\n in position \n\n{pos:?}\n\n, the \
         state transition should be Terminal \n\n{node:?}\n\n, and visa versa"
    );

    if let Some(expected_result) = expected_result {
        let node = node.terminal().expect("startpos should have branches");
        let result = DummyEvaluator::eval_terminal(node.clone(), &pos);
        assert_eq!(result, Evaluation::Terminal(expected_result));
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

    for _ in 0..3 {
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
