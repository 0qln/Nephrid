use std::ops::ControlFlow;

use crate::{core::{color::Color, move_iter::{fold_legal_moves, sliding_piece::magics}, position::Position, search::mcts::{NodeState, Evaluation}, zobrist}, uci::tokens::Tokenizer};

fn test(fen: &str, expected_result: Option<Evaluation>) {
    magics::init();
    zobrist::init();

    let mut fen = Tokenizer::new(fen);
    let pos = Position::try_from(&mut fen).unwrap();
    let mut moves = Vec::new();
    _ = fold_legal_moves(&pos, &mut moves, |acc, m| {
        ControlFlow::Continue::<(), _>({
            acc.push(m);
            acc
        })
    });
    // let result = PlayoutResult::maybe_new(&pos, if moves.is_empty() { NodeState::Terminal } else { NodeState::Branch });
    // assert_eq!(result, expected_result);
}

#[test]
fn normal() {
    test(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 
        None
    );
}

#[test]
fn stalemate() {
    test(
        "K7/3r4/2k5/1r6/8/8/8/8 w - - 0 1",
        Some(Evaluation::Draw)
    );
}
    
#[test]
fn fifty_move_rule() {
    test(
        "8/8/k7/3r4/8/5K2/8/8 b - - 100 54",
        Some(Evaluation::Draw)
    );
}

#[test]
fn checkmate_for_black() {
    test(
        "K2r4/2r5/2k5/8/8/8/8/8 w - - 0 1",
        Some(Evaluation::Win { relative_to: Color::BLACK})
    );
}

#[test]
fn checkmate_for_white() {
    test(
        "2k2R2/4R3/K7/8/8/8/8/8 b - - 0 1",
        Some(Evaluation::Win { relative_to: Color::WHITE})
    );
}

// todo: thee fold repetition
// todo: insufficent material