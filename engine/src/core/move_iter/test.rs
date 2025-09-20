use std::cell::UnsafeCell;

use crate::{
    core::{
        depth::Depth, move_iter::sliding_piece::magics, position::Position, search::Search, zobrist,
    },
    uci::{sync::CancellationToken, tokens::Tokenizer},
};
fn test_pos(mut fen: Tokenizer, depth: Depth, expected: u64) {
    magics::init();
    zobrist::init();

    let pos = Position::try_from(&mut fen).unwrap();
    let result = Search::perft(
        &mut UnsafeCell::new(pos),
        depth,
        CancellationToken::new(),
        |_, _| {},
    );
    assert_eq!(expected, result);
}

#[test]
fn test_legal_moves_0() {
    test_pos(
        Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        Depth::new(5),
        4865609,
    );
}

#[test]
fn test_legal_moves_1() {
    test_pos(
        Tokenizer::new("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        Depth::new(5),
        193690690,
    );
}

#[test]
fn test_legal_moves_2() {
    test_pos(
        Tokenizer::new("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        Depth::new(6),
        11030083,
    );
}

#[test]
fn test_legal_moves_3() {
    test_pos(
        Tokenizer::new("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
        Depth::new(5),
        15833292,
    );
}

#[test]
fn test_legal_moves_4() {
    test_pos(
        Tokenizer::new("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  "),
        Depth::new(5),
        89941194,
    );
}

#[test]
fn test_legal_moves_5() {
    test_pos(
        Tokenizer::new("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 "),
        Depth::new(5),
        164075551,
    );
}
