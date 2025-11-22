use crate::{
    core::{
        color::colors, move_iter::sliding_piece::magics, ply::Ply, position::Position, zobrist,
    },
    uci::tokens::Tokenizer,
};

#[test]
fn cloning() {
    let fens = ["r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"];

    zobrist::init();
    magics::init();

    for fen in fens {
        let pos = Position::try_from(&mut Tokenizer::new(fen)).unwrap();
        let cloned = pos.clone();
        assert_eq!(pos, cloned);
    }
}

#[test]
fn fen_decoding() {
    zobrist::init();
    magics::init();

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let pos = Position::try_from(&mut Tokenizer::new(fen)).expect("Should not fail.");

    assert_eq!(pos.get_turn(), colors::WHITE);
    assert_eq!(pos.plys_50(), Ply { v: 0 });
    assert_eq!(pos.ply(), Ply { v: 2 });
}
