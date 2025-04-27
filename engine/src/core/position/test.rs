use crate::{core::{move_iter::sliding_piece::magics, position::Position, zobrist}, uci::tokens::Tokenizer};

#[test]
fn cloning() {
    let fens = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    ];
    
    zobrist::init();
    magics::init();
    
    for fen in fens {
        let pos = Position::try_from(&mut Tokenizer::new(fen)).unwrap();
        let cloned = pos.clone();
        assert_eq!(pos, cloned);
    }
}