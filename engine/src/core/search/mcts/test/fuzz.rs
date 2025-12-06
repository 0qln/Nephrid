use std::error::Error;

use super::*;
use crate::{
    core::{move_iter::sliding_piece::magics, position::Position, zobrist},
    uci::tokens::Tokenizer,
};

#[test]
pub fn brrr() -> Result<(), Box<dyn Error>> {
    magics::init();
    zobrist::init();

    let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut pos = Position::try_from(&mut fen).unwrap();

    let eval = DummyEvaluator::default();
    let mut tree = Tree::new(&pos, &eval);

    for _ in 0..1_000_000 {
        tree.grow(&mut pos, &eval);
    }

    Ok(())
}
