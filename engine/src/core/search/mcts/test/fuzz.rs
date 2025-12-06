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

    let mut eval = DummyEvaluator::default();
    let limiter = NoopLimiter::default();
    let mut tree = Tree::new(&pos, &mut eval, &limiter);

    for _ in 0..1_000_000 {
        tree.grow(&mut pos, &mut eval, &limiter);
    }

    Ok(())
}
