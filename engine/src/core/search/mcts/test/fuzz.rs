use std::error::Error;

use super::*;
use crate::{
    core::{
        move_iter::sliding_piece::magics,
        position::Position,
        search::mcts::{
            back::DefaultBackuper, limiter::NoopLimiter, node::Tree, search::TreeSearcher,
            select::PuctSelector,
        },
        zobrist,
    },
    uci::tokens::Tokenizer,
};

#[test]
pub fn brrr() -> Result<(), Box<dyn Error>> {
    magics::init();
    zobrist::init();

    let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();

    let eval = DummyEvaluator::<1>::default();
    let limiter = NoopLimiter::default();
    let mut tree = Tree::new();

    for _ in 0..1_000_000 {
        let mut searcher = TreeSearcher::<1, _, _, PuctSelector<1>, DefaultBackuper>::new(
            &mut tree,
            pos.clone(),
            limiter.clone(),
            eval.clone(),
        );
        searcher.grow();
    }

    Ok(())
}
