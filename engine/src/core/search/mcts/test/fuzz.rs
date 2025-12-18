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

fn brrr<const X: usize>(pos: &str) {
    magics::init();
    zobrist::init();

    let mut fen = Tokenizer::new(pos);
    let pos = Position::try_from(&mut fen).unwrap();

    let eval = DummyEvaluator::<X>::default();
    let limiter = NoopLimiter::default();
    let mut tree = Tree::new();

    for _i in 0..(50_000 / X) {
        let mut searcher = TreeSearcher::<X, _, _, PuctSelector<X>, DefaultBackuper>::new(
            &mut tree,
            pos.clone(),
            limiter.clone(),
            eval.clone(),
        );
        searcher.grow();
    }
}

#[test]
pub fn brrr_bs_1() -> Result<(), Box<dyn Error>> {
    brrr::<1>("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Ok(())
}

#[test]
pub fn brrr_bs_8() -> Result<(), Box<dyn Error>> {
    brrr::<8>("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Ok(())
}

#[test]
pub fn brrr_bs_64() -> Result<(), Box<dyn Error>> {
    brrr::<64>("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Ok(())
}
