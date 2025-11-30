use std::error::Error;

use super::super::*;
use crate::{
    core::{
        move_iter::sliding_piece::magics, position::Position,
        search::mcts::eval::model::POLICY_OUTPUTS, zobrist,
    },
    uci::tokens::Tokenizer,
};

#[test]
pub fn brrr() -> Result<(), Box<dyn Error>> {
    magics::init();
    zobrist::init();

    let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut pos = Position::try_from(&mut fen).unwrap();

    #[derive(Default)]
    struct DummyEvaluator;
    impl Evaluator for DummyEvaluator {
        fn evaluate(&self, _pos: &Position) -> (f32, &'_ [f32; POLICY_OUTPUTS]) {
            const P: [f32; POLICY_OUTPUTS] = [0.0; POLICY_OUTPUTS];
            (0.0, &P)
        }
    }

    let eval = DummyEvaluator::default();
    let mut tree = Tree::new(&pos, &eval);

    for _ in 0..100_000 {
        tree.grow(&mut pos, &eval);
    }

    Ok(())
}
