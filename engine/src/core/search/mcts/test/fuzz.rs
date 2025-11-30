use std::cell::RefCell;
use std::error::Error;

use rand::{Rng, SeedableRng, rngs::SmallRng};

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

    let seed = 0xdead_beef;
    let rng = SmallRng::seed_from_u64(seed);

    struct DummyEvaluator(RefCell<SmallRng>);
    impl Evaluator for DummyEvaluator {
        fn evaluate(&self, _pos: &Position) -> (f32, [f32; POLICY_OUTPUTS]) {
            let mut rng = self.0.borrow_mut();

            let quality = rng.random_range(-1.0..=1.0);

            let policies: [f32; POLICY_OUTPUTS] = {
                let mut p = [0.2; POLICY_OUTPUTS];
                let policy_idx = rng.random_range(0..POLICY_OUTPUTS);
                p[policy_idx] = 1.0;
                p
            };

            (quality, policies)
        }
    }

    let eval = DummyEvaluator(RefCell::new(rng));
    let mut tree = Tree::new(&pos, &eval);

    for _ in 0..1_000_000 {
        tree.grow(&mut pos, &eval);
    }

    Ok(())
}
