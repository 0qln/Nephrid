use crate::core::{
    config::Configuration,
    move_iter::sliding_piece::magics,
    position::Position,
    search::mcts::{
        MctsParts, NullNoiser, StaticParts, back::DefaultBackuper, limiter::NoopLimiter,
        node::Tree, search::TreeSearcher, select::ucb::UcbSelector, test::DummyEvaluator,
    },
    zobrist,
};

use std::{error::Error, thread};

fn fuzz<const X: usize, P: Send + 'static>(pos: &'static str, parts: P, rounds: usize)
where
    for<'a> &'a P: MctsParts,
{
    magics::init();
    zobrist::init();

    thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let pos = Position::from_fen(pos).unwrap();

            let mut tree = Tree::default();

            let mut pos_clone = pos.clone();

            let parts = &parts;

            let mut searcher = TreeSearcher::<X, _, _, _, _, _>::new(
                &mut pos_clone,
                parts.selector(),
                NoopLimiter,
                parts.evaluator(),
                parts.backprop(),
                parts.noiser(),
            );

            searcher.init_root(&mut tree);

            for _i in 0..(rounds / X) {
                searcher.grow(&mut tree);
            }

            assert_eq!(&pos, &pos_clone);
            assert_eq!(tree.size(), tree.compute_size());
            // assert_eq!(tree.mindepth(), tree.compute_mindepth());
            assert_eq!(tree.maxdepth(), tree.compute_maxheight());
        })
        .expect("Couldn't spawn thread")
        .join()
        .expect("Should be able to join thread");
}

#[test]
pub fn sa_fuzz_bs_1() -> Result<(), Box<dyn Error>> {
    fuzz::<1, _>(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        StaticParts::default(),
        50_000,
    );
    Ok(())
}

#[test]
pub fn sa_fuzz_bs_8() -> Result<(), Box<dyn Error>> {
    fuzz::<8, _>(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        StaticParts::default(),
        50_000,
    );
    Ok(())
}

#[test]
pub fn sa_fuzz_bs_64() -> Result<(), Box<dyn Error>> {
    fuzz::<64, _>(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        StaticParts::default(),
        50_000,
    );
    Ok(())
}

#[derive(Default)]
struct NoAnalysisParts;

impl MctsParts for &NoAnalysisParts {
    type Selector = UcbSelector;
    type Evaluator = DummyEvaluator;
    type Backprop = DefaultBackuper;
    type Noiser = NullNoiser;
    type Instance = NoAnalysisParts;

    fn selector(&self) -> Self::Selector {
        Default::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        Default::default()
    }

    fn backprop(&self) -> Self::Backprop {
        Default::default()
    }

    fn noiser(&self) -> Self::Noiser {
        Default::default()
    }
}

impl From<&Configuration> for NoAnalysisParts {
    fn from(_: &Configuration) -> Self {
        Default::default()
    }
}

#[test]
pub fn na_fuzz_bs_1() -> Result<(), Box<dyn Error>> {
    fuzz::<1, _>(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        NoAnalysisParts::default(),
        500_000,
    );
    Ok(())
}
