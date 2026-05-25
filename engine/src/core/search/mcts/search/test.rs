use crate::core::{
    config::Configuration,
    move_iter::sliding_piece::magics,
    params::{IParams, Params, ParamsRef},
    position::Position,
    search::mcts::{
        HceParts, MctsParts, NullNoiser, node::DAG, search::TreeSearcher,
        select::ucb::UcbSelector, test::DummyEvaluator,
    },
    zobrist,
};

use std::{error::Error, thread};

fn fuzz<const X: usize, P: MctsParts + Default>(pos: &'static str, rounds: usize) {
    magics::init();
    zobrist::init();

    thread::Builder::new()
        .name("MCTS fuzz test thread".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let pos = Position::from_fen(pos).unwrap();

            let mut tree = DAG::default();

            let mut pos_clone = pos.clone();

            let parts = P::default();

            let mut searcher = TreeSearcher::<X, _, _, _>::new(
                &mut pos_clone,
                Params::default().shared(),
                parts.selector(),
                parts.evaluator(),
                parts.noiser(),
            );

            searcher.init_root(&mut tree);

            let iterations = rounds / X;

            for _i in 0..(rounds / X) {
                searcher.grow(&mut tree);
            }

            assert_eq!(&pos, &pos_clone);
            // With a transposition table, evicted nodes stay counted in
            // tree.size() but become unreachable from root. Allow size >=
            // subtree size; equality holds only when there are no collisions.
            assert!(
                tree.size() >= tree.compute_subtree_size(tree.root()),
                "tree.size() ({}) should be >= compute_subtree_size ({}) ",
                tree.size(),
                tree.compute_subtree_size(tree.root())
            );
            assert_eq!(
                tree.terminal_nodes(),
                tree.compute_subtree_terminal_nodes_count()
            );
            // todo: fix
            // assert_eq!(
            //     tree.maxheight(),
            //     tree.compute_subtree_maxheight(tree.root())
            // );
            assert!(
                tree.size() > iterations,
                "Tree should have more nodes than iterations, but it has {} nodes and after {} \
                 iterations",
                tree.size(),
                iterations
            );
        })
        .expect("Couldn't spawn thread")
        .join()
        .expect("Should be able to join thread");
}

#[test]
pub fn sa_fuzz_bs_1() -> Result<(), Box<dyn Error>> {
    fuzz::<1, HceParts>(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        1_000,
    );
    Ok(())
}

#[test]
pub fn sa_fuzz_bs_8() -> Result<(), Box<dyn Error>> {
    fuzz::<8, HceParts>(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        1_000,
    );
    Ok(())
}

#[test]
pub fn sa_fuzz_bs_64() -> Result<(), Box<dyn Error>> {
    fuzz::<64, HceParts>(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        1_000,
    );
    Ok(())
}

#[derive(Default)]
struct NoAnalysisParts;

impl MctsParts for NoAnalysisParts {
    type Selector = UcbSelector;
    type Evaluator = DummyEvaluator;
    type Noiser = NullNoiser;

    fn params(&self) -> ParamsRef {
        Default::default()
    }

    fn selector(&self) -> Self::Selector {
        Default::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
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
    fuzz::<1, NoAnalysisParts>(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        1_000,
    );
    Ok(())
}
