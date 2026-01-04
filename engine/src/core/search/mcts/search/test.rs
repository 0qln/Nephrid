use crate::core::{
    move_iter::sliding_piece::magics,
    position::Position,
    search::mcts::{
        back::DefaultBackuper, limiter::NoopLimiter, node::Tree, noise::NullNoiser,
        search::TreeSearcher, select::PuctSelector, test::DummyEvaluator,
    },
    zobrist,
};

use std::{error::Error, thread};

fn fuzz<const X: usize>(pos: &'static str) {
    magics::init();
    zobrist::init();

    thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let pos = Position::from_fen(pos).unwrap();

            let eval = DummyEvaluator::<X>::default();
            let limiter = NoopLimiter::default();
            let mut tree = Tree::new();

            for _i in 0..(50_000 / X) {
                let mut searcher = TreeSearcher::<X, _, _, PuctSelector<X>, _, _>::new(
                    &mut tree,
                    pos.clone(),
                    PuctSelector::default(),
                    limiter.clone(),
                    eval.clone(),
                    DefaultBackuper::default(),
                    NullNoiser,
                );
                searcher.grow();
            }
        })
        .expect("Couldn't spawn thread")
        .join()
        .expect("Should be able to join thread");
}

#[test]
pub fn fuzz_bs_1() -> Result<(), Box<dyn Error>> {
    fuzz::<1>("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Ok(())
}

#[test]
pub fn fuzz_bs_8() -> Result<(), Box<dyn Error>> {
    fuzz::<8>("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Ok(())
}

#[test]
pub fn fuzz_bs_64() -> Result<(), Box<dyn Error>> {
    fuzz::<64>("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Ok(())
}

#[test]
fn growth() {
    magics::init();
    zobrist::init();

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let pos = Position::from_fen(fen).unwrap();
    let mut tree = Tree::new();

    // Initial state checks
    assert_eq!(tree.get_root().borrow().visits(), 0);

    // Perform two growth iterations
    TreeSearcher::<1, DummyEvaluator<1>, NoopLimiter, PuctSelector<1>, DefaultBackuper, _>::new(
        &mut tree,
        pos.clone(),
        PuctSelector::default(),
        NoopLimiter,
        DummyEvaluator::default(),
        DefaultBackuper::default(),
        NullNoiser,
    )
    .grow();
    TreeSearcher::<1, DummyEvaluator<1>, NoopLimiter, PuctSelector<1>, DefaultBackuper, _>::new(
        &mut tree,
        pos.clone(),
        PuctSelector::default(),
        NoopLimiter,
        DummyEvaluator::default(),
        DefaultBackuper::default(),
        NullNoiser,
    )
    .grow();

    // Verify backpropagation
    assert!(tree.get_root().borrow().visits() >= 1);

    // Check that at least one child has been visited
    let root = tree.get_root();
    let mut child_visited = false;
    let mut i = 0;
    while let Some(branch) = root.borrow().get_branch(i) {
        if branch.node().borrow().visits() > 0 {
            child_visited = true;
            break;
        }
        i += 1;
    }
    assert!(child_visited, "At least one child should have visits");
}
