use crate::{
    core::{
        move_iter::sliding_piece::magics,
        position::Position,
        search::mcts::{
            back::DefaultBackuper, limiter::NoopLimiter, node::Tree, search::TreeSearcher,
            select::PuctSelector, test::DummyEvaluator,
        },
        zobrist,
    },
    uci::tokens::Tokenizer,
};

use std::{error::Error, thread};

fn fuzz<const X: usize>(pos: &'static str) {
    magics::init();
    zobrist::init();

    thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
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

    let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();
    let mut tree = Tree::new();

    // Create TreeSearcher with dummy evaluator
    let evaluator = DummyEvaluator::default();
    let limiter = NoopLimiter;
    let mut searcher = TreeSearcher::<
        1,
        DummyEvaluator<1>,
        NoopLimiter,
        PuctSelector<1>,
        DefaultBackuper,
    >::new(&mut tree, pos.clone(), limiter, evaluator);

    // Initial state checks
    assert_eq!(searcher.tree.get_root().borrow().visits(), 0);

    // Perform one growth iteration
    searcher.grow();

    // Verify backpropagation
    assert_eq!(tree.get_root().borrow().visits(), 1);

    // Check that at least one child has been visited
    let root = tree.get_root();
    root.borrow_mut().expand(&pos);
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
