use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use engine::core::{
    move_iter::sliding_piece::magics,
    position::Position,
    search::mcts::{
        back::DefaultBackuper, limiter::NoopLimiter, node::Tree, noise::NullNoiser,
        search::TreeSearcher, select::puct::PuctSelector, test::DummyEvaluator,
    },
    zobrist,
};

pub fn tree_grow(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    c.bench_function("mcts::tree::grow", |b| {
        b.iter_batched(
            || (Position::start_position(), Tree::default()),
            |(mut position, mut tree)| {
                let mut searcher = TreeSearcher::<1, _, _, _, _, _>::new(
                    &mut position,
                    PuctSelector::default(),
                    NoopLimiter::default(),
                    DummyEvaluator::default(),
                    DefaultBackuper::default(),
                    NullNoiser::default(),
                );
                searcher.init_root(&mut tree);
                for _ in 0..10_000 {
                    searcher.grow(&mut tree);
                }
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, tree_grow,);

criterion_main!(benches);
