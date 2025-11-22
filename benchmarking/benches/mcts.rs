use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use engine::core::{
    move_iter::sliding_piece::magics, position::Position, r#move::MoveList, search::mcts::Node,
    zobrist,
};
use rand::{rngs::SmallRng, SeedableRng};

pub fn node_simulate(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let pos = Position::start_position();
    let mut root = Node::root(&pos);
    let leaf = root.select_mut();

    c.bench_function("mcts::node_simulate", |b| {
        b.iter_batched(
            || {
                (
                    (
                        Vec::with_capacity(256),
                        MoveList::default(),
                        SmallRng::seed_from_u64(1),
                    ),
                    ({ (pos.clone(), leaf.clone()) }),
                )
            },
            |((mut stack, mut moves, mut rng), (mut pos, leaf))| {
                leaf.simulate(&mut pos, &mut stack, &mut moves, &mut rng)
            },
            BatchSize::PerIteration,
        )
    });
}

criterion_group!(benches, node_simulate,);

criterion_main!(benches);
