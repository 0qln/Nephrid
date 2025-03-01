use criterion::{criterion_group, criterion_main, Criterion};
use nephrid::engine::{
    move_iter::sliding_piece::magics, position::Position, search::mcts::Node, zobrist,
};

pub fn node_simulate(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let mut pos = Position::start_position();
    let mut root = Node::root(&pos);
    let leaf = root.select_mut();

    let mut stack = Vec::with_capacity(256);
    let mut moves = Vec::with_capacity(256);

    c.bench_function("mcts::node_simulate", |b| {
        b.iter(|| leaf.simulate(&mut pos, &mut stack, &mut moves))
    });
}

criterion_group!(benches, node_simulate,);

criterion_main!(benches);
