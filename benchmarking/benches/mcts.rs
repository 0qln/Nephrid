use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use engine::core::{
    move_iter::sliding_piece::magics,
    position::Position,
    search::mcts::{
        HceParts, MctsParts, limiter::DefaultLimiter, node::Tree, search::TreeSearcher,
    },
    zobrist,
};

fn bench_mcts<P: MctsParts>(mut pos: Position, mut tree: Tree, parts: P) {
    let mut searcher = TreeSearcher::<1, _, _, _, _, _>::new(
        &mut pos,
        parts.selector(),
        DefaultLimiter::default(),
        parts.evaluator(),
        parts.backprop(),
        parts.noiser(),
    );
    searcher.init_root(&mut tree);
    for _ in 0..20_000 {
        searcher.grow(&mut tree);
    }
}

pub fn mcts_benches(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let mut group = c.benchmark_group("mcts");
    group
        .measurement_time(Duration::from_secs(60))
        .sample_size(20);

    let csv_data = include_str!("../resources/positions.csv");

    for (i, line) in csv_data.lines().skip(1).enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let Some((_depth_str, fen_str)) = line.split_once(',')
        else {
            panic!("Invalid CSV line: {}", line);
        };

        let fen = fen_str.trim();

        let name = format!("fuzz_{i}");

        group.bench_with_input(BenchmarkId::new(name, fen), &fen, |b, &fen| {
            b.iter_batched(
                || {
                    (
                        Position::from_fen(fen).unwrap(),
                        Tree::default(),
                        HceParts::default(),
                    )
                },
                |(pos, tree, parts)| bench_mcts(pos, tree, parts),
                BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, mcts_benches);
criterion_main!(benches);
