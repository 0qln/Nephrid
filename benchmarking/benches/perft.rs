use std::{
    fmt::{self, Display, Formatter},
    time::Duration,
};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use engine::{
    core::{
        depth::Depth, move_iter::sliding_piece::magics, position::Position, search::limit::Limit,
        zobrist,
    },
    misc::DebugMode,
    uci::sync::CancellationToken,
};

fn bench_perft(pos: Position, depth: Depth) {
    let limit = Limit { depth, ..Default::default() };
    _ = engine::core::search::perft::perft(
        pos,
        limit,
        CancellationToken::new(),
        DebugMode::default(),
    );
}

#[derive(Debug, Clone, Copy)]
pub struct Pair<T1, T2>(T1, T2);

impl<T1: Display, T2: Display> Display for Pair<T1, T2> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

pub fn perft_benches(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let mut group = c.benchmark_group("perft");
    group
        .measurement_time(Duration::from_secs(60))
        .sample_size(20);

    let csv_data = include_str!("../resources/positions.csv");

    for (i, line) in csv_data.lines().skip(1).enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let Some((depth_str, fen_str)) = line.split_once(',')
        else {
            panic!("Invalid CSV line: {}", line);
        };

        let depth_val = depth_str
            .trim()
            .parse()
            .expect("Depth should be a valid number");
        let depth = Depth::new(depth_val);
        let fen = fen_str.trim();

        let name = format!("perft_{i}");

        group.bench_with_input(BenchmarkId::new(name, Pair(fen, depth)), &fen, |b, fen| {
            b.iter_batched(
                || Position::from_fen(fen).unwrap(),
                |pos| bench_perft(pos, depth),
                criterion::BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, perft_benches);
criterion_main!(benches);
