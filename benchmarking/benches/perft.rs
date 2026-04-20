use std::{
    fmt::{self, Display, Formatter},
    ops::ControlFlow,
    time::Duration,
};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use engine::{
    core::{
        depth::Depth,
        move_iter::{fold_legals, sliding_piece::magics},
        position::Position,
        search::{limit::UciLimit, perft::perft_inner_collect},
        zobrist,
    },
    misc::DebugMode,
    uci::sync::CancellationToken,
};

fn bench_perft<const Q: bool>(mut pos: Position, depth: Depth) {
    let limit = UciLimit { depth, ..Default::default() };
    _ = perft_inner_collect(
        &mut pos,
        limit.depth,
        &limit,
        &CancellationToken::default(),
        &DebugMode::default(),
        |_, _, _, _| {},
        |pos, list| {
            _ = fold_legals::<Q, _, _, _>(pos, (), |(), m| {
                list.push(m);
                ControlFlow::Continue::<(), ()>(())
            });
        },
    )
}

#[derive(Debug, Clone, Copy)]
pub struct Pair<T1, T2>(T1, T2);

impl<T1: Display, T2: Display> Display for Pair<T1, T2> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

fn perft(c: &mut Criterion) {
    perft_benches::<true>(c, "perft", include_str!("../resources/positions.csv"))
}

fn perft_pawn(c: &mut Criterion) {
    perft_benches::<true>(
        c,
        "perft_pawn",
        include_str!("../resources/pawn_positions.csv"),
    )
}

fn perft_rook(c: &mut Criterion) {
    perft_benches::<true>(
        c,
        "perft_rook",
        include_str!("../resources/rook_positions.csv"),
    )
}

fn perft_captures(c: &mut Criterion) {
    perft_benches::<false>(
        c,
        "perft_captures",
        include_str!("../resources/capture_positions.csv"),
    )
}

pub fn perft_benches<const Q: bool>(c: &mut Criterion, name: &str, csv_data: &str) {
    magics::init();
    zobrist::init();

    let mut group = c.benchmark_group(name);
    group
        .measurement_time(Duration::from_secs(60))
        .sample_size(20);

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

        let name = format!("{name}_{i}");

        group.bench_with_input(BenchmarkId::new(name, Pair(fen, depth)), &fen, |b, fen| {
            b.iter_batched(
                || Position::from_fen(fen).unwrap(),
                |pos| bench_perft::<Q>(pos, depth),
                criterion::BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, perft, perft_captures, perft_pawn, perft_rook);
criterion_main!(benches);
