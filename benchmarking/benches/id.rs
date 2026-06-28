use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use engine::{
    core::{
        depth::Depth,
        move_iter::sliding_piece::magics,
        params::C_IdHceParams,
        position::Position,
        search::{id, limit::UciLimit},
        zobrist,
    },
    misc::{CancellationToken, DebugMode},
};
use uom::si::{information::mebibyte, u64::Information};

const NODE_TARGET: u64 = 1_000_000;

#[rustfmt::skip]
const POSITIONS: &[(&str, &str)] = &[
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
];

fn search_with_node_target(pos: &mut Position) {
    let limit = UciLimit {
        is_active: true,
        nodes: NODE_TARGET,
        depth: Depth::MAX,
        lag_buf: 0,
        ..UciLimit::max()
    };
    let debug = DebugMode::off();
    let ct = CancellationToken::new();
    let hash_size = Information::new::<mebibyte>(16);

    id::go(pos, limit, &debug, ct, hash_size, C_IdHceParams);
}

pub fn id_nps(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let mut group = c.benchmark_group("id");
    group
        .throughput(Throughput::Elements(NODE_TARGET))
        .measurement_time(Duration::from_secs(30))
        .sample_size(10);

    for (name, fen) in POSITIONS {
        let pos = Position::from_fen(fen).unwrap();
        group.bench_with_input(BenchmarkId::new("search", name), &pos, |b, pos| {
            b.iter_batched(|| pos.clone(), |mut pos| search_with_node_target(&mut pos), BatchSize::SmallInput)
        });
    }

    group.finish();
}

criterion_group!(benches, id_nps);
criterion_main!(benches);
