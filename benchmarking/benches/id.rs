use std::{fmt, time::Duration};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use engine::{
    core::{
        chrono::{ChronoParams, TimeMan},
        depth::Depth,
        eval::StaticEvaluator,
        move_iter::sliding_piece::magics,
        params::{C_IdHceParams, C_IdNnueParams, IParams},
        position::Position,
        search::{
            id::{self, HceEvaluator, IdParams, NnueEvaluator, ScorerParams},
            limit::UciLimit,
            quiesce::QSearchParams,
        },
        zobrist,
    }, math, misc::{CancellationToken, DebugMode}
};
use uom::si::{information::mebibyte, u64::Information};

const NODE_TARGET: u64 = 1_000_000;

#[rustfmt::skip]
const POSITIONS: &[(&str, &str)] = &[
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
];

fn search_with_node_target<E: StaticEvaluator + Default, X: IParams>(pos: &mut Position, params: X)
where
    X::Ref: IdParams + QSearchParams + ChronoParams + ScorerParams + Clone + fmt::Debug,
{
    let params = params.shared();
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
    let mut tt = id::TT::new_of_size(hash_size);
    let mut hh = id::HH::new();
    let mut eval = E::default();
    let mut timeman = TimeMan::<X>::new_with_limits(&limit, pos, params.clone());

    id::go::<X>(pos, limit, &mut timeman, &debug, ct, &mut tt, &mut hh, &mut eval, params);
}

pub fn id_hce_nps(c: &mut Criterion) {
    math::init();
    magics::init();
    zobrist::init();

    let mut group = c.benchmark_group("id-hce");
    group
        .throughput(Throughput::Elements(NODE_TARGET))
        .measurement_time(Duration::from_secs(30))
        .sample_size(10);

    for (name, fen) in POSITIONS {
        let pos = Position::from_fen(fen).unwrap();
        group.bench_with_input(BenchmarkId::new("search", name), &pos, |b, pos| {
            b.iter_batched(
                || pos.clone(),
                |mut pos| search_with_node_target::<HceEvaluator, _>(&mut pos, C_IdHceParams),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn id_nnue_nps(c: &mut Criterion) {
    math::init();
    magics::init();
    zobrist::init();

    let mut group = c.benchmark_group("id-nnue");
    group
        .throughput(Throughput::Elements(NODE_TARGET))
        .measurement_time(Duration::from_secs(30))
        .sample_size(10);

    for (name, fen) in POSITIONS {
        let pos = Position::from_fen(fen).unwrap();
        group.bench_with_input(BenchmarkId::new("search", name), &pos, |b, pos| {
            b.iter_batched(
                || pos.clone(),
                |mut pos| search_with_node_target::<NnueEvaluator, _>(&mut pos, C_IdNnueParams),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, id_hce_nps, id_nnue_nps);
criterion_main!(benches);
