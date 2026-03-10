use std::fmt::{self, Display, Formatter};

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

fn run(c: &mut Criterion, name: &str, Pair(fen, depth): Pair<&str, Depth>) {
    magics::init();
    zobrist::init();

    let mut group = c.benchmark_group("perft");

    group.sample_size(10).bench_with_input(
        BenchmarkId::new(name, Pair(fen, depth)),
        &fen,
        |b, fen| {
            b.iter_batched(
                || Position::from_fen(fen).unwrap(),
                |pos| bench_perft(pos, depth),
                criterion::BatchSize::LargeInput,
            )
        },
    );
}

pub fn perft_0(c: &mut Criterion) {
    run(
        c,
        "perft_0",
        Pair(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            Depth::new(5),
        ),
    );
}
pub fn perft_1(c: &mut Criterion) {
    run(
        c,
        "perft_1",
        Pair(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            Depth::new(5),
        ),
    );
}
pub fn perft_2(c: &mut Criterion) {
    run(
        c,
        "perft_2",
        Pair("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", Depth::new(6)),
    );
}

pub fn perft_3(c: &mut Criterion) {
    run(
        c,
        "perft_3",
        Pair(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            Depth::new(5),
        ),
    );
}

pub fn perft_4(c: &mut Criterion) {
    run(
        c,
        "perft_4",
        Pair(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ",
            Depth::new(5),
        ),
    );
}
pub fn perft_5(c: &mut Criterion) {
    run(
        c,
        "perft_5",
        Pair(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ",
            Depth::new(5),
        ),
    );
}

pub fn pawn_perft(c: &mut Criterion) {
    run(
        c,
        "pawn_perft",
        Pair("K7/3p4/k7/4P3/5P2/8/3p4/4R3 b - - 0 1", Depth::new(3)),
    );
}

pub fn rook_perft(c: &mut Criterion) {
    run(
        c,
        "rook_perft",
        Pair("1r2n2N/7r/1pP1R2p/8/5R2/k7/8/K7 w - - 0 1", Depth::new(3)),
    );
}

#[derive(Debug, Clone, Copy)]
pub struct Pair<T1, T2>(T1, T2);

impl<T1: Display, T2: Display> Display for Pair<T1, T2> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

criterion_group!(
    benches, perft_0, perft_1, perft_2, perft_3, perft_4, perft_5, pawn_perft, rook_perft,
);
criterion_main!(benches);
