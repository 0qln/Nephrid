use std::cell::UnsafeCell;

use criterion::{criterion_group, criterion_main, Criterion};
use engine::core::depth::Depth;
use engine::core::fen::Fen;
use engine::core::move_iter::sliding_piece::magics;
use engine::core::position::Position;
use engine::core::search::Search;
use engine::uci::sync::CancellationToken;

fn bench_perft(pos: Position, depth: Depth) {
    _ = Search::perft(
        &mut UnsafeCell::new(pos),
        depth,
        CancellationToken::new(),
        |_, _| {},
    );
}

fn bench_pos(c: &mut Criterion, name: &str, mut fen: Fen, depth: Depth) {
    magics::init();
    let pos = Position::try_from(&mut fen).unwrap();
    c.bench_function(name, |b| b.iter(|| bench_perft(pos.clone(), depth)));
}

pub fn perft_0_benchmark(c: &mut Criterion) {
    bench_pos(c, "perft::0",
        Fen::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        Depth::new(5),
    )
}

pub fn perft_1_benchmark(c: &mut Criterion) {
    bench_pos(c, "perft::1",
        Fen::new("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        Depth::new(5),
    )
}

pub fn perft_2_benchmark(c: &mut Criterion) {
    bench_pos(c, "perft::2",
        Fen::new("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        Depth::new(6),
    )
}

pub fn perft_3_benchmark(c: &mut Criterion) {
    bench_pos(c, "perft::3",
        Fen::new("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
        Depth::new(5),
    )
}

pub fn perft_4_benchmark(c: &mut Criterion) {
    bench_pos(c, "perft::4",
        Fen::new("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  "),
        Depth::new(5),
    )
}

pub fn perft_5_benchmark(c: &mut Criterion) {
    bench_pos(c, "perft::5",
        Fen::new("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 "),
        Depth::new(5),
    )
}

pub fn pawn_general_pos0_benchmark(c: &mut Criterion) {
    bench_pos(c, "perft::pawns::general::pos0",
        Fen::new("K7/3p4/k7/4P3/5P2/8/3p4/4R3 b - - 0 1") ,
        Depth::new(3),      
    )
}

pub fn rook_general_pos0_benchmark(c: &mut Criterion) {
    bench_pos(c, "perft::rook::general::pos_0",
        Fen::new("1r2n2N/7r/1pP1R2p/8/5R2/k7/8/K7 w - - 0 1"),
        Depth::new(3),
    )
}

criterion_group!(
    benches,
    pawn_general_pos0_benchmark,
    rook_general_pos0_benchmark,
    perft_0_benchmark,
    perft_1_benchmark,
    perft_2_benchmark,
    perft_3_benchmark,
    perft_4_benchmark,
    perft_5_benchmark
);
criterion_main!(benches);
