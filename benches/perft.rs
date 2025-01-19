use std::cell::UnsafeCell;

use criterion::{criterion_group, criterion_main, Criterion};
use nephrid::engine::depth::Depth;
use nephrid::engine::fen::Fen;
use nephrid::engine::position::Position;
use nephrid::engine::search::Search;
use nephrid::uci::sync::CancellationToken;

fn bench_perft(pos: Position, depth: Depth) {
    _ = Search::perft(
        &mut UnsafeCell::new(pos),
        depth,
        CancellationToken::new(),
        |_, _| {},
    );
}

pub fn pawn_general_pos0_benchmark(c: &mut Criterion) {
    let mut fen = Fen::new("K7/3p4/k7/4P3/5P2/8/3p4/4R3 b - - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();
    c.bench_function("perft::pawns::general::pos0", |b| {
        b.iter(|| bench_perft(pos.clone(), Depth::new(3)))
    });
}

pub fn rook_general_pos0_benchmark(c: &mut Criterion) {
    let mut fen = Fen::new("1r2n2N/7r/1pP1R2p/8/5R2/k7/8/K7 w - - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();
    c.bench_function("perft::rook::general::pos_0", |b| {
        b.iter(|| bench_perft(pos.clone(), Depth::new(3)))
    });
}

criterion_group!(
    benches,
    pawn_general_pos0_benchmark,
    rook_general_pos0_benchmark
);
criterion_main!(benches);
