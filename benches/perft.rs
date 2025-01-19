use std::cell::UnsafeCell;

use criterion::{criterion_group, criterion_main, Criterion};
use nephrid::engine::depth::Depth;
use nephrid::engine::fen::Fen;
use nephrid::engine::position::Position;
use nephrid::engine::search::Search;
use nephrid::uci::sync::CancellationToken;

fn bench_perft(mut fen: Fen, depth: Depth) {
    let pos = Position::try_from(&mut fen).unwrap();
    _ = Search::perft(
        &mut UnsafeCell::new(pos),
        depth,
        CancellationToken::new(),
        |_, _| {},
    );
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("perft::pos_0", |b| {
        b.iter(|| {
            bench_perft(
                Fen::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                Depth::new(3),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
