use std::cell::UnsafeCell;
use std::fmt::{self, Display, Formatter};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use engine::core::depth::Depth;
use engine::core::move_iter::sliding_piece::magics;
use engine::core::position::Position;
use engine::core::search::Search;
use engine::core::zobrist;
use engine::uci::sync::CancellationToken;
use engine::uci::tokens::Tokenizer;

fn bench_perft(pos: Position, depth: Depth) {
    println!("{pos:?}");
    _ = Search::perft(
        &mut UnsafeCell::new(pos),
        depth,
        CancellationToken::new(),
        |m, _| println!("{m:?}"),
    );
}

fn run(c: &mut Criterion, name: &str, pairs: &[Pair<&str, Depth>]) {
    magics::init();
    zobrist::init();

    for p in pairs {
        let mut fen = Tokenizer::new(p.0);
        let pos = Position::try_from(&mut fen).unwrap();
        c.bench_with_input(BenchmarkId::new(name, p), &pos, |b, x| {
            b.iter(|| bench_perft(x.clone(), p.1))
        });
    }
}

pub fn perft(c: &mut Criterion) {
    run(
        c,
        "perft",
        &[
            Pair(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                Depth::new(5),
            ),
            Pair(
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                Depth::new(5),
            ),
            Pair("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", Depth::new(6)),
            Pair(
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                Depth::new(5),
            ),
            Pair(
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ",
                Depth::new(5),
            ),
            Pair(
                "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ",
                Depth::new(5),
            ),
        ],
    );
}

pub fn pawn_perft(c: &mut Criterion) {
    run(
        c,
        "pawn_perft",
        &[Pair("K7/3p4/k7/4P3/5P2/8/3p4/4R3 b - - 0 1", Depth::new(3))],
    );
}

pub fn rook_perft(c: &mut Criterion) {
    run(
        c,
        "rook_perft",
        &[Pair(
            "1r2n2N/7r/1pP1R2p/8/5R2/k7/8/K7 w - - 0 1",
            Depth::new(3),
        )],
    );
}

#[derive(Debug, Clone, Copy)]
pub struct Pair<T1, T2>(T1, T2);

impl<T1: Display, T2: Display> Display for Pair<T1, T2> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

criterion_group!(benches, perft, pawn_perft, rook_perft,);
criterion_main!(benches);
