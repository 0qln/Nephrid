use std::ops::ControlFlow;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engine::core::coordinates::Square;
use engine::core::r#move::Move;
use engine::core::move_iter::king::{King, compute_attacks, lookup_attacks};
use engine::core::move_iter::sliding_piece::magics;
use engine::core::move_iter::{FoldMoves, NoCheck, SingleCheck, king};
use engine::core::position::Position;
use engine::core::zobrist;
use engine::uci::tokens::Tokenizer;

pub fn king_attacks(c: &mut Criterion) {
    let king = Square::E4;
    let mut group = c.benchmark_group("king::attacks");

    group.bench_function("king::attacks::lookup", |b| {
        b.iter(|| lookup_attacks(black_box(king)))
    });
    group.bench_function("king::attacks::compute", |b| {
        b.iter(|| compute_attacks(black_box(king)))
    });

    group.finish();
}

pub fn king_move_iter_check_none(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let mut fen = Tokenizer::new("7k/8/8/8/8/7b/5n2/4K3 w - - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();

    c.bench_function("king::move_iter::check_none", |b| {
        b.iter(|| {
            <King as FoldMoves<NoCheck>>::fold_moves(
                black_box(&pos),
                black_box(0),
                black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc + m.get_to().v())),
            )
        })
    });
}

pub fn king_move_iter_check_some(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let mut fen = Tokenizer::new("4r2k/8/8/8/8/5b2/4K3/8 w - - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();

    c.bench_function("king::move_iter::check_some", |b| {
        b.iter(|| {
            <King as FoldMoves<SingleCheck>>::fold_moves(
                black_box(&pos),
                black_box(0),
                black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc + m.get_to().v())),
            )
        })
    });
}

pub fn king_move_iter_castling(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let inputs = [
        "4r2k/8/8/8/8/5b2/4K3/8 w KQ - 0 1",
        "2r3k1/8/8/8/8/8/8/R3K2R w KQ - 0 1",
        "6k1/8/8/8/8/8/8/R3K2R w Q - 0 1",
    ];

    for &input in &mut inputs.iter() {
        let mut fen = Tokenizer::new(input);
        let pos = Position::try_from(&mut fen).unwrap();
        c.bench_with_input(
            BenchmarkId::new("king::move_iter::castling", input),
            &pos,
            |b, pos| {
                b.iter(|| {
                    king::fold_legal_castling(
                        black_box(pos),
                        black_box(0),
                        black_box(|acc, m: Move| {
                            ControlFlow::Continue::<(), _>(acc + m.get_to().v())
                        }),
                    )
                })
            },
        );
    }
}

criterion_group!(
    benches,
    king_attacks,
    king_move_iter_check_none,
    king_move_iter_check_some,
    king_move_iter_castling,
);
criterion_main!(benches);
