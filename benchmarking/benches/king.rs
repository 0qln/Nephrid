#![allow(deprecated)]

use std::{hint::black_box, ops::ControlFlow};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use engine::core::{
    color::perspectives,
    coordinates::squares,
    r#move::Move,
    move_iter::{
        NoCheck, SingleCheck, dbg_fold_moves,
        king::{self, King, compute_attacks, lookup_attacks},
        opt::AllLegal,
        sliding_piece::magics,
    },
    position::Position,
    zobrist,
}, math};

pub fn king_attacks(c: &mut Criterion) {
    let king = squares::E4;
    let mut group = c.benchmark_group("king::attacks");

    group.bench_function("king::attacks::lookup", |b| b.iter(|| lookup_attacks(black_box(king))));
    group.bench_function("king::attacks::compute", |b| b.iter(|| compute_attacks(black_box(king))));

    group.finish();
}

pub fn king_move_iter_check_none(c: &mut Criterion) {
    math::init();
    magics::init();
    zobrist::init();

    let fen = "7k/8/8/8/8/7b/5n2/4K3 w - - 0 1";
    let pos = Position::from_fen(fen).unwrap();

    c.bench_function("king::move_iter::check_none", |b| {
        b.iter(|| {
            dbg_fold_moves::<King, NoCheck, AllLegal, _, _, _>(
                black_box(&pos),
                black_box(0),
                black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc ^ m.get_to().v())),
            )
        })
    });
}

pub fn king_move_iter_check_some(c: &mut Criterion) {
    math::init();
    magics::init();
    zobrist::init();

    let fen = "4r2k/8/8/8/8/5b2/4K3/8 w - - 0 1";
    let pos = Position::from_fen(fen).unwrap();

    c.bench_function("king::move_iter::check_some", |b| {
        b.iter(|| {
            dbg_fold_moves::<King, SingleCheck, AllLegal, _, _, _>(
                black_box(&pos),
                black_box(0),
                black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc ^ m.get_to().v())),
            )
        })
    });
}

pub fn king_move_iter_castling(c: &mut Criterion) {
    math::init();
    magics::init();
    zobrist::init();

    let inputs = [
        "4r2k/8/8/8/8/5b2/4K3/8 w KQ - 0 1",
        "2r3k1/8/8/8/8/8/8/R3K2R w KQ - 0 1",
        "6k1/8/8/8/8/8/8/R3K2R w Q - 0 1",
    ];

    for &input in &mut inputs.iter() {
        let pos = Position::from_fen(input).unwrap();
        c.bench_with_input(BenchmarkId::new("king::move_iter::castling", input), &pos, |b, pos| {
            b.iter(|| {
                king::fold_legal_castling::<perspectives::White, _, _, _>(
                    black_box(pos),
                    black_box(0),
                    black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc ^ m.get_to().v())),
                    king::nstm_attacks_for::<perspectives::White>(pos, pos.get_occupancy()),
                )
            })
        });
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
