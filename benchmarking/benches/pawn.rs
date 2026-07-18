#![allow(deprecated)]

use std::{hint::black_box, ops::ControlFlow};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use engine::core::{
    color::colors,
    coordinates::squares,
    r#move::Move,
    move_iter::{
        NoCheck, dbg_fold_moves,
        opt::AllLegal,
        pawn::{Pawn, lookup_attacks},
        sliding_piece::magics,
    },
    position::Position,
    zobrist,
};

pub fn pawn_attacks(c: &mut Criterion) {
    let pawn = squares::E4;
    let mut group = c.benchmark_group("pawn::attacks");

    group.bench_function("pawn::attacks::lookup", |b| {
        b.iter(|| lookup_attacks(black_box(pawn), black_box(colors::WHITE)))
    });

    group.finish();
}

pub fn move_iter_check_none(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let inputs = [
        "K4n1n/6P1/k7/1p1pP3/2P5/8/5P2/8 w - d6 0 1",
        "k7/6P1/8/3pP3/1n1rr1bP/2PP3P/4P1P1/3K4 w - d6 0 1",
    ];

    for &input in &mut inputs.iter() {
        let pos = Position::from_fen(input).unwrap();
        c.bench_with_input(BenchmarkId::new("pawn::move_iter::check_none", input), &pos, |b, pos| {
            b.iter(|| {
                dbg_fold_moves::<Pawn, NoCheck, AllLegal, _, _, _>(
                    black_box(pos),
                    black_box(0),
                    black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc ^ m.get_to().v())),
                )
            })
        });
    }
}

pub fn move_iter_ep_overhead(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let mut group = c.benchmark_group("pawn::move_iter::ep");

    // Active En Passant Square
    let fen_with_ep = "k7/8/8/3pP3/8/8/8/3K4 w - d6 0 1";
    let pos_with_ep = Position::from_fen(fen_with_ep).unwrap();

    group.bench_with_input(BenchmarkId::new("active_ep_validation", fen_with_ep), &pos_with_ep, |b, pos| {
        b.iter(|| {
            dbg_fold_moves::<Pawn, NoCheck, AllLegal, _, _, _>(
                black_box(pos),
                black_box(0),
                black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc ^ m.get_to().v())),
            )
        })
    });

    // No En Passant Square
    let fen_no_ep = "k7/8/8/3pP3/8/8/8/3K4 w - - 0 1";
    let pos_no_ep = Position::from_fen(fen_no_ep).unwrap();

    group.bench_with_input(BenchmarkId::new("early_exit_ep", fen_no_ep), &pos_no_ep, |b, pos| {
        b.iter(|| {
            dbg_fold_moves::<Pawn, NoCheck, AllLegal, _, _, _>(
                black_box(pos),
                black_box(0),
                black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc ^ m.get_to().v())),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, pawn_attacks, move_iter_check_none, move_iter_ep_overhead,);
criterion_main!(benches);
