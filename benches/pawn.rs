use std::ops::ControlFlow;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nephrid::engine::{
    color::Color, coordinates::Square, fen::Fen, r#move::Move, move_iter::{
        pawn::{self, lookup_attacks, Pawn},
        sliding_piece::magics, FoldMoves, NoCheck,
    }, position::Position
};

pub fn pawn_attacks(c: &mut Criterion) {
    let pawn = Square::E4;
    let mut group = c.benchmark_group("pawn::attacks");

    group.bench_function("pawn::attacks::lookup", |b| {
        b.iter(|| lookup_attacks(black_box(pawn), black_box(Color::WHITE)))
    });

    group.finish();
}

pub fn move_iter_check_none(c: &mut Criterion) {
    magics::init(0xdead_beef);

    let inputs = [
        "K4n1n/6P1/k7/1p1pP3/2P5/8/5P2/8 w - d6 0 1",
    ];

    for &input in &mut inputs.iter() {
        let mut fen = Fen::new(input);
        let pos = Position::try_from(&mut fen).unwrap();
        c.bench_with_input(
            BenchmarkId::new("pawn::move_iter::check_none", input),
            &pos,
            |b, pos| {
                b.iter(|| {
                    <Pawn as FoldMoves<NoCheck>>::fold_moves(
                        black_box(&pos),
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

criterion_group!(benches, pawn_attacks, move_iter_check_none);
criterion_main!(benches);
