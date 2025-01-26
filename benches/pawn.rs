use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nephrid::engine::{color::Color, coordinates::Square, move_iter::pawn::lookup_attacks};

pub fn pawn_attacks(c: &mut Criterion) {
    let pawn = Square::E4;
    let mut group = c.benchmark_group("pawn::attacks");

    group.bench_function("pawn::attacks::lookup", |b| {
        b.iter(|| lookup_attacks(black_box(pawn), black_box(Color::WHITE)))
    });

    group.finish();
}

criterion_group!(
    benches,
    pawn_attacks,
);
criterion_main!(benches);
