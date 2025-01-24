use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nephrid::engine::coordinates::Square;
use nephrid::engine::move_iter::knight::{compute_attacks, lookup_attacks};

pub fn criterion_benchmark(c: &mut Criterion) {
    let knight = Square::E4;
    let mut group = c.benchmark_group("knight::attacks");

    group.bench_function("knight::attacks::lookup", |b| {
        b.iter(|| lookup_attacks(black_box(knight)))
    });
    group.bench_function("knight::attacks::compute", |b| {
        b.iter(|| compute_attacks(black_box(knight)))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);