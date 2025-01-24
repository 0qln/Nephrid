use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nephrid::engine::coordinates::Square;
use nephrid::engine::move_iter::king::{compute_attacks, lookup_attacks};

pub fn criterion_benchmark(c: &mut Criterion) {
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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);