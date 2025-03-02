use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine::core::bitboard::Bitboard;
use engine::core::coordinates::Square;
use engine::core::move_iter::bishop::Bishop;
use engine::core::move_iter::sliding_piece::magics;
use engine::core::move_iter::sliding_piece::SlidingAttacks;

pub fn criterion_benchmark(c: &mut Criterion) {
    let bishop = Square::E4;
    let occupancy = Bitboard {
        v: 0x2400482000044c0_u64,
    };
    magics::init();

    let mut group = c.benchmark_group("bishop::attacks");

    group.bench_function("bishop::attacks::lookup", |b| {
        b.iter(|| Bishop::lookup_attacks(black_box(bishop), black_box(occupancy)))
    });
    group.bench_function("bishop::attacks::compute", |b| {
        b.iter(|| Bishop::compute_attacks(black_box(bishop), black_box(occupancy)))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);