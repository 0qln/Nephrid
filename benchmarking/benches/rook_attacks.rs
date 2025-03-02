use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine::core::bitboard::Bitboard;
use engine::core::coordinates::Square;
use engine::core::move_iter::rook::Rook;
use engine::core::move_iter::sliding_piece::magics;
use engine::core::move_iter::sliding_piece::SlidingAttacks;

pub fn criterion_benchmark(c: &mut Criterion) {
    let rook = Square::E4;
    let occupancy = Bitboard {
        v: 0xff08104424013410_u64,
    };
    magics::init();

    let mut group = c.benchmark_group("rook::attacks");

    group.bench_function("rook::attacks::lookup", |b| {
        b.iter(|| Rook::lookup_attacks(black_box(rook), black_box(occupancy)))
    });
    group.bench_function("rook::attacks::compute", |b| {
        b.iter(|| Rook::compute_attacks(black_box(rook), black_box(occupancy)))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
