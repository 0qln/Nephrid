use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use engine::core::{
    bitboard::Bitboard,
    coordinates::squares,
    move_iter::{
        rook::Rook,
        sliding_piece::{SlidingAttacks, magics},
    },
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let rook = squares::E4;
    let occupancy = Bitboard { v: 0xff08104424013410_u64 };
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
