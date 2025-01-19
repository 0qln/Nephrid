use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nephrid::engine::bitboard::Bitboard;
use nephrid::engine::coordinates::Square;
use nephrid::engine::move_iter::rook::compute_attacks;

pub fn criterion_benchmark(c: &mut Criterion) {
    let rook = Square::E4;
    let occupancy = Bitboard { v: 0xff08104424013410_u64 };

    c.bench_function("rook::compute_attacks", |b| b.iter(|| {
        compute_attacks(black_box(rook), black_box(occupancy))
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);