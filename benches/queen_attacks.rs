use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nephrid::engine::bitboard::Bitboard;
use nephrid::engine::coordinates::Square;
use nephrid::engine::move_iter::queen::Queen;
use nephrid::engine::move_iter::sliding_piece::magics;
use nephrid::engine::move_iter::sliding_piece::SlidingAttacks;

pub fn criterion_benchmark(c: &mut Criterion) {
    let queen = Square::E4;
    let occupancy = Bitboard {
        v: 0x24814c6240174d0_u64,
    };
    magics::init();

    let mut group = c.benchmark_group("queen::attacks");

    group.bench_function("queen::attacks::lookup", |b| {
        b.iter(|| Queen::lookup_attacks(black_box(queen), black_box(occupancy)))
    });
    group.bench_function("queen::attacks::compute", |b| {
        b.iter(|| Queen::compute_attacks(black_box(queen), black_box(occupancy)))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);