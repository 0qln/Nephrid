use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nephrid::engine::bitboard::Bitboard;

pub fn pop_cnt(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitboard::pop_cnt");
    
    let inputs = [0, 0x01000, 0x011000, 0x183247];

    for input in inputs {
        let bb = Bitboard { v: input };
        group.bench_with_input(BenchmarkId::new("bitboard::pop_cnt::normal", input), &bb, |b, &bb| {
            b.iter(|| bb.pop_cnt() == 1)
        });
        group.bench_with_input(BenchmarkId::new("bitboard::pop_cnt::specialized", input), &bb, |b, &bb| {
            b.iter(|| bb.pop_cnt_eq_1())
        });
    }

    group.finish();
}

criterion_group!(
    benches, 
    pop_cnt
);
criterion_main!(benches);