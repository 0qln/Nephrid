use core::fmt;
use std::fmt::Display;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use engine::core::{
    bitboard::Bitboard,
    coordinates::{Square, squares},
};

pub fn pop_cnt(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitboard::pop_cnt");

    let inputs = [0, 0x01000, 0x011000, 0x183247];

    for input in inputs {
        let bb = Bitboard { v: input };
        group.bench_with_input(
            BenchmarkId::new("bitboard::pop_cnt::normal", input),
            &bb,
            |b, &bb| b.iter(|| bb.pop_cnt() == 1),
        );
        group.bench_with_input(
            BenchmarkId::new("bitboard::pop_cnt::specialized", input),
            &bb,
            |b, &bb| b.iter(|| bb.pop_cnt_eq_1()),
        );
    }

    group.finish();
}

pub fn between(c: &mut Criterion) {
    let inputs = [
        Pair(squares::B2, squares::G7),
        Pair(squares::E1, squares::E8),
        Pair(squares::G7, squares::B2),
        Pair(squares::E8, squares::E1),
    ];

    for pair in inputs {
        c.bench_with_input(
            BenchmarkId::new("bitboard::between", pair),
            &pair,
            |b, &pair| b.iter(|| Bitboard::between(pair.0, pair.1)),
        );
    }
}

pub fn ray(c: &mut Criterion) {
    let inputs = [
        Pair(squares::B2, squares::G7),
        Pair(squares::E1, squares::E8),
        Pair(squares::G7, squares::B2),
        Pair(squares::E8, squares::E1),
        Pair(squares::A8, squares::E1),
    ];

    for pair in inputs {
        c.bench_with_input(
            BenchmarkId::new("bitboard::ray", pair),
            &pair,
            |b, &pair| b.iter(|| Bitboard::ray(pair.0, pair.1)),
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct Pair<T>(T, T);

impl<T: Display> fmt::Display for Pair<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

criterion_group!(benches, pop_cnt, between, ray,);
criterion_main!(benches);
