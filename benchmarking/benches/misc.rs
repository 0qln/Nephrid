use rand::seq::SliceRandom;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use engine::{core::r#move::MAX_LEGAL_MOVES, math, misc::List};

pub fn softmax(c: &mut Criterion) {
    let data = List::<{ MAX_LEGAL_MOVES }, f32>::repeat(2., 30);
    let mut buf = Box::new(List::<{ MAX_LEGAL_MOVES }, f32>::new());
    c.bench_function("softmax", |b| {
        b.iter(|| math::softmax(black_box(data.clone()), black_box(1.), black_box(&mut buf)))
    });
}

pub fn ln_i32(c: &mut Criterion) {
    math::init();

    let mut inputs: Vec<u8> = (0..=255).collect();
    inputs.shuffle(&mut rand::rng());

    let mut group = c.benchmark_group("ln_i32_comparison");
    group.bench_function("runtime_calculation", |b| {
        b.iter(|| {
            for &x in &inputs {
                black_box(math::ln_i32_rt(black_box(x)));
            }
        })
    });
    group.bench_function("table_lookup", |b| {
        b.iter(|| {
            for &x in &inputs {
                black_box(math::ln_i32(black_box(x)));
            }
        })
    });
    group.finish();
}

pub fn lmr_u8(c: &mut Criterion) {
    math::init();

    let mut inputs: Vec<(u8, u8)> = (0..=255).zip((0..=255).rev()).collect();
    inputs.shuffle(&mut rand::rng());

    let mut group = c.benchmark_group("lmr_u8_comparison");
    group.bench_function("runtime_calculation", |b| {
        b.iter(|| {
            for &(d, m) in &inputs {
                black_box(math::lmr_u8_rt(black_box(d), black_box(m)));
            }
        })
    });
    group.bench_function("table_lookup", |b| {
        b.iter(|| {
            for &(d, m) in &inputs {
                black_box(math::lmr_u8(black_box(d), black_box(m)));
            }
        })
    });
    group.finish();
}

criterion_group!(benches, softmax, ln_i32, lmr_u8);
criterion_main!(benches);
