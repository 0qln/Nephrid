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

criterion_group!(benches, softmax, ln_i32);
criterion_main!(benches);
