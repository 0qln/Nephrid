use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use engine::{
    core::{
        r#move::MAX_LEGAL_MOVES,
        search::mcts::eval::{self},
    },
    misc::List,
};

pub fn softmax(c: &mut Criterion) {
    let data = List::<{ MAX_LEGAL_MOVES }, f32>::repeat(2., 30);
    let mut buf = Box::new(List::<{ MAX_LEGAL_MOVES }, f32>::new());
    c.bench_function("softmax", |b| {
        b.iter(|| eval::softmax(black_box(data.clone()), black_box(1.), black_box(&mut buf)))
    });
}

criterion_group!(benches, softmax);
criterion_main!(benches);
