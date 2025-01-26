use std::ops::ControlFlow;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nephrid::engine::coordinates::Square;
use nephrid::engine::fen::Fen;
use nephrid::engine::r#move::Move;
use nephrid::engine::move_iter::knight::{self, compute_attacks, lookup_attacks};
use nephrid::engine::move_iter::sliding_piece::magics;
use nephrid::engine::position::Position;

pub fn knight_attacks(c: &mut Criterion) {
    let knight = Square::E4;
    let mut group = c.benchmark_group("knight::attacks");

    group.bench_function("knight::attacks::lookup", |b| {
        b.iter(|| lookup_attacks(black_box(knight)))
    });
    group.bench_function("knight::attacks::compute", |b| {
        b.iter(|| compute_attacks(black_box(knight)))
    });

    group.finish();
}

pub fn move_iter_check_none(c: &mut Criterion) {
    magics::init(0xdead_beef);

    let inputs = [
        "N3N3/6p1/1pp2p2/8/3N4/7k/8/7K w - - 0 1",
    ];
    
    for &input in &mut inputs.iter() {
        let mut fen = Fen::new(input);
        let pos = Position::try_from(&mut fen).unwrap();
        c.bench_with_input(
            BenchmarkId::new("knight::move_iter::check_none", input), &pos, |b, pos| {
            b.iter(|| {
                knight::fold_legals_check_none(
                    black_box(&pos),
                    black_box(0),
                    black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc + m.get_to().v())),
                )
            })
        });
    }
    
}

criterion_group!(benches, knight_attacks, move_iter_check_none);
criterion_main!(benches);