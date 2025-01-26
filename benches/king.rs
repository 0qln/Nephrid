use std::ops::ControlFlow;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use nephrid::engine::coordinates::Square;
use nephrid::engine::fen::Fen;
use nephrid::engine::move_iter::king::{compute_attacks, lookup_attacks};
use nephrid::engine::move_iter::sliding_piece::magics;
use nephrid::engine::move_iter::{fold_legal_move, king};
use nephrid::engine::position::Position;
use nephrid::engine::r#move::Move;

pub fn king_attacks(c: &mut Criterion) {
    let king = Square::E4;
    let mut group = c.benchmark_group("king::attacks");

    group.bench_function("king::attacks::lookup", |b| {
        b.iter(|| lookup_attacks(black_box(king)))
    });
    group.bench_function("king::attacks::compute", |b| {
        b.iter(|| compute_attacks(black_box(king)))
    });

    group.finish();
}

pub fn king_move_iter_check_none(c: &mut Criterion) {
    magics::init(0xdead_beef);

    let mut fen = Fen::new("7k/8/8/8/8/7b/5n2/4K3 w - - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();

    c.bench_function("king::move_iter::check_none", |b| {
        b.iter(|| {
            king::fold_legals_check_none(
                black_box(&pos),
                black_box(0),
                black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc + m.get_to().v())),
            )
        })
    });
}

pub fn king_move_iter_check_some(c: &mut Criterion) {
    magics::init(0xdead_beef);

    let mut fen = Fen::new("4r2k/8/8/8/8/5b2/4K3/8 w - - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();

    c.bench_function("king::move_iter::check_some", |b| {
        b.iter(|| {
            king::fold_legals_check_some(
                black_box(&pos),
                black_box(0),
                black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc + m.get_to().v())),
            )
        })
    });
}
criterion_group!(benches, king_attacks, king_move_iter_check_none, king_move_iter_check_some);
criterion_main!(benches);
