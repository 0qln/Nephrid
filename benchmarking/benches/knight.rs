use std::ops::ControlFlow;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use engine::core::coordinates::Square;
use engine::core::r#move::Move;
use engine::core::move_iter::knight::{compute_attacks, lookup_attacks, Knight};
use engine::core::move_iter::sliding_piece::magics;
use engine::core::move_iter::{FoldMoves, NoCheck, SingleCheck};
use engine::core::position::Position;
use engine::uci::tokens::Tokenizer;

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
    magics::init();

    let inputs = [
        "N3N3/6p1/1pp2p2/8/3N4/7k/8/7K w - - 0 1",
        "N7/6p1/1pp2p2/3b4/4N3/7k/8/7K w - - 0 1",
    ];
    
    for &input in &mut inputs.iter() {
        let mut fen = Tokenizer::new(input);
        let pos = Position::try_from(&mut fen).unwrap();
        c.bench_with_input(
            BenchmarkId::new("knight::move_iter::check_none", input), &pos, |b, pos| {
            b.iter(|| {
                <Knight as FoldMoves<NoCheck>>::fold_moves(
                    black_box(pos),
                    black_box(0),
                    black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc + m.get_to().v())),
                )
            })
        });
    }
    
}

pub fn move_iter_check_single(c: &mut Criterion) {
    magics::init();

    let inputs = [
        "N7/6p1/1pp2p2/5r2/8/3b1N1k/8/2N2K2 w - - 0 1",
    ];
    
    for &input in &mut inputs.iter() {
        let mut fen = Tokenizer::new(input);
        let pos = Position::try_from(&mut fen).unwrap();
        c.bench_with_input(
            BenchmarkId::new("knight::move_iter::check_single", input), &pos, |b, pos| {
            b.iter(|| {
                <Knight as FoldMoves<SingleCheck>>::fold_moves(
                    black_box(pos),
                    black_box(0),
                    black_box(|acc, m: Move| ControlFlow::Continue::<(), _>(acc + m.get_to().v())),
                )
            })
        });
    }
    
}

criterion_group!(benches, knight_attacks, move_iter_check_none, move_iter_check_single);
criterion_main!(benches);