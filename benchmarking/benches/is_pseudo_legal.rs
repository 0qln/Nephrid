use std::{hint::black_box, ops::ControlFlow, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use engine::core::{
    move_iter::{fold_legal_moves, sliding_piece::magics},
    position::Position,
    r#move::Move,
    zobrist,
};

fn collect_legal_moves(pos: &Position) -> Vec<Move> {
    fold_legal_moves(pos, Vec::new(), |mut acc, m| {
        acc.push(m);
        ControlFlow::Continue::<(), _>(acc)
    })
    .continue_value()
    .unwrap()
}

pub fn is_pseudo_legal(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let csv_data = include_str!("../resources/positions.csv");

    let mut group = c.benchmark_group("is_pseudo_legal");
    group
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);

    for (i, line) in csv_data.lines().skip(1).enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let Some((_, fen_str)) = line.split_once(',')
        else {
            panic!("Invalid CSV line: {}", line);
        };
        let fen = fen_str.trim();

        let pos = Position::from_fen(fen).unwrap();
        let moves = collect_legal_moves(&pos);

        let name = format!("is_pseudo_legal_{i}");
        group.bench_with_input(BenchmarkId::new(name, fen), &moves, |b, moves| {
            b.iter(|| {
                for &m in moves {
                    black_box(pos.is_pseudo_legal(black_box(m)));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, is_pseudo_legal);
criterion_main!(benches);
