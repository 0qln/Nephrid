use std::{hint::black_box, time::Duration};

use criterion::{BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime};
use engine::core::{
    color::{Perspective, colors, perspectives},
    eval::hce::TaperValue,
    r#move::Move,
    move_iter::sliding_piece::magics,
    position::Position,
    search::{
        id::{RbSet, Scorer},
        ordering::{MoveGenerator, MoveScorer, RtStage},
    },
    zobrist,
};

/// Builds a `MoveGenerator` already advanced up to (but not including)
/// `target`, so that the next `next_for` call enters the `target` stage.
///
/// Each `next_for` call advances the generator by exactly one stage, so priming
/// requires `target as u8` calls starting from the initial `YieldHashMove`
/// stage.
fn primed_generator<P: Perspective>(pos: &Position, scorer: &impl MoveScorer, target: RtStage) -> MoveGenerator {
    let mut move_gen = MoveGenerator::new(Move::null(), RbSet::default());
    while move_gen.stage() != target {
        let _ = move_gen.next_for::<P>(pos, scorer);
    }
    move_gen
}

fn bench_stage<P: Perspective>(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, fen: &str, pos: &Position, scorer: &Scorer, target: RtStage) {
    group.bench_with_input(BenchmarkId::new(name, fen), pos, |b, pos| {
        b.iter_batched(
            || primed_generator::<P>(pos, scorer, target),
            |mut move_gen| black_box(move_gen.next_for::<P>(black_box(pos), black_box(scorer))),
            BatchSize::SmallInput,
        )
    });
}

/// Benchmarks a single `MoveGenerator::next_for` stage across the positions in
/// `capture_positions.csv`.
fn bench_positions(c: &mut Criterion, group_name: &str, target: RtStage) {
    magics::init();
    zobrist::init();

    let csv_data = include_str!("../resources/capture_positions.csv");

    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(10)).sample_size(100);

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
        let scorer = Scorer {
            tt_move: Move::null(),
            killers: RbSet::default(),
            color: pos.get_turn(),
            phase: TaperValue::from_position(pos.piece_info()),
        };

        let name = format!("{group_name}_{i}");
        match pos.get_turn() {
            colors::WHITE => bench_stage::<perspectives::White>(&mut group, &name, fen, &pos, &scorer, target),
            colors::BLACK => bench_stage::<perspectives::Black>(&mut group, &name, fen, &pos, &scorer, target),
            _ => unreachable!(),
        }
    }

    group.finish();
}

pub fn generate_captures_and_promos(c: &mut Criterion) { bench_positions(c, "generate_captures_and_promos", RtStage::GenerateCapturesAndPromos); }

pub fn generate_quiets(c: &mut Criterion) { bench_positions(c, "generate_quiets", RtStage::GenerateQuiets); }

criterion_group!(benches, generate_captures_and_promos, generate_quiets);
criterion_main!(benches);
