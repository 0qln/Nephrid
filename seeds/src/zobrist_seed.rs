use std::ops::ControlFlow;

use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};

use engine::{core::{r#move::MoveList, move_iter::sliding_piece::magics, position::Position, zobrist}, math};

fn find_seeds() {
    math::init();
    magics::init();

    let mut seed = 14278029879823863027;
    let mut min = usize::MAX;
    loop {
        zobrist::force_init(seed);
        let r = test_seed(2000, &mut SmallRng::seed_from_u64(0xdeadbeef), min);
        if r.total_collisions < min {
            min = r.total_collisions;
            println!("collisions: {}, seed: {}", r.total_collisions, seed);
        }
        seed = SmallRng::seed_from_u64(seed).next_u64();
    }
}

struct SeedTestResult {
    total_collisions: usize,
}

fn test_seed(rounds: usize, rng: &mut SmallRng, min: usize) -> SeedTestResult {
    let pos = Position::start_position();
    let collisions = (0..rounds)
        .try_fold(0, |acc, _| {
            if acc >= min {
                return ControlFlow::Break(());
            }

            let pos = &mut pos.clone();

            // simulate a random game...
            loop {
                if let Some(_result) = pos.game_result() {
                    break;
                }

                let buffer = pos.collect_legals(MoveList::new());

                let mov = buffer.as_slice()[rng.random_range(0..buffer.len()) as usize];
                pos.make_move(mov, &mut ());
            }

            todo!(
                "The repetition_table does not exist anymore. If you want to tune the zobrist seeds, you're gonna have to implement your own \
                 hash-collision counting and/or statistics here."
            );
            // let collisions_per_ply = pos.repetition_table_collisions() as f32
            // / pos.ply().v as f32;
            // let collisions_per_ply_threshold = 0.002142857 *
            // pos.repetition_table_capacity() as f32;
            // if collisions_per_ply > collisions_per_ply_threshold {
            //     return ControlFlow::Break(());
            // }
            // ControlFlow::Continue(acc + pos.repetition_table_collisions())
        })
        .continue_value()
        .unwrap_or(usize::MAX);
    SeedTestResult { total_collisions: collisions }
}

fn main() { find_seeds(); }
