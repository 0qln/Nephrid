use std::ops::ControlFlow;

use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};

use engine::core::{
    move_iter::{fold_legal_moves, sliding_piece::magics},
    position::Position,
    search::mcts,
    zobrist,
};

fn find_seeds() {
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
    let buffer = &mut vec![];
    let collisions = (0..rounds)
        .try_fold(0, |acc, _| {
            if acc >= min {
                return ControlFlow::Break(());
            }

            let pos = &mut pos.clone();

            // simulate a random game...
            loop {
                buffer.clear();
                fold_legal_moves(pos, &mut *buffer, |acc, m| {
                    ControlFlow::Continue::<(), _>({
                        acc.push(m);
                        acc
                    })
                });

                if mcts::PlayoutResult::maybe_new(pos, buffer.len() as u8).is_some() {
                    break;
                }

                let mov = buffer[rng.random_range(0..buffer.len())];
                pos.make_move(mov);
            }

            let collisions_per_ply = pos.repetition_table_collisions() as f32 / pos.ply().v as f32;
            let collisions_per_ply_threshold = 0.002142857 * pos.repetition_table_capacity() as f32;
            if collisions_per_ply > collisions_per_ply_threshold {
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(acc + pos.repetition_table_collisions())
        })
        .continue_value()
        .unwrap_or(usize::MAX);
    SeedTestResult {
        total_collisions: collisions,
    }
}

fn main() {
    find_seeds();
}
