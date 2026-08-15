use std::{collections::HashMap, env::var, fs, ops::ControlFlow, path::PathBuf};

use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};

use engine::{
    core::{
        castling::CastlingRights,
        coordinates::EpCaptureSquare,
        r#move::MoveList,
        move_iter::sliding_piece::magics,
        piece::Piece,
        position::{EpdLineImport, Position},
        turn::Turn,
        zobrist,
    },
    uci::tokens::Tokenizer,
};

fn find_seeds() {
    magics::init();

    let epd_lines = {
        let mut path = PathBuf::new();
        path.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
        path.push("ccbench/in/books/popularpos_lichess_v3.epd");
        fs::read_to_string(&path).expect("Couldn't read path")
    };

    let all_positions: Vec<Position> = epd_lines
        .lines()
        .filter_map(|l| {
            let mut tok = Tokenizer::new(l);
            let (pos, _ops) = EpdLineImport(&mut tok).try_into().ok()?;
            Some(pos)
        })
        .collect();

    println!("Loaded {} positions into memory.", all_positions.len());

    let get_positions = |n: usize| all_positions[..n].iter().cloned();
    let moves_rng = || SmallRng::seed_from_u64(0x_dead_beef);

    let mut num_positions = 1;
    let mut min_collisions = usize::MAX;
    let mut seed = 9612274973016456243;

    loop {
        if num_positions > all_positions.len() {
            println!("reached max available dataset positions ({})!", all_positions.len());
            break;
        }

        zobrist::force_init(seed);

        let r = test_seed(get_positions(num_positions), &mut moves_rng(), min_collisions);
        if r.total_collisions < min_collisions {
            min_collisions = r.total_collisions;
            println!("positions: {}, collisions: {}, seed: {}", num_positions, r.total_collisions, seed);
        }

        // too good?
        if min_collisions == 0 {
            println!("seed {seed} perfect for {num_positions} positions. escalating...");
            num_positions += 1;
            min_collisions = test_seed(get_positions(num_positions), &mut moves_rng(), usize::MAX).total_collisions;
        }
        // too bad?
        else {
            seed = SmallRng::seed_from_u64(seed).next_u64();
        }
    }
}

struct SeedTestResult {
    total_collisions: usize,
}

#[derive(PartialEq, Eq)]
struct ZobristSource {
    pieces: [Piece; 64],
    turn: Turn,
    castling: CastlingRights,
    ep_capture_square: EpCaptureSquare,
}

impl From<&Position> for ZobristSource {
    fn from(pos: &Position) -> Self {
        Self {
            pieces: pos.get_pieces().clone(),
            turn: pos.get_turn(),
            castling: pos.get_castling(),
            ep_capture_square: pos.get_ep_capture_square(),
        }
    }
}

fn test_seed(mut positions: impl Iterator<Item = Position>, rng: &mut SmallRng, min: usize) -> SeedTestResult {
    const MAX_DEPTH: usize = 10; // 2^10 = 1024[nodes/position]

    let mut seen_positions: HashMap<zobrist::Hash, ZobristSource> = HashMap::new();

    let collisions = positions
        .try_fold(0, |mut collisions, mut pos| {
            if collisions >= min {
                return ControlFlow::Break(());
            }

            simulate_search(&mut pos, 0, MAX_DEPTH, rng, &mut seen_positions, &mut collisions, min);

            if collisions >= min {
                ControlFlow::Break(())
            }
            else {
                ControlFlow::Continue(collisions)
            }
        })
        .continue_value()
        .unwrap_or(usize::MAX);

    SeedTestResult { total_collisions: collisions }
}

fn simulate_search(
    pos: &mut Position,
    depth: usize,
    max_depth: usize,
    rng: &mut SmallRng,
    seen_positions: &mut HashMap<zobrist::Hash, ZobristSource>,
    collisions: &mut usize,
    min: usize,
) {
    if *collisions >= min {
        return;
    }

    let hash = pos.get_key();
    let current_source = ZobristSource::from(&*pos);

    match seen_positions.get(&hash) {
        Some(existing_source) if existing_source != &current_source => {
            *collisions += 1;
            if *collisions >= min {
                return;
            }
        }
        None => {
            seen_positions.insert(hash, current_source);
        }
        _ => {}
    }

    if depth >= max_depth || pos.game_result().is_some() {
        return;
    }

    let moves = pos.collect_legals(MoveList::new());
    let slice = moves.as_slice();

    if moves.is_empty() {
        return;
    }

    // Pick up to 2 distinct random moves at this node
    let selected_indices: [isize; 2] = match slice.len() {
        0 => unreachable!(),
        1 => [0, -1],
        2 => [0, 1],
        n @ 3.. => {
            let idx1 = rng.random_range(0..n) as isize;
            let mut idx2 = rng.random_range(0..n - 1) as isize;
            if idx2 >= idx1 {
                idx2 += 1;
            }
            [idx1, idx2]
        }
    };

    for idx in selected_indices.iter().filter_map(|&i| usize::try_from(i).ok()) {
        let mov = slice[idx];

        pos.make_move(mov, &mut ());
        simulate_search(pos, depth + 1, max_depth, rng, seen_positions, collisions, min);
        pos.unmake_move(mov, &mut ());

        if *collisions >= min {
            break;
        }
    }
}

fn main() { find_seeds(); }
