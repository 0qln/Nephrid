#![feature(const_default)]
#![feature(const_trait_impl)]
#![feature(derive_const)]
#![allow(clippy::too_many_arguments)]

use engine::{
    core::{
        castling::CastlingRights,
        coordinates::EpCaptureSquare,
        r#move::MoveList,
        move_iter::sliding_piece::magics,
        piece::Piece,
        position::{EpdLineImport, Position},
        search::data::{ReplacementStrategy, TTKey, TranspositionTable},
        turn::Turn,
        zobrist,
    },
    uci::tokens::Tokenizer,
};
use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};
use std::{env::var, fs, io::Write, path::PathBuf};

const SEED_BASELINE: u64 = 14933185240663103696;
const EVAL_POSITIONS: usize = 500;
const TT_SIZE: usize = 22;
const MAX_DEPTH: usize = 10; // 2^10 = 1024 [nodes/position]

fn load_positions(limit: usize) -> Vec<Position> {
    let epd_lines = {
        let mut path = PathBuf::new();
        path.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
        path.push("ccbench/in/books/popularpos_lichess_v3.epd");
        fs::read_to_string(&path).expect("Couldn't read path")
    };

    epd_lines
        .lines()
        .filter_map(|l| {
            let mut tok = Tokenizer::new(l);
            let (pos, _ops) = EpdLineImport(&mut tok).try_into().ok()?;
            Some(pos)
        })
        .take(limit)
        .collect()
}

fn load_seed() -> u64 {
    // Allow resuming from command-line argument, defaulting to the initial seed
    std::env::args().nth(1).and_then(|arg| arg.parse::<u64>().ok()).unwrap_or(SEED_BASELINE)
}

fn find_seeds() {
    magics::init();

    let mut all_positions: Vec<Position> = load_positions(EVAL_POSITIONS);
    println!("Loaded {} positions into memory.", all_positions.len());

    let moves_rng = || SmallRng::seed_from_u64(0x_dead_beef);

    let mut tt = TT::new(1 << TT_SIZE);
    let mut seed = load_seed();
    let mut seed_i = 0;

    zobrist::force_init(SEED_BASELINE);
    let mut best_r = test_seed(&mut all_positions, &mut moves_rng(), &mut tt, &SeedTestResult::worst());
    println!(
        "\r\x1b[2K[+] base: {SEED_BASELINE}, collisions: {}, insertions: {}, collisions/insertion: {:.8e}",
        best_r.total_collisions,
        best_r.total_insertions,
        best_r.avg_collisions_per_insertion().unwrap_or(0.)
    );

    loop {
        // Print the active seed on the bottom line (\r moves cursor to start, \x1b[2K
        // clears line)
        print!("\r\x1b[2KTesting seed {seed_i}: {seed}");
        let _ = std::io::stdout().flush();

        zobrist::force_init(seed);
        let r = test_seed(&mut all_positions, &mut moves_rng(), &mut tt, &best_r);

        if r.bound == Bound::Exact && r < best_r {
            let avg_rate = r.avg_collisions_per_insertion().unwrap_or(0.);

            // Clear the status line before logging the finding so it prints cleanly above
            println!(
                "\r\x1b[2K[+] seed: {seed}, collisions: {}, insertions: {}, collisions/insertion: {:.8e}",
                r.total_collisions, r.total_insertions, avg_rate
            );

            best_r = r;
        }

        seed = SmallRng::seed_from_u64(seed).next_u64();
        seed_i += 1;
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
enum Bound {
    #[default]
    Exact,
    Lower,
}

#[derive(Default)]
struct SeedTestResult {
    total_collisions: usize,
    total_insertions: usize,
    bound: Bound,
}

impl PartialOrd for SeedTestResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self.avg_collisions_per_insertion(), other.avg_collisions_per_insertion()) {
            (Some(a), Some(b)) => a.partial_cmp(&b),
            _ => None,
        }
    }
}

impl PartialEq for SeedTestResult {
    fn eq(&self, other: &Self) -> bool { self.avg_collisions_per_insertion() == other.avg_collisions_per_insertion() }
}

impl SeedTestResult {
    fn worst() -> Self {
        Self {
            total_collisions: usize::MAX,
            total_insertions: 1,
            bound: Bound::Lower,
        }
    }

    fn avg_collisions_per_insertion(&self) -> Option<f64> {
        if self.total_insertions > 0 {
            Some(self.total_collisions as f64 / self.total_insertions as f64)
        }
        else {
            None
        }
    }
}

type TT = TranspositionTable<OccupancyIndicator, AlwaysReplace>;

#[derive(PartialEq, Eq)]
struct ZobristSource {
    pieces: [Piece; 64],
    turn: Turn,
    castling: CastlingRights,
    ep_capture_square: EpCaptureSquare,
}

impl Clone for ZobristSource {
    fn clone(&self) -> Self {
        Self {
            pieces: self.pieces,
            turn: self.turn,
            castling: self.castling,
            ep_capture_square: self.ep_capture_square,
        }
    }
}

const impl Default for ZobristSource {
    fn default() -> Self {
        Self {
            pieces: [Default::default(); 64],
            turn: Default::default(),
            castling: Default::default(),
            ep_capture_square: Default::default(),
        }
    }
}

impl From<&Position> for ZobristSource {
    fn from(pos: &Position) -> Self {
        Self {
            pieces: *pos.get_pieces(),
            turn: pos.get_turn(),
            castling: pos.get_castling(),
            ep_capture_square: pos.get_ep_capture_square(),
        }
    }
}

#[derive(Clone)]
#[derive_const(Default)]
struct OccupancyIndicator {
    src: ZobristSource,
    key: zobrist::Hash,
}

impl TTKey for OccupancyIndicator {
    fn key(&self) -> zobrist::Hash { self.key }
}

struct AlwaysReplace;

impl ReplacementStrategy for AlwaysReplace {
    type Data = OccupancyIndicator;
    fn should_replace(_existing: &Self::Data, _new: &Self::Data) -> bool { true }
}

fn test_seed(positions: &mut [Position], rng: &mut SmallRng, tt: &mut TT, max: &SeedTestResult) -> SeedTestResult {
    tt.clear();

    let mut r = SeedTestResult::default();

    for pos in positions.iter_mut() {
        if r.total_collisions >= max.total_collisions {
            r.bound = Bound::Lower;
            break;
        }

        simulate_search(
            pos,
            0,
            MAX_DEPTH,
            rng,
            tt,
            &mut r.total_collisions,
            &mut r.total_insertions,
            max.total_collisions,
        );
    }

    r
}

fn simulate_search(
    pos: &mut Position,
    depth: usize,
    max_depth: usize,
    rng: &mut SmallRng,
    tt: &mut TT,
    collisions: &mut usize,
    insertions: &mut usize,
    max_collisions: usize,
) {
    if *collisions >= max_collisions {
        return;
    }

    let hash = pos.get_key();
    let current_source = ZobristSource::from(&*pos);

    match tt.get(hash) {
        Some(existing_source) => {
            if existing_source.src != current_source && existing_source.src != Default::default() {
                *collisions += 1;
                if *collisions >= max_collisions {
                    return;
                }
            }
        }
        None => {
            *insertions += 1;
            tt.try_insert(OccupancyIndicator { src: current_source, key: hash });
        }
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
        simulate_search(pos, depth + 1, max_depth, rng, tt, collisions, insertions, max_collisions);
        pos.unmake_move(mov, &mut ());

        if *collisions >= max_collisions {
            break;
        }
    }
}

fn main() { find_seeds(); }
