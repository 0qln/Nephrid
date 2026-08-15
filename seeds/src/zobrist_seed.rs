#![feature(const_default)]
#![feature(const_trait_impl)]
#![feature(control_flow_into_value)]
#![feature(derive_const)]

use std::{env::var, fs, ops::ControlFlow, path::PathBuf};

use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};

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
    let mut tt = TT::new(1 << 22);

    loop {
        if num_positions > all_positions.len() {
            println!("reached max available dataset positions ({})!", all_positions.len());
            break;
        }

        zobrist::force_init(seed);

        let r = test_seed(get_positions(num_positions), &mut moves_rng(), &mut tt, min_collisions);
        if r.total_collisions < min_collisions {
            min_collisions = r.total_collisions;
            println!(
                "[+] seed: {seed}, positions: {num_positions}, collisions: {} ({:?})",
                r.total_collisions, r.bound
            );
        }

        // too good?
        if min_collisions == 0 {
            print!("[ ] seed: {seed} perfect for {num_positions} positions. escalating to ");
            num_positions += 1;
            min_collisions = test_seed(get_positions(num_positions), &mut moves_rng(), &mut tt, usize::MAX).total_collisions;
            println!("{num_positions} positions with {min_collisions} collisions...");
        }
        // too bad?
        else {
            seed = SmallRng::seed_from_u64(seed).next_u64();
        }
    }
}

#[derive(Debug)]
enum Bound {
    Exact,
    Lower,
}

struct SeedTestResult {
    total_collisions: usize,
    bound: Bound,
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
            pieces: self.pieces.clone(),
            turn: self.turn.clone(),
            castling: self.castling.clone(),
            ep_capture_square: self.ep_capture_square.clone(),
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
            pieces: pos.get_pieces().clone(),
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

fn test_seed(mut positions: impl Iterator<Item = Position>, rng: &mut SmallRng, tt: &mut TT, min: usize) -> SeedTestResult {
    const MAX_DEPTH: usize = 10; // 2^10 = 1024[nodes/position]

    tt.clear();

    let mut bound = Bound::Exact;
    let collisions = positions
        .try_fold(0, |mut collisions, mut pos| {
            if collisions >= min {
                bound = Bound::Lower;
                return ControlFlow::Break(collisions);
            }

            simulate_search(&mut pos, 0, MAX_DEPTH, rng, tt, &mut collisions, min);

            if collisions >= min {
                bound = Bound::Lower;
                ControlFlow::Break(collisions)
            }
            else {
                ControlFlow::Continue(collisions)
            }
        })
        .into_value();

    SeedTestResult {
        total_collisions: collisions,
        bound,
    }
}

fn simulate_search(pos: &mut Position, depth: usize, max_depth: usize, rng: &mut SmallRng, tt: &mut TT, collisions: &mut usize, min: usize) {
    if *collisions >= min {
        return;
    }

    let hash = pos.get_key();
    let current_source = ZobristSource::from(&*pos);

    match tt.get(hash) {
        Some(existing_source) => {
            // Note: Since `tt.get(hash)` guarantees `existing_source.key() == hash`,
            // we only check if the physical board attributes differ to confirm a Zobrist
            // collision.
            if &existing_source.src != &current_source && &existing_source.src != &Default::default() {
                *collisions += 1;
                if *collisions >= min {
                    return;
                }
            }
        }
        None => {
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
        simulate_search(pos, depth + 1, max_depth, rng, tt, collisions, min);
        pos.unmake_move(mov, &mut ());

        if *collisions >= min {
            break;
        }
    }
}

fn main() { find_seeds(); }
