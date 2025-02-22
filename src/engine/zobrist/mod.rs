use core::hash;
use std::cell::LazyCell;
use std::mem;
use std::ops::ControlFlow;
use std::sync::atomic::AtomicBool;
use std::sync::{LazyLock, Once};
use std::{ops, sync::Arc};

use rand::{rngs::SmallRng, RngCore, SeedableRng};
use rand::{thread_rng, Rng};

use crate::misc::ConstFrom;
use crate::uci::sync::CancellationToken;

use super::move_iter::fold_legal_moves;
use super::position::repetitions::RepetitionTable;
use super::r#move::Move;
use super::search::mcts::{self, PlayoutResult};
use super::{
    bitboard::Bitboard,
    castling::CastlingRights,
    coordinates::{EpCaptureSquare, File, Square},
    piece::Piece,
    position::Position,
    search::{
        limit::{self, Limit},
        mode::Mode,
        target::Target,
        Search,
    },
    turn::Turn,
};

#[cfg(test)]
mod seeding;

/// Note: the default hash is equivalent to the hash of the default (empty) position.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct Hash {
    v: u64,
}

impl Hash {
    pub const fn v(self) -> u64 {
        self.v
    }

    pub fn from_v(v: u64) -> Self {
        Self { v }
    }

    #[inline]
    pub fn set_turn(&mut self, turn: Turn) -> Self {
        self.v ^= match turn {
            Turn::BLACK => unsafe { HASHER.stm },
            _ => 0,
        };
        *self
    }

    #[inline]
    pub fn toggle_turn(&mut self) -> Self {
        // xor is it's own inverse.
        self.v ^= unsafe { HASHER.stm };
        *self
    }

    #[inline]
    pub fn toggle_castling(&mut self, castling: CastlingRights) -> Self {
        self.v ^= unsafe { HASHER.castling }[castling.v() as usize];
        *self
    }

    #[inline]
    pub fn toggle_ep_square(&mut self, ep_sq: EpCaptureSquare) -> Self {
        self.v ^= ep_sq.v().map_or(0, |sq| {
            let file = File::from_c(sq);
            unsafe { HASHER.en_passant[file.v() as usize] }
        });
        *self
    }

    #[inline]
    pub fn toggle_piece_sq(&mut self, sq: Square, piece: Piece) -> Self {
        self.v ^= unsafe { HASHER.piece_sq[sq.v() as usize][piece.v() as usize] };
        *self
    }

    #[inline]
    pub fn move_piece_sq(&mut self, from: Square, to: Square, piece: Piece) -> Self {
        self.toggle_piece_sq(from, piece).toggle_piece_sq(to, piece)
    }
}

impl_op!(^ |l: Hash, r: u64| -> Hash { Hash { v: l.v ^ r } });

impl From<&Position> for Hash {
    fn from(pos: &Position) -> Self {
        Bitboard::full()
            .fold(Hash::default(), |mut acc, sq| {
                acc.toggle_piece_sq(sq, pos.get_piece(sq))
            })
            .toggle_ep_square(pos.get_ep_capture_square())
            .toggle_castling(pos.get_castling())
            .set_turn(pos.get_turn())
    }
}

static mut HASHER: Hasher = unsafe { mem::zeroed() };

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| unsafe {
        let mut seed = 11459972786511497394;
        let mut min = usize::MAX;
        loop {
            HASHER.init(seed);
            let r = test_seed(20000, &mut SmallRng::seed_from_u64(0xdeadbeef), min);
            if r.total_collisions < min {
                min = r.total_collisions;
                println!(
                    "collisions: {}, seed: {}",
                    r.total_collisions, seed
                );
            }
            seed = SmallRng::seed_from_u64(seed).next_u64();
        }
    });
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
                let buffer = &mut vec![];
                fold_legal_moves(pos, &mut *buffer, |acc, m| {
                    ControlFlow::Continue::<(), _>({
                        acc.push(m);
                        acc
                    })
                });

                if mcts::PlayoutResult::maybe_new(pos, &buffer).is_some() {
                    break;
                }

                let mov = buffer[rng.gen_range(0..buffer.len())];
                pos.make_move(mov);
            }

            let collisions_per_ply = pos.repetition_table_collisions() as f32 / pos.ply().v as f32;
            let collisions_per_ply_threshold = 0.0021428571 * pos.repetition_table_capacity() as f32;
            if collisions_per_ply > collisions_per_ply_threshold {
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(acc + pos.repetition_table_collisions())
        }).continue_value().unwrap_or(usize::MAX);
    SeedTestResult { total_collisions: collisions, }
}

struct Hasher {
    piece_sq: [[u64; 14]; 64],
    en_passant: [u64; 8],
    castling: [u64; 16],
    stm: u64,
}

impl Hasher {
    pub fn new(seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        Self {
            piece_sq: [[0; 14]; 64].map(|pieces| pieces.map(|_| rng.next_u64())),
            en_passant: [0; 8].map(|_| rng.next_u64()),
            castling: [0; 16].map(|_| rng.next_u64()),
            stm: rng.next_u64(),
        }
    }

    pub fn init(&mut self, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        self.piece_sq = [[0; 14]; 64].map(|pieces| pieces.map(|_| rng.next_u64()));
        self.en_passant = [0; 8].map(|_| rng.next_u64());
        self.castling = [0; 16].map(|_| rng.next_u64());
        self.stm = rng.next_u64();
    }
}
