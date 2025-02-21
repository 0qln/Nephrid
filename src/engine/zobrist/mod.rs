use core::hash;
use std::mem;
use std::sync::atomic::AtomicBool;
use std::sync::{LazyLock, Once};
use std::{ops, sync::Arc};

use rand::{rngs::SmallRng, RngCore, SeedableRng};

use crate::misc::ConstFrom;
use crate::uci::sync::CancellationToken;

use super::position::repetitions::RepetitionTable;
use super::search::mcts;
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

/// Note: the default hash is equivalent to the hash of the default (empty) position.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct Hash {
    v: u64,
}

impl Hash {
    pub fn v(self) -> u64 {
        self.v
    }

    pub fn from_v(v: u64) -> Self {
        Self { v }
    }

    #[inline]
    pub fn set_turn(&mut self, turn: Turn) -> Self {
        self.v ^= match turn {
            Turn::BLACK => HASHER.stm,
            _ => 0,
        };
        *self
    }

    #[inline]
    pub fn toggle_turn(&mut self) -> Self {
        // xor is it's own inverse.
        self.v ^= HASHER.stm;
        *self
    }

    #[inline]
    pub fn toggle_castling(&mut self, castling: CastlingRights) -> Self {
        self.v ^= HASHER.castling[castling.v() as usize];
        *self
    }

    #[inline]
    pub fn toggle_ep_square(&mut self, ep_sq: EpCaptureSquare) -> Self {
        self.v ^= ep_sq.v().map_or(0, |sq| {
            let file = File::from_c(sq);
            HASHER.en_passant[file.v() as usize]
        });
        *self
    }

    #[inline]
    pub fn toggle_piece_sq(&mut self, sq: Square, piece: Piece) -> Self {
        self.v ^= HASHER.piece_sq[sq.v() as usize][piece.v() as usize];
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
                acc.toggle_piece_sq(sq, pos.get_piece(sq, ))
            })
            .toggle_ep_square(pos.get_ep_capture_square())
            .toggle_castling(pos.get_castling())
            .set_turn(pos.get_turn())
    }
}

static HASHER: Hasher = unsafe { mem::zeroed() };

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| unsafe {
        // Safety: This is the only place that mutates the tables, and it is only done once.
        let global_hasher = (&HASHER as *const _ as *mut Hasher).as_mut().unwrap();

        let mut rng = SmallRng::seed_from_u64(0xdeadbeef);
        let mut min = usize::MAX;
        loop {
            global_hasher.init(rng.next_u64());
            let collisions = collisions(global_hasher, 10_000);
            if collisions < min {
                min = collisions;
                println!("new min collisions: {collisions}");
            }
        }
    });
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

fn collisions(hasher: &Hasher, rounds: usize) -> usize {
    let mut pos = Position::start_position();
    let mut tree = mcts::Tree::new(&pos);
    tree.select_leaf_mut(&mut pos);
    let leaf = unsafe { tree.selected_leaf().as_mut() };
    (0..rounds).map(|_| {
        leaf.simulate(&mut pos, &mut vec![], &mut vec![]);
        pos.repetition_table_collisions()
    }).sum()
}
