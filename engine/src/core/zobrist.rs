use std::sync::Once;
use std::{mem, ops};

use rand::{RngCore, SeedableRng, rngs::SmallRng};

use crate::misc::ConstFrom;

use super::{
    bitboard::Bitboard,
    castling::CastlingRights,
    coordinates::{EpCaptureSquare, File, Square},
    piece::Piece,
    position::Position,
    turn::Turn,
};

/// Note: the default hash is equivalent to the hash of the default (empty) position.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct Hash {
    v: u64,
}

#[inline]
#[allow(static_mut_refs)]
fn hasher() -> &'static Hasher {
    debug_assert!(INIT.is_completed(), "Hasher not initialized!");
    unsafe { &HASHER }
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
        (turn == Turn::BLACK).then(|| self.toggle_turn());
        *self
    }

    #[inline]
    pub fn toggle_turn(&mut self) -> Self {
        self.v ^= hasher().stm;
        *self
    }

    #[inline]
    pub fn toggle_castling(&mut self, castling: CastlingRights) -> Self {
        self.v ^= hasher().castling[castling.v() as usize];
        *self
    }

    #[inline]
    pub fn toggle_ep_square(&mut self, ep_sq: EpCaptureSquare) -> Self {
        if let Some(sq) = ep_sq.v() {
            self.v ^= hasher().en_passant[File::from_c(sq).v() as usize];
        }
        *self
    }

    #[inline]
    pub fn toggle_piece_sq(&mut self, sq: Square, piece: Piece) -> Self {
        self.v ^= hasher().piece_sq[sq.v() as usize][piece.v() as usize];
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

#[allow(static_mut_refs)]
pub fn init() {
    INIT.call_once(|| unsafe { HASHER.init(14278029879823863027) });
}

#[allow(static_mut_refs)]
pub fn force_init(seed: u64) {
    unsafe { HASHER.init(seed) };
}

struct Hasher {
    piece_sq: [[u64; 14]; 64],
    en_passant: [u64; 8],
    castling: [u64; 16],
    stm: u64,
}

impl Hasher {
    pub fn init(&mut self, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        self.piece_sq = [[0; 14]; 64].map(|pieces| pieces.map(|_| rng.next_u64()));
        self.en_passant = [0; 8].map(|_| rng.next_u64());
        self.castling = [0; 16].map(|_| rng.next_u64());
        self.stm = rng.next_u64();
    }
}
