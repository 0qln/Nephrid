use std::ops;
use std::sync::LazyLock;

use rand::{rngs::SmallRng, RngCore, SeedableRng};

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

impl Hash {
    #[inline]
    pub fn set_turn(&mut self, turn: Turn) -> Self {
        self.v ^= match turn { Turn::BLACK => HASHER.stm, _ => 0, };
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
                acc.toggle_piece_sq(sq, pos.get_piece(sq))
            })
            .toggle_ep_square(pos.get_ep_capture_square())
            .toggle_castling(pos.get_castling())
            .set_turn(pos.get_turn())
    }
}

static HASHER: LazyLock<Hasher> = LazyLock::new(|| Hasher::new(0xdead_beef));

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
}
