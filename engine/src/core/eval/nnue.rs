use std::mem;

use crate::core::{
    color::{Color, Perspective, colors},
    coordinates::{Square, squares},
    piece::{Piece, PieceType, piece_type},
    position::{PieceInfo, PieceInfoObserver},
};

pub type TValue = i16;
pub type TEval = i32;

const PIECES: usize = piece_type::N_VARIANTS - 1;
const COLORS: usize = colors::N_VARIANTS;
const SQUARES: usize = squares::N_VARIANTS;

pub const INPUT_SIZE: usize = PIECES * COLORS * SQUARES;
pub const HIDDEN_SIZE: usize = 2 << 9;
pub const OUTPUT_SIZE: usize = 1;

// todo: use Cp::SCALE?
pub const SCALE: TValue = 400;
pub const QA: TValue = 255;
pub const QB: TValue = 64;

pub const NNUE: Network = unsafe { mem::transmute(*include_bytes!("../../../../checkpoints/simple-40/quantised.bin")) };

#[repr(C)]
pub struct Network {
    acc_weights: [HiddenLayer; INPUT_SIZE],
    acc_biases: [TValue; HIDDEN_SIZE],
    out_weights: [TValue; colors::N_VARIANTS * HIDDEN_SIZE],
    out_bias: [TValue; OUTPUT_SIZE],
}

#[repr(C, align(64))]
pub struct HiddenLayer {
    vals: [TValue; HIDDEN_SIZE],
}

impl Network {
    pub fn forward(&self, acc_stm: &Accumulator, acc_nstm: &Accumulator) -> i32 {
        let mut eval: i32 = 0;

        for (&value, &weight) in acc_stm.values.iter().zip(&self.out_weights[..HIDDEN_SIZE]) {
            eval += activation(value) * TEval::from(weight);
        }

        for (&value, &weight) in acc_nstm.values.iter().zip(&self.out_weights[HIDDEN_SIZE..]) {
            eval += activation(value) * TEval::from(weight);
        }

        eval /= TEval::from(QA);
        eval += TEval::from(self.out_bias[0]);
        eval *= TEval::from(SCALE);
        eval /= TEval::from(QA) * TEval::from(QB);

        eval
    }
}

pub struct Accumulator {
    values: [TValue; HIDDEN_SIZE],
}

impl Accumulator {
    pub fn init(net: &Network) -> Self {
        //
        Self { values: net.acc_biases }
    }

    /// Add a feature to an accumulator.
    pub fn add_feature(&mut self, idx: usize, net: &Network) {
        for (i, d) in self.values.iter_mut().zip(&net.acc_weights[idx].vals) {
            *i += *d
        }
    }

    /// Remove a feature from an accumulator.
    pub fn remove_feature(&mut self, idx: usize, net: &Network) {
        for (i, d) in self.values.iter_mut().zip(&net.acc_weights[idx].vals) {
            *i -= *d
        }
    }
}

pub struct AccumulatorPair {
    pub white: Accumulator,
    pub black: Accumulator,
}

impl PieceInfoObserver for AccumulatorPair {
    fn on_init(&mut self, pos: &PieceInfo) {
        for sq in squares::A1..squares::H8 {
            // todo
            // if let Some(p) = pos.piece_at(sq) {
            //     self.on_piece_put(sq, p);
            // }
        }
    }

    fn on_piece_put(&mut self, sq: Square, p: Piece) {
        // let idx = input_index::<colors::WHITE>(sq, pt, c);
        // self.white.add_feature(idx, &NNUE);
        // let idx = input_index::<colors::BLACK>(sq, pt, c);
        // self.black.add_feature(idx, &NNUE);
    }

    fn on_piece_removed(&mut self, sq: Square, p: Piece) {
        // let idx = input_index::<colors::WHITE>(sq, pt, c);
        // self.white.remove_feature(idx, &NNUE);
        // let idx = input_index::<colors::BLACK>(sq, pt, c);
        // self.black.remove_feature(idx, &NNUE);
    }

    fn on_piece_moved(&mut self, from: Square, to: Square, p: Piece) {
        todo!()
        // let idx = input_index::<colors::WHITE>(from, pt, c);
        // self.white.remove_feature(idx, &NNUE);
        // let idx = input_index::<colors::WHITE>(to, pt, c);
        // self.white.add_feature(idx, &NNUE);
        // let idx = input_index::<colors::BLACK>(from, pt, c);
        // self.black.remove_feature(idx, &NNUE);
        // let idx = input_index::<colors::BLACK>(to, pt, c);
        // self.black.add_feature(idx, &NNUE);
    }
}

pub fn input_index<P: Perspective>(sq: Square, pt: PieceType, c: Color) -> usize {
    let (mut sq, mut c) = (sq, c);

    if P::COLOR == colors::BLACK {
        c = !c;
        sq = sq.flip_v();
    }

    let c = c.v() as usize;
    let pt = pt.v() as usize;
    let sq = sq.v() as usize;

    c * PIECES * SQUARES + pt * SQUARES + sq
}

const fn activation(x: TValue) -> TEval { screlu(x) }
const fn crelu(x: TValue) -> TEval { TEval::from(x).clamp(0, TEval::from(QA)) }
const fn screlu(x: TValue) -> TEval { crelu(x).pow(2) }
