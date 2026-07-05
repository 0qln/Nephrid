use std::mem;

use thiserror::Error;

use crate::{
    core::{
        color::{
            Color, Perspective, colors,
            perspectives::{Black, White},
        },
        coordinates::{Square, squares},
        piece::{Piece, PieceType, piece_type},
        position::{PieceInfo, PieceInfoObserver},
    },
    misc::{CheckHealth, CheckHealthResult},
};

pub type TValue = i16;
pub type TEval = i32;

const PIECES: usize = piece_type::N_VARIANTS - 1;
const COLORS: usize = colors::N_VARIANTS;
const SQUARES: usize = squares::N_VARIANTS;

pub const INPUT_SIZE: usize = PIECES * COLORS * SQUARES;
pub const HIDDEN_SIZE: usize = 2 << 7;
pub const OUTPUT_SIZE: usize = 1;

// todo: use Cp::SCALE?
pub const SCALE: TValue = 400;
pub const QA: TValue = 255;
pub const QB: TValue = 64;

pub const NNUE: Network = unsafe {
    mem::transmute(*include_bytes!(
        "../../../../checkpoints/nnue-1783259751942-768-256_400*255*64-40/quantised.bin"
    ))
};

#[repr(C, align(64))]
pub struct HiddenLayer {
    vals: [TValue; HIDDEN_SIZE],
}

#[repr(C)]
pub struct Network {
    acc_weights: [HiddenLayer; INPUT_SIZE],
    acc_biases: HiddenLayer,
    out_weights: [TValue; colors::N_VARIANTS * HIDDEN_SIZE],
    out_bias: [TValue; OUTPUT_SIZE],
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

#[derive(Debug, Error)]
pub enum CheckNnueHealthError {
    #[error("NNUE network is all zeros. likely corrupted!")]
    AllZero,

    #[error("Value {value} at index {idx} in {field} is out of expected range")]
    OutOfRange { field: &'static str, idx: usize, value: i16 },
}

impl CheckHealth for Network {
    type Error = CheckNnueHealthError;

    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        const MAX_ABS: i16 = 2000;

        fn check_slice(slice: &[i16], field: &'static str) -> Result<(), CheckNnueHealthError> {
            for (idx, &val) in slice.iter().enumerate() {
                if val.abs() > MAX_ABS {
                    return Err(CheckNnueHealthError::OutOfRange { field, idx, value: val });
                }
            }
            Ok(())
        }

        // acc_weights
        for (layer_idx, layer) in self.acc_weights.iter().enumerate() {
            let field = format!("acc_weights[{}]", layer_idx);
            check_slice(&layer.vals, Box::leak(field.into_boxed_str()))?;
        }

        // acc_biases
        check_slice(&self.acc_biases.vals, "acc_biases")?;

        // out_weights
        check_slice(&self.out_weights, "out_weights")?;

        // out_bias
        check_slice(&self.out_bias, "out_bias")?;

        // verify not all zeros (sum of absolute values > 0)
        let total_abs: i64 = self
            .acc_weights
            .iter()
            .flat_map(|layer| layer.vals.iter())
            .chain(self.acc_biases.vals.iter())
            .chain(self.out_weights.iter())
            .chain(self.out_bias.iter())
            .map(|&x| x.abs() as i64)
            .sum();

        if total_abs == 0 {
            return Err(CheckNnueHealthError::AllZero);
        }

        Ok(())
    }
}

pub struct Accumulator {
    values: [TValue; HIDDEN_SIZE],
}

impl Accumulator {
    pub fn init(net: &Network) -> Self {
        //
        Self { values: net.acc_biases.vals }
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
        for sq in squares::A1..=squares::H8 {
            let p = pos.get_piece(sq);
            if p.piece_type() != piece_type::NONE {
                self.on_piece_put(sq, p);
            }
        }
    }

    fn on_piece_put(&mut self, sq: Square, p: Piece) {
        let (c, pt) = p.unpack();
        self.white.add_feature(input_index::<White>(sq, pt, c), &NNUE);
        self.black.add_feature(input_index::<Black>(sq, pt, c), &NNUE);
    }

    fn on_piece_removed(&mut self, sq: Square, p: Piece) {
        let (c, pt) = p.unpack();
        self.white.remove_feature(input_index::<White>(sq, pt, c), &NNUE);
        self.black.remove_feature(input_index::<Black>(sq, pt, c), &NNUE);
    }

    fn on_piece_moved(&mut self, from: Square, to: Square, p: Piece) {
        self.on_piece_removed(from, p);
        self.on_piece_put(to, p);
    }
}

#[inline(always)]
pub fn input_index<P: Perspective>(sq: Square, pt: PieceType, c: Color) -> usize {
    let (mut sq, mut c) = (sq, c);

    if P::COLOR == colors::BLACK {
        c = !c;
        sq = sq.flip_v();
    }

    let c = c.v() as usize;
    let sq = sq.v() as usize;
    let pt = pt.v() as usize - 1;

    c * PIECES * SQUARES + pt * SQUARES + sq
}

const fn activation(x: TValue) -> TEval { screlu(x) }
const fn crelu(x: TValue) -> TEval { TEval::from(x).clamp(0, TEval::from(QA)) }
const fn screlu(x: TValue) -> TEval { crelu(x).pow(2) }
