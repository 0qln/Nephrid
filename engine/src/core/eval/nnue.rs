use crate::core::{depth::Depth, id::SearchStack};
use std::{
    fs::File,
    hint::unreachable_unchecked,
    io::{self, Read},
    mem,
    path::Path,
};

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

pub type NetworkBytes = [u8; size_of::<Network>()];

pub static DEFAULT_NNUE: NetworkBytes = *include_bytes!("../../../../checkpoints/nnue-1783259751942-768-256_400*255*64-40/quantised.bin");

static mut NNUE: Network = unsafe { mem::transmute::<NetworkBytes, Network>(DEFAULT_NNUE) };

/// Gets a nnue, defaulting to some default net if the user net paniced during
/// initialization.
#[allow(static_mut_refs)]
pub fn get_nnue() -> &'static Network { unsafe { &NNUE } }

pub fn read_net_bytes(path: &Path) -> Result<NetworkBytes, io::Error> {
    let mut f = File::open(path)?;
    let mut bytes = [0u8; size_of::<Network>()];
    f.read_exact(&mut bytes)?;
    Ok(bytes)
}

#[allow(static_mut_refs)]
pub fn set_nnue(bytes: NetworkBytes) -> Result<(), CheckNnueHealthError> {
    let net: Network = unsafe { mem::transmute::<NetworkBytes, Network>(bytes) };
    net.check_health()?;

    unsafe {
        NNUE = net;
    }

    Ok(())
}

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
    pub fn forward(&self, acc_stm: &mut Accumulator, acc_nstm: &mut Accumulator) -> i32 {
        #[cfg(debug_assertions)]
        {
            acc_stm.check_health().expect("Unhealthy accumulator");
            acc_nstm.check_health().expect("Unhealthy accumulator");
        }

        let mut eval: i32 = 0;

        for (&value, &weight) in acc_stm.values().iter().zip(&self.out_weights[..HIDDEN_SIZE]) {
            eval += activation(value) * TEval::from(weight);
        }

        for (&value, &weight) in acc_nstm.values().iter().zip(&self.out_weights[HIDDEN_SIZE..]) {
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

#[derive(Clone, Copy)]
pub struct Accumulator {
    values: [TValue; HIDDEN_SIZE],

    /// This field is only used in debug builds to verify that the accumulator
    /// is consistent with the features added/removed.
    #[cfg(debug_assertions)]
    inputs: [TValue; INPUT_SIZE],
}

impl Accumulator {
    pub fn init(net: &Network) -> Self {
        Self {
            values: net.acc_biases.vals,

            #[cfg(debug_assertions)]
            inputs: [0; INPUT_SIZE],
        }
    }

    /// Add a feature to an accumulator.
    pub fn add_feature(&mut self, idx: usize, net: &Network) {
        #[cfg(debug_assertions)]
        {
            self.inputs[idx] += 1;
        }

        for (val, weight) in self.values.iter_mut().zip(&net.acc_weights[idx].vals) {
            *val += *weight
        }
    }

    /// Remove a feature from an accumulator.
    pub fn remove_feature(&mut self, idx: usize, net: &Network) {
        #[cfg(debug_assertions)]
        {
            self.inputs[idx] -= 1;
        }

        for (val, weight) in self.values.iter_mut().zip(&net.acc_weights[idx].vals) {
            *val -= *weight
        }
    }

    pub fn update_feature(&mut self, idx: usize, net: &Network, update: i16) {
        #[cfg(debug_assertions)]
        {
            self.inputs[idx] += update;
        }

        for (val, weight) in self.values.iter_mut().zip(&net.acc_weights[idx].vals) {
            *val += *weight * update;
        }
    }

    pub fn values(&mut self) -> [i16; HIDDEN_SIZE] { self.values }
}

#[derive(Debug, Error)]
pub enum CheckAccumulatorHealthError {
    #[error("Accumulator input at index {idx} wasn't 0 or 1")]
    InputIsntOne { idx: usize, value: i16 },
}

#[cfg(debug_assertions)]
impl CheckHealth for Accumulator {
    type Error = CheckAccumulatorHealthError;

    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        for (idx, &val) in self.inputs.iter().enumerate() {
            if val != 0 && val != 1 {
                return Err(CheckAccumulatorHealthError::InputIsntOne { idx, value: val });
            }
        }

        // todo: check that each accumulated val is a multiple of its inputs * w - bias

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FeatureUpdates {
    counts: [[[i8; SQUARES]; PIECES]; COLORS],
}

impl Default for FeatureUpdates {
    fn default() -> Self {
        Self {
            counts: [[[0; SQUARES]; PIECES]; COLORS],
        }
    }
}

impl FeatureUpdates {
    pub fn get_mut(&mut self, sq: Square, pt: PieceType, c: Color) -> &mut i8 {
        let c = c.v() as usize;
        let pt = pt.v() as usize - 1;
        let sq = sq.v() as usize;
        unsafe { self.counts.get_unchecked_mut(c).get_unchecked_mut(pt).get_unchecked_mut(sq) }
    }
}

#[derive(Clone, Default)]
struct AccUpdates {
    /// Before you apply anything, reset everything.
    reset: bool,

    /// The pieces that were put(+) or removed(-) on the board.
    updates: FeatureUpdates,

    /// Which features have been updated?
    updated: Vec<IndexInfo>,
}

impl AccUpdates {
    fn new() -> Self {
        Self {
            reset: false,
            updates: Default::default(),
            updated: Vec::with_capacity(16),
        }
    }

    fn put(&mut self, sq: Square, p: Piece) { self.update(sq, p, 1); }
    fn remove(&mut self, sq: Square, p: Piece) { self.update(sq, p, -1); }

    fn update(&mut self, sq: Square, p: Piece, delta: i8) {
        let (c, pt) = p.unpack();

        let cnt = self.updates.get_mut(sq, pt, c);

        if *cnt == 0 {
            self.updated.push(IndexInfo { sq, pt, c });
        }

        *cnt += delta;
    }

    fn reset(&mut self) {
        self.updates = Default::default();
        self.reset = true;
    }

    fn apply(&mut self, acc_white: &mut Accumulator, acc_black: &mut Accumulator, net: &Network) {
        if self.reset {
            *acc_white = Accumulator::init(net);
            *acc_black = Accumulator::init(net);
            self.reset = false;
        }

        for IndexInfo { sq, pt, c } in self.updated.drain(..) {
            let cnt = {
                let ptr = self.updates.get_mut(sq, pt, c);
                let val = *ptr;
                *ptr = 0;
                val as i16
            };

            if cnt == 0 {
                continue;
            }

            let idx_w = input_index_for::<White>(sq, pt, c);
            let idx_b = input_index_for::<Black>(sq, pt, c);

            match cnt {
                // +1 or -1 generates much better assembly so switch here
                1 => {
                    acc_white.add_feature(idx_w, net);
                    acc_black.add_feature(idx_b, net);
                }
                -1 => {
                    acc_white.remove_feature(idx_w, net);
                    acc_black.remove_feature(idx_b, net);
                }
                _ => {
                    acc_white.update_feature(idx_w, net, cnt);
                    acc_black.update_feature(idx_b, net, cnt);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct IndexInfo {
    sq: Square,
    pt: PieceType,
    c: Color,
}

#[derive(Clone)]
pub struct AccumulatorPair {
    updates: AccUpdates,
    white: Accumulator,
    black: Accumulator,
}

impl Default for AccumulatorPair {
    fn default() -> Self { Self::new(get_nnue()) }
}

impl AccumulatorPair {
    pub fn new(net: &Network) -> Self {
        Self {
            updates: AccUpdates::new(),
            white: Accumulator::init(net),
            black: Accumulator::init(net),
        }
    }

    pub fn inherit_from(&mut self, parent: &Self) {
        self.white = parent.white;
        self.black = parent.black;

        self.updates.reset = parent.updates.reset;
        self.updates.updates = parent.updates.updates;

        self.updates.updated.clear();
        self.updates.updated.extend_from_slice(&parent.updates.updated);
    }

    pub fn sync(&mut self, net: &Network) { self.updates.apply(&mut self.white, &mut self.black, net); }

    pub fn get_mut_for<P: Perspective>(&mut self, net: &Network) -> (&mut Accumulator, &mut Accumulator) {
        self.sync(net);

        match P::COLOR {
            colors::WHITE => (&mut self.white, &mut self.black),
            colors::BLACK => (&mut self.black, &mut self.white),
            _ => unsafe { unreachable_unchecked() },
        }
    }
}

impl PieceInfoObserver for AccumulatorPair {
    fn on_init(&mut self, pos: &PieceInfo) {
        // reset
        self.updates.reset();

        // put pieces
        for sq in squares::A1..=squares::H8 {
            let p = pos.get_piece(sq);
            if p.piece_type() != piece_type::NONE {
                self.updates.put(sq, p);
            }
        }
    }

    fn on_piece_put(&mut self, sq: Square, p: Piece) { self.updates.put(sq, p); }

    fn on_piece_removed(&mut self, sq: Square, p: Piece) { self.updates.remove(sq, p); }

    fn on_piece_moved(&mut self, from: Square, to: Square, p: Piece) {
        self.on_piece_removed(from, p);
        self.on_piece_put(to, p);
    }
}

#[derive(Default)]
pub struct AccumulatorStack {
    accs: SearchStack<AccumulatorPair>,
}

impl AccumulatorStack {
    pub fn get_accs_mut(&mut self, idx: Depth) -> &mut AccumulatorPair { self.accs.get_mut(idx) }

    pub fn propagate(&mut self, old: Depth, new: Depth) {
        self.accs.propagate(old, new, |parent, child| {
            child.inherit_from(parent);
        })
    }
}

#[inline(always)]
fn input_index_for<P: Perspective>(sq: Square, pt: PieceType, c: Color) -> usize {
    let (mut sq, mut c) = (sq, c);

    if P::COLOR == colors::BLACK {
        c = !c;
        sq = sq.flip_v();
    }

    input_index(sq, pt, c)
}

#[inline(always)]
fn input_index(sq: Square, pt: PieceType, c: Color) -> usize {
    let c = c.v() as usize;
    let sq = sq.v() as usize;
    let pt = pt.v() as usize - 1;

    c * PIECES * SQUARES + pt * SQUARES + sq
}

const fn activation(x: TValue) -> TEval { screlu(x) }
const fn crelu(x: TValue) -> TEval { TEval::from(x).clamp(0, TEval::from(QA)) }
const fn screlu(x: TValue) -> TEval { crelu(x).pow(2) }
