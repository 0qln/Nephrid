use burn::record::{CompactRecorder, Recorder, RecorderError};
use std::path::PathBuf;

use burn::{
    config::Config,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu, Tanh,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::Tensor,
};
use itertools::Itertools;
use thiserror::Error;

use crate::core::{
    bitboard,
    castling::castling_sides,
    color::colors,
    coordinates::{files, ranks, squares},
    piece::{piece_type, promo_piece_type},
    position::Position,
};

#[cfg(test)]
pub mod test;

pub const BOARD_INPUT_HISTORY: usize = 8;

// +1 for the possible ep capture square.
pub const BOARD_INPUT_CHANNELS: usize = colors::N_VARIANTS * piece_type::N_VARIANTS;

pub const BOARD_INPUT_TENSOR_DIM: usize = {
    // Batch
    1 +
    // Channel
    1 +
    // Rank
    1 +
    // File
    1
};

pub const STATE_INPUT_LEN: usize = {
    // Castling rights
    4 +
    // Plys after last capture or pawn move (should be normalized / devided by 50)
    1
};

pub const STATE_INPUT_TENSOR_DIM: usize = {
    // Batch
    1 +
    // Values
    1
};

pub const POLICY_OUTPUT_TENSOR_DIM: usize = {
    // Batch
    1 +
    // Move
    1
};

/// Policy target dimension for single-label-classification.
pub const POLICY_SLC_TARGET_TENSOR_DIM: usize = {
    POLICY_OUTPUT_TENSOR_DIM -
    // Just use the index, no need to bring all the zeros.
    1
};

/// Policy target dimension for multi-label-classification.
pub const POLICY_MLC_TARGET_TENSOR_DIM: usize = POLICY_OUTPUT_TENSOR_DIM;

pub const VALUE_OUTPUT_TENSOR_DIM: usize = {
    // Batch
    1 +
    // singleton
    1
};

pub type BoardInputTensor<B> = Tensor<B, BOARD_INPUT_TENSOR_DIM>;

pub type BoardInputFloats = [bitboard::Floats; BOARD_INPUT_CHANNELS];

pub fn board_input(pos: &Position) -> BoardInputFloats {
    let us = pos.get_turn();
    let flip = us == colors::BLACK;
    let them = !us;
    [
        pos.get_bitboard(piece_type::PAWN, us).into_floats(flip),
        pos.get_bitboard(piece_type::PAWN, them).into_floats(flip),
        pos.get_bitboard(piece_type::KNIGHT, us).into_floats(flip),
        pos.get_bitboard(piece_type::KNIGHT, them).into_floats(flip),
        pos.get_bitboard(piece_type::BISHOP, us).into_floats(flip),
        pos.get_bitboard(piece_type::BISHOP, them).into_floats(flip),
        pos.get_bitboard(piece_type::ROOK, us).into_floats(flip),
        pos.get_bitboard(piece_type::ROOK, them).into_floats(flip),
        pos.get_bitboard(piece_type::QUEEN, us).into_floats(flip),
        pos.get_bitboard(piece_type::QUEEN, them).into_floats(flip),
        pos.get_bitboard(piece_type::KING, us).into_floats(flip),
        pos.get_bitboard(piece_type::KING, them).into_floats(flip),
        pos.get_ep_capture_bitboard(us).into_floats(flip),
        pos.get_ep_capture_bitboard(them).into_floats(flip),
    ]
}

pub fn board_history_input<B: Backend>(
    history: &[BoardInputFloats],
    device: &B::Device,
) -> BoardInputTensor<B> {
    let history_len = history.len();
    let padding_len = BOARD_INPUT_HISTORY - history_len;
    debug_assert_eq!(padding_len + history_len, BOARD_INPUT_HISTORY);

    // pad missing history info with zeroes.
    let padding_tensor = BoardInputTensor::<B>::zeros(
        [
            1,
            padding_len * BOARD_INPUT_CHANNELS,
            ranks::N_VARIANTS,
            files::N_VARIANTS,
        ],
        device,
    );

    // convert input floats to tensors
    let history_tensor = Tensor::cat(
        history
            .iter()
            .map(|b| Tensor::from_floats([*b], device))
            .collect_vec(),
        1,
    );

    // concat padding with history
    // burn throws error "assertion failed: divisor != 0" if padding is empty, so
    // we branch here manually.
    if padding_len == 0 {
        debug_assert_eq!(
            history_tensor.shape()[1],
            BOARD_INPUT_HISTORY * BOARD_INPUT_CHANNELS,
            "if we need no padding the history tensor should be full"
        );
        history_tensor
    }
    else if history_len == 0 {
        debug_assert_eq!(
            padding_tensor.shape()[1],
            BOARD_INPUT_HISTORY * BOARD_INPUT_CHANNELS,
            "if we have no history the padding tensor should be full"
        );
        padding_tensor
    }
    else {
        Tensor::cat(vec![padding_tensor, history_tensor], 1)
    }
}

pub type StateInputTensor<B> = Tensor<B, STATE_INPUT_TENSOR_DIM>;

pub type StateInputFloats = [f32; STATE_INPUT_LEN];

pub fn state_input(pos: &Position) -> StateInputFloats {
    let us = pos.get_turn();
    let them = !us;
    let castling = pos.get_castling();
    [
        castling.get_float(castling_sides::KING_SIDE, us),
        castling.get_float(castling_sides::QUEEN_SIDE, us),
        castling.get_float(castling_sides::KING_SIDE, them),
        castling.get_float(castling_sides::QUEEN_SIDE, them),
        // todo: figure out what the right scale is. maybe make it dependent on the max depth at
        // which the limiter stops the mcts search? maybe just a sigmoid or something?
        pos.plys_50().v as f32 / 100.0,
    ]
}

pub fn input_batched<const N: usize, B: Backend>(
    inputs: [BoardInputFloats; N],
    device: &B::Device,
) -> Tensor<B, BOARD_INPUT_TENSOR_DIM> {
    Tensor::from_floats(inputs, device)
}

pub const VALUE_OUTPUTS: usize = 1;

// because our value_output_layer is uses tanh
pub const VALUE_WIN: f32 = 1.0;
pub const VALUE_DRAW: f32 = 0.0;
pub const VALUE_LOSE: f32 = -1.0;

pub type ValueOutputTensor<B> = Tensor<B, VALUE_OUTPUT_TENSOR_DIM>;

pub const POLICY_OUTPUTS: usize = {
    // from * to
    squares::N_VARIANTS * squares::N_VARIANTS +
    // Possible promotions
    promo_piece_type::N_VARIANTS * files::N_VARIANTS
};

pub type PolicyOutputTensor<B> = Tensor<B, POLICY_OUTPUT_TENSOR_DIM>;

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    b_norm: BatchNorm<B>,
    activation: Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, { BOARD_INPUT_TENSOR_DIM }>,
    ) -> Tensor<B, { BOARD_INPUT_TENSOR_DIM }> {
        let x = self.conv.forward(x);
        let x = self.b_norm.forward(x);
        self.activation.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    channels_in: usize,
    channels_out: usize,
    kernel_size: [usize; 2],
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock<B> {
        ConvBlock {
            conv: Conv2dConfig::new([self.channels_in, self.channels_out], self.kernel_size)
                // PaddingConfig2d::Same ensures our 8x8 spatial grid is preserved.
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            b_norm: BatchNormConfig::new(self.channels_out).init(device),
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    relu1: Relu,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    relu2: Relu,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, { BOARD_INPUT_TENSOR_DIM }>,
    ) -> Tensor<B, { BOARD_INPUT_TENSOR_DIM }> {
        let identity = x.clone();

        let out = self.conv1.forward(x);
        let out = self.bn1.forward(out);
        let out = self.relu1.forward(out);

        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        // Residual skip-connection: add the original input before the final activation
        let out = out + identity;
        self.relu2.forward(out)
    }
}

#[derive(Config, Debug)]
pub struct ResidualBlockConfig {
    channels: usize,
}

impl ResidualBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        ResidualBlock {
            conv1: Conv2dConfig::new([self.channels, self.channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn1: BatchNormConfig::new(self.channels).init(device),
            relu1: Relu::new(),
            conv2: Conv2dConfig::new([self.channels, self.channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn2: BatchNormConfig::new(self.channels).init(device),
            relu2: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    // Shared Layers (The Tower)
    initial_conv: ConvBlock<B>,
    res_blocks: Vec<ResidualBlock<B>>,

    // Value Head
    value_conv: Conv2d<B>,
    value_bn: BatchNorm<B>,
    value_relu: Relu,
    value_dense1: Linear<B>,
    value_dense2: Linear<B>,
    value_tanh: Tanh,

    // Policy Head
    policy_conv: Conv2d<B>,
    policy_bn: BatchNorm<B>,
    policy_relu: Relu,
    policy_dense: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(
        &self,
        board_input: Tensor<B, 4>,
        state_input: Tensor<B, 2>,
    ) -> (
        Tensor<B, 2>, // VALUE_OUTPUT_TENSOR_DIM
        Tensor<B, 2>, // POLICY_OUTPUT_TENSOR_DIM
    ) {
        // --- 1. SHARED TOWER ---
        let mut x = self.initial_conv.forward(board_input);

        for block in self.res_blocks.iter() {
            x = block.forward(x);
        }

        // --- 2. VALUE HEAD ---
        let v = self.value_conv.forward(x.clone());
        let v = self.value_bn.forward(v);
        let v = self.value_relu.forward(v);

        // Flatten spatial dimensions
        let v = v.flatten(1, 3);

        // Inject state inputs (castling, fifty-move rule)
        let v = Tensor::cat(vec![v, state_input], 1);

        let v = self.value_dense1.forward(v);
        let v = self.value_relu.forward(v);

        let value_out = self.value_dense2.forward(v);
        let value_out = self.value_tanh.forward(value_out);

        // --- 3. POLICY HEAD ---
        let p = self.policy_conv.forward(x);
        let p = self.policy_bn.forward(p);
        let p = self.policy_relu.forward(p);

        // Flatten spatial dimensions
        let p = p.flatten(1, 3);

        let policy_out = self.policy_dense.forward(p);

        (value_out, policy_out)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 128)]
    pub channels: usize,
    #[config(default = 8)]
    pub num_res_blocks: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let input_channels = BOARD_INPUT_CHANNELS * BOARD_INPUT_HISTORY;

        // --- 1. SHARED TOWER INITIALIZATION ---
        let initial_conv = ConvBlockConfig::new(input_channels, self.channels, [3, 3]).init(device);

        let mut res_blocks = Vec::with_capacity(self.num_res_blocks);
        for _ in 0..self.num_res_blocks {
            res_blocks.push(ResidualBlockConfig::new(self.channels).init(device));
        }

        // --- 2. VALUE HEAD INITIALIZATION ---
        let value_head_channels = 1;
        let value_conv =
            Conv2dConfig::new([self.channels, value_head_channels], [1, 1]).init(device);
        let value_bn = BatchNormConfig::new(value_head_channels).init(device);

        // 8x8 squares * 1 channel + state info
        let value_flattened_size =
            (ranks::N_VARIANTS * files::N_VARIANTS * value_head_channels) + STATE_INPUT_LEN;
        let value_dense_hidden = 256;

        let value_dense1 = LinearConfig::new(value_flattened_size, value_dense_hidden).init(device);
        let value_dense2 = LinearConfig::new(value_dense_hidden, VALUE_OUTPUTS).init(device);

        // --- 3. POLICY HEAD INITIALIZATION ---
        let policy_head_channels = 2;
        let policy_conv =
            Conv2dConfig::new([self.channels, policy_head_channels], [1, 1]).init(device);
        let policy_bn = BatchNormConfig::new(policy_head_channels).init(device);

        // 8x8 squares * 2 channels
        let policy_flattened_size = ranks::N_VARIANTS * files::N_VARIANTS * policy_head_channels;

        let policy_dense = LinearConfig::new(policy_flattened_size, POLICY_OUTPUTS).init(device);

        Model {
            initial_conv,
            res_blocks,

            value_conv,
            value_bn,
            value_relu: Relu::new(),
            value_dense1,
            value_dense2,
            value_tanh: Tanh::new(),

            policy_conv,
            policy_bn,
            policy_relu: Relu::new(),
            policy_dense,
        }
    }
}

#[derive(Debug, Error)]
pub enum LoadNNError {
    #[error("Bad nn record: {0}")]
    BadRecord(#[from] RecorderError),
}

impl<B: Backend> TryFrom<(PathBuf, &B::Device)> for Model<B> {
    type Error = LoadNNError;

    fn try_from((path, device): (PathBuf, &B::Device)) -> Result<Self, Self::Error> {
        let record = CompactRecorder::new().load(path, device)?;
        let nn = ModelConfig::new().init(device).load_record(record);
        // todo: figure out a workaround for the lazy eval of the burn framework
        // let _warmup = nn
        //     .forward(
        //         BoardInputTensor::<B>::zeros(
        //             [
        //                 1,
        //                 BOARD_INPUT_HISTORY * BOARD_INPUT_CHANNELS,
        //                 ranks::N_VARIANTS,
        //                 files::N_VARIANTS,
        //             ],
        //             device,
        //         ),
        //         StateInputTensor::<B>::zeros([1, STATE_INPUT_LEN], device),
        //     )
        //     .1
        //     .to_data()
        //     .to_vec::<f32>();
        // B::sync(&device);
        // (the above does not work)
        Ok(nn)
    }
}
