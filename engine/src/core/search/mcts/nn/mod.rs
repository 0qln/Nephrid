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

use crate::{
    core::{
        bitboard,
        castling::castling_sides,
        color::colors,
        coordinates::{files, ranks, squares},
        piece::{piece_type, promo_piece_type},
        position::Position,
    },
    misc::{CheckHealth, CheckHealthResult},
};

#[cfg(test)]
pub mod test;

#[derive(Debug, Error)]
pub enum CheckTensorHealthError {
    #[error("Tensor value is nan: {0}")]
    Nan(f32),

    #[error("Tensor value is infinite: {0}")]
    Infinite(f32),
}

impl<const D: usize, B: Backend> CheckHealth for Tensor<B, D> {
    type Error = CheckTensorHealthError;

    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        for &value in self.clone().into_data().as_slice::<f32>().unwrap() {
            if value.is_nan() {
                return Err(CheckTensorHealthError::Nan(value));
            }
            if value.is_infinite() {
                return Err(CheckTensorHealthError::Infinite(value));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum CheckLinearHealthError {
    #[error("Linear weight tensor is unhealthy: {0}")]
    Weight(CheckTensorHealthError),

    #[error("Linear bias tensor is unhealthy: {0}")]
    Bias(CheckTensorHealthError),
}

impl<B: Backend> CheckHealth for Linear<B> {
    type Error = CheckLinearHealthError;
    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        if let Err(e) = self.weight.clone().into_value().check_health() {
            return Err(CheckLinearHealthError::Weight(e));
        }

        if let Some(bias) = self.bias.clone()
            && let Err(e) = bias.into_value().check_health()
        {
            return Err(CheckLinearHealthError::Bias(e));
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum CheckConv2dHealthError {
    #[error("Conv weight tensor is unhealthy: {0}")]
    Weight(CheckTensorHealthError),

    #[error("Conv bias tensor is unhealthy: {0}")]
    Bias(CheckTensorHealthError),
}

impl<B: Backend> CheckHealth for Conv2d<B> {
    type Error = CheckConv2dHealthError;

    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        if let Err(e) = self.weight.clone().into_value().check_health() {
            return Err(CheckConv2dHealthError::Weight(e));
        }

        if let Some(bias) = self.bias.clone()
            && let Err(e) = bias.into_value().check_health()
        {
            return Err(CheckConv2dHealthError::Bias(e));
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum CheckBatchnormHealthError {
    #[error("BatchNorm gamma tensor is unhealthy: {0}")]
    Gamma(CheckTensorHealthError),

    #[error("BatchNorm beta tensor is unhealthy: {0}")]
    Beta(CheckTensorHealthError),
}

impl<B: Backend> CheckHealth for BatchNorm<B> {
    type Error = CheckBatchnormHealthError;

    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        if let Err(e) = self.gamma.clone().into_value().check_health() {
            return Err(CheckBatchnormHealthError::Gamma(e));
        }

        if let Err(e) = self.beta.clone().into_value().check_health() {
            return Err(CheckBatchnormHealthError::Beta(e));
        }

        Ok(())
    }
}

pub const INPUT_CHANNELS: usize = BOARD_INPUT_CHANNELS * BOARD_INPUT_HISTORY;

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

    // todo: more inputs (e.g. pins, checks, attacks)
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

pub const VALUE_OUTPUTS: usize = 1;

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

#[derive(Debug, Error)]
pub enum CheckConvBlockHealthError {
    #[error("Conv block is unhealthy: {0}")]
    BadConv(#[from] CheckConv2dHealthError),

    #[error("BatchNorm in conv block is unhealthy: {0}")]
    BadBatchNorm(#[from] CheckBatchnormHealthError),
}

impl<B: Backend> CheckHealth for ConvBlock<B> {
    type Error = CheckConvBlockHealthError;
    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        self.conv.check_health()?;
        self.b_norm.check_health()?;
        Ok(())
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

#[derive(Debug, Error)]
pub enum CheckResidualBlockHealthError {
    #[error("Conv block is unhealthy: {0}")]
    BadConv(#[from] CheckConv2dHealthError),

    #[error("BatchNorm in conv block is unhealthy: {0}")]
    BadBatchNorm(#[from] CheckBatchnormHealthError),
}

impl<B: Backend> CheckHealth for ResidualBlock<B> {
    type Error = CheckResidualBlockHealthError;
    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        self.conv1.check_health()?;
        self.bn1.check_health()?;
        self.conv2.check_health()?;
        self.bn2.check_health()?;
        Ok(())
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
    initial_conv: ConvBlock<B>,
    res_blocks: Vec<ResidualBlock<B>>,

    // Value Head
    value_conv: ConvBlock<B>,
    value_relu: Relu,
    value_dense1: Linear<B>,
    value_dense2: Linear<B>,
    value_tanh: Tanh,

    // Policy Head
    policy_conv: ConvBlock<B>,
    policy_dense: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(
        &self,
        board_input: BoardInputTensor<B>,
        state_input: StateInputTensor<B>,
    ) -> (ValueOutputTensor<B>, PolicyOutputTensor<B>) {
        let mut x = self.initial_conv.forward(board_input);

        for block in self.res_blocks.iter() {
            x = block.forward(x);
        }

        // Value Head
        let v = self.value_conv.forward(x.clone());
        let v = v.flatten(1, 3);

        // State Inputs
        let v = Tensor::cat(vec![v, state_input], 1);
        let v = self.value_dense1.forward(v);
        let v = self.value_relu.forward(v);
        let v = self.value_dense2.forward(v);
        let v = self.value_tanh.forward(v);

        // Policy Head
        // todo: maybe also inject state data here?
        let p = self.policy_conv.forward(x);
        let p = p.flatten(1, 3);
        let p = self.policy_dense.forward(p);

        (v, p)
    }

    pub fn warmup(&self, batch_size_max: usize, device: &B::Device) {
        // todo:
        // this or just doing one batchsize and then filling with zeroes if not full?
        for batch_size in 1..=batch_size_max {
            let dummy_state = StateInputTensor::<B>::zeros([batch_size, STATE_INPUT_LEN], device);
            let dummy_board = BoardInputTensor::<B>::zeros(
                [
                    batch_size,
                    INPUT_CHANNELS,
                    ranks::N_VARIANTS,
                    files::N_VARIANTS,
                ],
                device,
            );

            let output = self.forward(dummy_board, dummy_state);

            let _warmup = output.0.sum().into_scalar();
            let _warmup = output.1.sum().into_scalar();
        }

        B::sync(device);
    }
}

#[derive(Debug, Error)]
pub enum CheckModelHealthError {
    #[error("Initial Conv Block is unhealthy: {0}")]
    InitialConv(CheckConvBlockHealthError),

    #[error("Residual Block {index} is unhealthy: {error}")]
    ResidualBlock {
        index: usize,
        #[source]
        error: CheckResidualBlockHealthError,
    },

    #[error("Value Head is unhealthy: {0}")]
    ValueHeadConv(CheckConvBlockHealthError),

    #[error("Value Head is unhealthy: {0}")]
    ValueHeadDense(CheckLinearHealthError),

    #[error("Policy Head is unhealthy: {0}")]
    PolicyHeadConv(CheckConvBlockHealthError),

    #[error("Policy Head is unhealthy: {0}")]
    PolicyHeadDense(CheckLinearHealthError),
}

impl<B: Backend> CheckHealth for Model<B> {
    type Error = CheckModelHealthError;
    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        self.initial_conv
            .check_health()
            .map_err(Self::Error::InitialConv)?;

        for (index, block) in self.res_blocks.iter().enumerate() {
            block
                .check_health()
                .map_err(|e| CheckModelHealthError::ResidualBlock { index, error: e })?;
        }

        self.value_conv
            .check_health()
            .map_err(Self::Error::ValueHeadConv)?;
        self.value_dense1
            .check_health()
            .map_err(Self::Error::ValueHeadDense)?;
        self.value_dense2
            .check_health()
            .map_err(Self::Error::ValueHeadDense)?;

        self.policy_conv
            .check_health()
            .map_err(CheckModelHealthError::PolicyHeadConv)?;
        self.policy_dense
            .check_health()
            .map_err(Self::Error::PolicyHeadDense)?;

        Ok(())
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 128)]
    pub channels: usize,

    #[config(default = 8)]
    pub num_res_blocks: usize,

    #[config(default = 256)]
    pub value_dense_hidden: usize,

    #[config(default = 1)]
    pub value_head_channels: usize,

    #[config(default = 2)]
    pub policy_head_channels: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let Self {
            channels,
            value_dense_hidden,
            num_res_blocks,
            value_head_channels,
            policy_head_channels,
        } = *self;

        let initial_conv = ConvBlockConfig::new(INPUT_CHANNELS, channels, [3, 3]).init(device);

        let mut res_blocks = Vec::with_capacity(num_res_blocks);
        for _ in 0..num_res_blocks {
            res_blocks.push(ResidualBlockConfig::new(channels).init(device));
        }

        // Value Head
        let value_conv = ConvBlockConfig::new(channels, value_head_channels, [1, 1]).init(device);
        let value_flattened_size = (squares::N_VARIANTS * value_head_channels) + STATE_INPUT_LEN;
        let value_dense1 = LinearConfig::new(value_flattened_size, value_dense_hidden).init(device);
        let value_dense2 = LinearConfig::new(value_dense_hidden, VALUE_OUTPUTS).init(device);

        // Policy Head
        let policy_conv = ConvBlockConfig::new(channels, policy_head_channels, [1, 1]).init(device);
        let policy_flattened_size = squares::N_VARIANTS * policy_head_channels;
        let policy_dense = LinearConfig::new(policy_flattened_size, POLICY_OUTPUTS).init(device);

        Model {
            initial_conv,
            res_blocks,

            value_conv,
            value_relu: Relu::new(),
            value_dense1,
            value_dense2,
            value_tanh: Tanh::new(),

            policy_conv,
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
        Ok(nn)
    }
}
