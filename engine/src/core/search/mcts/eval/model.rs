use burn::{
    config::Config,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
        Relu, Tanh,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::Backend,
    tensor::{Tensor, activation::softmax},
};

use crate::core::{
    bitboard,
    castling::castling_sides,
    color::colors,
    coordinates::{files, ranks, squares},
    piece::{piece_type, promo_piece_type},
    position::Position,
};

// +1 for the possible ep capture square.
const BOARD_INPUT_CHANNELS: usize = colors::N_VARIANTS * (piece_type::N_VARIANTS);

const BOARD_INPUT_TENSOR_DIM: usize = {
    // Batch
    1 +
    // Channel
    1 +
    // Rank
    1 +
    // File
    1
};

const STATE_INPUT_LEN: usize = {
    // Castling rights
    4 +
    // Plys after last capture or pawn move (should be normalized / devided by 50)
    1
};

const STATE_INPUT_TENSOR_DIM: usize = {
    // Batch
    1 +
    // Values
    1
};

type BoardInputFloats = [bitboard::Floats; BOARD_INPUT_CHANNELS];

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

type StateInputFloats = [f32; STATE_INPUT_LEN];

pub fn state_input(pos: &Position) -> StateInputFloats {
    let us = pos.get_turn();
    let them = !us;
    let castling = pos.get_castling();
    [
        castling.get_float(castling_sides::KING_SIDE, us),
        castling.get_float(castling_sides::QUEEN_SIDE, us),
        castling.get_float(castling_sides::KING_SIDE, them),
        castling.get_float(castling_sides::QUEEN_SIDE, them),
        pos.plys_50().v as f32 / 50.0,
    ]
}

pub fn input_batched<const N: usize, B: Backend>(
    inputs: [BoardInputFloats; N],
    device: &B::Device,
) -> Tensor<B, BOARD_INPUT_TENSOR_DIM> {
    Tensor::from_floats(inputs, device)
}

const VALUE_OUTPUTS: usize = 1;

pub const POLICY_OUTPUTS: usize = {
    // from * to
    squares::N_VARIANTS * squares::N_VARIANTS +
    // Possible promotions
    promo_piece_type::N_VARIANTS * files::N_VARIANTS
};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    b_norm: BatchNorm<B, { BOARD_INPUT_TENSOR_DIM - 2 }>,
    activation: Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, { BOARD_INPUT_TENSOR_DIM }>,
    ) -> Tensor<B, { BOARD_INPUT_TENSOR_DIM }> {
        let x = self.conv.forward(x);
        let x = self.b_norm.forward(x);
        let x = self.activation.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    channels: [usize; 2],
    kernel_size: [usize; 2],
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock<B> {
        ConvBlock {
            conv: Conv2dConfig::new(self.channels, self.kernel_size)
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            activation: Default::default(),
            b_norm: BatchNormConfig::new(self.channels[1]).init(device),
        }
    }
}

const B1_HEADS: usize = 6;
const B1_CHANNELS: usize = 16;
const B1_KERNELS: [[usize; 2]; B1_HEADS] = [[3, 3], [5, 5], [9, 9], [15, 15], [1, 7], [7, 1]];

const B2_HEADS: usize = 2;
const B2_CHANNELS: usize = 32;
const B2_KERNELS: [[usize; 2]; B2_HEADS] = [[3, 3], [5, 5]];
const B2_ADAPTER_CHANNELS: usize = B2_KERNELS.len() * B2_CHANNELS;

const B3_HEADS: usize = 2;
const B3_CHANNELS: usize = 32;
const B3_KERNELS: [[usize; 2]; B3_HEADS] = [[3, 3], [5, 5]];
const B3_ADAPTER_CHANNELS: usize = B3_KERNELS.len() * B3_CHANNELS;

const B4_HEADS: usize = 1;
const B4_CHANNELS: usize = 64;
const B4_KERNELS: [[usize; 2]; B4_HEADS] = [[3, 3]];
const B4_ADAPTER_CHANNELS: usize = B4_KERNELS.len() * B4_CHANNELS;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    b1_conv_0: ConvBlock<B>,
    b1_conv_1: ConvBlock<B>,
    b1_conv_2: ConvBlock<B>,
    b1_conv_3: ConvBlock<B>,
    b1_conv_4: ConvBlock<B>,
    b1_conv_5: ConvBlock<B>,

    b2_adaper: ConvBlock<B>,
    b2_conv_0: ConvBlock<B>,
    b2_conv_1: ConvBlock<B>,
    b2_pool: MaxPool2d,

    b3_adaper: ConvBlock<B>,
    b3_conv_0: ConvBlock<B>,
    b3_conv_1: ConvBlock<B>,
    b3_pool: MaxPool2d,

    b4_adaper: ConvBlock<B>,
    b4_conv_0: ConvBlock<B>,
    b4_pool: MaxPool2d,

    dropout: Dropout,

    dense0: Linear<B>,
    dense1: Linear<B>,
    dense2: Linear<B>,
    dense3: Linear<B>,

    value_dense: Linear<B>,
    value_out: Linear<B>,
    value_activ: Tanh,

    policy_dense: Linear<B>,
    policy_out: Linear<B>,
    // (softmax)
}

impl<B: Backend> Model<B> {
    /// # Shapes
    /// - `board_input`: (batch_size, see: `BOARD_INPUT_LEN`, ranks, files)
    /// - `state_input`: (batch_size, see: `STATE_INPUT_LEN`)
    /// - `value_out`: (batch_size, 1)
    /// - `policy_out`: (batch_size, num_moves)
    pub fn forward(
        &self,
        board_input: Tensor<B, 4>,
        state_input: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [bi_batch_size, bil, rs, fs] = board_input.dims();
        assert_eq!(bil, BOARD_INPUT_CHANNELS);
        assert_eq!(rs, ranks::N_VARIANTS);
        assert_eq!(fs, files::N_VARIANTS);

        let [si_batch_size, sil] = state_input.dims();
        assert_eq!(si_batch_size, bi_batch_size);
        assert_eq!(sil, STATE_INPUT_LEN);

        let x = board_input;
        let x = Tensor::cat(
            vec![
                self.b1_conv_0.forward(x.clone()),
                self.b1_conv_1.forward(x.clone()),
                self.b1_conv_2.forward(x.clone()),
                self.b1_conv_3.forward(x.clone()),
                self.b1_conv_4.forward(x.clone()),
                self.b1_conv_5.forward(x),
            ],
            1,
        );

        let x = self.b2_adaper.forward(x);
        let x = x.clone()
            + Tensor::cat(
                vec![self.b2_conv_0.forward(x.clone()), self.b2_conv_1.forward(x)],
                1,
            );
        let x = self.b2_pool.forward(x);

        let x = self.b3_adaper.forward(x);
        let x = x.clone()
            + Tensor::cat(
                vec![self.b3_conv_0.forward(x.clone()), self.b3_conv_1.forward(x)],
                1,
            );
        let x = self.b3_pool.forward(x);

        let x = self.b4_adaper.forward(x);
        let x = x.clone() + self.b4_conv_0.forward(x);
        let x = self.b4_pool.forward(x);

        let x = x.flatten(1, 3);

        // todo: should we always use a dropout layer?
        let x = self.dropout.forward(x);

        let x = self.dense0.forward(x);
        let x = self.dense1.forward(x);
        let x = self.dense2.forward(x);
        let x = self.dense3.forward(x);

        let value_out = self.value_dense.forward(x.clone());
        let value_out = self.value_out.forward(value_out);
        let value_out = self.value_activ.forward(value_out);

        let policy_out = self.policy_dense.forward(x.clone());
        let policy_out = self.policy_out.forward(policy_out);
        let policy_out = softmax(policy_out, 1);

        (value_out, policy_out)
    }

    pub fn forward_with_loss(
        &self,
        board_input: Tensor<B, 4>,
        state_input: Tensor<B, 2>,
        target_value: Tensor<B, 2>,
        target_policy: Tensor<B, 2>,
    ) -> () {
        unimplemented!()
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 0.5)]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            b1_conv_0: ConvBlockConfig::new([BOARD_INPUT_CHANNELS, B1_CHANNELS], B1_KERNELS[0])
                .init(device),
            b1_conv_1: ConvBlockConfig::new([BOARD_INPUT_CHANNELS, B1_CHANNELS], B1_KERNELS[1])
                .init(device),
            b1_conv_2: ConvBlockConfig::new([BOARD_INPUT_CHANNELS, B1_CHANNELS], B1_KERNELS[2])
                .init(device),
            b1_conv_3: ConvBlockConfig::new([BOARD_INPUT_CHANNELS, B1_CHANNELS], B1_KERNELS[3])
                .init(device),
            b1_conv_4: ConvBlockConfig::new([BOARD_INPUT_CHANNELS, B1_CHANNELS], B1_KERNELS[4])
                .init(device),
            b1_conv_5: ConvBlockConfig::new([BOARD_INPUT_CHANNELS, B1_CHANNELS], B1_KERNELS[5])
                .init(device),

            b2_adaper: ConvBlockConfig::new([B1_HEADS * B1_CHANNELS, B2_ADAPTER_CHANNELS], [1, 1])
                .init(device),
            b2_conv_0: ConvBlockConfig::new([B2_ADAPTER_CHANNELS, B2_CHANNELS], B2_KERNELS[0])
                .init(device),
            b2_conv_1: ConvBlockConfig::new([B2_ADAPTER_CHANNELS, B2_CHANNELS], B2_KERNELS[1])
                .init(device),
            b2_pool: MaxPool2dConfig::new([2, 2]).init(),

            b3_adaper: ConvBlockConfig::new([B2_HEADS * B2_CHANNELS, B3_ADAPTER_CHANNELS], [1, 1])
                .init(device),
            b3_conv_0: ConvBlockConfig::new([B3_ADAPTER_CHANNELS, B3_CHANNELS], B3_KERNELS[0])
                .init(device),
            b3_conv_1: ConvBlockConfig::new([B3_ADAPTER_CHANNELS, B3_CHANNELS], B3_KERNELS[1])
                .init(device),
            b3_pool: MaxPool2dConfig::new([2, 2]).init(),

            b4_adaper: ConvBlockConfig::new([B3_HEADS * B3_CHANNELS, B4_ADAPTER_CHANNELS], [1, 1])
                .init(device),
            b4_conv_0: ConvBlockConfig::new([B4_ADAPTER_CHANNELS, B4_CHANNELS], B4_KERNELS[0])
                .init(device),
            b4_pool: MaxPool2dConfig::new([2, 2]).init(),

            dropout: DropoutConfig::new(self.dropout).init(),

            // todo: i got no clue where the 5x5 comes from
            // (probably from b4 filter >>> pool >>> ..., but that should be 4x4?)
            dense0: LinearConfig::new(B4_CHANNELS * B4_HEADS * 5 * 5, 64 << 4).init(device),
            dense1: LinearConfig::new(64 << 4, 64 << 3).init(device),
            dense2: LinearConfig::new(64 << 3, 64 << 2).init(device),
            dense3: LinearConfig::new(64 << 2, 64 << 2).init(device),

            value_dense: LinearConfig::new(64 << 2, 64 << 1).init(device),
            value_out: LinearConfig::new(64 << 1, VALUE_OUTPUTS).init(device),
            value_activ: Tanh::new(),

            policy_dense: LinearConfig::new(64 << 2, 64 << 4).init(device),
            policy_out: LinearConfig::new(64 << 4, POLICY_OUTPUTS).init(device),
        }
    }
}
