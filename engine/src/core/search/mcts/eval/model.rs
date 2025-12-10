use crate::core::zobrist;
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
use itertools::Itertools;

use crate::core::{
    bitboard,
    castling::castling_sides,
    color::colors,
    coordinates::{files, ranks, squares},
    piece::{piece_type, promo_piece_type},
    position::Position,
    search::mcts,
};

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

pub const POLICY_TARGET_TENSOR_DIM: usize = {
    POLICY_OUTPUT_TENSOR_DIM -
    // Just use the index, no need to bring all the zeros.
    1
};

pub const VALUE_OUTPUT_TENSOR_DIM: usize = {
    // Batch
    1 +
    // singleton
    1
};

pub type BoardInputTensor<B: Backend> = Tensor<B, 4>;

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

pub type StateInputTensor<B: Backend> = Tensor<B, 2>;

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
        // which the limiter stops the mcts search?
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

// because our value_output_layer is uses tanh
pub const VALUE_WIN: f32 = 1.0;
pub const VALUE_DRAW: f32 = 0.0;
pub const VALUE_LOSE: f32 = -1.0;

pub const POLICY_OUTPUTS: usize = {
    // from * to
    squares::N_VARIANTS * squares::N_VARIANTS +
    // Possible promotions
    promo_piece_type::N_VARIANTS * files::N_VARIANTS
};

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
        let x = self.activation.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    channels_in: usize,
    channels: usize,
    kernel_size: [usize; 2],
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock<B> {
        ConvBlock {
            conv: Conv2dConfig::new([self.channels_in, self.channels], self.kernel_size)
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            activation: Default::default(),
            b_norm: BatchNormConfig::new(self.channels).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiConvBlock<B: Backend> {
    convs: Vec<ConvBlock<B>>,
}

impl<B: Backend> MultiConvBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, { BOARD_INPUT_TENSOR_DIM }>,
    ) -> Tensor<B, { BOARD_INPUT_TENSOR_DIM }> {
        // println!("m-conv in: {:?}", x.shape());
        let x = Tensor::cat(
            self.convs
                .iter()
                .map(|conv| conv.forward(x.clone()))
                .collect_vec(),
            1,
        );
        // println!("m-conv out: {:?}", x.shape());
        x
    }
}

#[derive(Config, Debug)]
pub struct MultiConvBlockConfig {
    heads: Vec<[usize; 2]>,
    channels_in: usize,
    channels: usize,
}

impl MultiConvBlockConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> MultiConvBlock<B> {
        let convs = self
            .heads
            .into_iter()
            .map(|k| ConvBlockConfig::new(self.channels_in, self.channels, k).init(device))
            .collect_vec();

        MultiConvBlock { convs }
    }
}

#[derive(Module, Debug)]
pub struct ResidualConvBlock<B: Backend> {
    // If we want this to be a residual layer, we need to transform the output
    // to have as many features as the input
    adapter: ConvBlock<B>,
    conv_block: MultiConvBlock<B>,
}

impl<B: Backend> ResidualConvBlock<B> {
    pub fn forward(
        &self,
        mut x: Tensor<B, { BOARD_INPUT_TENSOR_DIM }>,
    ) -> Tensor<B, { BOARD_INPUT_TENSOR_DIM }> {
        x = self.adapter.forward(x);
        x = x.clone() + self.conv_block.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct ResidualConvBlockConfig {
    conv_block: MultiConvBlockConfig,
    channels_in: usize,
}

impl ResidualConvBlockConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> ResidualConvBlock<B> {
        // need an adapter layer to go from [n,c_0,w,h] to [n,c_1,w,h]
        let adapter = ConvBlockConfig::new(
            self.channels_in,
            self.conv_block.channels * self.conv_block.heads.len(),
            [1, 1],
        )
        .init(device);

        let conv_block = self.conv_block.init(device);

        ResidualConvBlock { adapter, conv_block }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pool: MaxPool2d,

    convs1: MultiConvBlock<B>,
    convs2: MultiConvBlock<B>,
    convs3: MultiConvBlock<B>,
    convs4: MultiConvBlock<B>,
    convs5: MultiConvBlock<B>,
    convs6: MultiConvBlock<B>,
    convs7: MultiConvBlock<B>,
    convs8: MultiConvBlock<B>,
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
    /// - `value_out`: (batch_size, 1)
    /// - `policy_out`: (batch_size, num_moves)
    pub fn forward(
        &self,
        // getting some kind of recursive evaluation error here, so inline the constants （´＿｀）
        board_input: Tensor<B, 4>, // BOARD_INPUT_TENSOR_DIM
        state_input: Tensor<B, 2>, // STATE_INPUT_TENSOR_DIM
    ) -> (
        Tensor<B, 2>, // VALUE_OUTPUT_TENSOR_DIM
        Tensor<B, 2>, // POLICY_OUTPUT_TENSOR_DIM
    ) {
        let [bi_batch_size, bil, rs, fs] = board_input.dims();
        assert_eq!(bil, BOARD_INPUT_CHANNELS * BOARD_INPUT_HISTORY);
        assert_eq!(rs, ranks::N_VARIANTS);
        assert_eq!(fs, files::N_VARIANTS);

        let [si_batch_size, sil] = state_input.dims();
        assert_eq!(si_batch_size, bi_batch_size);
        assert_eq!(sil, STATE_INPUT_LEN);

        let x = board_input;

        let x = self.convs1.forward(x);

        let x = self.convs2.forward(x);
        let x = self.convs3.forward(x);
        let x = self.convs4.forward(x);
        let x = self.convs5.forward(x);
        let x = self.convs6.forward(x);
        let x = self.pool.forward(x);

        let x = self.convs7.forward(x);
        let x = self.pool.forward(x);

        let x = self.convs8.forward(x);
        let x = self.pool.forward(x);

        let x = x.flatten(1, 3);

        // println!("{:?}", x.shape());

        // todo: should we always use a dropout layer?
        // let x = self.dropout.forward(x);

        let x = Tensor::cat(vec![x, state_input], 1);

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
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 0.2)]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let pool = MaxPool2dConfig::new([2, 2]);

        const B1_CHANNELS: usize = 16;
        const B1_HEADS: [[usize; 2]; 6] = [[3, 3], [5, 5], [9, 9], [15, 15], [1, 7], [7, 1]];

        const B6_CHANNELS: usize = 32;
        const B6_HEADS: [[usize; 2]; 6] = B1_HEADS; // [[3, 3], [5, 5]];

        const B7_CHANNELS: usize = 256;
        const B7_HEADS: [[usize; 2]; 2] = [[3, 3], [5, 5]];

        const B8_CHANNELS: usize = 1024;
        const B8_HEADS: [[usize; 2]; 2] = [[1, 1], [3, 3]];

        let convs1_in = BOARD_INPUT_CHANNELS * BOARD_INPUT_HISTORY;
        let convs1 = MultiConvBlockConfig::new(B1_HEADS.to_vec(), convs1_in, B1_CHANNELS);

        let convs2_in = B1_CHANNELS * B1_HEADS.len();
        let convs2 = MultiConvBlockConfig::new(B1_HEADS.to_vec(), convs2_in, B1_CHANNELS);

        let convs6_in = B1_CHANNELS * B1_HEADS.len();
        let convs6 = MultiConvBlockConfig::new(B6_HEADS.to_vec(), convs6_in, B6_CHANNELS);
        // let convs2 = ResidualConvBlockConfig::new(convs2, convs2_in);

        let convs7_in = B6_CHANNELS * B6_HEADS.len();
        let convs7 = MultiConvBlockConfig::new(B7_HEADS.to_vec(), convs7_in, B7_CHANNELS);
        // let convs3 = ResidualConvBlockConfig::new(convs3, convs3_in);

        let convs8_in = B7_CHANNELS * B7_HEADS.len();
        let convs8 = MultiConvBlockConfig::new(B8_HEADS.to_vec(), convs8_in, B8_CHANNELS);
        // let convs4 = ResidualConvBlockConfig::new(convs4, convs4_in);

        let dropout = DropoutConfig::new(self.dropout);

        let dense0 = LinearConfig::new(B8_CHANNELS * B8_HEADS.len() + STATE_INPUT_LEN, 64 << 4);
        let dense1 = LinearConfig::new(64 << 4, 64 << 3);
        let dense2 = LinearConfig::new(64 << 3, 64 << 2);
        let dense3 = LinearConfig::new(64 << 2, 64 << 2);

        let value_dense = LinearConfig::new(64 << 2, 64 << 1);
        let value_out = LinearConfig::new(64 << 1, VALUE_OUTPUTS);
        let value_activ = Tanh::new();

        let policy_dense = LinearConfig::new(64 << 2, 64 << 4);
        let policy_out = LinearConfig::new(64 << 4, POLICY_OUTPUTS);

        Model {
            convs1: convs1.clone().init(device),
            convs2: convs2.clone().init(device),
            convs3: convs2.clone().init(device),
            convs4: convs2.clone().init(device),
            convs5: convs2.clone().init(device),
            convs6: convs6.init(device),
            convs7: convs7.init(device),
            convs8: convs8.init(device),

            pool: pool.init(),

            dropout: dropout.init(),

            dense0: dense0.init(device),
            dense1: dense1.init(device),
            dense2: dense2.init(device),
            dense3: dense3.init(device),

            value_dense: value_dense.init(device),
            value_out: value_out.init(device),
            value_activ,

            policy_dense: policy_dense.init(device),
            policy_out: policy_out.init(device),
        }
    }
}
