use std::error::Error;

use burn::{
    backend,
    data::dataloader::batcher::Batcher,
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use burn_cuda::{Cuda, CudaDevice};
use engine::{
    core::{
        bitboard,
        coordinates::{files, ranks},
        move_iter::sliding_piece::magics,
        position::Position,
        search::{
            self,
            limit::Limit,
            mcts::eval::model::{
                BOARD_INPUT_CHANNELS, BOARD_INPUT_LEN, BOARD_INPUT_TENSOR_DIM, Model, ModelConfig,
                POLICY_OUTPUT_TENSOR_DIM, POLICY_OUTPUTS, STATE_INPUT_LEN, STATE_INPUT_TENSOR_DIM,
                VALUE_OUTPUT_TENSOR_DIM,
            },
        },
        zobrist,
    },
    misc::DebugMode,
    uci::{sync::CancellationToken, tokens::Tokenizer},
};

fn main() {
    magics::init();
    zobrist::init();

    let device = CudaDevice::default();
    println!("Device: {:?}", device);

    type Backend = Cuda<f32>;
    let model = ModelConfig::new().init::<Backend>(&device);
    println!("Model: {:#?}", model);

    // let batch = generate_batch(&model);
    let result = self_play(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &model,
    );
    println!("{result:?}");
}

#[derive(Clone, Debug)]
pub struct SelfplayBatch<B: Backend> {
    pub board_inputs: Tensor<B, BOARD_INPUT_TENSOR_DIM>,
    pub state_inputs: Tensor<B, STATE_INPUT_TENSOR_DIM>,
    pub value_targets: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    pub policy_targets: Tensor<B, POLICY_OUTPUT_TENSOR_DIM>,
}

pub struct SelfplayItem {
    pub board_input: [bitboard::Floats; BOARD_INPUT_CHANNELS],
    pub state_input: [f32; STATE_INPUT_LEN],
    pub value_target: f32,
    pub policy_target: [f32; POLICY_OUTPUTS],
}

#[derive(Clone, Default)]
pub struct SelfplayBatcher;

impl<B: Backend> Batcher<SelfplayItem, SelfplayBatch<B>> for SelfplayBatcher {
    fn batch(&self, items: Vec<SelfplayItem>) -> SelfplayBatch<B> {
        let device = &<B as Backend>::Device::default();
        let boards = items
            .iter()
            .map(|x| TensorData::from(x.board_input))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let states = items
            .iter()
            .map(|x| TensorData::from(x.state_input))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let values = items
            .iter()
            .map(|x| TensorData::from([x.value_target]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let policies = items
            .iter()
            .map(|x| TensorData::from(x.policy_target))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        SelfplayBatch {
            board_inputs: Tensor::cat(boards, 0),
            state_inputs: Tensor::cat(states, 0),
            value_targets: Tensor::cat(values, 0),
            policy_targets: Tensor::cat(policies, 0),
        }
    }
}

fn self_play<B: Backend>(pos: &str, model: &Model<B>) -> Result<(), Box<dyn Error>> {
    let limit = Limit {
        is_active: true,
        winc: 100,
        binc: 100,
        wtime: 0,
        btime: 0,
        ..Default::default()
    };
    let debug = DebugMode::default();
    let ct = CancellationToken::new();

    let tok = &mut Tokenizer::new(pos);
    let mut pos: Position = tok
        .try_into()
        .expect(format!("Invalid FEN: {pos}").as_str());

    println!("{pos}");

    loop {
        let mov = search::mcts(pos.clone(), model, limit.clone(), debug.clone(), ct.clone());

        if mov.is_none() || pos.fifty_move_rule() || pos.has_threefold_repetition() {
            break;
        }

        pos.make_move(mov.unwrap());

        println!("{pos}");
    }

    Ok(())
}
