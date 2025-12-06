use super::model::*;
use burn_cuda::{Cuda, CudaDevice};

use crate::core::{
    move_iter::sliding_piece::magics,
    position::Position,
    search::mcts::eval::model::{ModelConfig, board_input, state_input},
    zobrist,
};

#[test]
fn inference() {
    zobrist::init();
    magics::init();

    let device = CudaDevice::default();
    println!("Device: {:?}", device);

    type Backend = Cuda<f32>;
    let model = ModelConfig::new().init::<Backend>(&device);

    let pos = Position::start_position();

    let board_input: BoardInputFloats = board_input(&pos);
    let state_input: StateInputFloats = state_input(&pos);

    for i in 0..10 {
        let (value, policy) =
            model.forward([board_input.clone()].into(), [state_input.clone()].into());

        let value = value.to_data().into_vec::<f32>();
        let policy = policy.to_data().into_vec::<f32>();
        println!("[{i}] Value: {:?}", value);
        println!("[{i}] Policy: {:?}", policy);
    }
}
