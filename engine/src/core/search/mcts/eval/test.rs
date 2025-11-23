use burn_cuda::{Cuda, CudaDevice};

use crate::core::{move_iter::sliding_piece::magics, position::Position, search::mcts::eval::model::{board_input, state_input, ModelConfig}, zobrist};

#[test]
fn inference() {
    zobrist::init();
    magics::init();

    let device = CudaDevice::default();
    println!("Device: {:?}", device);
    
    type Backend = Cuda<f32>;
    let model = ModelConfig::new().init::<Backend>(&device);
    println!("Model: {:#?}", model);
    
    let pos = Position::start_position();

    let board_input = [board_input(&pos)].into();
    let state_input = [state_input(&pos)].into();
    
    let (value, policy) = model.forward(board_input, state_input);
    
    let value = value.to_data().into_vec::<f32>();
    let policy = policy.to_data().into_vec::<f32>();
    
    println!("Value: {:?}", value);
    println!("Policy: {:?}", policy);
}
