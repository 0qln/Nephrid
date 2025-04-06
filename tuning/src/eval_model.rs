use burn_cuda::{Cuda, CudaDevice};
use engine::core::search::mcts::eval::model::ModelConfig;

fn main() {
    let device = CudaDevice::default();
    println!("Device: {:?}", device);
    
    type Backend = Cuda<f32>;
    let model = ModelConfig::new().init::<Backend>(&device);
    println!("Model: {:#?}", model);
}