use super::*;
use burn_cuda::{Cuda, CudaDevice};

use crate::core::{move_iter::sliding_piece::magics, zobrist};

fn make_input<B: Backend>(
    batch_size: usize,
    device: &B::Device,
) -> (BoardInputTensor<B>, StateInputTensor<B>) {
    let board_input = BoardInputTensor::<B>::zeros(
        [
            batch_size,
            BOARD_INPUT_HISTORY * BOARD_INPUT_CHANNELS,
            ranks::N_VARIANTS,
            files::N_VARIANTS,
        ],
        device,
    );

    let state_input = StateInputTensor::<B>::zeros([batch_size, STATE_INPUT_LEN], device);

    (board_input, state_input)
}

#[test]
fn inference() {
    zobrist::init();
    magics::init();

    let device = CudaDevice::default();
    println!("Device: {:?}", device);

    type Backend = Cuda<f32>;
    let model = ModelConfig::new().init::<Backend>(&device);

    for i in [1, 2, 4, 8, 16, 32, 64] {
        println!("Running `forward` with batchsize {i}");
        let (b_in, s_in) = make_input(i, &device);
        let (_value, _policy) = model.forward(b_in, s_in);
    }
}
