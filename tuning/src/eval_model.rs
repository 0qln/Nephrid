use burn::tensor::{Float, Tensor, TensorKind};
use burn_cuda::{Cuda, CudaDevice};

fn main() {
    let device = CudaDevice::default();
    
    let t = Tensor::<Cuda, 2, Float>::zeros([10, 10], &device);
    
    println!("Tensor: {:?}", t);

    println!("Running on {:?}", device);
}