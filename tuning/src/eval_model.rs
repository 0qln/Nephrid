use burn_cuda::CudaDevice;

fn main() {
    let device = CudaDevice::default();
    println!("Running on {:?}", device);
}