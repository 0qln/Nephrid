use burn_cuda::Cuda;

pub mod model;

#[cfg(test)]
pub mod test;

pub type Backend = Cuda<f32>;