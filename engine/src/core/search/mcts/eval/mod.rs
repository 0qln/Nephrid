use std::sync::Once;

use burn_cuda::Cuda;
use model::Model;

pub mod model;

#[cfg(test)]
pub mod test;

pub type Backend = Cuda<f32>;