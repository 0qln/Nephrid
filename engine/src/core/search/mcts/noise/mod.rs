use itertools::Itertools;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, num_traits::Zero};
use thiserror::Error;

use crate::core::search::mcts::node::{CtNodeRef, node_state::Evaluated};

#[cfg(test)]
pub mod test;

pub trait Noiser {
    type Error;

    /// Apply noise to the polices of this node.
    fn apply_noise(&mut self, node: &mut CtNodeRef<Evaluated>) -> Result<(), Self::Error>;
}

#[derive(Debug, Clone)]
pub struct DirichletNoiser {
    alpha: f32,
    eps: f32,
    rng: SmallRng,
}

impl DirichletNoiser {
    pub fn new(alpha: f32, eps: f32, rng: SmallRng) -> Self {
        Self { alpha, eps, rng }
    }

    pub fn gamma(&self) -> Result<Gamma<f32>, DirichletNoiseError> {
        let alpha = self.alpha.into();
        Gamma::new(alpha, 1.).map_err(|_| DirichletNoiseError::BadAlpha(alpha))
    }
}

#[derive(Debug, Error)]
pub enum DirichletNoiseError {
    #[error("Bad alpha value: {0}")]
    BadAlpha(f32),

    #[error("Total noise was too low.")]
    LowNoise,
}

impl Noiser for DirichletNoiser {
    type Error = DirichletNoiseError;

    fn apply_noise(&mut self, node: &mut CtNodeRef<Evaluated>) -> Result<(), Self::Error> {
        // Generate noise
        let noise = {
            let node = node.borrow();
            let branches = node.branches();
            let gamma = self.gamma()?;
            let distr = gamma.sample_iter(&mut self.rng);
            let noise = distr.take(branches.len()).collect_vec();
            let total = noise.iter().sum::<f32>();

            if total.is_zero() {
                return Err(DirichletNoiseError::LowNoise);
            }

            noise
        };

        // Apply noise
        let eps = self.eps;
        let mut node = node.borrow_mut();
        node.apply_policy_noise(&noise, eps);

        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct NullNoiser;

impl Noiser for NullNoiser {
    type Error = ();

    fn apply_noise(&mut self, _node: &mut CtNodeRef<Evaluated>) -> Result<(), Self::Error> {
        Ok(())
    }
}
