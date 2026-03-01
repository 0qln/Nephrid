use itertools::Itertools;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, num_traits::Zero};
use thiserror::Error;

use crate::core::search::mcts::node::Node;

// todo: fix test
// #[cfg(test)]
// pub mod test;

pub trait Noiser {
    type Error;

    /// Apply noise to the polices of this node.
    fn apply_noise(&mut self, node: &mut Node) -> Result<(), Self::Error>;
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

    fn apply_noise(&mut self, node: &mut Node) -> Result<(), Self::Error> {
        // Generate noise
        let alpha = self.alpha;
        let gamma = Gamma::new(alpha, 1.0_f32).map_err(|_| DirichletNoiseError::BadAlpha(alpha))?;
        let distr = gamma.sample_iter(&mut self.rng);
        let noise = distr.take(node.num_branches()).collect_vec();
        let total = noise.iter().sum::<f32>();

        if total.is_zero() {
            return Err(DirichletNoiseError::LowNoise);
        }

        // Apply noise
        for (branch, noise) in node.iter_branches_mut().zip(noise) {
            let eps = self.eps;
            let norm_noise = noise / total;
            let policy = branch.policy();
            branch.set_policy(policy * (1. - eps) + eps * norm_noise);
        }

        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct NullNoiser;

impl Noiser for NullNoiser {
    type Error = ();

    fn apply_noise(&mut self, _node: &mut Node) -> Result<(), Self::Error> {
        Ok(())
    }
}
