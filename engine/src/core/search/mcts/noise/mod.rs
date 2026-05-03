use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, num_traits::Zero};
use std::convert::Infallible;
use thiserror::Error;

use crate::{
    core::{
        r#move::MAX_LEGAL_MOVES,
        search::mcts::{
            eval::{Policy, Probability},
            node::{NodeId, Tree, node_state::Evaluated},
        },
    },
    misc::List,
};

#[cfg(test)]
pub mod test;

pub trait Noiser {
    type Error: std::error::Error;

    /// Apply noise to the polices of this node.
    fn apply_noise(&mut self, node: NodeId<Evaluated>, tree: &mut Tree) -> Result<(), Self::Error>;
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
        let alpha = self.alpha;
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

    fn apply_noise(&mut self, node: NodeId<Evaluated>, tree: &mut Tree) -> Result<(), Self::Error> {
        // Generate noise
        let noise = {
            let node = tree.node(node);
            let branches = node.branches();
            let gamma = self.gamma()?;
            let distr = gamma.sample_iter(&mut self.rng);
            let noise = distr
                .take(branches.len())
                .collect::<List<{ MAX_LEGAL_MOVES }, _>>();
            let total = noise.iter().sum::<f32>();

            if total.is_zero() {
                return Err(DirichletNoiseError::LowNoise);
            }

            let normalized_noise = noise
                .iter()
                .map(|noise_val| Probability::new(noise_val / total))
                .collect::<List<{ MAX_LEGAL_MOVES }, _>>();

            Policy::new(normalized_noise)
        };

        // Apply noise
        let eps = self.eps;
        tree.apply_policy_noise(node, &noise, eps);

        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct NullNoiser;

impl Noiser for NullNoiser {
    type Error = Infallible;

    fn apply_noise(
        &mut self,
        _node: NodeId<Evaluated>,
        _tree: &mut Tree,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}
