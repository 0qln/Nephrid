use burn::{
    Tensor,
    config::Config,
    prelude::Backend,
    train::metric::{Adaptor, LossInput},
};
use engine::{
    core::{
        search::mcts::{
            eval::{GameResult, Quality},
            nn::{CheckTensorHealthError, VALUE_OUTPUT_TENSOR_DIM},
        },
        turn::Turn,
    },
    misc::{CheckHealth, CheckHealthResult},
};
use thiserror::Error;

use crate::self_play::Outcome;

pub mod el;

#[derive(Config, Debug)]
pub struct LossConfig {
    pub value_loss_weight: f32,
    pub policy_loss_weight: f32,
}

/// label loss output
pub struct LossOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    // Value output
    pub value_loss: Tensor<B, 1>,
    // Quality Output
    pub policy_loss: Tensor<B, 1>,
}

impl<B: Backend> LossOutput<B> {
    pub fn new(value_loss: Tensor<B, 1>, policy_loss: Tensor<B, 1>) -> Self {
        Self {
            loss: value_loss.clone() + policy_loss.clone(),
            value_loss,
            policy_loss,
        }
    }

    pub fn new_weighted(
        value_loss: Tensor<B, 1>,
        policy_loss: Tensor<B, 1>,
        value_weight: f32,
        policy_weight: f32,
    ) -> Self {
        let weighted_loss = value_loss.clone() * value_weight + policy_loss.clone() * policy_weight;
        Self {
            loss: weighted_loss,
            // not applying the weights to the loss segments to keep logging clean.
            value_loss,
            policy_loss,
        }
    }
}

#[derive(Debug, Error)]
pub enum CheckLossOutputHealthError {
    #[error("Value loss tensor is unhealthy: {0}")]
    ValueLoss(CheckTensorHealthError),

    #[error("Policy loss tensor is unhealthy: {0}")]
    PolicyLoss(CheckTensorHealthError),

    #[error("Total loss tensor is unhealthy: {0}")]
    Loss(CheckTensorHealthError),
}

impl<B: Backend> CheckHealth for LossOutput<B> {
    type Error = CheckLossOutputHealthError;

    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        self.value_loss
            .check_health()
            .map_err(CheckLossOutputHealthError::ValueLoss)?;

        self.policy_loss
            .check_health()
            .map_err(CheckLossOutputHealthError::PolicyLoss)?;

        self.loss
            .check_health()
            .map_err(CheckLossOutputHealthError::Loss)?;

        Ok(())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for LossOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

#[derive(Clone, Default)]
pub struct PlayoutBatcher;

#[derive(Clone, Debug)]
pub struct ValueTarget(pub Quality);

impl From<(Outcome, Turn)> for ValueTarget {
    /// value target depending on the game result and the current moving player.
    fn from((result, moving_color): (Outcome, Turn)) -> Self {
        match result {
            Outcome::Discrete(GameResult::Draw) => Self(Quality::draw()),
            Outcome::Discrete(GameResult::Win { relative_to }) => {
                if relative_to == moving_color {
                    Self(Quality::win())
                }
                else {
                    Self(Quality::loss())
                }
            }
            Outcome::Continuous { relative_to, quality } => {
                if relative_to == moving_color {
                    Self(quality)
                }
                else {
                    Self(quality.inverse())
                }
            }
        }
    }
}

pub type ValueTargetTensor<B> = Tensor<B, VALUE_OUTPUT_TENSOR_DIM>;
