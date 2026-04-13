use burn::{
    Tensor,
    prelude::Backend,
    train::metric::{Adaptor, LossInput},
};
use engine::core::{
    search::mcts::{
        eval::GameResult,
        nn::{VALUE_DRAW, VALUE_LOSE, VALUE_OUTPUT_TENSOR_DIM, VALUE_WIN},
    },
    turn::Turn,
};

pub mod el;

// not up to date. e.g. they don't handle setting the policy to a 1-hot when
// tree root has a mate-in-1 and such.
#[deprecated]
pub mod mlc;
#[deprecated]
pub mod slc;

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
}

impl<B: Backend> Adaptor<LossInput<B>> for LossOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

#[derive(Clone, Default)]
pub struct PlayoutBatcher;

#[derive(Clone, Debug)]
pub struct ValueTarget(pub f32);

impl From<(GameResult, Turn)> for ValueTarget {
    /// value target depending on the game result and the current moving player.
    fn from((result, moving_color): (GameResult, Turn)) -> Self {
        match result {
            GameResult::Draw => Self(VALUE_DRAW),
            GameResult::Win { relative_to } => {
                if relative_to == moving_color {
                    Self(VALUE_WIN)
                }
                else {
                    Self(VALUE_LOSE)
                }
            }
        }
    }
}

pub type ValueTargetTensor<B> = Tensor<B, VALUE_OUTPUT_TENSOR_DIM>;
