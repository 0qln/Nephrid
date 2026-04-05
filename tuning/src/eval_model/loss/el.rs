//! Exact loss. (KL-divergence between the policy target and the predicted
//! policy, and MSE between the value target and the predicted value)

use burn::{
    data::dataloader::batcher::Batcher,
    module::Module,
    nn::loss::{MseLoss, Reduction},
    tensor::{Float, TensorData, backend::AutodiffBackend},
    train::{RegressionOutput, TrainOutput, TrainStep},
};
use engine::core::search::mcts::{
    eval::{GameResult, RawPolicy, softmax},
    nn::{
        BoardInputTensor, Model, POLICY_OUTPUT_TENSOR_DIM, POLICY_OUTPUTS, STATE_INPUT_TENSOR_DIM,
        StateInputTensor, VALUE_OUTPUT_TENSOR_DIM, board_history_input,
    },
    node::node_state::Evaluated,
};

use burn::tensor::{Tensor, backend::Backend};

use engine::core::search::mcts::node::Tree;
use itertools::Itertools;

use crate::{
    Decision, LossOutput, PlayoutBatcher, Target,
    data::{BoardInput, StateInput},
    loss::ValueTargetTensor,
    self_play::PlayoutItem,
};

#[derive(Clone, Debug)]
pub struct ExactLossPlayoutBatch<B: Backend> {
    pub board_inputs: BoardInputTensor<B>,
    pub state_inputs: StateInputTensor<B>,
    pub value_targets: ValueTargetTensor<B>,
    pub policy_targets: PolicyTargetTensor<B>,
}

pub type ValueTarget = super::ValueTarget;

#[derive(Clone, Debug)]
pub struct PolicyTarget(Vec<f32>);

impl From<&ExactLossTarget> for PolicyTarget {
    fn from(target: &ExactLossTarget) -> Self {
        Self(target.raw_policy.iter().collect_vec())
    }
}

pub type PolicyTargetTensor<B> = Tensor<B, POLICY_OUTPUT_TENSOR_DIM, Float>;

#[derive(Clone, Debug)]
pub struct ExactLossPlayoutItem {
    pub board_input: BoardInput,
    pub state_input: StateInput,
    pub value_target: ValueTarget,
    pub policy_target: PolicyTarget,
}

impl<'a> From<(GameResult, &'a [Decision<ExactLossTarget>])> for ExactLossPlayoutItem {
    fn from((result, decisions): (GameResult, &'a [Decision<ExactLossTarget>])) -> Self {
        let decision = &decisions[decisions.len() - 1];
        let player = decision.state.moving_color;
        Self {
            board_input: BoardInput::from(decisions),
            state_input: StateInput::from(decision),
            value_target: ValueTarget::from((result, player)),
            policy_target: PolicyTarget::from(&decision.target),
        }
    }
}

impl PlayoutItem for ExactLossPlayoutItem {
    type Target = ExactLossTarget;
}

#[derive(Debug, Clone)]
pub struct ExactLossTarget {
    raw_policy: RawPolicy,
}

impl From<&Tree> for ExactLossTarget {
    fn from(tree: &Tree) -> Self {
        Self {
            raw_policy: {
                let root = tree.node_switch(tree.root()).get::<Evaluated>();
                let root = root.expect("Root should be evaluated");

                let branches = tree.branches(root);

                let mut raw_policy = RawPolicy::null();
                for branch in branches.iter() {
                    raw_policy.set(
                        usize::from(branch.mov()),
                        tree.node(branch.node()).visits() as f32,
                    );
                }
                softmax(raw_policy.inner_mut(), 10.);
                raw_policy
            },
        }
    }
}

impl Target for ExactLossTarget {}

/// Calculate the KL divergence loss from predictions and probabilistic targets.
#[derive(Module, Debug, Clone)]
pub struct KLDivergenceLoss {}

impl Default for KLDivergenceLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl KLDivergenceLoss {
    pub fn new() -> Self {
        Self {}
    }

    /// Compute the KL divergence loss.
    ///
    /// `predictions` is expected to be a probability distribution.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size]` or `[batch_size, num_classes]`
    /// - targets: `[batch_size]` or `[batch_size, num_classes]` (probabilities)
    pub fn forward<B: Backend, const D: usize>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        // For numerical stability, clamp values to avoid log(0)
        let epsilon = 1e-8;
        let predictions_clamped = predictions.clone().clamp(epsilon, 1.0);
        let targets_clamped = targets.clone().clamp(epsilon, 1.0);

        // KL(P||Q) = sum[ P * log(P) - P * log(Q) ]
        // where P = targets, Q = predictions
        let kl_div = targets_clamped.clone() * targets_clamped.clone().log()
            - targets_clamped * predictions_clamped.log();

        // Sum over class dimension if multi-dimensional
        let loss_per_sample = if D > 1 { kl_div.sum_dim(D - 1) } else { kl_div };

        // Average over batch
        loss_per_sample.mean()
    }
}

/// Foward with loss. (Policy loss: exact probabilities)
pub fn forward_with_loss_exact_loss<B: Backend>(
    this: &Model<B>,
    board_input: BoardInputTensor<B>,
    state_input: Tensor<B, STATE_INPUT_TENSOR_DIM>,
    target_value: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    target_policy: Tensor<B, 2, Float>,
) -> LossOutput<B> {
    let (value_output, policy_output) = this.forward(board_input, state_input);

    let value_loss =
        MseLoss::new().forward(value_output.clone(), target_value.clone(), Reduction::Auto);

    let policy_loss = KLDivergenceLoss::new().forward(policy_output.clone(), target_policy.clone());

    LossOutput::new(
        RegressionOutput::new(value_loss, value_output, target_value).loss,
        // ExactProbsClassificationOutput::new(policy_loss, policy_output, target_policy).loss,
        policy_loss,
    )
}

impl<B: AutodiffBackend> TrainStep<ExactLossPlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: ExactLossPlayoutBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss_exact_loss(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> Batcher<B, ExactLossPlayoutItem, ExactLossPlayoutBatch<B>> for PlayoutBatcher {
    fn batch(
        &self,
        items: Vec<ExactLossPlayoutItem>,
        device: &B::Device,
    ) -> ExactLossPlayoutBatch<B> {
        let boards = items
            .iter()
            .map(|x| board_history_input(&x.board_input.0, device))
            .collect();

        let states = items
            .iter()
            .map(|x| TensorData::from([x.state_input.0]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let values = items
            .iter()
            .map(|x| TensorData::from([[x.value_target.0]]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let policies = items
            .into_iter()
            .map(|x| {
                TensorData::from([
                    TryInto::<[f32; POLICY_OUTPUTS]>::try_into(x.policy_target.0).expect("")
                ])
            })
            .map(|x| Tensor::from_data(x, device))
            .collect();

        ExactLossPlayoutBatch {
            board_inputs: Tensor::cat(boards, 0),
            state_inputs: Tensor::cat(states, 0),
            value_targets: Tensor::cat(values, 0),
            policy_targets: Tensor::cat(policies, 0),
        }
    }
}
