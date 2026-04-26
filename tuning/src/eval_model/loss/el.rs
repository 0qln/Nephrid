//! Exact loss. (Soft cross-entropy loss between the policy target and the
//! predicted policy, and MSE between the value target and the predicted value)

use burn::{
    data::dataloader::batcher::Batcher,
    nn::loss::{MseLoss, Reduction},
    tensor::{Float, TensorData, activation::log_softmax, backend::AutodiffBackend},
    train::{TrainOutput, TrainStep, ValidStep},
};
use engine::{
    core::search::mcts::{
        eval::{RawPolicy, VisitCounts, normalize_visits},
        nn::{
            BoardInputTensor, CheckTensorHealthError, Model, POLICY_OUTPUT_TENSOR_DIM,
            POLICY_OUTPUTS, STATE_INPUT_TENSOR_DIM, StateInputTensor, VALUE_OUTPUT_TENSOR_DIM,
            board_history_input,
        },
        node::node_state::Evaluated,
    },
    misc::{CheckHealth, CheckHealthResult},
};

use burn::tensor::{Tensor, backend::Backend};

use engine::core::search::mcts::node::Tree;
use itertools::Itertools;
use thiserror::Error;

use crate::{
    Decision, LossOutput, PlayoutBatcher, Target,
    data::{BoardInput, StateInput},
    loss::ValueTargetTensor,
    self_play::{Outcome, PlayoutItem},
};

#[derive(Clone, Debug)]
pub struct ExactLossPlayoutBatch<B: Backend> {
    pub board_inputs: BoardInputTensor<B>,
    pub state_inputs: StateInputTensor<B>,
    pub value_targets: ValueTargetTensor<B>,
    pub policy_targets: PolicyTargetTensor<B>,
}

#[derive(Error, Debug)]
pub enum CheckExactLossPlayoutBatchHealthError {
    #[error("Board input health error: {0}")]
    BoardInputHealth(CheckTensorHealthError),
    #[error("State input health error: {0}")]
    StateInputHealth(CheckTensorHealthError),
    #[error("Value target health error: {0}")]
    ValueTargetHealth(CheckTensorHealthError),
    #[error("Policy target health error: {0}")]
    PolicyTargetHealth(CheckTensorHealthError),
}

impl<B: Backend> CheckHealth for ExactLossPlayoutBatch<B> {
    type Error = CheckExactLossPlayoutBatchHealthError;

    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        self.board_inputs
            .check_health()
            .map_err(CheckExactLossPlayoutBatchHealthError::BoardInputHealth)?;
        self.state_inputs
            .check_health()
            .map_err(CheckExactLossPlayoutBatchHealthError::StateInputHealth)?;
        self.value_targets
            .check_health()
            .map_err(CheckExactLossPlayoutBatchHealthError::ValueTargetHealth)?;
        self.policy_targets
            .check_health()
            .map_err(CheckExactLossPlayoutBatchHealthError::PolicyTargetHealth)?;
        Ok(())
    }
}

pub type ValueTarget = super::ValueTarget;

#[derive(Clone, Debug)]
pub struct PolicyTarget(pub RawPolicy);

impl From<&ExactLossTarget> for PolicyTarget {
    fn from(target: &ExactLossTarget) -> Self {
        let mut visit_counts = [0.; POLICY_OUTPUTS];
        for &(mov, visits) in &target.visit_counts.0 {
            visit_counts[usize::from(mov)] = visits as f32;
        }

        let raw_policy = normalize_visits(&visit_counts, 1.);

        Self(RawPolicy::new(raw_policy))
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

impl<'a> From<(Outcome, &'a [Decision<ExactLossTarget>])> for ExactLossPlayoutItem {
    fn from((result, decisions): (Outcome, &'a [Decision<ExactLossTarget>])) -> Self {
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
    /// vec<(move, visit_count)> on all legal moves in the position
    visit_counts: VisitCounts,
}

impl From<&Tree> for ExactLossTarget {
    fn from(tree: &Tree) -> Self {
        Self {
            visit_counts: {
                let root_id = tree.root();
                let root = tree.node_switch(root_id).get::<Evaluated>();
                let root_node = tree.node(root_id);
                let root_value = root_node.value();
                let root_evaluated = root.expect("Root should be evaluated");

                let branches = tree.branches(root_evaluated);

                // override for proven wins (proven loss for parent node is a proven win for our
                // current root)
                if root_value.is_proven_loss() {
                    VisitCounts(
                        branches
                            .iter()
                            .map(|branch| {
                                let child_node = tree.node(branch.node());
                                let is_winning_move = child_node.value().is_proven_win();
                                (branch.mov(), if is_winning_move { 1 } else { 0 })
                            })
                            .collect_vec(),
                    )
                }
                // otherwise use visit counts
                else {
                    VisitCounts(
                        branches
                            .iter()
                            .map(|branch| (branch.mov(), tree.node(branch.node()).visits()))
                            .collect_vec(),
                    )
                }
            },
        }
    }
}

impl Target for ExactLossTarget {
    fn from_visit_counts(visit_counts: VisitCounts) -> Self {
        Self { visit_counts }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SoftCrossEntropyLoss;

impl SoftCrossEntropyLoss {
    /// Creates a new SoftCrossEntropyLoss.
    pub fn new() -> Self {
        Self
    }

    /// Computes the soft cross entropy loss between raw logits and target
    /// probabilities.
    ///
    /// # Arguments
    ///
    /// * `logits` - The raw unnormalized predictions from the model. Expected
    ///   shape: `[batch_size, num_classes]`
    /// * `targets` - The target probability distributions from MCTS. Expected
    ///   shape: `[batch_size, num_classes]`
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the batch-averaged loss.
    pub fn forward<B: Backend, const D: usize>(
        &self,
        logits: Tensor<B, D, Float>,
        targets: Tensor<B, D, Float>,
    ) -> Tensor<B, 1, Float> {
        // 1. Convert raw logits to log-probabilities safely using the LogSumExp trick.
        // We apply it over the last dimension (the class/move dimension).
        let class_dim = D - 1;
        let log_preds = log_softmax(logits, class_dim);

        // 2. Calculate the cross-entropy: -sum(target * log(pred))
        // This computes the loss for each individual board in the batch.
        let batch_losses = -(targets * log_preds).sum_dim(class_dim);

        // 3. Average the loss across the entire batch to get a single scalar
        batch_losses.mean()
    }
}

/// Foward with loss. (Policy loss: exact probabilities)
pub fn forward_with_loss<B: Backend>(
    this: &Model<B>,
    board_input: BoardInputTensor<B>,
    state_input: Tensor<B, STATE_INPUT_TENSOR_DIM>,
    target_value: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    target_policy: Tensor<B, 2, Float>,
) -> LossOutput<B> {
    let (value_output, policy_output) = this.forward(board_input, state_input);
    let value_loss = MseLoss::new().forward(value_output, target_value, Reduction::Auto);
    let policy_loss = SoftCrossEntropyLoss::new().forward(policy_output, target_policy);
    LossOutput::new(value_loss, policy_loss)
}

pub fn forward_with_loss_weighted<B: Backend>(
    model: &Model<B>,
    board_input: BoardInputTensor<B>,
    state_input: Tensor<B, STATE_INPUT_TENSOR_DIM>,
    target_value: Tensor<B, VALUE_OUTPUT_TENSOR_DIM>,
    target_policy: Tensor<B, 2, Float>,
    value_weight: f32,
    policy_weight: f32,
) -> LossOutput<B> {
    let (value_output, policy_output) = model.forward(board_input, state_input);
    let value_loss = MseLoss::new().forward(value_output, target_value, Reduction::Auto);
    let policy_loss = SoftCrossEntropyLoss::new().forward(policy_output, target_policy);
    LossOutput::new_weighted(value_loss, policy_loss, value_weight, policy_weight)
}

impl<B: Backend> ValidStep<ExactLossPlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: ExactLossPlayoutBatch<B>) -> LossOutput<B> {
        forward_with_loss(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        )
    }
}

impl<B: AutodiffBackend> TrainStep<ExactLossPlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: ExactLossPlayoutBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

pub struct WeightedModel<B: Backend> {
    pub model: Model<B>,
    pub value_weight: f32,
    pub policy_weight: f32,
}

impl<B: Backend> WeightedModel<B> {
    pub fn new(model: Model<B>, value_weight: f32, policy_weight: f32) -> Self {
        Self {
            model,
            value_weight,
            policy_weight,
        }
    }
}

impl<B: Backend> ValidStep<ExactLossPlayoutBatch<B>, LossOutput<B>> for WeightedModel<B> {
    fn step(&self, batch: ExactLossPlayoutBatch<B>) -> LossOutput<B> {
        forward_with_loss_weighted(
            &self.model,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
            self.value_weight,
            self.policy_weight,
        )
    }
}

impl<B: AutodiffBackend> TrainStep<ExactLossPlayoutBatch<B>, LossOutput<B>> for WeightedModel<B> {
    fn step(&self, batch: ExactLossPlayoutBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss_weighted(
            &self.model,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
            self.value_weight,
            self.policy_weight,
        );
        TrainOutput::new(&self.model, item.loss.backward(), item)
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
            .map(|x| TensorData::from([[x.value_target.0.v()]]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        let policies = items
            .into_iter()
            .map(|x| TensorData::from([x.policy_target.0.into_inner()]))
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
