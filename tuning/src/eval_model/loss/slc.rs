//! Single-label classification

use burn::train::TrainOutput;
use engine::core::search::mcts::{eval::GameResult, nn::board_history_input};

use burn::tensor::{Tensor, backend::Backend};

use burn::tensor::backend::AutodiffBackend;

use burn::{
    data::dataloader::batcher::Batcher,
    nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction},
    tensor::{Int, TensorData},
    train::{ClassificationOutput, RegressionOutput, TrainStep},
};
use engine::core::{
    r#move::Move,
    search::mcts::{nn::Model, node::Tree},
};

use engine::core::search::mcts::nn::*;

use crate::{
    Decision, LossOutput, PlayoutBatcher, Target,
    data::{BoardInput, StateInput},
    loss::ValueTargetTensor,
};

#[derive(Clone, Debug)]
pub struct SLCPlayoutBatch<B: Backend> {
    pub board_inputs: BoardInputTensor<B>,
    pub state_inputs: StateInputTensor<B>,
    pub value_targets: ValueTargetTensor<B>,
    pub policy_targets: PolicyTargetTensor<B>,
}

pub type ValueTarget = super::ValueTarget;

#[derive(Clone, Debug)]
pub struct PolicyTarget(usize);

impl From<&SLCTarget> for PolicyTarget {
    fn from(target: &SLCTarget) -> Self {
        Self(usize::from(target.mov))
    }
}

pub type PolicyTargetTensor<B> = Tensor<B, 1, Int>;

#[derive(Clone, Debug)]
pub struct SLCPlayoutItem {
    pub board_input: BoardInput,
    pub state_input: StateInput,
    pub value_target: ValueTarget,
    pub policy_target: PolicyTarget,
}

impl<'a> From<(GameResult, &'a [Decision<SLCTarget>])> for SLCPlayoutItem {
    fn from((result, decisions): (GameResult, &'a [Decision<SLCTarget>])) -> Self {
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

impl<B: AutodiffBackend> TrainStep<SLCPlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: SLCPlayoutBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss_slc(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> Batcher<B, SLCPlayoutItem, SLCPlayoutBatch<B>> for PlayoutBatcher {
    fn batch(&self, items: Vec<SLCPlayoutItem>, device: &B::Device) -> SLCPlayoutBatch<B> {
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
            .iter()
            .map(|x| TensorData::from([x.policy_target.0]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        SLCPlayoutBatch {
            board_inputs: Tensor::cat(boards, 0),
            state_inputs: Tensor::cat(states, 0),
            value_targets: Tensor::cat(values, 0),
            policy_targets: Tensor::cat(policies, 0),
        }
    }
}

// Info to help find the training target.
// 0: Most visited move / best_move.
#[derive(Debug, Clone)]
struct SLCTarget {
    mov: Move,
}

impl<'a> From<&'a Tree> for SLCTarget {
    fn from(tree: &'a Tree) -> Self {
        Self {
            mov: tree
                .maybe_best_move(tree.root())
                .expect("Tree should have a bestmove"),
        }
    }
}

impl Target for SLCTarget {}

/// Foward with loss. (Policy loss: single label classification)
pub fn forward_with_loss_slc<B: Backend>(
    this: &Model<B>,
    board_input: BoardInputTensor<B>,
    state_input: StateInputTensor<B>,
    target_value: ValueTargetTensor<B>,
    target_policy: PolicyTargetTensor<B>,
) -> LossOutput<B> {
    let (value_output, policy_output) = this.forward(board_input, state_input);

    let value_loss =
        MseLoss::new().forward(value_output.clone(), target_value.clone(), Reduction::Auto);

    let policy_loss = CrossEntropyLossConfig::new()
        .init(&policy_output.device())
        .forward(policy_output.clone(), target_policy.clone());

    LossOutput::new(
        RegressionOutput::new(value_loss, value_output, target_value).loss,
        ClassificationOutput::new(policy_loss, policy_output, target_policy).loss,
    )
}
