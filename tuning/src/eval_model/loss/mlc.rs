//! Multi-label classification

use burn::{
    nn::loss::BinaryCrossEntropyLossConfig,
    train::{MultiLabelClassificationOutput, TrainOutput},
};
use engine::core::search::mcts::{
    eval::GameResult,
    nn::{BoardInputTensor, StateInputTensor, board_history_input},
    node::{Value, node_state::Evaluated},
};

use burn::tensor::{Tensor, backend::Backend};

use burn::tensor::backend::AutodiffBackend;

use burn::{
    data::dataloader::batcher::Batcher,
    nn::loss::{MseLoss, Reduction},
    tensor::{Int, TensorData},
    train::{RegressionOutput, TrainStep},
};
use engine::core::{
    r#move::Move,
    search::mcts::{nn::Model, node::Tree},
};
use itertools::Itertools;

use crate::{
    Decision, LossOutput, PlayoutBatcher, Target,
    data::{BoardInput, StateInput},
    loss::ValueTargetTensor,
};

#[derive(Clone, Debug)]
pub struct MLCPlayoutBatch<B: Backend> {
    pub board_inputs: BoardInputTensor<B>,
    pub state_inputs: StateInputTensor<B>,
    pub value_targets: ValueTargetTensor<B>,
    pub policy_targets: PolicyTargetTensor<B>,
}

pub type ValueTarget = super::ValueTarget;

#[derive(Clone, Debug)]
pub struct PolicyTarget(Vec<usize>);

impl From<&MLCTarget> for PolicyTarget {
    fn from(target: &MLCTarget) -> Self {
        Self(target.moves.iter().cloned().map(usize::from).collect_vec())
    }
}

pub type PolicyTargetTensor<B> = Tensor<B, 2, Int>;

#[derive(Clone, Debug)]
pub struct MLCPlayoutItem {
    pub board_input: BoardInput,
    pub state_input: StateInput,
    pub value_target: ValueTarget,
    pub policy_target: PolicyTarget,
}

impl<'a> From<(GameResult, &'a [Decision<MLCTarget>])> for MLCPlayoutItem {
    fn from((result, decisions): (GameResult, &'a [Decision<MLCTarget>])) -> Self {
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

impl<B: AutodiffBackend> TrainStep<MLCPlayoutBatch<B>, LossOutput<B>> for Model<B> {
    fn step(&self, batch: MLCPlayoutBatch<B>) -> TrainOutput<LossOutput<B>> {
        let item = forward_with_loss_mlc(
            self,
            batch.board_inputs,
            batch.state_inputs,
            batch.value_targets,
            batch.policy_targets,
        );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

/// Foward with loss. (Policy loss: multi label classification)
pub fn forward_with_loss_mlc<B: Backend>(
    this: &Model<B>,
    board_input: BoardInputTensor<B>,
    state_input: StateInputTensor<B>,
    target_value: ValueTargetTensor<B>,
    target_policy: PolicyTargetTensor<B>,
) -> LossOutput<B> {
    let (value_output, policy_output) = this.forward(board_input, state_input);

    let value_loss =
        MseLoss::new().forward(value_output.clone(), target_value.clone(), Reduction::Auto);

    let policy_loss = BinaryCrossEntropyLossConfig::new()
        .init(&policy_output.device())
        .forward(policy_output.clone(), target_policy.clone());

    LossOutput::new(
        RegressionOutput::new(value_loss, value_output, target_value).loss,
        MultiLabelClassificationOutput::new(policy_loss, policy_output, target_policy).loss,
    )
}

impl<B: Backend> Batcher<B, MLCPlayoutItem, MLCPlayoutBatch<B>> for PlayoutBatcher {
    fn batch(&self, items: Vec<MLCPlayoutItem>, device: &B::Device) -> MLCPlayoutBatch<B> {
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
            .map(|x| TensorData::from(&x.policy_target.0[..]))
            .map(|x| Tensor::from_data(x, device))
            .collect();

        MLCPlayoutBatch {
            board_inputs: Tensor::cat(boards, 0),
            state_inputs: Tensor::cat(states, 0),
            value_targets: Tensor::cat(values, 0),
            policy_targets: Tensor::cat(policies, 0),
        }
    }
}

#[derive(Debug, Clone)]
struct MLCTarget {
    moves: Vec<Move>,
}

impl<'a> From<&'a Tree> for MLCTarget {
    fn from(tree: &'a Tree) -> Self {
        Self {
            moves: tree
                .best_moves(
                    tree.node_switch(tree.root())
                        .get::<Evaluated>()
                        .expect("fuck"),
                    Value(0.5),
                )
                .collect_vec(),
        }
    }
}

impl Target for MLCTarget {}

// impl<B: Backend> ValidStep<PlayoutBatch<B>, MLCLossOutput<B>> for Model<B> {
//     fn step(&self, batch: PlayoutBatch<B>) -> MLCLossOutput<B> {
//         forward_with_loss_mlc(
//             self,
//             batch.board_inputs,
//             batch.state_inputs,
//             batch.value_targets,
//             batch.policy_targets,
//         )
//     }
// }

// /// Multi-label classification output adapted for multiple metrics.
// pub struct ExactProbsClassificationOutput<B: Backend> {
//     /// The loss.
//     pub loss: Tensor<B, 1>,

//     /// The output.
//     pub output: Tensor<B, 2>,

//     /// The targets.
//     pub targets: Tensor<B, 2>,
// }

// impl<B: Backend> ExactProbsClassificationOutput<B> {
//     pub fn new
// }
// impl<B: Backend> ItemLazy for MultiLabelClassificationOutput<B> {
//     type ItemSync = MultiLabelClassificationOutput<NdArray>;

//     fn sync(self) -> Self::ItemSync {
//         let [output, loss, targets] = Transaction::default()
//             .register(self.output)
//             .register(self.loss)
//             .register(self.targets)
//             .execute()
//             .try_into()
//             .expect("Correct amount of tensor data");

//         let device = &Default::default();

//         MultiLabelClassificationOutput {
//             output: Tensor::from_data(output, device),
//             loss: Tensor::from_data(loss, device),
//             targets: Tensor::from_data(targets, device),
//         }
//     }
// }
