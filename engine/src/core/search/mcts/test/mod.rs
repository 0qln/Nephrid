use crate::core::search::mcts::{
    eval::{Policy, RawPolicy},
    node::{NodeId, Tree, node_state::HasBranches},
    search::{BatchItem, Selection},
};
use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::core::{
    position::Position,
    search::mcts::{
        eval::{Evaluation, Evaluator, Guess, Quality},
        nn::POLICY_OUTPUTS,
    },
    turn::Turn,
};

#[derive(Clone)]
pub struct DummyTraceData {
    turn: Turn,
}

/// A dummy evaluator that returns random values.
/// Useful for testing the search tree mechanics without a trained network.
#[derive(Debug, Clone)]
pub struct DummyEvaluator {
    rng: SmallRng,
}

impl DummyEvaluator {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for DummyEvaluator {
    fn default() -> Self {
        // Use a fixed seed for deterministic testing behavior.
        let seed = 0xdead_beef;
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl Evaluator for DummyEvaluator {
    type TraceData = DummyTraceData;

    fn trace<S: HasBranches>(
        &self,
        _node: NodeId<S>,
        _tree: &Tree,
        pos: &mut Position,
    ) -> Self::TraceData {
        DummyTraceData { turn: pos.get_turn() }
    }

    fn eval_batch<const X: usize>(
        &mut self,
        tree: &Tree,
        _selection: &Selection<X, Self::TraceData>,
        leafs: &[&BatchItem<Self::TraceData>],
    ) -> impl Iterator<Item = Evaluation> {
        let mut evaluations = Vec::with_capacity(leafs.len());

        for leaf in leafs {
            let trace_data = &leaf.data;

            let quality = self.rng.random_range(-1.0..=1.0);

            let mut policy_arr = [0.01; POLICY_OUTPUTS];
            let spike_index = self.rng.random_range(0..POLICY_OUTPUTS);
            policy_arr[spike_index] = 1.0;

            let raw_policy = RawPolicy::new(policy_arr);

            let moves = tree.move_indices(leaf.node);

            evaluations.push(Evaluation::Guess(Box::new(Guess {
                relative_to: trace_data.turn,
                quality: Quality::new(quality),
                policy: Policy::from_raw(&raw_policy, moves).expect("a policy"),
            })));
        }

        evaluations.into_iter()
    }
}
