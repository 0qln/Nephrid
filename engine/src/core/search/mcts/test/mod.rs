use crate::{core::search::mcts::{
    eval::Policy, nn::RawLogits, node::{NodeId, Tree, node_state::HasBranches}, search::{BatchItem, Selection}
}, misc::List};
use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::core::{
    position::Position,
    search::mcts::{
        eval::{Evaluator, Guess, Quality},
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

    fn eval_batch(
        &mut self,
        tree: &Tree,
        _selection: &Selection<Self::TraceData>,
        leafs: &[&BatchItem<Self::TraceData>],
    ) -> impl Iterator<Item = Guess> {
        let mut evaluations = Vec::with_capacity(leafs.len());

        for leaf in leafs {
            let trace_data = &leaf.trace;

            let quality = self.rng.random_range(-1.0..=1.0);

            let mut policy_arr = [0.01; POLICY_OUTPUTS];
            let spike_index = self.rng.random_range(0..POLICY_OUTPUTS);
            policy_arr[spike_index] = 2.0;
            let raw_logits = RawLogits::new(policy_arr);

            let moves = tree.policy_indeces(leaf.node);

            evaluations.push(Guess {
                relative_to: trace_data.turn,
                quality: Quality::new(quality),
                policy: Policy::from_raw_logits(&raw_logits, moves, 1.0, &mut List::new()),
            });
        }

        evaluations.into_iter()
    }
}
