use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::core::{
    position::Position,
    search::mcts::{
        eval::{Evaluation, Evaluator, Guess, RawPolicy},
        nn::POLICY_OUTPUTS,
        node::NodeRef,
        search::SelectionNodeRef,
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

impl<'a> Evaluator<'a> for DummyEvaluator {
    type TraceData = DummyTraceData;

    fn trace(&self, _node: NodeRef, pos: &Position) -> Self::TraceData {
        DummyTraceData { turn: pos.get_turn() }
    }

    fn eval_batch(
        &mut self,
        leafs: &[SelectionNodeRef<Self::TraceData>],
    ) -> impl Iterator<Item = Evaluation> {
        let mut evaluations = Vec::with_capacity(leafs.len());

        for leaf in leafs {
            let leaf = leaf.borrow();
            let trace_data = &leaf.data().trace_data;

            let quality = self.rng.random_range(-1.0..=1.0);

            let mut policy_arr = [0.01; POLICY_OUTPUTS];
            let spike_index = self.rng.random_range(0..POLICY_OUTPUTS);
            policy_arr[spike_index] = 1.0;

            let raw_policy = RawPolicy::new(policy_arr);

            evaluations.push(Evaluation::Guess(Box::new(Guess {
                relative_to: trace_data.turn,
                quality,
                policy: raw_policy,
            })));
        }

        evaluations.into_iter()
    }
}
