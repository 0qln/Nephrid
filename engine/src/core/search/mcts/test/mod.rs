use crate::{
    core::search::mcts::{
        eval::Policy,
        nn::RawLogits,
        node::{NodeId, Tree, node_state::HasBranches},
        search::{BatchItem, Selection},
    },
    misc::List,
};
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
                policy: Policy::from_raw_logits(
                    &raw_logits,
                    moves,
                    1.0,
                    &mut Box::new(List::new()),
                ),
            });
        }

        evaluations.into_iter()
    }
}

#[cfg(test)]
#[test]
#[ignore = "implementation in the search is missing and test is not complete yet"]
pub fn chooses_shortest_mate() {
    use crate::{
        core::search::{
            limit::UciLimit,
            mcts::{HceParts, MctsConfig, SearchState, strategy::MctsUci},
        },
        misc::{CancellationToken, DebugMode},
    };

    struct Config;

    impl MctsConfig for Config {
        type Parts = HceParts;
        type Strat = MctsUci;
    }

    // this position was reached during training with the following moves
    // 47. Qf6+ Kh6 48. Qg5+ Kg7 49. Qf6+ Kxf6 50. Kg1 Bh2+ 51. Kh1 Bg1 *
    // the nn failed to play the mate in 1, likely because it saw that the played
    // sequence of moves was proven and did not search further for a quicker
    // mate. consequently the selfplay loop cut the game of too early and even
    // tho it would've been a mate in 1, it was declared a draw.
    let mut pos = Position::from_fen("8/2P3kp/6p1/4p1Q1/P6P/6b1/1rq3B1/7K w - - 8 47").unwrap();
    let mut state = SearchState::default();
    let mut strat = MctsUci::new(
        UciLimit {
            iterations: 300,
            ..Default::default()
        },
        DebugMode::off(),
        CancellationToken::new(),
        None,
    );
    let parts = HceParts::default();

    super::mcts::<1, Config, _>(&mut pos, &parts, &mut state, &mut strat);

    todo!()
}
