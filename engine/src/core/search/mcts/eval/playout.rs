use itertools::Itertools;
use rand::prelude::*;

use crate::core::{
    color::colors,
    depth::Depth,
    move_iter::fold_legal_moves,
    search::mcts::node::node_state::{Branching, Valid},
};

use super::*;

/// Evaluator that performs random playouts (rollouts) to determine the value of
/// a position. Suitable for pure MCTS approaches.
#[derive(Debug)]
pub struct PlayoutEvaluator {
    rng: SmallRng,
}

impl PlayoutEvaluator {
    pub fn new(rng: SmallRng) -> Self {
        Self { rng }
    }

    /// Executes a random playout from the given position until a terminal state
    /// or depth limit.
    fn playout(&mut self, mut pos: Position) -> GameResult {
        let mut depth = Depth::ROOT;

        loop {
            let moves = {
                let mut moves = Vec::new();
                _ = fold_legal_moves(&pos, &mut moves, |acc, m| {
                    ControlFlow::Continue::<(), _>({
                        acc.push(m);
                        acc
                    })
                });
                moves
            };

            // 1. Check for Terminal State / Draw Rules
            if let Some(result) = pos.search_result_with(!moves.is_empty(), depth) {
                return result;
            }

            // 2. Check Depth Limit (treat as Draw)
            if depth >= Depth::MAX {
                return GameResult::Draw;
            }

            // 3. Make a random move
            let mov = moves[self.rng.random_range(0..moves.len())];
            pos.make_move(mov);

            depth += 1;
        }
    }
}

/// Trace data required to run a playout later in the batch phase.
#[derive(Clone)]
pub struct PlayoutTraceData {
    /// A snapshot of the position at the leaf node.
    start_pos: Position,
}

impl Evaluator for PlayoutEvaluator {
    type TraceData = Option<PlayoutTraceData>;

    /// Captures the position at the current node.
    fn trace<S: const Valid + HasBranches>(
        &self,
        node: NodeId<S>,
        _tree: &Tree,
        pos: &mut Position,
    ) -> Self::TraceData {
        node.try_into::<Branching>()
            .map(|_node| PlayoutTraceData { start_pos: pos.clone() })
    }

    /// Runs playouts for all collected leaves in the batch.
    fn eval_batch(
        &mut self,
        _tree: &Tree,
        _selection: &Selection<Self::TraceData>,
        leafs: &[&BatchItem<Self::TraceData>],
    ) -> impl Iterator<Item = Guess> {
        leafs
            .iter()
            .filter_map(|leaf| leaf.trace.as_ref())
            .map(|eval_info| {
                let result = self.playout(eval_info.start_pos.clone());
                // todo: make policy optional or smth
                let policy = Policy::new_even(0);
                match result {
                    GameResult::Draw => Guess {
                        policy,
                        relative_to: colors::WHITE,
                        quality: Quality::draw(),
                    },
                    GameResult::Win { relative_to } => Guess {
                        policy,
                        relative_to,
                        quality: Quality::win(),
                    },
                }
            })
            .collect_vec()
            .into_iter()
    }
}
