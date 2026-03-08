use itertools::Itertools;
use rand::prelude::*;

use crate::core::{move_iter::fold_legal_moves, search::mcts::node::node_state::Branching};

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
        let mut depth = 0;
        const MAX_DEPTH: usize = 256;

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
            if let Some(result) = pos.game_result_with(!moves.is_empty()) {
                return result;
            }

            // 2. Check Depth Limit (treat as Draw)
            if depth >= MAX_DEPTH {
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
    fn trace<S: HasBranches>(&self, node: CtNodeRef<S>, pos: &Position) -> Self::TraceData {
        node.try_into::<Branching>()
            .map(|_node| PlayoutTraceData { start_pos: pos.clone() })
    }

    /// Runs playouts for all collected leaves in the batch.
    fn eval_batch<const X: usize>(
        &mut self,
        _selection: &Selection<X, Self::TraceData>,
        leafs: &[&SelectionLeaf<Self::TraceData>],
    ) -> impl Iterator<Item = Evaluation> {
        leafs
            .iter()
            .filter_map(|&leaf| {
                leaf.leaf_data
                    .as_ref()
                    .map(|l| (&l.trace_data).as_ref())
                    .flatten()
            })
            .map(|eval_info| {
                let result = self.playout(eval_info.start_pos.clone());
                Evaluation::Terminal(result)
            })
            .collect_vec()
            .into_iter()
    }
}
