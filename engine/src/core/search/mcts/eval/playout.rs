use itertools::Itertools;
use rand::prelude::*;
use std::{cell::RefCell, rc::Rc};

use crate::core::{move_iter::fold_legal_moves, search::mcts::search::SelectionNodeRef};

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
            if let Some(result) = pos.game_result_with(moves.len() > 0) {
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
    type TraceData = PlayoutTraceData;

    /// Captures the position at the current node.
    fn trace(&self, _node: Rc<RefCell<Node>>, pos: &Position) -> Self::TraceData {
        PlayoutTraceData { start_pos: pos.clone() }
    }

    /// Runs playouts for all collected leaves in the batch.
    fn eval_batch(
        &mut self,
        leafs: &[SelectionNodeRef<Self::TraceData>],
    ) -> impl Iterator<Item = Evaluation> {
        leafs
            .iter()
            .map(|leaf| {
                let leaf_borrow = leaf.borrow();
                let data = leaf_borrow.data();
                let info = &data.trace_data;

                let result = self.playout(info.start_pos.clone());

                Evaluation::Terminal(result)
            })
            .collect_vec()
            .into_iter()
    }
}
