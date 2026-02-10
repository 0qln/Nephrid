use itertools::Itertools;
use rand::prelude::*;
use std::{cell::RefCell, rc::Rc};

use crate::core::{
    move_iter::fold_legal_moves, piece::piece_type, search::mcts::search::SelectionNodeRef,
};

use super::*;

/// Evaluator that performs random playouts (rollouts) to determine the value of
/// a position. Suitable for pure MCTS approaches.
#[derive(Default, Debug)]
pub struct PlayoutEvaluator;

impl PlayoutEvaluator {
    pub fn new() -> Self {
        Self
    }

    /// Determines the result based on the current position state.
    pub fn check_terminal(pos: &Position, move_cnt: usize) -> Option<GameResult> {
        // Check for Draw by Repetition or 50-Move Rule
        if pos.has_threefold_repetition() || pos.fifty_move_rule() {
            return Some(GameResult::Draw);
        }

        // Check for Terminal State (Checkmate / Stalemate)
        if move_cnt == 0 {
            let us = pos.get_turn();
            // Use piece_type::KING (assuming it maps to King::ID from reference)
            let king = pos.get_bitboard(piece_type::KING, us);
            let nstm_attacks = pos.get_nstm_attacks();
            let in_check = !(king & nstm_attacks).is_empty();

            return Some(if in_check {
                // If in check and no moves, it's a loss for the current player
                GameResult::Win { relative_to: !us }
            }
            else {
                // Stalemate
                GameResult::Draw
            });
        }

        None
    }

    /// Executes a random playout from the given position until a terminal state
    /// or depth limit. Returns the score from White's perspective (1.0 win,
    /// -1.0 loss, 0.0 draw).
    fn playout(mut pos: Position) -> GameResult {
        let mut rng = rand::thread_rng();
        let mut depth = 0;
        const MAX_DEPTH: usize = 256; // Cutoff to prevent infinite loops

        loop {
            // Generate legal moves for the current position
            let moves = {
                let mut moves = Vec::new();
                fold_legal_moves(&pos, &mut moves, |acc, m| {
                    ControlFlow::Continue::<(), _>({
                        acc.push(m);
                        acc
                    })
                });
                moves
            };
            let move_cnt = moves.len();

            // 1. Check for Terminal State / Draw Rules
            if let Some(result) = Self::check_terminal(&pos, move_cnt) {
                return result;
            }

            // 2. Check Depth Limit (treat as Draw)
            if depth >= MAX_DEPTH {
                return GameResult::Draw;
            }

            // 3. Make a random move
            if let Some(mv) = moves.as_slice().choose(&mut rng) {
                pos.make_move(*mv);
            }
            else {
                break;
            }

            depth += 1;
        }

        GameResult::Draw
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

                let result = Self::playout(info.start_pos.clone());

                Evaluation::Terminal(result)
            })
            .collect_vec()
            .into_iter()
    }
}
