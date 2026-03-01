use itertools::Itertools;
use std::cmp::max;

use crate::core::{color::colors, piece::piece_type, search::mcts::search::SelectionNodeRef};

use super::*;

#[derive(Debug, PartialEq, Default)]
pub struct QualityInput {
    w_q: u32,
    b_q: u32,
}

impl QualityInput {
    fn material(pos: &Position, color: Color) -> u32 {
        const PIECE_VALUES: [u32; piece_type::N_VARIANTS] = [0, 1, 3, 3, 5, 8, 0];
        (piece_type::PAWN..piece_type::KING)
            .map(|p| pos.get_bitboard(p, color).pop_cnt() * PIECE_VALUES[p.v() as usize])
            .sum()
    }

    fn psqt(_pos: &Position, _color: Color) -> u32 {
        // todo
        0
    }

    fn value(pos: &Position, color: Color) -> u32 {
        Self::material(pos, color) + Self::psqt(pos, color)
    }

    fn new(pos: &Position) -> Self {
        Self {
            w_q: Self::value(pos, colors::WHITE),
            b_q: Self::value(pos, colors::BLACK),
        }
    }
}

#[derive(Debug, PartialEq, Default)]
pub struct PolicyInput {
    p: RawPolicy,
}

#[derive(PartialEq, Debug)]
pub struct EvalInfo<'node> {
    /// The node that this eval info is for.
    node: NodeRef<'node>,

    /// Quality info for static evaluation
    q_input: QualityInput,

    /// Policy for static evaluation
    p_input: PolicyInput,

    /// Turn of the current player.
    turn: Turn,
}

impl<'node> EvalInfo<'node> {
    pub fn new(node: NodeRef<'node>, pos: &Position) -> Self {
        Self {
            node,
            turn: pos.get_turn(),
            q_input: QualityInput::new(pos),
            p_input: PolicyInput::default(),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct StaticEvaluator;

impl StaticEvaluator {
    pub fn new() -> Self {
        Self
    }
}

impl<'trace> Evaluator<'trace> for StaticEvaluator {
    type TraceData = EvalInfo<'trace>;

    fn trace(&self, node: NodeRef<'trace>, pos: &Position) -> Self::TraceData {
        EvalInfo::new(node, pos)
    }

    fn eval_batch(
        &mut self,
        leafs: &[SelectionNodeRef<Self::TraceData>],
    ) -> impl Iterator<Item = Evaluation> {
        leafs
            .iter()
            .map(|leaf| {
                let leaf_borrow = leaf.borrow();
                let data = leaf_borrow.data();
                let eval_info = &data.trace_data;

                // Squish into a range from -1 to +1
                let w_q = eval_info.q_input.w_q;
                let b_q = eval_info.q_input.b_q;
                let d = w_q as i32 - b_q as i32;
                let m = max(w_q, b_q);
                // Prevent division by zero if board is empty or pieces have 0 value
                let q = if m == 0 { 0. } else { d as f32 / m as f32 };

                let quality = if eval_info.turn == colors::WHITE { q } else { -q };

                Evaluation::Guess(Box::new(Guess {
                    relative_to: eval_info.turn,
                    quality,
                    policy: eval_info.p_input.p.to_owned(),
                }))
            })
            .collect_vec()
            .into_iter()
    }
}
