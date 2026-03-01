use itertools::Itertools;
use std::assert_matches::debug_assert_matches;

use crate::{
    core::{
        color::colors,
        coordinates::{Square, squares},
        piece::{PieceType, piece_type},
        position::PieceInfo,
        search::mcts::search::SelectionNodeRef,
    },
    impl_variants,
};

use super::*;

pub struct Psqt([i32; squares::N_VARIANTS]);

impl Psqt {
    pub fn get(&self, sq: Square) -> i32 {
        // SAFETY: sq is in correct range
        unsafe { *self.0.get_unchecked(sq.v() as usize) }
    }

    pub const fn empty() -> Psqt {
        Self([const { 0 }; squares::N_VARIANTS])
    }
}

const PIECE_VALUES: [i32; piece_type::N_VARIANTS] = [0, 100, 300, 300, 500, 800, 0];

const MG_PAWN_TABLE: Psqt = Psqt([
    0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20, -14,
    13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12,
    -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0,
]);

const MG_KNIGHT_TABLE: Psqt = Psqt([
    -167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84,
    129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10,
    19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23,
]);

const MG_BISHOP_TABLE: Psqt = Psqt([
    -29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50,
    37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18,
    10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21,
]);

const MG_ROOK_TABLE: Psqt = Psqt([
    32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16,
    -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0,
    -5, -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26,
]);

const MG_QUEEN_TABLE: Psqt = Psqt([
    -28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47,
    57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2,
    14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50,
]);

const MG_KING_TABLE: Psqt = Psqt([
    -65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6,
    22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14,
    -14, -22, -46, -44, -30, -15, -27, 1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24,
    14,
]);

const EG_PAWN_TABLE: Psqt = Psqt([
    0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82,
    84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8, 13,
    8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0,
]);

const EG_KNIGHT_TABLE: Psqt = Psqt([
    -58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9,
    -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1,
    15, 10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50,
    -64,
]);

const EG_BISHOP_TABLE: Psqt = Psqt([
    -14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0,
    4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15,
    -14, -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17,
]);

const EG_ROOK_TABLE: Psqt = Psqt([
    13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13,
    1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9,
    -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20,
]);

const EG_QUEEN_TABLE: Psqt = Psqt([
    -9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9,
    3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5,
    -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41,
]);

const EG_KING_TABLE: Psqt = Psqt([
    -74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45,
    44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23,
    16, 7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43,
]);

const PSQT_MG: [Psqt; piece_type::N_VARIANTS] = [
    Psqt::empty(),
    MG_PAWN_TABLE,
    MG_KNIGHT_TABLE,
    MG_BISHOP_TABLE,
    MG_ROOK_TABLE,
    MG_QUEEN_TABLE,
    MG_KING_TABLE,
];

const PSQT_EG: [Psqt; piece_type::N_VARIANTS] = [
    Psqt::empty(),
    EG_PAWN_TABLE,
    EG_KNIGHT_TABLE,
    EG_BISHOP_TABLE,
    EG_ROOK_TABLE,
    EG_QUEEN_TABLE,
    EG_KING_TABLE,
];

const PSQT: [[Psqt; piece_type::N_VARIANTS]; game_phases::N_VARIANTS] = [PSQT_MG, PSQT_EG];

fn psqt_score(phase: GamePhase, piece: PieceType, sq: Square) -> i32 {
    PSQT[phase.v() as usize][piece.v() as usize].get(sq)
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GamePhase {
    v: TGamePhase,
}

pub type TGamePhase = u8;

impl_variants! {
    TGamePhase as GamePhase in game_phases {
        MG, EG
    }
}

pub struct PiecePhase {
    v: TPiecePhase,
}

pub type TPiecePhase = u32;

impl_variants! {
    TPiecePhase as PiecePhase in piece_phases {
        TOTAL = (
            PAWN_C * 16 +
            KNIGHT_C * 4 +
            BISHOP_C * 4 +
            ROOK_C * 4 +
            QUEEN_C * 2
        ),
        NONE = 0,
        PAWN = 0,
        KNIGHT = 1,
        BISHOP = 1,
        ROOK = 2,
        QUEEN = 4,
    }
}

const PIECE_PHASES: [PiecePhase; piece_type::N_VARIANTS] = {
    use piece_phases::*;
    [NONE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, NONE]
};

/// Tapered Evaluation Phase value.
/// Where:
///  0 => early game
/// 24 => late game
#[derive(PartialEq, Debug, Default, Copy, Clone)]
pub struct TaperValue(u32);

impl TaperValue {
    pub fn from_position(pos: &Position) -> Self {
        let inv_phase = (piece_type::PAWN..piece_type::KING)
            .map(|p| pos.get_piece_bb(p).pop_cnt() * PIECE_PHASES[p.v() as usize].v())
            .sum::<u32>();

        Self(piece_phases::TOTAL_C - inv_phase)
    }

    pub fn weighted_eval(&self, mg_eval: i32, eg_eval: i32) -> i32 {
        let phase = self.0 as i32;
        let total = piece_phases::TOTAL_C as i32;
        ((mg_eval * (total - phase)) + (eg_eval * phase)) / total
    }
}

#[derive(Debug, PartialEq, Default)]
pub struct QualityInput {}

impl QualityInput {
    fn material(pos: &PieceInfo, color: Color) -> i32 {
        (piece_type::PAWN..piece_type::KING)
            .map(|p| pos.get_bitboard(p, color).pop_cnt() as i32 * PIECE_VALUES[p.v() as usize])
            .sum()
    }

    fn psqt(pos: &PieceInfo, color: Color, phase: TaperValue) -> i32 {
        fn score(pos: &PieceInfo, color: Color, phase: GamePhase) -> i32 {
            (piece_type::PAWN..=piece_type::KING)
                .map(|piece| {
                    pos.get_bitboard(piece, color)
                        .map(|sq| psqt_score(phase, piece, sq))
                        .sum::<i32>()
                })
                .sum()
        }

        let mg = score(pos, color, game_phases::MG);
        let eg = score(pos, color, game_phases::EG);
        phase.weighted_eval(mg, eg)
    }

    fn value(pos: &PieceInfo, color: Color, phase: TaperValue) -> i32 {
        Self::material(pos, color) + Self::psqt(pos, color, phase)
    }

    fn new() -> Self {
        Self {}
    }
}

/// Convert QualityInput into Quality, where the Quality is relative to white.
impl From<&EvalInfo> for Quality {
    fn from(input: &EvalInfo) -> Self {
        // get delta
        let w_q = QualityInput::value(&input.pos, colors::WHITE, input.phase);
        let b_q = QualityInput::value(&input.pos, colors::BLACK, input.phase);
        let d = (w_q - b_q) as f32;

        // normalize to [-1;1]
        let m = (w_q.abs() + b_q.abs()) as f32;
        Quality::new(if m == 0. { 0. } else { d / m })
    }
}

#[derive(Debug, PartialEq, Default)]
pub struct PolicyInput {}

impl PolicyInput {
    fn new() -> Self {
        Self {}
    }
}

impl From<&EvalInfo> for RawPolicy {
    fn from(input: &EvalInfo) -> Self {
        let node = &input.node.borrow();
        let pos = &input.pos;
        let phase = input.phase;

        debug_assert_matches!(node.state(), NodeState::Expanded | NodeState::Terminal);

        let mut policy = RawPolicy::null();
        for mov in node.iter_branches().map(|b| b.mov()) {
            // policy for each move in the position is the difference of the psqt score from
            // the previous position and the psqt score of the next position that is
            // achieved by the move.
            let from = mov.get_from();
            let to = mov.get_to();
            let piece = pos.get_piece(from).piece_type();
            let curr_score = {
                let mg = psqt_score(game_phases::MG, piece, from);
                let eg = psqt_score(game_phases::EG, piece, from);
                phase.weighted_eval(mg, eg)
            };
            let new_score = {
                let mg = psqt_score(game_phases::MG, piece, to);
                let eg = psqt_score(game_phases::EG, piece, to);
                phase.weighted_eval(mg, eg)
            };
            let score = (curr_score - new_score) as f32;
            policy.set(usize::from(mov), score);
        }

        policy
    }
}

#[derive(PartialEq, Debug)]
pub struct EvalInfo {
    /// The node that this eval info is for.
    node: Rc<RefCell<Node>>,

    /// Quality info for static evaluation
    q_input: QualityInput,

    /// Policy for static evaluation
    p_input: PolicyInput,

    /// Turn of the current player.
    turn: Turn,

    /// The current game phase taper value.
    phase: TaperValue,

    /// Piece infos of the position.
    pos: PieceInfo,
}

impl EvalInfo {
    pub fn new(node: Rc<RefCell<Node>>, pos: &Position) -> Self {
        Self {
            pos: pos.piece_info().clone(),
            phase: TaperValue::from_position(&pos),
            node: node.clone(),
            turn: pos.get_turn(),
            q_input: QualityInput::new(),
            p_input: PolicyInput::new(),
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

impl Evaluator for StaticEvaluator {
    type TraceData = EvalInfo;

    fn trace(&self, node: Rc<RefCell<Node>>, pos: &Position) -> Self::TraceData {
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

                Evaluation::Guess(Box::new(Guess {
                    relative_to: colors::WHITE,
                    quality: eval_info.into(),
                    policy: eval_info.into(),
                }))
            })
            .collect_vec()
            .into_iter()
    }
}
