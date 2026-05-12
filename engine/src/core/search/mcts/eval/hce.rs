use const_for::const_for;

use std::{
    cmp::{Reverse, min},
    marker::PhantomData,
    ops,
};

use crate::core::{
    bitboard::Bitboard,
    color::{Perspective, perspectives},
    coordinates::{EpTargetSquare, File, Rank, files, pawn_utils::single_step, ranks},
    move_iter::{
        bishop::Bishop, fold_legal_captures, fold_legal_moves, king, knight, pawn, queen::Queen,
        rook::Rook, sliding_piece::SlidingAttacks,
    },
    piece::PromoPieceType,
    position::CheckState,
    turn::Turn,
};

use crate::{
    core::{
        color::colors,
        coordinates::{Square, squares},
        r#move::Move,
        piece::{PieceType, piece_type},
        position::{PieceInfo, StateInfo},
        search::mcts::node::node_state::Branching,
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

const PIECE_SCORES: [i32; piece_type::N_VARIANTS] = [0, 100, 300, 300, 500, 800, 0];

pub fn piece_score(pt: PieceType) -> i32 {
    PIECE_SCORES[pt.v() as usize]
}

const MG_PAWN_TABLE: Psqt = Psqt([
    0, 0, 0, 0, 0, 0, 0, 0, //
    98, 134, 61, 95, 68, 126, 34, -11, //
    -6, 7, 26, 31, 65, 56, 25, -20, //
    -14, 13, 6, 21, 23, 12, 17, -23, //
    -27, -2, -5, 12, 17, 6, 10, -25, //
    -26, -4, -4, -10, 3, 3, 33, -12, //
    -35, -1, -20, -23, -15, 24, 38, -22, //
    0, 0, 0, 0, 0, 0, 0, 0, //
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
    -65, 23, 16, -15, -56, -34, 2, 13, //
    29, -1, -20, -7, -8, -4, -38, -29, //
    -9, 24, 2, -16, -20, 6, 22, -22, //
    -17, -20, -12, -27, -30, -25, -14, -36, //
    -49, -1, -27, -39, -46, -44, -33, -51, //
    -14, -14, -22, -46, -44, -30, -15, -27, //
    1, 7, -8, -64, -43, -16, 9, 8, //
    -15, 36, 12, -54, 8, -28, 24, 14, //
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
    -74, -35, -18, -18, -11, 15, 4, -17, //
    -12, 17, 14, 17, 17, 38, 23, 11, //
    10, 17, 23, 15, 20, 45, 44, 13, //
    -8, 22, 24, 27, 26, 33, 26, 3, //
    -18, -4, 21, 24, 27, 23, 9, -11, //
    -19, -3, 11, 21, 23, 16, 7, -9, //
    -27, -11, 4, 13, 14, 4, -5, -17, //
    -53, -34, -21, -11, -28, -14, -24, -43, //
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

fn psqt_score(phase: GamePhase, piece: PieceType, sq: Square, color: Color) -> i32 {
    let sq = if color == colors::WHITE { sq.flip_v() } else { sq };
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
#[derive(PartialEq, Debug, Default, Copy, Clone, PartialOrd)]
pub struct TaperValue(u32);

impl TaperValue {
    pub fn from_position(pos: &PieceInfo) -> Self {
        let inv_phase = (piece_type::PAWN..piece_type::KING)
            .map(|p| pos.get_piece_bb(p).pop_cnt() * PIECE_PHASES[p.v() as usize].v())
            .sum::<u32>();

        Self(piece_phases::TOTAL_C.saturating_sub(inv_phase))
    }

    pub fn weighted_eval(&self, mg_eval: i32, eg_eval: i32) -> i32 {
        let phase = self.0 as i32;
        let total = piece_phases::TOTAL_C as i32;
        ((mg_eval * (total - phase)) + (eg_eval * phase)) / total
    }
}

/// A penalty for `P`
pub struct Penalty<P: Perspective>(pub i32, PhantomData<P>);

impl<P: Perspective> fmt::Display for Penalty<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Penalty<{}>({})", P::COLOR, self.0)
    }
}

impl<P: Perspective> From<Penalty<P>> for Score<P> {
    #[inline(always)]
    fn from(val: Penalty<P>) -> Self {
        Score::new(-val.0)
    }
}

/// A bonus for `P`
#[derive(Debug, Copy, Clone)]
pub struct Score<P: Perspective>(pub i32, PhantomData<P>);

impl<P: Perspective> fmt::Display for Score<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Score<{}>({})", P::COLOR, self.0)
    }
}

impl<P: Perspective, Rhs: Into<Score<P>>> ops::Add<Rhs> for Score<P> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output {
        Self(self.0 + rhs.into().0, PhantomData)
    }
}

impl<P: Perspective> Eq for Score<P> {}

impl<P: Perspective> Ord for Score<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0).then_with(|| self.1.cmp(&other.1))
    }
}

impl<P: Perspective> PartialOrd for Score<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Perspective> PartialEq for Score<P> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<P: Perspective> Score<P> {
    pub const POS_INF: Self = Self::new(30_000);
    pub const NEG_INF: Self = Self::new(-30_000);

    #[inline(always)]
    pub const fn new(val: i32) -> Self {
        Self(val, PhantomData)
    }
}

impl<P: Perspective> Penalty<P> {
    #[inline(always)]
    pub const fn new(val: i32) -> Self {
        Self(val, PhantomData)
    }
}

// not using the `-` operator because this is not really just arithmetic
// negation, but also a perspective flip.
impl<P: Perspective> ops::Not for Score<P> {
    type Output = Score<P::Opponent>;

    /// Negate the score and flip the perspective to the opponent.
    #[inline(always)]
    fn not(self) -> Self::Output {
        Score::new(-self.0)
    }
}

impl<P: Perspective> From<Score<P>> for Cp {
    #[inline(always)]
    fn from(value: Score<P>) -> Self {
        if P::IS_WHITE {
            Cp { v: value.0 as i16 }
        }
        else {
            Cp { v: (-value.0) as i16 }
        }
    }
}

/// # Q-Search
///
/// Make the position quiet.
///
/// [q-search](https://www.chessprogramming.org/Quiescence_Search)
fn qsearch<P: Perspective>(pos: &mut Position, mut alpha: Score<P>, beta: Score<P>) -> Score<P> {
    let in_check = pos.get_check_state() != CheckState::None;

    let mut best_value = Score::NEG_INF;

    let piece_info = pos.piece_info();
    let phase = TaperValue::from_position(piece_info);

    // stand pad if not in check
    if !in_check {
        let static_eval = static_eval(
            piece_info,
            pos.get_ep_target_square(),
            pos.get_turn(),
            phase,
        );

        best_value = static_eval;

        if best_value >= beta {
            return best_value;
        }
        if best_value > alpha {
            alpha = best_value;
        }
    }

    // consider captures (and quiets if in check)
    let mut move_list = List::<{ MAX_LEGAL_MOVES }, (Move, i32)>::new();
    if in_check {
        _ = fold_legal_moves::<_, _, _>(pos, (), |_, m| {
            move_list.push((m, 0));
            ControlFlow::Continue::<(), ()>(())
        });
    }
    else {
        _ = fold_legal_captures::<_, _, _>(pos, (), |_, m| {
            move_list.push((m, 0));
            ControlFlow::Continue::<(), ()>(())
        });
    };

    /*\                                             /*\
    |*|---------------------------------------------|*|
    |*| generate the see score outside of the move  |*|
    |*| generation and the sorting, such that it    |*|
    |*| isn't computed for each comparison and when |*|
    |*| don't break cache locality.                 |*|
    |*|---------------------------------------------|*|
    \*/                                             \*/
    for &mut (m, ref mut see_score) in move_list.as_mut_slice() {
        *see_score = see(pos.piece_info(), m, P::COLOR);
    }

    // move ordering
    move_list
        .as_mut_slice()
        .sort_unstable_by_key(|&(_, see)| Reverse(see));

    // recurse
    for &(m, _) in move_list.iter() {
        // delta pruning
        if !in_check && phase < TaperValue(16) {
            let value_bonus = if let Ok(promo) = TryInto::<PromoPieceType>::try_into(m.get_flag()) {
                piece_score(promo.into()) - piece_score(piece_type::PAWN)
            }
            else {
                0
            };

            // SAFETY: we know this is a capture move.
            let capture_square = unsafe { m.get_capture_sq().unwrap_unchecked() };
            let captured_piece = pos.get_piece(capture_square);
            let captured_value = piece_score(captured_piece.piece_type());

            let futility_margin = 200;
            let futility_score = captured_value + value_bonus + futility_margin;

            if best_value + Score::new(futility_score) < alpha {
                continue;
            }
        }

        pos.make_move_for::<P>(m);

        let score = !qsearch(pos, !beta, !alpha);

        pos.unmake_move_for::<P>(m);

        if score >= beta {
            return score;
        }
        if score > best_value {
            best_value = score;
        }
        if score > alpha {
            alpha = score;
        }
    }

    best_value
}

pub fn material<P: Perspective>(pos: &PieceInfo) -> Score<P> {
    let score = (piece_type::PAWN..piece_type::KING)
        .map(|p| pos.get_bitboard(p, P::COLOR).pop_cnt() as i32 * piece_score(p))
        .sum();

    Score::new(score)
}

#[allow(clippy::identity_op)]
pub fn mobility<P: Perspective>(pos: &PieceInfo, phase: TaperValue) -> Score<P> {
    let color = P::COLOR;
    let occ = pos.get_occupancy();
    let score = (piece_type::KNIGHT..piece_type::KING)
        .map(|pt| {
            let pieces = pos.get_bitboard(pt, color);
            let scores: i32 = pieces
                .map(|sq| match pt {
                    piece_type::KNIGHT => {
                        let attacks = knight::lookup_attacks(sq);
                        let score = attacks.pop_cnt() * 5;
                        let score = score as i32;
                        phase.weighted_eval(score, score)
                    }
                    piece_type::BISHOP => {
                        let attacks = <Bishop as SlidingAttacks>::lookup_attacks(sq, occ);
                        let score = attacks.pop_cnt() * 5;
                        let score = score as i32;
                        phase.weighted_eval(score, score)
                    }
                    piece_type::ROOK => {
                        let attacks = <Rook as SlidingAttacks>::lookup_attacks(sq, occ);
                        let vertical = attacks & Bitboard::from(File::from(sq));
                        let horizontal = attacks & Bitboard::from(Rank::from(sq));
                        let vert_cnt = vertical.pop_cnt();
                        let hort_cnt = horizontal.pop_cnt();
                        phase.weighted_eval(
                            // vertical mobility is more valuable in the opening
                            (vert_cnt * 3 + hort_cnt * 1) as i32,
                            (vert_cnt * 5 + hort_cnt * 5) as i32,
                        )
                    }
                    piece_type::QUEEN => {
                        let attacks = <Queen as SlidingAttacks>::lookup_attacks(sq, occ);
                        let score = attacks.pop_cnt() * 5;
                        score as i32
                    }
                    _ => unreachable!(),
                })
                .sum();

            scores
        })
        .sum();

    Score::new(score)
}

pub fn pawn_shield<P: Perspective>(pos: &PieceInfo, phase: TaperValue, king: Square) -> Score<P> {
    let pawns = pos.get_bitboard(piece_type::PAWN, P::COLOR);

    let p1_squares = king::lookup_attacks(king);
    let p2_squares = king::lookup_attacks(king).shift(single_step(P::COLOR));
    let p2_protected = pawn::compute_attacks(pawns, P::COLOR);

    let p1_shield = pawns & p1_squares;
    let p2_shield_strong = pawns & p2_squares & p2_protected;
    let p2_shield_weak = pawns & p2_squares & !p2_protected;

    let p1_score = p1_shield.pop_cnt() as i32;
    let p2_score_strong = p2_shield_strong.pop_cnt() as i32;
    let p2_score_weak = p2_shield_weak.pop_cnt() as i32;

    let score = p1_score * 10 + p2_score_strong * 5 + p2_score_weak * 4;

    // we don't want the pawns from trying to promote in the endgame
    let score = phase.weighted_eval(score, 0);

    Score::new(score)
}

/// Evaluates the safety of our king's position by looking at enemy pawn storm.
pub fn pawn_storm_penalty<P: Perspective>(
    pos: &PieceInfo,
    ep_sq: EpTargetSquare,
    turn: Turn,
    king: Square,
) -> Penalty<P> {
    const DANGER_ZONES: [[Bitboard; squares::N_VARIANTS]; colors::N_VARIANTS] = {
        let step_b = 1;
        let step_w = -1;

        let zones_w = {
            let mut zones = [Bitboard::empty(); squares::N_VARIANTS];
            const_for!(king_sq_v in squares::A1_C..(squares::H8_C+1) => {
                // SAFETY: correct range
                let king_sq = unsafe { Square::from_v(king_sq_v) };
                let king_rank = Rank::from(king_sq);
                let king_file = File::from(king_sq);

                let danger_ranks = {
                    Bitboard::from(king_rank.saturating_shift(step_b))
                    | Bitboard::from(king_rank.saturating_shift(step_b * 2))
                    | Bitboard::from(king_rank.saturating_shift(step_b * 3))
                    | Bitboard::from(king_rank.saturating_shift(step_b * 4))
                };
                let danger_files = {
                    Bitboard::from(king_file)
                    | Bitboard::from(king_file.saturating_shift(1))
                    | Bitboard::from(king_file.saturating_shift(-1))
                };

                zones[king_sq_v as usize] = danger_ranks & danger_files;
            });
            zones
        };

        let zones_b = {
            let mut zones = [Bitboard::empty(); squares::N_VARIANTS];
            const_for!(king_sq_v in squares::A1_C..(squares::H8_C+1) => {
                // SAFETY: correct range
                let king_sq = unsafe { Square::from_v(king_sq_v) };
                let king_rank = Rank::from(king_sq);
                let king_file = File::from(king_sq);

                let danger_ranks = {
                    Bitboard::from(king_rank.saturating_shift(step_w))
                    | Bitboard::from(king_rank.saturating_shift(step_w * 2))
                    | Bitboard::from(king_rank.saturating_shift(step_w * 3))
                    | Bitboard::from(king_rank.saturating_shift(step_w * 4))
                };
                let danger_files = {
                    Bitboard::from(king_file)
                    | Bitboard::from(king_file.saturating_shift(1))
                    | Bitboard::from(king_file.saturating_shift(-1))
                };

                zones[king_sq_v as usize] = danger_ranks & danger_files;
            });
            zones
        };

        let mut zones = [[Bitboard::empty(); squares::N_VARIANTS]; colors::N_VARIANTS];
        zones[colors::WHITE.v() as usize] = zones_w;
        zones[colors::BLACK.v() as usize] = zones_b;

        zones
    };

    let us = P::COLOR;
    let them = !us;
    let danger_zone = DANGER_ZONES[us.v() as usize][king.v() as usize];

    let enemy_pawns = pos.get_bitboard(piece_type::PAWN, them);
    let relevant_pawns = enemy_pawns & danger_zone;
    let ally_pawns = pos.get_bitboard(piece_type::PAWN, us);
    let allies = pos.get_color_bb(us);

    // we only consider the ep a valid capture if it's the opponents turn.
    let ep_target = if turn == them { Some(ep_sq) } else { None };
    let ep_target_bb = Bitboard::from(ep_target.and_then(|x| x.v()));

    let capture_sq = allies | ep_target_bb;
    let nomnom_pawns = relevant_pawns & pawn::compute_attacks(capture_sq, us);
    let unblocked_pawns = relevant_pawns & !ally_pawns.shift(single_step(us)) & !nomnom_pawns;

    let storm_danger_penalty = unblocked_pawns.pop_cnt() * 10 + nomnom_pawns.pop_cnt() * 30;

    Penalty::<P>::new(storm_danger_penalty as i32)
}

pub fn open_king_file_penalty<P: Perspective>(
    pos: &PieceInfo,
    phase: TaperValue,
    king: Square,
) -> Penalty<P> {
    // [[start, end], king_file]
    const DANGER_FILES: [[File; 2]; files::N_VARIANTS] = {
        let mut files = [[files::A; 2]; files::N_VARIANTS];
        const_for!(king_file_v in files::A_C..(files::H_C+1) => {
            // SAFETY: correct range
            let king_file = unsafe { File::from_v(king_file_v) };

            let left = king_file.saturating_shift(-1);
            let midd = king_file;
            let right = king_file.saturating_shift(1);

            files[king_file_v as usize] = [
                if midd.v() == files::A_C { midd } else { left },
                if midd.v() == files::H_C { midd } else { right },
            ];
        });
        files
    };

    let mut penalty = 0;

    let king_file = File::from(king);
    let us = P::COLOR;
    let them = !us;
    let enemy_pawns = pos.get_bitboard(piece_type::PAWN, them);
    let ally_pawns = pos.get_bitboard(piece_type::PAWN, us);
    let [f_min, f_max] = DANGER_FILES[king_file.v() as usize];
    for file in f_min..f_max {
        let file_bb = Bitboard::from(file);
        let has_ally = !(ally_pawns & file_bb).is_empty();
        let has_enemy = !(enemy_pawns & file_bb).is_empty();

        match (has_ally, has_enemy) {
            (true, false) => penalty += 15,  // enemy has a semi-open file
            (false, true) => penalty += 35,  // pawn shield is gone
            (false, false) => penalty += 60, // fully open file
            (true, true) => {}               // closed file
        }
    }

    let score = phase.weighted_eval(penalty, 0);

    Penalty::<P>::new(score)
}

pub fn king_safety<P: Perspective>(
    pos: &PieceInfo,
    _ep_sq: EpTargetSquare,
    _turn: Turn,
    phase: TaperValue,
) -> Score<P> {
    if let Some(king) = pos.get_bitboard(piece_type::KING, P::COLOR).lsb() {
        pawn_shield::<P>(pos, phase, king) + open_king_file_penalty::<P>(pos, phase, king)
        // + pawn_storm_penalty::<P>(pos, ep_sq, turn, king)
    }
    else {
        Score::new(0)
    }
}

pub fn passed_pawns<P: Perspective>(
    pos: &PieceInfo,
    ep_sq: EpTargetSquare,
    turn: Turn,
) -> Score<P> {
    let us = P::COLOR;
    let them = !us;

    let our_pawns = pos.get_bitboard(piece_type::PAWN, us);
    let our_attacks = pawn::compute_attacks(our_pawns, us);
    let their_pawns = pos.get_bitboard(piece_type::PAWN, them);
    let their_attacks = pawn::compute_attacks(their_pawns, them);

    // we only consider the ep a valid capture if it's the opponents turn.
    let ep_target = if turn == them { Some(ep_sq) } else { None };
    let ep_target_bb = Bitboard::from(ep_target.and_then(|x| x.v()));

    // inlined compassrose because the constant 'NORT_C' wichi is literally just '8'
    // is 'unconstrained' -_-, thanks rust
    //
    // if this gets ever fixed, we should just be able to pass in
    // single_step::<P::COLOR>() as the direction without trouble...
    //
    let their_frontfill = match P::COLOR {
        colors::WHITE => (their_pawns | their_attacks).fill::<-8 /*south*/>(),
        colors::BLACK => (their_pawns | their_attacks).fill::<8 /* north*/>(),
        _ => unreachable!(),
    };

    // if the passed pawn can be captured en passant, don't count him
    let their_ep_capture = their_attacks & ep_target_bb;
    let their_ep_capt_sq = their_ep_capture.shift(single_step(us));

    let passed_pawns = our_pawns & !(their_frontfill | their_ep_capt_sq);

    let our_passer_rearspan = match P::COLOR {
        colors::WHITE => passed_pawns.span::<-8 /*south*/>(),
        colors::BLACK => passed_pawns.span::<8 /* north*/>(),
        _ => unreachable!(),
    };

    // normal or doubled passed pawns
    let secondary_passed_pawns = passed_pawns & our_passer_rearspan;
    let primary_passed_pawns = passed_pawns & !our_passer_rearspan;

    // protected passed pawn
    let protected_passed_pawns = passed_pawns & our_attacks;

    // tarrasch rule
    let our_rooks = pos.get_bitboard(piece_type::ROOK, us);
    let their_rooks = pos.get_bitboard(piece_type::ROOK, them);
    let protective_rooks = our_rooks & our_passer_rearspan;
    let aggressor_rooks = their_rooks & our_passer_rearspan;

    // score primary passed pawns higher than secondary/doubled passed pawns
    // give a bonus for protected passed pawns.
    let score = protected_passed_pawns.pop_cnt() as i32 * 100
        + primary_passed_pawns.pop_cnt() as i32 * 100
        + secondary_passed_pawns.pop_cnt() as i32 * 20
        + protective_rooks.pop_cnt() as i32 * 50
        - aggressor_rooks.pop_cnt() as i32 * 50;

    Score::new(score as i32)
}

fn bishop_pair<P: Perspective>(pos: &PieceInfo) -> Score<P> {
    let bishop_cnt = pos.get_bitboard(piece_type::BISHOP, P::COLOR).pop_cnt();
    let score = if bishop_cnt >= 2 { 75 } else { 0 };
    Score::new(score)
}

fn psqt<P: Perspective>(pos: &PieceInfo, phase: TaperValue) -> Score<P> {
    fn score(pos: &PieceInfo, color: Color, phase: GamePhase) -> i32 {
        (piece_type::PAWN..=piece_type::KING)
            .map(|piece| {
                pos.get_bitboard(piece, color)
                    .map(|sq| psqt_score(phase, piece, sq, color))
                    .sum::<i32>()
            })
            .sum()
    }

    let mg = score(pos, P::COLOR, game_phases::MG);
    let eg = score(pos, P::COLOR, game_phases::EG);
    Score::new(phase.weighted_eval(mg, eg))
}

fn static_value<P: Perspective>(
    pos: &PieceInfo,
    ep_sq: EpTargetSquare,
    turn: Turn,
    phase: TaperValue,
) -> Score<P> {
    material::<P>(pos)
        + mobility::<P>(pos, phase)
        + psqt::<P>(pos, phase)
        + bishop_pair::<P>(pos)
        + king_safety::<P>(pos, ep_sq, turn, phase)
        + passed_pawns::<P>(pos, ep_sq, turn)
}

fn find_smallest_attacker(pos: &PieceInfo, to: Square, us: Color, occ: Bitboard) -> Option<Square> {
    let all_pawns = pos.get_bitboard(piece_type::PAWN, us);
    let available_pawns = occ & all_pawns;
    let attacking_pawns = pawn::lookup_attacks(to, !us) & available_pawns;
    if let pawn @ Some(_) = attacking_pawns.lsb() {
        return pawn;
    }

    let all_knights = pos.get_bitboard(piece_type::KNIGHT, us);
    let available_knights = occ & all_knights;
    let attacking_knights = knight::lookup_attacks(to) & available_knights;
    if let knight @ Some(_) = attacking_knights.lsb() {
        return knight;
    }

    let all_bishops = pos.get_bitboard(piece_type::BISHOP, us);
    let available_bishops = occ & all_bishops;
    let attacking_bishops = <Bishop as SlidingAttacks>::lookup_attacks(to, occ) & available_bishops;
    if let bishop @ Some(_) = attacking_bishops.lsb() {
        return bishop;
    }

    let all_rooks = pos.get_bitboard(piece_type::ROOK, us);
    let available_rooks = occ & all_rooks;
    let attacking_rooks = <Rook as SlidingAttacks>::lookup_attacks(to, occ) & available_rooks;
    if let rook @ Some(_) = attacking_rooks.lsb() {
        return rook;
    }

    let all_queens = pos.get_bitboard(piece_type::QUEEN, us);
    let available_queens = occ & all_queens;
    let attacking_queens = <Queen as SlidingAttacks>::lookup_attacks(to, occ) & available_queens;
    if let queen @ Some(_) = attacking_queens.lsb() {
        return queen;
    }

    let all_kings = pos.get_bitboard(piece_type::KING, us);
    let available_kings = occ & all_kings;
    let attacking_kings = king::lookup_attacks(to) & available_kings;
    if let king @ Some(_) = attacking_kings.lsb() {
        return king;
    }

    None
}

/// MVV-LVA inspired bonus for capturing high-value pieces with low-value
/// pieces.
pub fn mvv_lva(pos: &PieceInfo, mov: Move) -> i32 {
    if let Some(capture_sq) = mov.get_capture_sq() {
        let capturing = pos.get_piece(mov.get_from()).piece_type();
        let captured = pos.get_piece(capture_sq).piece_type();
        let score = piece_score(captured) - piece_score(capturing);
        return score;
    }
    0
}

/// Static Exchange Evaluation (SEE) for captures.
pub fn see(pos: &PieceInfo, mov: Move, mut us: Color) -> i32 {
    let to = mov.get_to();
    let from = mov.get_from();

    let mut gain = [0; 32];
    let mut depth = Depth::ROOT;

    let mut occupancy = pos.get_occupancy();
    let mut attacker_sq = from;
    let mut attacker_pt = pos.get_piece(from).piece_type();

    // initial gain
    gain[0] = {
        let mut initial_gain = 0;

        // en passant
        if let Some(sq) = mov.get_capture_sq() {
            initial_gain = piece_score(pos.get_piece(sq).piece_type());
            occupancy &= !Bitboard::from(sq);
        }

        // promos
        if let Ok(promo) = PromoPieceType::try_from(mov.get_flag()) {
            initial_gain += piece_score(promo.v()) - piece_score(piece_type::PAWN);
            attacker_pt = promo.v();
        }

        // return early on quiet moves
        if initial_gain == 0 && mov.get_capture_sq().is_none() {
            return 0;
        }

        initial_gain
    };

    loop {
        depth += 1;
        us = !us;

        occupancy ^= Bitboard::from(attacker_sq);

        let next_attacker = find_smallest_attacker(pos, to, us, occupancy);

        match next_attacker {
            Some(sq) => {
                attacker_sq = sq;
                let next_attacker_pt = pos.get_piece(attacker_sq).piece_type();

                // Gain at this depth is the value of the piece we just exposed to capture,
                // minus the value we give up if the opponent recaptures.
                gain[depth.index()] = piece_score(attacker_pt) - gain[depth.index() - 1];

                // If the piece that just attacked is a pawn, and 'to' is a promotion rank,
                // it promotes. We assume it promotes to a Queen for SEE purposes.
                if next_attacker_pt == piece_type::PAWN
                    && matches!(Rank::from(to), ranks::_1 | ranks::_8)
                {
                    attacker_pt = piece_type::QUEEN;
                }
                else {
                    attacker_pt = next_attacker_pt;
                }
            }
            None => break,
        }
    }

    // Negamax propagation back up the sequence
    while depth > Depth::new(1) {
        depth -= 1;
        gain[depth.index() - 1] = min(gain[depth.index() - 1], -gain[depth.index()]);
    }

    gain[0]
}

fn static_eval<P: Perspective>(
    pos: &PieceInfo,
    ep_sq: EpTargetSquare,
    turn: Turn,
    phase: TaperValue,
) -> Score<P> {
    let (ep_w, ep_b) = if turn == colors::WHITE {
        (ep_sq, EpTargetSquare::none())
    }
    else {
        (EpTargetSquare::none(), ep_sq)
    };
    let w_q = static_value::<P>(pos, ep_w, turn, phase);
    let b_q = static_value::<P::Opponent>(pos, ep_b, turn, phase);
    w_q + !b_q
}

#[derive(Debug, PartialEq, Default)]
pub struct PolicyInput;

impl PolicyInput {
    pub fn psqt(
        phase: TaperValue,
        piece: PieceType,
        from: Square,
        to: Square,
        color: Color,
    ) -> i32 {
        let curr_score = {
            let mg = psqt_score(game_phases::MG, piece, from, color);
            let eg = psqt_score(game_phases::EG, piece, from, color);
            phase.weighted_eval(mg, eg)
        };
        let new_score = {
            let mg = psqt_score(game_phases::MG, piece, to, color);
            let eg = psqt_score(game_phases::EG, piece, to, color);
            phase.weighted_eval(mg, eg)
        };
        new_score - curr_score
    }

    pub fn meta(_pos: &PieceInfo, _mov: Move, _state: &StateInfo) -> i32 {
        // todo: give bonus for promotions etc.
        0
    }
}

pub struct EvalInfo<Moves: AsRef<[Move]>> {
    /// The to-be-evaluated that this eval info is for.
    moves: Moves,

    /// Turn of the current player.
    turn: Turn,

    /// The current game phase taper value.
    phase: TaperValue,

    /// Piece infos of the position.
    pos: PieceInfo,

    /// Current position state info.
    state: StateInfo,

    /// State info of the position after quieting it.
    quality: Cp,
}

impl<Moves: AsRef<[Move]>> EvalInfo<Moves> {
    pub fn new(moves: Moves, pos: &mut Position) -> Self {
        let quality: Cp = match pos.get_turn().v() {
            colors::WHITE_C => {
                qsearch::<perspectives::White>(pos, Score::NEG_INF, Score::POS_INF).into()
            }
            colors::BLACK_C => {
                qsearch::<perspectives::Black>(pos, Score::NEG_INF, Score::POS_INF).into()
            }
            _ => unreachable!(),
        };
        Self {
            quality,
            pos: pos.piece_info().clone(),
            state: pos.state_info().clone(),
            phase: TaperValue::from_position(pos.piece_info()),
            moves,
            turn: pos.get_turn(),
        }
    }

    /// Convert QualityInput into Quality, where the Quality is relative to
    /// white.
    pub fn quality(&self) -> Quality {
        Quality::from(self.quality)
    }

    pub fn policy(&self, buf: &mut List<218 /* inlined MAX_LEGAL_MOVES */, f32>) -> Policy {
        let pos = &self.pos;
        let phase = self.phase;
        let state = &self.state;
        let color = self.turn;

        // let mut logits = unsafe {
        //     let pointer = &mut policy.0 as *mut List<{ MAX_LEGAL_MOVES },
        // Probability>;     let logits = ptr::read(pointer.cast::<List<_,
        // f32>>());     ManuallyDrop::new(logits)
        // };

        let mut logits = List::new();

        for &mov in self.moves.as_ref().iter() {
            // policy for each move in the position is the difference of the psqt score from
            // the previous position and the psqt score of the next position that is
            // achieved by the move.
            let from = mov.get_from();
            let to = mov.get_to();
            let piece = pos.get_piece(from).piece_type();
            let score = PolicyInput::psqt(phase, piece, from, to, color)
                + see(pos, mov, color)
                + PolicyInput::meta(pos, mov, state);

            logits.push(score as f32);
        }

        // todo: setting the temperature to 20 showed a huge improvement in commit
        // 8646dd8d554d
        Policy::from_logits(Logits(logits), 10., buf)
    }
}

#[derive(Debug, Default, Clone)]
pub struct HceEvaluator {
    policy_buf: Box<List<{ MAX_LEGAL_MOVES }, f32>>,
}

impl HceEvaluator {
    pub fn new() -> Self {
        Self {
            policy_buf: Box::new(List::new()),
        }
    }
}

impl Evaluator for HceEvaluator {
    type TraceData = Option<EvalInfo<Vec<Move>>>;

    fn trace<S: const Valid + HasBranches>(
        &self,
        node: NodeId<S>,
        tree: &Tree,
        pos: &mut Position,
    ) -> Self::TraceData {
        node.try_into::<Branching>()
            .map(|node| EvalInfo::new(tree.branches(node).iter().map(|b| b.mov()).collect(), pos))
    }

    fn eval_batch(
        &mut self,
        _tree: &Tree,
        _selection: &Selection<Self::TraceData>,
        leafs: &[&BatchItem<Self::TraceData>],
    ) -> impl Iterator<Item = Guess> {
        leafs.iter().filter_map(|&leaf| {
            let eval_info = leaf.trace.as_ref()?;
            Some(Guess {
                relative_to: colors::WHITE,
                quality: eval_info.quality(),
                policy: eval_info.policy(&mut self.policy_buf),
            })
        })
    }
}
