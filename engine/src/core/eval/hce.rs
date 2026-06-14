use crate::{
    core::{
        bitboard::Bitboard,
        color::{Color, Perspective, colors},
        coordinates::{
            EpTargetSquare, File, Rank, Square, files, pawn_utils::single_step, squares,
        },
        move_iter::{
            bishop::Bishop, king, knight, pawn, queen::Queen, rook::Rook,
            sliding_piece::SlidingAttacks,
        },
        piece::{PieceType, piece_type},
        position::PieceInfo,
        search::score::{Penalty, Score},
        turn::Turn,
    },
    impl_variants,
};
use const_for::const_for;

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

pub const fn piece_score(pt: PieceType) -> i32 {
    PIECE_SCORES[pt.v() as usize]
}

#[rustfmt::skip]
const MG_PAWN_TABLE: Psqt = Psqt([
      0,    0,    0,    0,    0,    0,    0,    0,
     98,  134,   61,   95,   68,  126,   34,  -11,
     -6,    7,   26,   31,   65,   56,   25,  -20,
    -14,   13,    6,   21,   23,   12,   17,  -23,
    -27,   -2,   -5,   12,   17,    6,   10,  -25,
    -26,   -4,   -4,  -10,    3,    3,   33,  -12,
    -35,   -1,  -20,  -23,  -15,   24,   38,  -22,
      0,    0,    0,    0,    0,    0,    0,    0,
]);

#[rustfmt::skip]
const MG_KNIGHT_TABLE: Psqt = Psqt([
   -167,  -89,  -34,  -49,   61,  -97,  -15, -107,
    -73,  -41,   72,   36,   23,   62,    7,  -17,
    -47,   60,   37,   65,   84,  129,   73,   44,
     -9,   17,   19,   53,   37,   69,   18,   22,
    -13,    4,   16,   13,   28,   19,   21,   -8,
    -23,   -9,   12,   10,   19,   17,   25,  -16,
    -29,  -53,  -12,   -3,   -1,   18,  -14,  -19,
   -105,  -21,  -58,  -33,  -17,  -28,  -19,  -23,
]);

#[rustfmt::skip]
const MG_BISHOP_TABLE: Psqt = Psqt([
    -29,    4,  -82,  -37,  -25,  -42,    7,   -8,
    -26,   16,  -18,  -13,   30,   59,   18,  -47,
    -16,   37,   43,   40,   35,   50,   37,   -2,
     -4,    5,   19,   50,   37,   37,    7,   -2,
     -6,   13,   13,   26,   34,   12,   10,    4,
      0,   15,   15,   15,   14,   27,   18,   10,
      4,   15,   16,    0,    7,   21,   33,    1,
    -33,   -3,  -14,  -21,  -13,  -12,  -39,  -21,
]);

#[rustfmt::skip]
const MG_ROOK_TABLE: Psqt = Psqt([
     32,   42,   32,   51,   63,    9,   31,   43,
     27,   32,   58,   62,   80,   67,   26,   44,
     -5,   19,   26,   36,   17,   45,   61,   16,
    -24,  -11,    7,   26,   24,   35,   -8,  -20,
    -36,  -26,  -12,   -1,    9,   -7,    6,  -23,
    -45,  -25,  -16,  -17,    3,    0,   -5,  -33,
    -44,  -16,  -20,   -9,   -1,   11,   -6,  -71,
    -19,  -13,    1,   17,   16,    7,  -37,  -26,
]);

#[rustfmt::skip]
const MG_QUEEN_TABLE: Psqt = Psqt([
    -28,    0,   29,   12,   59,   44,   43,   45,
    -24,  -39,   -5,    1,  -16,   57,   28,   54,
    -13,  -17,    7,    8,   29,   56,   47,   57,
    -27,  -27,  -16,  -16,   -1,   17,   -2,    1,
     -9,  -26,   -9,  -10,   -2,   -4,    3,   -3,
    -14,    2,  -11,   -2,   -5,    2,   14,    5,
    -35,   -8,   11,    2,    8,   15,   -3,    1,
     -1,  -18,   -9,   10,  -15,  -25,  -31,  -50,
]);

#[rustfmt::skip]
const MG_KING_TABLE: Psqt = Psqt([
    -65,   23,   16,  -15,  -56,  -34,    2,   13,
     29,   -1,  -20,   -7,   -8,   -4,  -38,  -29,
     -9,   24,    2,  -16,  -20,    6,   22,  -22,
    -17,  -20,  -12,  -27,  -30,  -25,  -14,  -36,
    -49,   -1,  -27,  -39,  -46,  -44,  -33,  -51,
    -14,  -14,  -22,  -46,  -44,  -30,  -15,  -27,
      1,    7,   -8,  -64,  -43,  -16,    9,    8,
    -15,   36,   12,  -54,    8,  -28,   24,   14,
]);

#[rustfmt::skip]
const EG_PAWN_TABLE: Psqt = Psqt([
      0,    0,    0,    0,    0,    0,    0,    0,
    178,  173,  158,  134,  147,  132,  165,  187,
     94,  100,   85,   67,   56,   53,   82,   84,
     32,   24,   13,    5,   -2,    4,   17,   17,
     13,    9,   -3,   -7,   -7,   -8,    3,   -1,
      4,    7,   -6,    1,    0,   -5,   -1,   -8,
     13,    8,    8,   10,   13,    0,    2,   -7,
      0,    0,    0,    0,    0,    0,    0,    0,
]);

#[rustfmt::skip]
const EG_KNIGHT_TABLE: Psqt = Psqt([
    -58,  -38,  -13,  -28,  -31,  -27,  -63,  -99,
    -25,   -8,  -25,   -2,   -9,  -25,  -24,  -52,
    -24,  -20,   10,    9,   -1,   -9,  -19,  -41,
    -17,    3,   22,   22,   22,   11,    8,  -18,
    -18,   -6,   16,   25,   16,   17,    4,  -18,
    -23,   -3,   -1,   15,   10,   -3,  -20,  -22,
    -42,  -20,  -10,   -5,   -2,  -20,  -23,  -44,
    -29,  -51,  -23,  -15,  -22,  -18,  -50,  -64,
]);

#[rustfmt::skip]
const EG_BISHOP_TABLE: Psqt = Psqt([
    -14,  -21,  -11,   -8,   -7,   -9,  -17,  -24,
     -8,   -4,    7,  -12,   -3,  -13,   -4,  -14,
      2,   -8,    0,   -1,   -2,    6,    0,    4,
     -3,    9,   12,    9,   14,   10,    3,    2,
     -6,    3,   13,   19,    7,   10,   -3,   -9,
    -12,   -3,    8,   10,   13,    3,   -7,  -15,
    -14,  -18,   -7,   -1,    4,   -9,  -15,  -27,
    -23,   -9,  -23,   -5,   -9,  -16,   -5,  -17,
]);

#[rustfmt::skip]
const EG_ROOK_TABLE: Psqt = Psqt([
     13,   10,   18,   15,   12,   12,    8,    5,
     11,   13,   13,   11,   -3,    3,    8,    3,
      7,    7,    7,    5,    4,   -3,   -5,   -3,
      4,    3,   13,    1,    2,    1,   -1,    2,
      3,    5,    8,    4,   -5,   -6,   -8,  -11,
     -4,    0,   -5,   -1,   -7,  -12,   -8,  -16,
     -6,   -6,    0,    2,   -9,   -9,  -11,   -3,
     -9,    2,    3,   -1,   -5,  -13,    4,  -20,
]);

#[rustfmt::skip]
const EG_QUEEN_TABLE: Psqt = Psqt([
     -9,   22,   22,   27,   27,   19,   10,   20,
    -17,   20,   32,   41,   58,   25,   30,    0,
    -20,    6,    9,   49,   47,   35,   19,    9,
      3,   22,   24,   45,   57,   40,   57,   36,
    -18,   28,   19,   47,   31,   34,   39,   23,
    -16,  -27,   15,    6,    9,   17,   10,    5,
    -22,  -23,  -30,  -16,  -16,  -23,  -36,  -32,
    -33,  -28,  -22,  -43,   -5,  -32,  -20,  -41,
]);

#[rustfmt::skip]
const EG_KING_TABLE: Psqt = Psqt([
    -74,  -35,  -18,  -18,  -11,   15,    4,  -17,
    -12,   17,   14,   17,   17,   38,   23,   11,
     10,   17,   23,   15,   20,   45,   44,   13,
     -8,   22,   24,   27,   26,   33,   26,    3,
    -18,   -4,   21,   24,   27,   23,    9,  -11,
    -19,   -3,   11,   21,   23,   16,    7,   -9,
    -27,  -11,    4,   13,   14,    4,   -5,  -17,
    -53,  -34,  -21,  -11,  -28,  -14,  -24,  -43,
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

#[inline]
pub fn psqt_score(phase: GamePhase, piece: PieceType, sq: Square, color: Color) -> i32 {
    let sq = if color == colors::WHITE { sq.flip_v() } else { sq };
    PSQT[phase.v() as usize][piece.v() as usize].get(sq)
}

#[inline]
pub fn tapered_psqt(phase: TaperValue, piece: PieceType, sq: Square, color: Color) -> i32 {
    let mg = psqt_score(game_phases::MG, piece, sq, color);
    let eg = psqt_score(game_phases::EG, piece, sq, color);
    phase.weighted_eval(mg, eg)
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
    pub const fn new(val: u32) -> Self {
        debug_assert!(val <= piece_phases::TOTAL_C);
        Self(val)
    }

    pub fn from_position(pos: &PieceInfo) -> Self {
        let inv_phase = (piece_type::PAWN..piece_type::KING)
            .map(|p| pos.get_piece_bb(p).pop_cnt() * PIECE_PHASES[p.v() as usize].v())
            .sum::<u32>();

        Self(piece_phases::TOTAL_C.saturating_sub(inv_phase))
    }

    pub const fn weighted_eval(&self, mg_eval: i32, eg_eval: i32) -> i32 {
        let phase = self.0 as i32;
        let total = piece_phases::TOTAL_C as i32;
        ((mg_eval * (total - phase)) + (eg_eval * phase)) / total
    }

    pub const fn v(&self) -> u32 {
        self.0
    }
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

// todo: ep capture possibilities are subject of qsearch anyway, so maybe it's
// better to just ignore that possibility.
#[allow(clippy::erasing_op)]
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
    let score = protected_passed_pawns.pop_cnt() as i32 * 50
        + primary_passed_pawns.pop_cnt() as i32 * 30
        + secondary_passed_pawns.pop_cnt() as i32 * 0
        + protective_rooks.pop_cnt() as i32 * 20
        - aggressor_rooks.pop_cnt() as i32 * 15;

    Score::new(score as i32)
}

pub fn bishop_pair<P: Perspective>(pos: &PieceInfo) -> Score<P> {
    let bishop_cnt = pos.get_bitboard(piece_type::BISHOP, P::COLOR).pop_cnt();
    let score = if bishop_cnt >= 2 { 75 } else { 0 };
    Score::new(score)
}

pub fn psqt<P: Perspective>(pos: &PieceInfo, phase: TaperValue) -> Score<P> {
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

pub fn hygge_king<P: Perspective>(pos: &PieceInfo, phase: TaperValue) -> Score<P> {
    let us = P::COLOR;
    let them = !us;

    if let Some(their_king) = pos.get_bitboard(piece_type::KING, them).lsb() {
        let our_knights = pos.get_bitboard(piece_type::KNIGHT, us);
        let knight_bonuses = our_knights
            .map(|knight| knight.distance(their_king))
            .map(|dist| match dist {
                0 => 0,      // can't happen
                1 => 5,      // probably too close
                2 | 3 => 20, // knight attacks are effective here
                4 | 5 => 10, // we're getting there :D
                _ => 0,
            })
            .sum::<i32>();

        let our_queens = pos.get_bitboard(piece_type::QUEEN, us);
        let queen_bonuses = our_queens
            .map(|queen| queen.distance(their_king))
            .map(|dist| match dist {
                0 => 0, // can't happen
                1 => 50,
                2 => 40,
                3 => 30,
                4 => 20,
                5 => 10,
                6 => 5,
                _ => 0,
            })
            .sum::<i32>();

        let our_king = pos.get_bitboard(piece_type::KING, us).lsb();
        let king_bonus = our_king
            .map(|k| k.distance(their_king))
            .map(|dist| match dist {
                0 | 1 => 0, // can't happen
                2 => 30,
                3 => 20,
                4 => 10,
                5 => 5,
                _ => 0,
            })
            .unwrap_or(0);

        // pawns walk up the board, which is handled in psqt

        // bishop should move to center, which is handled in psqt

        // rook distance is not that important for mating in the eg

        let score = knight_bonuses + queen_bonuses + king_bonus;

        Score::new(phase.weighted_eval(0, score))
    }
    else {
        Score::new(0)
    }
}
