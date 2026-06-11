use std::{cmp::min, ops::Range};

use std::ops::ControlFlow;

use crate::{
    core::{
        bitboard::Bitboard,
        color::{Color, Perspective, colors, perspectives},
        coordinates::{Rank, Square, ranks},
        depth::Depth,
        eval::hce::{TaperValue, piece_score, tapered_psqt},
        r#move::{MAX_LEGAL_MOVES, Move},
        move_iter::{self, fold_moves},
        piece::{PieceType, PromoPieceType, piece_type},
        position::{PieceInfo, Position},
        search::{
            id::RbSet,
            ordering::stages::{GenerateCapturesAndPromos, GenerateQuiets},
        },
    },
    misc::List,
};

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

        let next_attacker = pos
            .smallest_attackers(to, us, occupancy)
            .and_then(|bb| bb.lsb());

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

pub fn psqt(phase: TaperValue, piece: PieceType, from: Square, to: Square, color: Color) -> i32 {
    let curr_score = tapered_psqt(phase, piece, from, color);

    // todo: change piece type for promotions
    let new_score = tapered_psqt(phase, piece, to, color);

    new_score - curr_score
}

#[derive(Debug, Clone)]
pub struct ScoredMove {
    pub score: i32,
    pub mov: Move,
}

impl ScoredMove {
    #[inline]
    pub fn new(m: Move, score: i32) -> Self {
        Self { score, mov: m }
    }

    #[inline]
    pub fn mov(&self) -> Move {
        self.mov
    }

    #[inline]
    pub fn score(&self) -> i32 {
        self.score
    }

    #[inline]
    pub fn set_score(&mut self, score: i32) {
        self.score = score;
    }
}

pub trait MoveScorer {
    fn score<S: Stage>(&self, pos: &Position, mov: Move) -> i32;
}

#[derive(Debug)]
pub struct MovePicker {
    move_gen: MoveGenerator,
    slice: Range<usize>,
    curr: usize,
    max_stage: RtStage,
}

impl MovePicker {
    pub fn from_scored(scored: impl Iterator<Item = ScoredMove>) -> Self {
        let mut moves = List::new();

        for item in scored {
            moves.push(item);
        }

        let len = moves.len();

        Self {
            move_gen: MoveGenerator::new_precomputed(RtStage::Done, moves),
            slice: 0..len,
            curr: 0,
            max_stage: RtStage::Done,
        }
    }

    pub fn new(hash_move: Move, killers: RbSet<Move, 2>) -> Self {
        Self {
            move_gen: MoveGenerator::new(hash_move, killers),
            slice: Range::default(),
            curr: 0,
            max_stage: RtStage::Done,
        }
    }

    pub fn new_with_max_stage(
        hash_move: Move,
        killers: RbSet<Move, 2>,
        max_stage: RtStage,
    ) -> Self {
        Self {
            move_gen: MoveGenerator::new(hash_move, killers),
            slice: Range::default(),
            curr: 0,
            max_stage,
        }
    }

    #[inline]
    pub fn next(&mut self, pos: &Position, scorer: &impl MoveScorer) -> Option<Move> {
        match pos.get_turn() {
            colors::WHITE => self.next_for::<perspectives::White>(pos, scorer),
            colors::BLACK => self.next_for::<perspectives::Black>(pos, scorer),
            _ => unreachable!(),
        }
    }

    pub fn next_for<P: Perspective>(
        &mut self,
        pos: &Position,
        scorer: &impl MoveScorer,
    ) -> Option<Move> {
        // try to generate new moves if we've exhausted the current slice. `curr` is an
        // absolute index into the generator's buffer.
        while self.curr >= self.slice.end {
            if self.move_gen.stage > self.max_stage {
                // we'd have to generate moves past the max stage
                return None;
            }
            match self.move_gen.next_for::<P>(pos, scorer) {
                Err(MoveGenExhausted) => {
                    // move gen is done, there are no more moves
                    return None;
                }
                Ok(s) => {
                    self.curr = s.start;
                    self.slice = s;
                }
            }
        }

        let slice = self.move_gen.buf.as_mut_subslice(self.curr..self.slice.end);
        partial_sort_desc(slice);

        let m = slice[0].mov();

        self.curr += 1;

        Some(m)
    }
}

/// Brings the highest score to the front of the slice
#[inline]
pub fn partial_sort_desc(slice: &mut [ScoredMove]) {
    let len = slice.len();
    let mut best_idx = 0;
    for i in 1..len {
        if slice[i].score() > slice[best_idx].score() {
            best_idx = i;
        }
    }

    slice.swap(0, best_idx);
}

/// If true, generates legals, if false, generates pseudo legals.
pub const LEGAL: bool = true;

mod scores {
    pub const HASH_MOVE: i32 = 300_000;
    pub const GOOD_CAPTURES_AND_PROMOS: i32 = 210_000; // + see
    pub const KILLER: i32 = 200_000; // - age * 10_000
    pub const BAD_CAPTURES_AND_PROMOS: i32 = 100_000; // + see
    pub const QUIET: i32 = 0; // + psqt diff
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RtStage {
    YieldHashMove,
    GenerateCapturesAndPromos,
    YieldGoodCapturesAndPromos,
    YieldKillers,
    YieldBadCaptures,
    GenerateQuiets,
    YieldQuiets,
    Done,
}

impl TryFrom<u8> for RtStage {
    type Error = ();

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::YieldHashMove),
            1 => Ok(Self::GenerateCapturesAndPromos),
            2 => Ok(Self::YieldGoodCapturesAndPromos),
            3 => Ok(Self::YieldKillers),
            4 => Ok(Self::YieldBadCaptures),
            5 => Ok(Self::GenerateQuiets),
            6 => Ok(Self::YieldQuiets),
            7 => Ok(Self::Done),
            _ => Err(()),
        }
    }
}

impl RtStage {
    #[inline]
    fn next(&mut self) {
        if *self != Self::Done {
            // Safety: We don't increment if we are at the max.
            *self = unsafe { (*self as u8 + 1).try_into().unwrap_unchecked() };
        }
    }
}

pub const trait Stage {
    fn stage() -> RtStage;
}

#[rustfmt::skip]
pub mod stages {
    use super::*;

    pub struct YieldHashMove;
    impl Stage for YieldHashMove {
        fn stage() -> RtStage { RtStage::YieldHashMove }
    }

    pub struct GenerateCapturesAndPromos;
    impl Stage for GenerateCapturesAndPromos {
        fn stage() -> RtStage { RtStage::GenerateCapturesAndPromos }
    }
    #[rustfmt::skip]
    impl const move_iter::Options for GenerateCapturesAndPromos {
        fn gen_quiets() -> bool { false }
        fn gen_captures() -> bool { true }
        fn gen_promos() -> bool { true }
        fn legal() -> bool { LEGAL }
    }

    pub struct YieldGoodCapturesAndPromos;
    impl Stage for YieldGoodCapturesAndPromos {
        fn stage() -> RtStage { RtStage::YieldGoodCapturesAndPromos }
    }

    pub struct YieldKillers;
    impl Stage for YieldKillers {
        fn stage() -> RtStage { RtStage::YieldKillers }
    }

    pub struct YieldBadCaptures;
    impl Stage for YieldBadCaptures {
        fn stage() -> RtStage { RtStage::YieldBadCaptures }
    }

    pub struct YieldQuiets;
    impl Stage for YieldQuiets {
        fn stage() -> RtStage { RtStage::YieldQuiets }
    }

    pub struct GenerateQuiets;
    impl Stage for GenerateQuiets {
        fn stage() -> RtStage { RtStage::GenerateQuiets }
    }
    #[rustfmt::skip]
    impl const move_iter::Options for GenerateQuiets {
        fn gen_quiets() -> bool { true }
        fn gen_captures() -> bool { false }
        fn gen_promos() -> bool { false }
        fn legal() -> bool { LEGAL }
    }

    pub struct Done;
    impl Stage for Done {
        fn stage() -> RtStage { RtStage::Done }
    }
}

#[derive(Debug)]
pub struct MoveGenerator {
    stage: RtStage,
    hash_move: Move,
    killers: RbSet<Move, 2>,
    buf: List<{ MAX_LEGAL_MOVES }, ScoredMove>,
    start_good_capt_and_promos: usize,
    num_good_capt_and_promos: usize,
    num_capt_and_promos: usize,
}

pub struct MoveGenExhausted;

impl MoveGenerator {
    pub fn new(hash_move: Move, killers: RbSet<Move, 2>) -> Self {
        Self {
            buf: List::new(),
            stage: RtStage::YieldHashMove,
            hash_move,
            killers,
            start_good_capt_and_promos: 0,
            num_good_capt_and_promos: 0,
            num_capt_and_promos: 0,
        }
    }

    pub fn new_precomputed(stage: RtStage, buf: List<{ MAX_LEGAL_MOVES }, ScoredMove>) -> Self {
        Self {
            buf,
            stage,
            hash_move: Move::null(),
            killers: RbSet::default(),
            start_good_capt_and_promos: 0,
            num_good_capt_and_promos: 0,
            num_capt_and_promos: 0,
        }
    }

    /// Pushes the next stage of moves into the list. Returns the new slice that
    /// could be consumed. slice.
    pub fn next_for<P: Perspective>(
        &mut self,
        pos: &Position,
        scorer: &impl MoveScorer,
    ) -> Result<Range<usize>, MoveGenExhausted> {
        match self.stage {
            RtStage::YieldHashMove => {
                if self.hash_move != Move::null()
                    && pos.is_pseudo_legal_for::<P>(self.hash_move)
                    && pos.is_legal_for::<P>(self.hash_move)
                {
                    let score = scorer.score::<stages::YieldHashMove>(pos, self.hash_move);
                    self.buf
                        .push(ScoredMove::new(self.hash_move, scores::HASH_MOVE + score));

                    self.stage.next();
                    Ok(0..1)
                }
                else {
                    self.stage.next();
                    Ok(0..0)
                }
            }
            RtStage::GenerateCapturesAndPromos => {
                let start = self.buf.len();

                // todo: if the killer is Move::null() then we don't need to check inside this
                // loop, same with the killers below
                _ = fold_moves::<GenerateCapturesAndPromos, _, _, _>(pos, (), |_, m| {
                    // todo: this could be slow... maybe check this somewhere else?
                    if m != self.hash_move && self.killers._position(&m).is_none() {
                        // todo: do not filter here, just filter before yielding and don't generate
                        // a second time below
                        self.buf.push(ScoredMove::new(m, 0));
                    }
                    ControlFlow::Continue::<(), _>(())
                });

                let end = self.buf.len();
                let num = end - start;

                // generate the score outside of the move generation and the sorting, such
                // that it isn't computed for each comparison and we don't distrurb cache
                // locality.
                //
                // single-pass in-place partition: good captures/promos (see >= 0) are moved
                // to the front of the slice, bad ones stay at the back. this keeps the
                // YieldGoodCapturesAndPromos / YieldBadCaptures ranges accurate without a
                // separate sort or buffer.
                let slice = self.buf.as_mut_subslice(start..end);
                let mut num_good = 0;
                for i in 0..num {
                    let s = scorer.score::<GenerateCapturesAndPromos>(pos, slice[i].mov);
                    // todo: maybe its faster to keep a back pointer and just insert the bad ones in
                    // the back and the good ones in the front. that way we don't need to swap any
                    // elements? (using this for now though because it just works and is an
                    // improvement to the previous version...)
                    if s >= 0 {
                        slice[i].score = s + scores::GOOD_CAPTURES_AND_PROMOS;
                        slice.swap(num_good, i);
                        num_good += 1;
                    }
                    else {
                        slice[i].score = s + scores::BAD_CAPTURES_AND_PROMOS;
                    }
                }

                self.start_good_capt_and_promos = start;
                self.num_good_capt_and_promos = num_good;
                self.num_capt_and_promos = num;
                self.stage.next();

                Ok(0..0)
            }
            RtStage::YieldGoodCapturesAndPromos => {
                let start = self.start_good_capt_and_promos;
                let end = start + self.num_good_capt_and_promos;
                self.stage.next();
                Ok(start..end)
            }
            RtStage::YieldKillers => {
                let start = self.buf.len();

                for &killer in self.killers.as_slice() {
                    debug_assert!(
                        killer.get_capture_sq().is_none(),
                        "Killers should not be captures"
                    );

                    if killer != Move::null()
                        && pos.is_pseudo_legal_for::<P>(killer)
                        && pos.is_legal_for::<P>(killer)
                    {
                        let s = scorer.score::<stages::YieldKillers>(pos, killer);
                        self.buf.push(ScoredMove::new(killer, scores::KILLER + s));
                    }
                }

                let end = self.buf.len();

                self.stage.next();
                Ok(start..end)
            }
            RtStage::YieldBadCaptures => {
                let num = self.num_capt_and_promos;

                let start_good = self.start_good_capt_and_promos;
                let end_good = start_good + self.num_good_capt_and_promos;
                let num_good = end_good - start_good;

                let start_bad = end_good;
                let num_bad = num - num_good;
                let end_bad = start_bad + num_bad;

                self.stage.next();
                Ok(start_bad..end_bad)
            }
            RtStage::GenerateQuiets => {
                let start = self.buf.len();

                _ = fold_moves::<GenerateQuiets, _, _, _>(pos, (), |_, m| {
                    // todo: this could be slow... maybe check this somewhere else?
                    if m != self.hash_move && self.killers._position(&m).is_none() {
                        // todo: do not filter here, just filter before yielding and don't generate
                        // a second time below
                        self.buf.push(ScoredMove::new(m, 0));
                    }
                    ControlFlow::Continue::<(), _>(())
                });

                let end = self.buf.len();

                for &mut ScoredMove { mov, ref mut score } in self.buf.as_mut_subslice(start..end) {
                    let s = scorer.score::<GenerateQuiets>(pos, mov);
                    *score = s + scores::QUIET;
                }

                self.stage.next();
                Ok(start..end)
            }
            RtStage::YieldQuiets => {
                // todo: we already gave the range above so no need to yield those again. tbh
                // this whole yield-quiets stage can just be removed lol
                self.stage.next();
                Ok(0..0)
            }
            RtStage::Done => Err(MoveGenExhausted),
        }
    }
}

#[cfg(test)]
pub mod test {
    use std::collections::HashSet;

    use itertools::Itertools;
    use rand::{
        SeedableRng,
        rngs::SmallRng,
        seq::{IndexedRandom, IteratorRandom},
    };

    use crate::core::{
        color::colors,
        coordinates::squares,
        r#move::{MoveList, move_flags},
        move_iter::sliding_piece::magics,
        position::Position,
        search::{id, ordering},
        zobrist,
    };

    use super::*;

    fn run_see_test(fen: &str, mov: Move, us: Color, expected: i32) {
        magics::init();
        zobrist::init();

        let pos = Position::from_fen(fen).unwrap();

        let actual_score = ordering::see(pos.piece_info(), mov, us);

        assert_eq!(
            actual_score, expected,
            "SEE failed for move {:?} in FEN {}. Expected {}, got {}",
            mov, fen, expected, actual_score
        );
    }

    #[test]
    fn see_quiet_move() {
        // e4 move, no captures
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let mov = Move::new(squares::E2, squares::E4, move_flags::QUIET);
        run_see_test(fen, mov, colors::WHITE, 0);
    }

    #[test]
    fn see_undefended_capture() {
        // White knight takes undefended black pawn on d5
        let fen = "8/8/8/3p4/4N3/8/8/8 w - - 0 1";
        let mov = Move::new(squares::E4, squares::D5, move_flags::CAPTURE);

        // Expected: Gains the pawn
        run_see_test(fen, mov, colors::WHITE, piece_score(piece_type::PAWN));
    }

    #[test]
    fn see_equal_trade() {
        // White pawn takes black pawn on d5, black recaptures
        let fen = "8/8/4p3/3p4/4P3/8/8/8 w - - 0 1";
        let mov = Move::new(squares::E4, squares::D5, move_flags::CAPTURE);

        // Expected: +100 for PxP, but opponent recaptures for -100. Net is 0.
        run_see_test(
            fen,
            mov,
            colors::WHITE,
            piece_score(piece_type::PAWN) - piece_score(piece_type::PAWN),
        );
    }

    #[test]
    fn see_losing_capture() {
        // White queen takes defended black pawn on d5
        let fen = "8/8/4p3/3p4/4Q3/8/8/8 w - - 0 1";
        let mov = Move::new(squares::E4, squares::D5, move_flags::CAPTURE);

        // Expected: QxP (+100). Black plays PxQ (-800). Net: -700.
        // (If your Negamax propagation is wrong, this test will fail!)
        run_see_test(
            fen,
            mov,
            colors::WHITE,
            -piece_score(piece_type::QUEEN) + piece_score(piece_type::PAWN),
        );
    }

    #[test]
    fn see_complex_xray_dogpile() {
        // Classic SEE test: White has Rooks on d1, d3. Black has Rook d8, Bishop d6.
        // White initiates: Rd3xd6.
        let fen = "1k1r4/1p5p/3b4/8/8/3R4/1PP4P/1K1R4 w - - 0 1";
        let mov = Move::new(squares::D3, squares::D6, move_flags::CAPTURE);

        // 1. White RxB (+300)
        // 2. Black RxR (+500) -> Net so far: -200
        // 3. White RxR (revealed by x-ray!) (+500) -> Net: +300
        run_see_test(
            fen,
            mov,
            colors::WHITE,
            piece_score(piece_type::BISHOP) - piece_score(piece_type::ROOK)
                + piece_score(piece_type::ROOK),
        );
    }

    #[test]
    fn see_en_passant() {
        // White pawn on e5 captures d5 pawn en passant
        let fen = "8/8/8/3pP3/8/8/8/8 w - d6 0 1";
        // Ensure you pass the EN_PASSANT flag so your code knows it's an EP move!
        let mov = Move::new(squares::E5, squares::D6, move_flags::EN_PASSANT);

        // Expected: +100 for the pawn.
        run_see_test(fen, mov, colors::WHITE, piece_score(piece_type::PAWN));
    }

    #[test]
    fn see_capture_promotion() {
        // White pawn on e7 captures Black Rook on d8 and promotes to Queen.
        // Black has a Rook on c8 ready to recapture the new Queen.
        let fen = "2rr4/4P3/8/8/8/8/8/8 w - - 0 1";
        // Pass the promotion-capture flag (e.g., PROMO_QUEEN_CAPTURE)
        let mov = Move::new(
            squares::E7,
            squares::D8,
            move_flags::CAPTURE_PROMOTION_QUEEN,
        );

        // 1. White captures Rook (+500) and promotes (+800) - loses pawn (-100).
        //    Initial gain: +1300.
        // 2. Black recaptures the newly promoted Queen (+800).
        run_see_test(
            fen,
            mov,
            colors::WHITE,
            piece_score(piece_type::ROOK) + piece_score(piece_type::QUEEN)
                - piece_score(piece_type::PAWN)
                - piece_score(piece_type::QUEEN),
        );
    }

    fn test_does_pick_all_legal_moves(fen: &str, depth: Depth) {
        magics::init();
        zobrist::init();

        let mut pos = Position::from_fen(fen).unwrap();
        recurse_test(&mut pos, &mut SmallRng::seed_from_u64(0), depth);

        fn recurse_test(pos: &mut Position, rng: &mut SmallRng, depth: Depth) -> u64 {
            if depth == Depth::new(0) {
                return 1;
            }

            let all_moves = pos.collect_moves(MoveList::new());

            let hash_move = *all_moves.as_slice().choose(rng).unwrap_or(&Move::null());

            let mut killers = RbSet::new();
            let mut get_killer = || {
                *all_moves
                    .iter()
                    .filter(|&m| !m.get_flag().is_capture() && hash_move != *m)
                    .choose(rng)
                    .unwrap_or(&Move::null())
            };
            killers.push(get_killer());
            killers.push(get_killer());

            let mut picker = MovePicker::new(hash_move, killers.clone());

            let scorer = id::Scorer {
                tt_move: hash_move,
                killers,
                color: pos.get_turn(),
                phase: TaperValue::from_position(pos.piece_info()),
            };

            let mut cnt = 0;
            let mut got = MoveList::new();
            while let Some(mov) = picker.next(pos, &scorer) {
                got.push(mov);
                cnt += 1;
                pos.make_move(mov);
                recurse_test(pos, rng, depth - 1);
                pos.unmake_move(mov);
            }

            let expected = all_moves.len() as u64;
            let result = cnt;

            assert!(
                result == expected,
                "Move count mismatch in position: {} \nExpected: {} \nGot: {}, \nDiff: {:?}",
                crate::core::position::FenExport(pos),
                expected,
                result,
                {
                    let expected = all_moves.iter().collect::<HashSet<_>>();
                    let result = got.iter().collect::<HashSet<_>>();
                    let diff = expected
                        .symmetric_difference(&result)
                        .cloned()
                        .collect_vec();

                    if diff.is_empty() {
                        format!("None, there's likely duplicates in one of the sets.")
                    }
                    else {
                        format!("{diff:?}")
                    }
                }
            );

            cnt
        }
    }

    #[test]
    fn does_pick_all_legal_moves_0() {
        test_does_pick_all_legal_moves(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            Depth::new(5),
        );
    }

    #[test]
    fn does_pick_all_legal_moves_1() {
        test_does_pick_all_legal_moves(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            Depth::new(5),
        );
    }

    #[test]
    fn does_pick_all_legal_moves_2() {
        test_does_pick_all_legal_moves("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", Depth::new(6));
    }

    #[test]
    fn does_pick_all_legal_moves_3() {
        test_does_pick_all_legal_moves(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            Depth::new(5),
        );
    }

    #[test]
    fn does_pick_all_legal_moves_4() {
        test_does_pick_all_legal_moves(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ",
            Depth::new(5),
        );
    }

    #[test]
    fn does_pick_all_legal_moves_5() {
        test_does_pick_all_legal_moves(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ",
            Depth::new(5),
        );
    }
}
