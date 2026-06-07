use std::{cmp::min, ops::ControlFlow};

use crate::{
    core::{
        bitboard::Bitboard,
        color::Color,
        coordinates::{Rank, Square, ranks},
        depth::Depth,
        eval::hce::{TaperValue, piece_score, tapered_psqt},
        r#move::{MAX_UNREACHABLE_MOVES, Move},
        move_iter::{self, fold_moves},
        piece::{PieceType, PromoPieceType, piece_type},
        position::{PieceInfo, Position},
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
    score: i32,
    mov: Move,
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
    fn score(&self, mov: Move) -> i32;
}

#[derive(Debug)]
pub struct MovePicker {
    moves: List<{ MAX_UNREACHABLE_MOVES }, ScoredMove>,
    curr: usize,
}

impl MovePicker {
    pub fn from_scored(scored: impl Iterator<Item = ScoredMove>) -> Self {
        let mut moves = List::new();

        for item in scored {
            moves.push(item);
        }

        Self { moves, curr: 0 }
    }

    pub fn from_position<O: move_iter::Options, S: MoveScorer>(pos: &Position, scorer: S) -> Self {
        let mut moves = List::new();

        _ = fold_moves::<O, _, _, _>(pos, (), |_, m| {
            moves.push(ScoredMove::new(m, 0));
            ControlFlow::Continue::<(), ()>(())
        });

        // generate the see score outside of the move generation and the sorting, such
        // that it isn't computed for each comparison and we don't distrurb cache
        // locality.
        for &mut ScoredMove { mov, ref mut score } in moves.as_mut_slice() {
            *score = scorer.score(mov);
        }

        Self { moves, curr: 0 }
    }
}

impl Iterator for MovePicker {
    type Item = (Move, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let len = self.moves.len();
        if self.curr >= len {
            return None;
        }

        // Partial selection sort: find the highest-scored move in curr..len and swap it
        // into position.
        let slice = self.moves.as_mut_slice();
        let mut best_idx = self.curr;
        for i in (self.curr + 1)..len {
            if slice[i].score() > slice[best_idx].score() {
                best_idx = i;
            }
        }
        slice.swap(self.curr, best_idx);

        let i = self.curr;
        let m = slice[i].mov();

        self.curr += 1;

        Some((m, i))
    }
}

#[cfg(test)]
pub mod test {
    use crate::core::{
        color::colors, coordinates::squares, r#move::move_flags, move_iter::sliding_piece::magics,
        position::Position, search::ordering, zobrist,
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
}
