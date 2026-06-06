use std::cmp::min;

use crate::core::{
    bitboard::Bitboard,
    color::Color,
    coordinates::{Rank, Square, ranks},
    depth::Depth,
    eval::hce::{TaperValue, piece_score, tapered_psqt},
    r#move::Move,
    move_iter::{
        bishop::Bishop, king, knight, pawn, queen::Queen, rook::Rook, sliding_piece::SlidingAttacks,
    },
    piece::{PieceType, PromoPieceType, piece_type},
    position::PieceInfo,
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

pub fn psqt(phase: TaperValue, piece: PieceType, from: Square, to: Square, color: Color) -> i32 {
    let curr_score = tapered_psqt(phase, piece, from, color);

    // todo: change piece type for promotions
    let new_score = tapered_psqt(phase, piece, to, color);

    new_score - curr_score
}

#[cfg(test)]
pub mod test {
    use crate::core::{color::colors, coordinates::squares, r#move::move_flags, move_iter::sliding_piece::magics, position::Position, search::ordering, zobrist};

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
