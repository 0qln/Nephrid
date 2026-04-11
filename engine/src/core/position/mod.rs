use core::fmt;
use std::{fmt::Write, ops::ControlFlow, ptr::NonNull};

use thiserror::Error;

use crate::{
    core::{
        bitboard::Bitboard,
        castling::{CastlingRights, CastlingSideTokenizationError},
        color::{Color, ColorTokenizationError, Perspective, colors, perspectives},
        coordinates::{
            EpTargetSquareTokenizationError, File, Rank, RankParseError, Square, files, ranks,
            squares,
        },
        depth::Depth,
        r#move::{Move, MoveParseError, SAN, move_flags},
        move_iter::{bishop, fold_legal_moves, king, knight, pawn, rook},
        piece::{Piece, PieceParseError, PieceType, PromoPieceType, piece_type},
        ply::{FullMoveCountParseError, FullMoveCountTokenizationError, PlyTokenizationError},
        search::mcts::eval::GameResult,
        turn::Turn,
        zobrist,
    },
    misc::{ConstFrom, ValueOutOfSetError},
    uci::tokens::Tokenizer,
};

use super::{
    coordinates::{EpCaptureSquare, EpTargetSquare},
    move_iter::{bishop::Bishop, queen::Queen, rook::Rook, sliding_piece::SlidingAttacks},
    ply::{FullMoveCount, Ply},
};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CheckState {
    #[default]
    None,
    Single,
    Double,
}

#[cfg(test)]
mod test;

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct StateInfo {
    // Memoized state
    pub checkers: Bitboard,
    pub blockers: Bitboard,
    pub nstm_attacks: Bitboard,
    pub check_state: CheckState,

    // Game history
    pub plys50: Ply,
    pub ply: Ply,
    pub turn: Turn,
    pub ep_capture_square: EpCaptureSquare,
    pub castling: CastlingRights,
    pub captured_piece: Piece,
    pub key: zobrist::Hash,
}

impl StateInfo {
    /// Initiate checkers, blockers, nstm_attacks, check_state
    pub fn init(&mut self, pos: &Position) {
        let stm = self.turn;
        let nstm = !stm;
        let king = pos.get_bitboard(piece_type::KING, stm);
        let occupancy = pos.get_occupancy();
        let enemies = pos.get_color_bb(nstm);
        (self.nstm_attacks, self.checkers) = {
            enemies.fold((Bitboard::empty(), Bitboard::empty()), |acc, enemy_sq| {
                let enemy = pos.get_piece(enemy_sq);
                let enemy_attacks = match enemy.piece_type() {
                    piece_type::PAWN => pawn::lookup_attacks(enemy_sq, nstm),
                    piece_type::KNIGHT => knight::lookup_attacks(enemy_sq),
                    piece_type::BISHOP => Bishop::lookup_attacks(enemy_sq, occupancy),
                    piece_type::ROOK => Rook::lookup_attacks(enemy_sq, occupancy),
                    piece_type::QUEEN => Queen::lookup_attacks(enemy_sq, occupancy),
                    piece_type::KING => king::lookup_attacks(enemy_sq),
                    _ => unreachable!(
                        "We are iterating the squares which contain enemies. piece_type::NONE \
                         should not be here."
                    ),
                };
                (
                    acc.0 | enemy_attacks,
                    match enemy_attacks & king {
                        Bitboard { v: 0 } => acc.1,
                        _ => acc.1 | Bitboard::from_c(enemy_sq),
                    },
                )
            })
        };

        if let Some(king_sq) = king.lsb() {
            let x_ray_checkers = pos.get_x_ray_checkers(king_sq, enemies);
            self.blockers = x_ray_checkers.fold(Bitboard::empty(), |acc, x_ray_checker| {
                let between_squares = Bitboard::between(x_ray_checker, king_sq);
                let between_occupancy = occupancy & between_squares;
                if between_occupancy.pop_cnt_eq_1() {
                    acc | between_squares
                }
                else {
                    acc
                }
            });
        }

        self.check_state = match self.checkers.pop_cnt() {
            1 => CheckState::Single,
            2 => CheckState::Double,
            _ => CheckState::None,
        };
    }
}

#[derive(Debug, Eq)]
pub struct StateStack {
    states: Vec<StateInfo>,
    current: usize,
}

impl Clone for StateStack {
    fn clone(&self) -> Self {
        let mut states = self.states.clone();
        states.reserve(self.states.capacity() - self.states.len());
        Self { states, current: self.current }
    }
}

impl PartialEq for StateStack {
    fn eq(&self, other: &Self) -> bool {
        self.states[..self.current] == other.states[..other.current]
    }
}

impl Default for StateStack {
    fn default() -> Self {
        Self::new(StateInfo::default())
    }
}

impl StateStack {
    pub fn new(initial_state: StateInfo) -> Self {
        let mut vec = Vec::with_capacity(16);
        vec.push(initial_state);
        Self { states: vec, current: 0 }
    }

    /// Returns a reference to the current state.
    #[inline]
    pub fn get_current(&self) -> &StateInfo {
        // Safety: The current index is always in range
        unsafe { self.states.get_unchecked(self.current) }
    }

    pub fn get_prev(&self, go_back: usize) -> Option<&StateInfo> {
        if go_back > self.current {
            return None;
        }
        // Safety: We checked above
        unsafe { Some(self.states.get_unchecked(self.current - go_back)) }
    }

    /// Returns a mutable reference to the current state.
    #[inline]
    pub fn get_current_mut(&mut self) -> &mut StateInfo {
        // Safety: The current index is always in range
        unsafe { self.states.get_unchecked_mut(self.current) }
    }

    /// Returns the pushed state.
    #[inline]
    pub fn get_next(&mut self, new: fn(&StateInfo) -> StateInfo) -> NonNull<StateInfo> {
        let next = self.current + 1;

        // self.current can only ever be one greater than the length of the vector.
        debug_assert!(self.states.len() >= next);

        if self.states.len() == next {
            // Safety: self.states.len() >= 1 => self.states.len() > self.current - 1 >= 0
            let prev = unsafe { self.states.get_unchecked(self.current) };
            self.states.push(new(prev));
        }

        // Safety: The current index is always in range.
        NonNull::from_ref(unsafe { self.states.get_unchecked(next) })
    }

    /// Increment the current index.
    #[inline]
    pub fn incr(&mut self) {
        self.current += 1;
    }

    /// Returns the popped state.
    #[inline]
    pub fn pop_current(&mut self) -> NonNull<StateInfo> {
        // Safety: The current index is always in range
        let ret = NonNull::from_ref(unsafe { self.states.get_unchecked(self.current) });
        self.current = self.current.saturating_sub(1);
        ret
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PieceInfo {
    c_bitboards: [Bitboard; 2],
    t_bitboards: [Bitboard; 7],
    pieces: [Piece; 64],
    piece_counts: [i8; 14],
}

impl Default for PieceInfo {
    /// Returns an empty position.
    fn default() -> Self {
        Self {
            c_bitboards: Default::default(),
            t_bitboards: Default::default(),
            pieces: [Piece::default(); 64],
            piece_counts: Default::default(),
        }
    }
}

impl PieceInfo {
    #[inline]
    pub fn get_bitboard(&self, piece_type: PieceType, color: Color) -> Bitboard {
        self.get_color_bb(color) & self.get_piece_bb(piece_type)
    }

    #[inline]
    pub fn get_color_bb(&self, color: Color) -> Bitboard {
        // Safety:
        // It's not possible to safely create an instance of Color,
        // without checking that the value is in range.
        unsafe { *self.c_bitboards.get_unchecked(color.v() as usize) }
    }

    #[inline]
    fn get_color_bb_mut(&mut self, color: Color) -> &mut Bitboard {
        // Safety:
        // It's not possible to safely create an instance of Color,
        // without checking that the value is in range.
        unsafe { self.c_bitboards.get_unchecked_mut(color.v() as usize) }
    }

    // todo: these get_unchecked's are not neccesary anymore.

    #[inline]
    pub fn get_piece_bb(&self, piece_type: PieceType) -> Bitboard {
        self.t_bitboards[piece_type.v() as usize]
    }

    #[inline]
    fn get_piece_bb_mut(&mut self, piece_type: PieceType) -> &mut Bitboard {
        // Safety:
        // It's not possible to safely create an instance of PieceType,
        // without checking that the value is in range.
        unsafe { self.t_bitboards.get_unchecked_mut(piece_type.v() as usize) }
    }

    #[inline]
    pub fn get_occupancy(&self) -> Bitboard {
        self.get_color_bb(colors::WHITE) | self.get_color_bb(colors::BLACK)
    }

    #[inline]
    pub fn get_piece(&self, sq: Square) -> Piece {
        // Safety: sq is in range 0..64
        unsafe { *self.pieces.get_unchecked(sq.v() as usize) }
    }

    #[inline]
    fn get_piece_mut(&mut self, sq: Square) -> &mut Piece {
        // Safety: sq is in range 0..64
        unsafe { self.pieces.get_unchecked_mut(sq.v() as usize) }
    }

    #[inline]
    pub fn get_piece_count(&self, piece: Piece) -> i8 {
        // Safety:
        // It's not possible to safely create an instance of Piece,
        // without checking that the value is in range.
        unsafe { *self.piece_counts.get_unchecked(piece.v() as usize) }
    }

    #[inline]
    fn get_piece_count_mut(&mut self, piece: Piece) -> &mut i8 {
        // Safety:
        // It's not possible to safely create an instance of Piece,
        // without checking that the value is in range.
        unsafe { self.piece_counts.get_unchecked_mut(piece.v() as usize) }
    }

    fn remove_piece(&mut self, sq: Square) {
        let target = Bitboard::from_c(sq);
        let piece = self.get_piece(sq);
        assert_ne!(piece, Piece::default(), "No piece at {sq}");
        *self.get_piece_bb_mut(piece.piece_type()) ^= target;
        *self.get_color_bb_mut(piece.color()) ^= target;
        *self.get_piece_mut(sq) = Piece::default();
        *self.get_piece_count_mut(piece) -= 1;
    }

    fn put_piece(&mut self, sq: Square, piece: Piece) {
        let target = Bitboard::from_c(sq);
        assert_eq!(
            self.get_piece(sq),
            Piece::default(),
            "Piece already at {sq}: {}",
            self.get_piece(sq)
        );
        *self.get_piece_bb_mut(piece.piece_type()) |= target;
        *self.get_color_bb_mut(piece.color()) |= target;
        *self.get_piece_mut(sq) = piece;
        *self.get_piece_count_mut(piece) += 1;
    }

    fn move_piece(&mut self, from: Square, to: Square) {
        assert!(
            self.get_piece(from) != Piece::default(),
            "No piece at {from}"
        );
        assert!(
            self.get_piece(to) == Piece::default(),
            "Piece already at {to}: {}",
            self.get_piece(to)
        );
        let piece = self.get_piece(from);
        let from_to = Bitboard::from_c(from) | Bitboard::from_c(to);
        *self.get_color_bb_mut(piece.color()) ^= from_to;
        *self.get_piece_bb_mut(piece.piece_type()) ^= from_to;
        *self.get_piece_mut(from) = Piece::default();
        *self.get_piece_mut(to) = piece;
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Position {
    piece_info: PieceInfo,
    state: StateStack,
}

impl Default for Position {
    /// Returns an empty position.
    fn default() -> Self {
        Self {
            piece_info: Default::default(),
            state: Default::default(),
        }
    }
}

impl Position {
    #[inline]
    pub fn get_turn(&self) -> Turn {
        self.state.get_current().turn
    }

    #[inline]
    pub fn get_ep_capture_square(&self) -> EpCaptureSquare {
        self.state.get_current().ep_capture_square
    }

    #[inline]
    pub fn get_ep_target_square(&self) -> EpTargetSquare {
        // the last ep capture was the last players turn, so we invert the turn...
        EpTargetSquare::from((self.get_ep_capture_square(), !self.get_turn()))
    }

    #[inline]
    pub fn get_ep_capture_bitboard(&self, c: Color) -> Bitboard {
        if let Some(sq) = self.get_ep_capture_square().v()
            && self.get_turn() == c
        {
            Bitboard::from_c(sq)
        }
        else {
            Default::default()
        }
    }

    #[inline]
    pub fn get_castling(&self) -> CastlingRights {
        self.state.get_current().castling
    }

    #[inline]
    pub fn get_key(&self) -> zobrist::Hash {
        self.state.get_current().key
    }

    #[inline]
    pub fn get_check_state(&self) -> CheckState {
        self.state.get_current().check_state
    }

    #[inline]
    pub fn get_checkers(&self) -> Bitboard {
        self.state.get_current().checkers
    }

    #[inline]
    pub fn get_nstm_attacks(&self) -> Bitboard {
        self.state.get_current().nstm_attacks
    }

    #[inline]
    pub fn get_blockers(&self) -> Bitboard {
        self.state.get_current().blockers
    }

    #[inline]
    pub fn has_threefold_repetition(&self) -> bool {
        // cannot have the same position 3 times if the same player has only moved
        // 4 times after the last irreversible move.
        let plys50 = self.plys_50().v;
        if plys50 < 8 {
            return false;
        }

        let mut i = 2;
        let current_key = self.get_key();
        let mut repetitions = 0;
        while let Some(state) = self.state.get_prev(i)
            && ((i as u16) <= plys50)
        {
            if state.key == current_key {
                repetitions += 1;
                if repetitions >= 2 {
                    return true;
                }
            }
            // only compare same color, so skip 2
            i += 2;
        }
        false
    }

    #[inline]
    pub fn has_twofold_repetition(&self) -> bool {
        let plys50 = self.plys_50().v;
        if plys50 < 4 {
            return false;
        }

        let mut i = 2;
        let current_key = self.get_key();
        while let Some(state) = self.state.get_prev(i)
            && ((i as u16) <= plys50)
        {
            if state.key == current_key {
                return true;
            }
            // only compare same color, so skip 2
            i += 2;
        }
        false
    }

    #[inline]
    pub fn is_insufficient_material(&self) -> bool {
        self.piece_info.piece_counts.iter().sum::<i8>() <= 2
    }

    #[inline]
    pub fn plys_50(&self) -> Ply {
        self.state.get_current().plys50
    }

    #[inline]
    pub fn ply(&self) -> Ply {
        self.state.get_current().ply
    }

    pub fn full_move(&self) -> FullMoveCount {
        self.ply().into()
    }

    #[inline]
    pub fn fifty_move_rule(&self) -> bool {
        self.plys_50() >= Ply { v: 100 }
    }

    /// Returns the game result if the position is in a terminal state, else
    /// None. has_moves: Whether this position has subsequent moves.
    pub fn game_result_with(&self, has_moves: bool) -> Option<GameResult> {
        let stm = self.get_turn();
        let check_state = self.get_check_state();

        // First check if the position is a normal game ending.
        if !has_moves {
            Some(if check_state != CheckState::None {
                // If in check and no moves, it's a loss for the current player
                GameResult::Win { relative_to: !stm }
            }
            else {
                // Stalemate
                GameResult::Draw
            })
        }
        // Then check if the position has reached some of the extra-rule endings.
        else if self.has_threefold_repetition()
            || self.fifty_move_rule()
            || self.is_insufficient_material()
        {
            Some(GameResult::Draw)
        }
        // Otherwise no game result.
        else {
            None
        }
    }

    /// Returns the game result if the position is in a terminal state or looks
    /// to be a terminal state found in search, else None.
    /// has_moves: Whether this position has subsequent moves.
    /// search_depth: The current search depth.
    pub fn search_result_with(&self, has_moves: bool, search_depth: Depth) -> Option<GameResult> {
        let stm = self.get_turn();
        let check_state = self.get_check_state();

        // First check if the position is a normal game ending.
        if !has_moves {
            Some(if check_state != CheckState::None {
                // If in check and no moves, it's a loss for the current player
                GameResult::Win { relative_to: !stm }
            }
            else {
                // Stalemate
                GameResult::Draw
            })
        }
        // Then check if the position has reached some of the extra-rule endings.
        //
        // regarding 2-fold usage:
        //   Apply the 2-fold heuristic deep in the tree.
        //   At the root, fall back to the strict 3-fold rule to catch actual draws.
        else if (search_depth > Depth::ROOT && self.has_twofold_repetition())
            || (search_depth == Depth::ROOT && self.has_threefold_repetition())
            || self.fifty_move_rule()
            || self.is_insufficient_material()
        {
            Some(GameResult::Draw)
        }
        // Otherwise no game result.
        else {
            None
        }
    }

    pub fn game_result(&self) -> Option<GameResult> {
        let has_moves = self.has_legal_moves();
        self.game_result_with(has_moves)
    }

    pub fn search_result(&self, search_depth: Depth) -> Option<GameResult> {
        let has_moves = self.has_legal_moves();
        self.search_result_with(has_moves, search_depth)
    }

    pub fn has_legal_moves(&self) -> bool {
        fold_legal_moves(self, false, |_, _| ControlFlow::Break(true)).into_value()
    }

    /// Returns the X-Ray checkers for the given king.
    /// X-Ray checkers are pieces which attack a king
    /// through zero or more pieces.
    #[inline]
    pub fn get_x_ray_checkers(&self, king: Square, enemies: Bitboard) -> Bitboard {
        let rooks = self.get_piece_bb(piece_type::ROOK);
        let bishops = self.get_piece_bb(piece_type::BISHOP);
        let queens = self.get_piece_bb(piece_type::QUEEN);
        let rook_checkers = rook::compute_attacks_0_occ(king) & (rooks | queens);
        let bishop_checkers = bishop::compute_attacks_0_occ(king) & (bishops | queens);
        (rook_checkers | bishop_checkers) & enemies
    }

    /// # Safety
    /// This is unsafe, because it allows you to modify the internal
    /// representation, without updating the state.
    ///
    /// This pub, because it is used for benchmarking.
    pub unsafe fn put_piece_unsafe(&mut self, sq: Square, piece: Piece) {
        self.piece_info.put_piece(sq, piece)
    }

    /// # Safety
    /// This is unsafe, because it allows you to modify the internal
    /// representation, without updating the state.
    ///
    /// This pub, because it is used for benchmarking.
    pub unsafe fn remove_piece_unsafe(&mut self, sq: Square) {
        self.piece_info.remove_piece(sq)
    }

    /// # Safety
    /// This is unsafe, because it allows you to modify the internal
    /// representation, without updating the state.
    ///
    /// This pub, because it is used for benchmarking.
    pub unsafe fn move_piece_unsafe(&mut self, from: Square, to: Square) {
        self.piece_info.move_piece(from, to)
    }

    /// Makes a move on the board.
    pub fn make_move(&mut self, m: Move) {
        if self.get_turn() == colors::WHITE {
            self.make_move_for::<perspectives::White>(m);
        }
        else {
            self.make_move_for::<perspectives::Black>(m);
        }
    }

    // todo: use these in the mcts searh functions instead of make_move(..)
    pub fn make_move_for<P: Perspective>(&mut self, m: Move) {
        let us = P::COLOR;
        debug_assert_eq!(
            us,
            self.get_turn(),
            "Color parameter C must match the current turn."
        );

        let (from, to, flag) = m.into();
        let moving_piece = self.get_piece(from);
        let target_piece = self.get_piece(to);

        // Safety: During the lifetime of this pointer, no other pointer
        // reads or writes to the memory location of the next state.
        let next_state = unsafe {
            self.state
                .get_next(|prev| {
                    StateInfo {
                        // These don't change across leafes on the same depth...
                        ply: prev.ply + 1,
                        turn: !prev.turn,
                        ..Default::default()
                    }
                })
                .as_mut()
        };

        // These might change across leafes on the same depth, so the
        // need to be reinitialized for each leaf.
        let curr_state = self.state.get_current();
        next_state.castling = curr_state.castling;
        next_state.plys50 = curr_state.plys50 + 1;
        next_state.ep_capture_square = EpCaptureSquare::default();
        next_state.key = curr_state.key;
        next_state
            .key
            .toggle_ep_square(curr_state.ep_capture_square);
        next_state.key.toggle_turn();
        next_state.captured_piece = Piece::default();

        // castling
        next_state
            .castling
            .apply_mask(CastlingRights::MASKS[from.index()]);
        next_state
            .castling
            .apply_mask(CastlingRights::MASKS[to.index()]);

        // captures
        if flag.is_capture() {
            let captured_piece = match flag {
                move_flags::EN_PASSANT => Piece::from_c((!us, piece_type::PAWN)),
                _ => target_piece,
            };

            let captured_sq = match flag {
                move_flags::EN_PASSANT => {
                    // Safety:
                    // If the move is an en passant, the `to` square is on the 3rd or 6th rank.
                    unsafe {
                        let target_sq = EpTargetSquare::try_from(to).unwrap_unchecked();
                        EpCaptureSquare::from((target_sq, !us))
                            .v()
                            .unwrap_unchecked()
                    }
                }
                _ => to,
            };

            self.piece_info.remove_piece(captured_sq);

            next_state.captured_piece = captured_piece;
            next_state.key.toggle_piece_sq(captured_sq, captured_piece);
            next_state.plys50 = Ply { v: 0 };
        }

        // move the piece
        self.piece_info.move_piece(from, to);
        next_state.key.move_piece_sq(from, to, moving_piece);

        match moving_piece.piece_type() {
            // castling
            piece_type::KING => match flag {
                move_flags::KING_CASTLE => {
                    let rank = Rank::from_c(to);
                    let rook_from = Square::from_c((files::H, rank));
                    let rook_to = Square::from_c((files::F, rank));
                    let rook = self.get_piece(rook_from);
                    self.piece_info.move_piece(rook_from, rook_to);
                    next_state.key.move_piece_sq(rook_from, rook_to, rook);
                }
                move_flags::QUEEN_CASTLE => {
                    let rank = Rank::from_c(to);
                    let rook_from = Square::from_c((files::A, rank));
                    let rook_to = Square::from_c((files::D, rank));
                    let rook = self.get_piece(rook_from);
                    self.piece_info.move_piece(rook_from, rook_to);
                    next_state.key.move_piece_sq(rook_from, rook_to, rook);
                }
                _ => (),
            },
            // pawns
            piece_type::PAWN => {
                match flag.v() {
                    move_flags::DOUBLE_PAWN_PUSH_C => {
                        // Safety: A double pawn push destination square is the definition of
                        // an en passant square.
                        next_state.ep_capture_square =
                            unsafe { EpCaptureSquare::try_from(to).unwrap_unchecked() };
                        next_state
                            .key
                            .toggle_ep_square(next_state.ep_capture_square);
                    }
                    move_flags::PROMOTION_KNIGHT_C..=move_flags::CAPTURE_PROMOTION_QUEEN_C => {
                        // Safety: We just checked, that the flag is in a valid range.
                        let promo_t = unsafe { PromoPieceType::try_from(flag).unwrap_unchecked() };
                        let promo = Piece::from_c((us, promo_t));
                        self.piece_info.remove_piece(to);
                        next_state.key.toggle_piece_sq(to, moving_piece);
                        self.piece_info.put_piece(to, promo);
                        next_state.key.toggle_piece_sq(to, promo);
                    }
                    _ => (),
                }

                next_state.plys50 = Ply { v: 0 };
            }
            _ => (),
        }

        // update castling rights in the hash, if they have changed.
        if next_state.castling != curr_state.castling {
            next_state
                .key
                .toggle_castling(curr_state.castling)
                .toggle_castling(next_state.castling);
        }

        // update state stack
        next_state.init(self);
        self.state.incr();
    }

    pub fn unmake_move(&mut self, m: Move) {
        if self.get_turn() == colors::BLACK {
            self.unmake_move_for::<perspectives::White>(m);
        }
        else {
            self.unmake_move_for::<perspectives::Black>(m);
        }
    }

    pub fn unmake_move_for<P: Perspective>(&mut self, m: Move) {
        let us = P::COLOR;
        debug_assert_eq!(
            !us,
            self.get_turn(),
            "Color parameter C must be the opposite of the current turn."
        );

        let (from, to, flag) = m.into();

        // Safety: During the lifetime of this pointer, no other pointer
        // writes to the memory location of the popped state.
        let popped_state = unsafe { self.state.pop_current().as_ref() };

        match flag.v() {
            // castling
            move_flags::KING_CASTLE_C => {
                let rank = Rank::from_c(to);
                let rook_from = Square::from_c((files::H, rank));
                let rook_to = Square::from_c((files::F, rank));
                self.piece_info.move_piece(rook_to, rook_from);
            }
            move_flags::QUEEN_CASTLE_C => {
                let rank = Rank::from_c(to);
                let rook_from = Square::from_c((files::A, rank));
                let rook_to = Square::from_c((files::D, rank));
                self.piece_info.move_piece(rook_to, rook_from);
            }
            // promotions
            move_flags::PROMOTION_KNIGHT_C..=move_flags::CAPTURE_PROMOTION_QUEEN_C => {
                let pawn = Piece::from_c((us, piece_type::PAWN));
                self.piece_info.remove_piece(to);
                self.piece_info.put_piece(to, pawn);
            }
            _ => {}
        }

        // move the piece
        self.piece_info.move_piece(to, from);

        // captures
        let captured_piece = popped_state.captured_piece;
        if captured_piece != Piece::default() {
            let captured_sq = match flag {
                move_flags::EN_PASSANT => {
                    // Safety:
                    // If the move is an en passant, the `to` square is on the 3rd or 6th rank.
                    unsafe {
                        let target_sq = EpTargetSquare::try_from(to).unwrap_unchecked();
                        EpCaptureSquare::from((target_sq, !us))
                            .v()
                            .unwrap_unchecked()
                    }
                }
                _ => to,
            };

            self.piece_info.put_piece(captured_sq, captured_piece);
        }
    }

    pub fn start_position() -> Self {
        // Safety: This FEN string is valid
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let fen = FenImport(&mut Tokenizer::new(fen));
        unsafe { Self::try_from(fen).unwrap_unchecked() }
    }

    // Convenience wrappers

    pub fn from_fen(fen: &str) -> Result<Self, FenParseError> {
        Self::try_from(FenImport(&mut Tokenizer::new(fen)))
    }

    pub fn from_pgn(pgn: &str) -> Result<Self, PgnImportError> {
        Self::try_from(PgnImport(&mut Tokenizer::new(pgn)))
    }

    #[inline]
    pub fn get_bitboard(&self, piece_type: PieceType, color: Color) -> Bitboard {
        self.piece_info.get_bitboard(piece_type, color)
    }

    #[inline]
    pub fn get_color_bb(&self, color: Color) -> Bitboard {
        self.piece_info.get_color_bb(color)
    }

    pub fn get_piece_bb(&self, piece_type: PieceType) -> Bitboard {
        self.piece_info.get_piece_bb(piece_type)
    }

    pub fn get_occupancy(&self) -> Bitboard {
        self.piece_info.get_occupancy()
    }

    #[inline]
    pub fn get_piece(&self, sq: Square) -> Piece {
        self.piece_info.get_piece(sq)
    }

    pub fn get_piece_count(&self, piece: Piece) -> i8 {
        self.piece_info.get_piece_count(piece)
    }

    pub fn piece_info(&self) -> &PieceInfo {
        &self.piece_info
    }

    pub fn state_info(&self) -> &StateInfo {
        self.state.get_current()
    }
}

pub struct PiecePlacementInfo<'a>(&'a Position);

impl<'a> fmt::Display for PiecePlacementInfo<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pos = &self.0;

        // 1. Piece Placement
        // ranks in big-endian order
        for rank in (ranks::_1_C..=ranks::_8_C).rev() {
            let rank = unsafe { Rank::from_v(rank) };

            // files in little-endian order from A to H
            let mut nones = 0;
            for file in files::A_C..=files::H_C {
                let file = unsafe { File::from_v(file) };
                let square = Square::from_c((file, rank));
                let piece = pos.get_piece(square);

                if piece.piece_type() == piece_type::NONE {
                    nones += 1;
                    continue;
                }
                if nones != 0 {
                    write!(f, "{nones}")?;
                    nones = 0;
                }
                write!(f, "{piece}")?;
            }
            if nones != 0 {
                write!(f, "{nones}")?;
            }
            if rank != ranks::_1 {
                f.write_char('/')?;
            }
        }

        Ok(())
    }
}

pub struct FenExport<'a>(pub &'a Position);

impl<'a> fmt::Display for FenExport<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pos = &self.0;

        write!(
            f,
            "{} {} {} {} {} {}",
            PiecePlacementInfo(pos),
            pos.get_turn(),
            pos.get_castling(),
            pos.get_ep_target_square(),
            pos.plys_50(),
            pos.full_move()
        )
    }
}

// pgn spec: https://www.thechessdrum.net/PGN_Reference.txt

/// 3.2.4: Reduced export format
///
/// A PGN game represented using export format is said to be in "reduced export
/// format" if all of the following hold: 1) it has no commentary, 2) it has
/// only the standard seven tag roster identification information ("STR", see
/// below), 3) it has no recursive annotation variations ("RAV", see below), and
/// 4) it has no numeric annotation glyphs ("NAG", see below).  Reduced export
/// format is used for bulk storage of unannotated games.  It represents a
/// minimum level of standard conformance for a PGN exporting application.
pub struct ReducedPgn(pub PgnTagPairSection, pub PgnMoveTextSection);

impl ReducedPgn {
    /// 9.7: Alternative starting positions
    ///
    /// There are two tags defined for assistance with describing games that did
    /// not start from the usual initial array.
    ///
    ///
    /// 9.7.1: Tag: SetUp
    ///
    /// This tag takes an integer that denotes the "set-up" status of the game.
    /// A value of "0" indicates that the game has started from the usual
    /// initial array. A value of "1" indicates that the game started from a
    /// set-up position; this position is given in the "FEN" tag pair.  This
    /// tag must appear for a game starting with a set-up position.  If it
    /// appears with a tag value of "1", a FEN tag pair must also appear.
    ///
    ///
    /// 9.7.2: Tag: FEN
    ///
    /// This tag uses a string that gives the Forsyth-Edwards Notation for the
    /// starting position used in the game.  FEN is described in a later
    /// section of this document.  If a SetUp tag appears with a tag value
    /// of "1", the FEN tag pair is also required.
    pub fn start_position(&self) -> Result<Position, FenParseError> {
        let mut has_setup = false;
        let mut fen_string = None;

        for PgnTagPair(key, value) in &self.0.0 {
            match *key {
                "SetUp" if value == "1" => has_setup = true,
                "FEN" => fen_string = Some(value),
                _ => {}
            }
        }

        // Apply the custom FEN if provided
        if has_setup || fen_string.is_some() {
            if let Some(fen) = fen_string {
                return Ok(Position::from_fen(&fen)?);
            }
        }

        Ok(Position::start_position())
    }

    /// Returns the moves in SAN format, without move numbers or annotations.
    pub fn moves<'a>(&'a self) -> impl Iterator<Item = &'a str> {
        self.1.0.iter().filter_map(|move_info| {
            if let PgnMoveInfo::Move(san_str) = move_info {
                Some(san_str.as_str())
            }
            else {
                None
            }
        })
    }

    /// From<(Current Position, Subsequent Moves)>
    pub fn from_current_pos(mut pos: Position, moves: &[Move]) -> Self {
        let (move_tokens, game_result) = {
            let mut xs: Vec<PgnMoveInfo> = vec![];

            let game_result = PgnResultValue(pos.game_result());
            xs.insert(0, PgnMoveInfo::GameTerminationMarker(game_result));

            for mov in moves.iter().rev().cloned() {
                pos.unmake_move(mov);

                let stm = pos.get_turn();
                let fmc = pos.full_move();

                let san = format!("{}", SAN { context: &pos, mov });
                xs.insert(0, PgnMoveInfo::Move(san));

                if stm == colors::WHITE || xs[0].is_annotation() {
                    xs.insert(0, PgnMoveInfo::MoveNumberIndication(fmc, stm));
                }
            }

            let stm = pos.get_turn();
            let fmc = pos.full_move();
            if stm == colors::BLACK {
                xs.insert(0, PgnMoveInfo::MoveNumberIndication(fmc, stm));
            }

            (xs, game_result)
        };

        let fen = format!("{}", FenExport(&pos));

        let moves_sec = PgnMoveTextSection(move_tokens);
        let tags_sec = PgnTagPairSection(vec![
            PgnTagPair("FEN", fen),
            PgnTagPair("Result", game_result.to_string()),
        ]);

        Self(tags_sec, moves_sec)
    }

    /// From<(Initial Position, Subsequent Moves)>
    pub fn from_initial_pos(pos: &mut Position, moves: &[Move]) -> Self {
        let (move_tokens, game_result) = {
            let mut xs: Vec<PgnMoveInfo> = vec![];
            for mov in moves.iter().cloned() {
                let stm = pos.get_turn();
                let fmc = pos.full_move();

                // 8.2.2.2: Export format move number indications
                //
                // [...]
                //
                // All white move elements have a preceding move number indication.  A black
                // move element has a preceding move number indication only in
                // two cases: first, if there is intervening annotation or
                // commentary between the black move and the previous white
                // move; and second, if there is no previous white move in the
                // special case where a game starts from a position where Black is the active
                // player.
                // There are no other cases where move number indications appear in PGN export
                // format.
                if stm == colors::WHITE || xs.is_empty() || xs[xs.len() - 1].is_annotation() {
                    xs.push(PgnMoveInfo::MoveNumberIndication(fmc, stm));
                }

                let san = format!("{}", SAN { context: pos, mov });
                xs.push(PgnMoveInfo::Move(san));

                pos.make_move(mov);
            }

            let game_result = PgnResultValue(pos.game_result());
            xs.push(PgnMoveInfo::GameTerminationMarker(game_result));

            (xs, game_result)
        };

        for mov in moves.iter().rev().cloned() {
            pos.unmake_move(mov);
        }

        let fen = format!("{}", FenExport(pos));

        let moves_sec = PgnMoveTextSection(move_tokens);
        let tags_sec = PgnTagPairSection(vec![
            PgnTagPair("FEN", fen),
            PgnTagPair("Result", game_result.to_string()),
        ]);

        Self(tags_sec, moves_sec)
    }
}

impl fmt::Display for ReducedPgn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)?;
        self.1.fmt(f)
    }
}

#[derive(Debug, Error)]
pub enum PgnParseError {
    #[error("Malformed tag: missing closing bracket or space separator")]
    MalformedTag,

    #[error("Unknown or invalid tag key (Reduced PGN only supports STR and FEN/SetUp)")]
    UnknownTagKey,

    #[error("Invalid PGN move text token: {0}")]
    InvalidMoveTextToken(#[from] PgnMoveInfoParseError),
}

impl TryFrom<&mut Tokenizer<'_>> for ReducedPgn {
    type Error = PgnParseError;

    fn try_from(value: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        let mut tags = Vec::new();
        let mut moves = Vec::new();

        loop {
            value.skip_ws();

            // not splitting the tag and moves section by newlines, such that we can also
            // parse a flattened pgn, like in the uci input format.
            match value.peek_next_char() {
                Some('[') => {
                    value.consume_char(); // Consume the '['

                    let inner = value.take_until(']').ok_or(PgnParseError::MalformedTag)?;
                    let (key, val) = inner
                        .trim()
                        .split_once(' ')
                        .ok_or(PgnParseError::MalformedTag)?;

                    // Strictly enforce the Reduced PGN specification keys.
                    let static_key: &'static str = match key {
                        "Event" => "Event",
                        "Site" => "Site",
                        "Date" => "Date",
                        "Round" => "Round",
                        "White" => "White",
                        "Black" => "Black",
                        "Result" => "Result",
                        "FEN" => "FEN",
                        "SetUp" => "SetUp",
                        _ => return Err(PgnParseError::UnknownTagKey),
                    };

                    let clean_val = val.trim().trim_matches('"').to_string();

                    tags.push(PgnTagPair(static_key, clean_val));
                }
                Some(_) if let Some(token) = value.next_token() => {
                    moves.push(PgnMoveInfo::try_from(token)?)
                }
                // end of stream
                _ => {
                    break;
                }
            }
        }

        Ok(Self(PgnTagPairSection(tags), PgnMoveTextSection(moves)))
    }
}

pub struct PgnImport<'a, 'b>(pub &'a mut Tokenizer<'b>);

#[derive(Debug, Error)]
pub enum PgnImportError {
    #[error("Invalid FEN provided in PGN tags: {0}")]
    FenError(#[from] FenParseError),

    #[error("Error parsing PGN: {0}")]
    PgnParseError(#[from] PgnParseError),

    #[error("Invalid SAN move: {0}")]
    InvalidSanMove(#[from] MoveParseError),
}

impl<'a, 'b> TryFrom<PgnImport<'a, 'b>> for Position {
    type Error = PgnImportError;

    fn try_from(pgn: PgnImport<'a, 'b>) -> Result<Self, Self::Error> {
        let pgn = ReducedPgn::try_from(pgn.0)?;

        let mut position = pgn.start_position()?;

        for san in pgn.moves() {
            position.make_move(Move::from_san(san, &position)?);
        }

        Ok(position)
    }
}

// For PGN export format, there are no white space characters between the left
// bracket and the tag name, there are no white space characters between the tag
// value and the right bracket, and there is a single space character between
// the tag name and the tag value.
pub struct PgnTagPair<A, B>(pub A, pub B);

impl<A: fmt::Display, B: fmt::Display> fmt::Display for PgnTagPair<A, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // todo: we could check the length of fmt(self.a) or fmt(self.b), and introduce
        // a linebreak if that would overflow the 255 char limit.
        write!(f, "[{} \"{}\"]", self.0, self.1)
    }
}

// 8.1.1.7: The Result tag
//
// The Result field value is the result of the game.  It is always exactly the
// same as the game termination marker that concludes the associated movetext.
// It is always one of four possible values: "1-0" (White wins), "0-1" (Black
// wins), "1/2-1/2" (drawn game), and "*" (game still in progress, game
// abandoned, or result otherwise unknown).  Note that the digit zero is used in
// both of the first two cases; not the letter "O".
//
// All possible examples:
//
// [Result "0-1"]
// [Result "1-0"]
// [Result "1/2-1/2"]
// [Result "*"]
#[derive(Clone, Copy)]
pub struct PgnResultValue(pub Option<GameResult>);

impl fmt::Display for PgnResultValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0 {
            Some(GameResult::Win { relative_to: colors::WHITE }) => "1-0",
            Some(GameResult::Win { relative_to: colors::BLACK }) => "0-1",
            Some(GameResult::Draw) => "1/2-1/2",
            None => "*",
            _ => unreachable!(),
        })
    }
}

#[derive(Debug, Error)]
pub enum PgnResultValueParseError {
    #[error("Invalid game termination marker: {0}")]
    InvalidValue(#[from] ValueOutOfSetError<&'static str>),
}

impl TryFrom<&str> for PgnResultValue {
    type Error = PgnResultValueParseError;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "1-0" => Ok(Self(Some(GameResult::Win { relative_to: colors::WHITE }))),
            "0-1" => Ok(Self(Some(GameResult::Win { relative_to: colors::BLACK }))),
            "1/2-1/2" => Ok(Self(Some(GameResult::Draw))),
            "*" => Ok(Self(None)),
            _ => Err(ValueOutOfSetError::new("The value", &["1-0", "0-1", "1/2-1/2", "*"]).into()),
        }
    }
}

// Some tag values may be composed of a sequence of items.  For example, a
// consultation game may have more than one player for a given side.  When this
// occurs, the single character ":" (colon) appears between adjacent items.
// Because of this use as an internal separator in strings, the colon should not
// otherwise appear in a string.
pub struct ColonSeparated<T>(pub T);

impl<I> fmt::Display for ColonSeparated<I>
where
    for<'a> &'a I: IntoIterator<Item: fmt::Display>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.0.into_iter();

        if let Some(first) = iter.next() {
            write!(f, "{first}")?;
        }

        for item in iter {
            write!(f, ":{item}")?;
        }

        f.write_char(']')
    }
}

/// 8.1: Tag pair section
///
/// The tag pair section is composed of a series of zero or more tag pairs.
pub struct PgnTagPairSection(pub Vec<PgnTagPair<&'static str, String>>);

impl fmt::Display for PgnTagPairSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // PGN import format may have multiple tag pairs on the same line and may even
        // have a tag pair spanning more than a single line.  Export format requires
        // each tag pair to appear left justified on a line by itself; a single
        // empty line follows the last tag pair.
        for tag_pair in self.0.iter() {
            writeln!(f, "{tag_pair}")?;
        }
        writeln!(f)
    }
}

/// The move and a optional annotation.
pub enum PgnMoveInfo {
    /// The fullmove number and the current player.
    MoveNumberIndication(FullMoveCount, Color),

    /// The move in standard algebraic notation (SAN).
    Move(String),

    /// An annotation.
    Annotation(String),

    /// The game termination marker.
    GameTerminationMarker(PgnResultValue),
}

impl PgnMoveInfo {
    pub fn is_annotation(&self) -> bool {
        matches!(self, Self::Annotation(_))
    }
}

impl fmt::Display for PgnMoveInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MoveNumberIndication(fmc, stm) => match *stm {
                colors::WHITE => write!(f, "{fmc}."),
                colors::BLACK => write!(f, "{fmc}..."),
                _ => unreachable!(),
            },
            Self::Move(san) => write!(f, "{san}"),
            Self::Annotation(text) => write!(f, "({text})"),
            Self::GameTerminationMarker(result) => write!(f, "{result}"),
        }
    }
}

#[derive(Debug, Error)]
pub enum PgnMoveInfoParseError {
    #[error("Invalid game termination marker: {0}")]
    InvalidGameTerminationMarker(PgnResultValueParseError),

    #[error("Invalid move number indication: {0}")]
    InvalidFmc(#[from] FullMoveCountParseError),

    #[error("Unknown move info token: {0}")]
    UnknownMoveInfoTokenError(String),
}

impl TryFrom<&str> for PgnMoveInfo {
    type Error = PgnMoveInfoParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.ends_with("...") {
            let fmc = FullMoveCount::try_from(&value[..value.len() - 3])?;
            Ok(Self::MoveNumberIndication(fmc, colors::BLACK))
        }
        else if value.ends_with('.') {
            let fmc = FullMoveCount::try_from(&value[..value.len() - 1])?;
            Ok(Self::MoveNumberIndication(fmc, colors::WHITE))
        }
        else if value.starts_with('(') && value.ends_with(')') {
            let annotation = value[1..value.len() - 1].trim();
            Ok(Self::Annotation(annotation.to_string()))
        }
        else if let Ok(termination) = PgnResultValue::try_from(value) {
            Ok(Self::GameTerminationMarker(termination))
        }
        else {
            Ok(PgnMoveInfo::Move(value.to_string()))
        }
    }
}

/// 8.2: Movetext section
///
/// The movetext section is composed of chess moves, move number indications,
/// optional annotations, and a single concluding game termination marker.
pub struct PgnMoveTextSection(pub Vec<PgnMoveInfo>);

impl fmt::Display for PgnMoveTextSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // In PGN export format, tokens in the movetext are placed left justified on
        // successive text lines each of which has less than 80 printing characters.  As
        // many tokens as possible are placed on a line with the remainder appearing on
        // successive lines.  A single space character appears between any two adjacent
        // symbol tokens on the same line in the movetext.  As with the tag pair
        // section, a single empty line follows the last line of data to
        // conclude the movetext section.
        let mut width = 0;
        let mut iter = self.0.iter();

        if let Some(first) = iter.next() {
            let text = first.to_string();
            f.write_str(&text)?;
            width = text.len();
        }

        for token in iter {
            let text = token.to_string();
            if width + 1 + text.len() >= 80 {
                f.write_char('\n')?;
                f.write_str(&text)?;
                width = text.len();
            }
            else {
                f.write_char(' ')?;
                f.write_str(&text)?;
                width += 1 + text.len();
            }
        }

        writeln!(f)
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("board:\n{}\n", String::from(self)))?;
        f.write_str(&format!("turn: {:?}\n", self.get_turn()))?;
        f.write_str(&format!("ply: {:?}\n", self.ply()))?;
        Ok(())
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&String::from(self))
    }
}
impl From<&Position> for String {
    fn from(val: &Position) -> Self {
        let mut result = String::new();
        for rank in (0..=7).rev() {
            result.push_str(&(rank + 1).to_string());
            result.push(' ');
            for file in 0..=7 {
                let sq =
                    Square::from_c((File::try_from(file).unwrap(), Rank::try_from(rank).unwrap()));
                let piece = val.get_piece(sq);
                let c: char = piece.into();
                result.push(c);
                result.push(' ');
            }
            result.push('\n');
        }
        result.push_str("  a b c d e f g h");
        result
    }
}

pub struct FenImport<'a, 'b>(pub &'a mut Tokenizer<'b>);

#[derive(Debug, Error)]
pub enum FenParseError {
    #[error("Invalid rank: {0}")]
    InvalidRank(RankParseError<char>),

    #[error("Invalid piece: {0}")]
    InvalidPiece(PieceParseError),

    #[error("Invalid square reached.")]
    InvalidFenSquare,

    #[error("Failed to parse turn part: {0}")]
    TurnPart(ColorTokenizationError),

    #[error("Failed to parse castling availability part: {0}")]
    CastlingPart(CastlingSideTokenizationError),

    #[error("Failed to parse en passant capture square part: {0}")]
    EpSquarePart(EpTargetSquareTokenizationError),

    #[error("Failed to parse half moves to 50 move rule: {0}")]
    Plys50Part(PlyTokenizationError),

    #[error("Failed to parse full move clock part: {0}")]
    FullMoveCountPart(FullMoveCountTokenizationError),
}

impl<'a, 'b> TryFrom<FenImport<'a, 'b>> for Position {
    type Error = FenParseError;

    fn try_from(fen: FenImport<'a, 'b>) -> Result<Self, Self::Error> {
        let fen = fen.0;
        let mut position = Position::default();
        let mut sq = squares::H8.v() as i8;

        // 1. Piece placement
        for char in fen.skip_ws().chars() {
            match char {
                '/' => continue,
                '1'..='8' => {
                    let rank = Rank::try_from(char).map_err(Self::Error::InvalidRank)?;
                    let rank = i8::from(rank);
                    sq -= rank + 1
                }
                _ => {
                    let piece = Piece::try_from(char).map_err(Self::Error::InvalidPiece)?;
                    let pos_sq = Square::try_from(sq as u8)
                        .map_err(|_| Self::Error::InvalidFenSquare)?
                        .flip_h();

                    position.piece_info.put_piece(pos_sq, piece);
                    sq -= 1;
                }
            }
            if sq < squares::A1.v() as i8 {
                break;
            }
        }

        let turn = Turn::try_from(&mut *fen).map_err(Self::Error::TurnPart)?;
        let mut state = StateInfo {
            // 2. Side to move
            turn,

            // 3. Castling ability
            castling: CastlingRights::try_from(&mut *fen).map_err(Self::Error::CastlingPart)?,

            // 4. En passant target square
            ep_capture_square: EpCaptureSquare::from((
                EpTargetSquare::try_from(fen.skip_ws()).map_err(Self::Error::EpSquarePart)?,
                !turn,
            )),

            // 5. Halfmove clock
            plys50: Ply::try_from(fen.skip_ws()).map_err(Self::Error::Plys50Part)?,

            // 6. Fullmove counter
            ply: Ply::from((
                FullMoveCount::try_from(fen.skip_ws()).map_err(Self::Error::FullMoveCountPart)?,
                turn,
            )),

            ..Default::default()
        };

        state.init(&position);
        position.state = StateStack::new(state);
        position.state.get_current_mut().key = zobrist::Hash::from(&position);

        Ok(position)
    }
}

pub struct EpdImport<'a, 'b>(pub &'a mut Tokenizer<'b>);

pub struct EpdLine(
    pub PiecePlacementInfo<'static>,
    pub Turn,
    pub CastlingRights,
    pub EpTargetSquare,
);

// ...
