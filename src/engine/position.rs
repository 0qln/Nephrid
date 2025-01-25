use core::fmt;
use std::ptr::NonNull;

use crate::{
    engine::{
        bitboard::Bitboard, castling::CastlingRights, color::Color, coordinates::{File, Rank, Square}, fen::Fen, r#move::Move, move_iter::{bishop, king, knight, pawn, queen, rook}, piece::{Piece, PieceType}, turn::Turn, zobrist
    }, misc::{ConstFrom, ParseError}
};

use super::{castling::CastlingSide, coordinates::{EpCaptureSquare, EpTargetSquare}, r#move::MoveFlag, move_iter::{bishop::Bishop, queen::Queen, rook::Rook, sliding_piece::Attacks}, piece::PromoPieceType, ply::{FullMoveCount, Ply}};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CheckState {
    #[default]
    None,
    Single,
    Double
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct StateInfo {
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
    pub has_threefold_repetition: bool
}

impl StateInfo {
    /// Initiate checkers, blockers, nstm_attacks, check_state
    pub fn init(&mut self, pos: &Position) {
        let stm = self.turn;
        let nstm = !stm;
        let king = pos.get_bitboard(PieceType::KING, stm);
        let occupancy = pos.get_occupancy();
        let enemies = pos.get_color_bb(nstm);
        (self.nstm_attacks, self.checkers) = {
            enemies.fold((Bitboard::empty(), Bitboard::empty()), |acc, enemy_sq| {
                let enemy = pos.get_piece(enemy_sq);     
                let enemy_attacks = match enemy.piece_type() {
                    PieceType::PAWN => pawn::lookup_attacks(enemy_sq, nstm),
                    PieceType::KNIGHT => knight::lookup_attacks(enemy_sq),
                    PieceType::BISHOP => Bishop::lookup_attacks(enemy_sq, occupancy),
                    PieceType::ROOK => Rook::lookup_attacks(enemy_sq, occupancy),
                    PieceType::QUEEN => Queen::lookup_attacks(enemy_sq, occupancy),
                    PieceType::KING => king::lookup_attacks(enemy_sq),
                    _ => unreachable!("We are iterating the squares which contain enemies. PieceType::NONE should not be here."),
                };
                (
                    acc.0 | enemy_attacks,
                    match enemy_attacks & king {
                        Bitboard { v: 0 } => acc.1,
                        _ => acc.1 | Bitboard::from_c(enemy_sq)
                    }
                )
            })
        };
        
        if let Some(king_sq) = king.lsb() {
            let x_ray_checkers = pos.get_x_ray_checkers(king_sq, enemies);
            self.blockers = x_ray_checkers.fold(Bitboard::empty(), |acc, x_ray_checker| {
                let between_squares = Bitboard::between(x_ray_checker, king_sq);
                let between_occupancy = occupancy & between_squares;
                between_occupancy.pop_cnt_eq_1().then_some(acc | between_squares).unwrap_or(acc)
            });
        }
        
        self.check_state = match self.checkers.pop_cnt() {
            1 => CheckState::Single,
            2 => CheckState::Double,
            _ => CheckState::None
        };
    }
}

#[derive(Clone)]
struct StateStack {
    states: Vec::<StateInfo>,
    current: usize,
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
        Self {
            states: vec,
            current: 0
        }
    }

    /// Returns a reference to the current state.
    #[inline]
    pub fn get_current(&self) -> &StateInfo {
        // Safety: The current index is always in range
        unsafe {
            self.states.get_unchecked(self.current)
        }
    }     
    
    /// Returns a mutable reference to the current state.
    #[inline]
    pub fn get_current_mut(&mut self) -> &mut StateInfo {
        // Safety: The current index is always in range
        unsafe {
            self.states.get_unchecked_mut(self.current)
        }
    }
    
    /// Returns a pointer to the current state.
    #[inline]
    pub fn get_current_ptr(&mut self) -> NonNull<StateInfo> {
        NonNull::from_ref(self.get_current_mut())
    }
    
    /// Returns the pushed state.
    #[inline]
    pub fn get_next(&mut self, new: fn(&StateInfo) -> StateInfo) -> NonNull<StateInfo> {
        let next = self.current + 1;
        
        // self.current can only ever be one greater than the length of the vector.
        assert!(self.states.len() >= next);

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

#[derive(Clone)]
pub struct Position {
    c_bitboards: [Bitboard; 2],
    t_bitboards: [Bitboard; 7],
    pieces: [Piece; 64],
    piece_counts: [i8; 14],
    state: StateStack,
}

impl Default for Position {
    /// Returns an empty position.
    fn default() -> Self {
        Self {
            c_bitboards: Default::default(),
            t_bitboards: Default::default(),
            pieces: [Piece::default(); 64],
            piece_counts: Default::default(),
            state: Default::default(),
        }
    }
}

impl Position {
    #[inline]
    pub fn get_bitboard(&self, piece_type: PieceType, color: Color) -> Bitboard {
        self.get_color_bb(color) & self.get_piece_bb(piece_type)
    }

    #[inline]
    pub fn get_color_bb(&self, color: Color) -> Bitboard {
        // Safety:
        // It's not possible to safely create an instance of Color,
        // without checking that the value is in range.
        unsafe {
            *self.c_bitboards.get_unchecked(color.v() as usize)
        }
    }
    
    #[inline]
    fn get_color_bb_mut(&mut self, color: Color) -> &mut Bitboard {
        // Safety:
        // It's not possible to safely create an instance of Color,
        // without checking that the value is in range.
        unsafe {
            self.c_bitboards.get_unchecked_mut(color.v() as usize)
        }
    }

    #[inline]
    pub fn get_piece_bb(&self, piece_type: PieceType) -> Bitboard {
        // Safety:
        // It's not possible to safely create an instance of PieceType,
        // without checking that the value is in range.
        unsafe {
            *self.t_bitboards.get_unchecked(piece_type.v() as usize)
        }
    }
    
    #[inline]
    fn get_piece_bb_mut(&mut self, piece_type: PieceType) -> &mut Bitboard {
        // Safety:
        // It's not possible to safely create an instance of PieceType,
        // without checking that the value is in range.
        unsafe {
            self.t_bitboards.get_unchecked_mut(piece_type.v() as usize)
        }
    }
    
    #[inline]
    pub fn get_occupancy(&self) -> Bitboard {
        self.get_color_bb(Color::WHITE) | self.get_color_bb(Color::BLACK)
    }
    
    #[inline]
    pub fn get_piece(&self, sq: Square) -> Piece {
        // Safety: sq is in range 0..64
        unsafe {
            *self.pieces.get_unchecked(sq.v() as usize)
        }
    }
    
    #[inline]
    fn get_piece_mut(&mut self, sq: Square) -> &mut Piece {
        // Safety: sq is in range 0..64
        unsafe {
            self.pieces.get_unchecked_mut(sq.v() as usize)
        }
    }
    
    #[inline]
    pub fn get_piece_count(&self, piece: Piece) -> i8 {
        // Safety:
        // It's not possible to safely create an instance of Piece,
        // without checking that the value is in range.
        unsafe {
            *self.piece_counts.get_unchecked(piece.v() as usize)
        }
    }
    
    #[inline]
    fn get_piece_count_mut(&mut self, piece: Piece) -> &mut i8 {
        // Safety:
        // It's not possible to safely create an instance of Piece,
        // without checking that the value is in range.
        unsafe {
            self.piece_counts.get_unchecked_mut(piece.v() as usize)
        }
    }

    #[inline]
    pub fn get_turn(&self) -> Turn {
        self.state.get_current().turn
    }
    
    #[inline]
    pub fn get_ep_capture_square(&self) -> EpCaptureSquare {
        self.state.get_current().ep_capture_square
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
    
    /// Returns the X-Ray checkers for the given king.
    /// X-Ray checkers are pieces which attack a king 
    /// through zero or more pieces.
    #[inline]
    pub fn get_x_ray_checkers(&self, king: Square, enemies: Bitboard) -> Bitboard {
        let rooks = self.get_piece_bb(PieceType::ROOK);
        let bishops = self.get_piece_bb(PieceType::BISHOP);
        let queens = self.get_piece_bb(PieceType::QUEEN);
        let rook_checkers = rook::compute_attacks_0_occ(king) & (rooks | queens);
        let bishop_checkers = bishop::compute_attacks_0_occ(king) & (bishops | queens);
        (rook_checkers | bishop_checkers) & enemies
    }

    #[inline]
    fn put_piece(&mut self, sq: Square, piece: Piece) {
        let target = Bitboard::from_c(sq);
        *self.get_piece_bb_mut(piece.piece_type()) |= target;
        *self.get_color_bb_mut(piece.color()) |= target;
        *self.get_piece_mut(sq) = piece;
        *self.get_piece_count_mut(piece) += 1;
    }
    
    /// # Safety
    /// This is unsafe, because it allows you to modify the internal
    /// representation, without updating the state.
    /// 
    /// This pub, because it is used for benchmarking.
    #[inline(never)]
    pub unsafe fn put_piece_unsafe(&mut self, sq: Square, piece: Piece) { 
        self.put_piece(sq, piece) 
    }
    
    #[inline]
    fn remove_piece(&mut self, sq: Square) {
        let target = Bitboard::from_c(sq);
        let piece = self.get_piece(sq);
        *self.get_piece_bb_mut(piece.piece_type()) ^= target;
        *self.get_color_bb_mut(piece.color()) ^= target;
        *self.get_piece_mut(sq) = Piece::default();
        *self.get_piece_count_mut(piece) -= 1;
    }  
    
    /// # Safety
    /// This is unsafe, because it allows you to modify the internal
    /// representation, without updating the state.
    /// 
    /// This pub, because it is used for benchmarking.
    #[inline(never)]
    pub unsafe fn remove_piece_unsafe(&mut self, sq: Square) { 
        self.remove_piece(sq) 
    }
    
    #[inline] 
    fn move_piece(&mut self, from: Square, to: Square) {
        debug_assert!(self.get_piece(from) != Piece::default());
        debug_assert!(self.get_piece(to) == Piece::default());
        let piece = self.get_piece(from);
        let from_to = Bitboard::from_c(from) | Bitboard::from_c(to);
        *self.get_color_bb_mut(piece.color()) ^= from_to;
        *self.get_piece_bb_mut(piece.piece_type()) ^= from_to;
        *self.get_piece_mut(from) = Piece::default();
        *self.get_piece_mut(to) = piece;
    }
    
    /// # Safety
    /// This is unsafe, because it allows you to modify the internal
    /// representation, without updating the state.
    /// 
    /// This pub, because it is used for benchmarking.
    #[inline(never)]
    pub unsafe fn move_piece_unsafe(&mut self, from: Square, to: Square) { 
        self.move_piece(from, to) 
    }
    
    /// Makes a move on the board.
    pub fn make_move(&mut self, m: Move) {
        let us = self.get_turn();
        let (from, to, flag) = m.into();
        let moving_piece = self.get_piece(from);
        let target_piece = self.get_piece(to);

        // Safety: During the lifetime of this pointer, no other pointer
        // reads or writes to the memory location of the next state. 
        let next_state = unsafe { 
            self.state.get_next(|prev| {
                StateInfo {
                    // These don't change across leafes on the same depth...
                    ply: prev.ply + 1,
                    turn: !prev.turn,
                    ..Default::default()
                }
            }).as_mut() 
        };

        // These might change across leafes on the same depth, so the 
        // need to be reinitialized for each leaf.
        next_state.castling = self.state.get_current().castling;
        next_state.plys50 = self.state.get_current().plys50 + 1;
        next_state.ep_capture_square = EpCaptureSquare::default();
        next_state.key = self.state.get_current().key;
        next_state.key.toggle_ep_square(self.state.get_current().ep_capture_square);
        next_state.key.toggle_turn();
        next_state.captured_piece = Piece::default();
        
        // captures
        if flag.is_capture() {
            let captured_piece = match flag {
                MoveFlag::EN_PASSANT => Piece::from_c((!us, PieceType::PAWN)),
                _ => target_piece,
            };
            
            let captured_sq = match flag {
                MoveFlag::EN_PASSANT => {
                    // Safety:
                    // If the move is an en passant, the `to` square is on the 3rd or 6th rank.
                    unsafe {
                        let target_sq = EpTargetSquare::try_from(to).unwrap_unchecked();
                        EpCaptureSquare::from((target_sq, !us)).v().unwrap_unchecked()
                    }
                }
                _ => to,
            };
            
            self.remove_piece(captured_sq);
            
            next_state.captured_piece = captured_piece;
            next_state.key.toggle_piece_sq(captured_sq, captured_piece);
            next_state.plys50 = Ply { v: 0 };
            update_castling(captured_sq, !us, &mut next_state.castling);
        }
        
        // move the piece
        self.move_piece(from, to);
        next_state.key.move_piece_sq(from, to, moving_piece);
        
        match moving_piece.piece_type() {
            // castling
            PieceType::KING => {
                next_state.castling.set_false(CastlingSide::QUEEN_SIDE, us);
                next_state.castling.set_false(CastlingSide::KING_SIDE, us);
                match flag {
                    MoveFlag::KING_CASTLE => {
                        let rank = Rank::from_c(to);
                        let rook_from = Square::from_c((File::H, rank));
                        let rook_to   = Square::from_c((File::F, rank));
                        let rook = self.get_piece(rook_from);
                        self.move_piece(rook_from, rook_to);
                        next_state.key.move_piece_sq(rook_from, rook_to, rook);
                    },
                    MoveFlag::QUEEN_CASTLE => {
                        let rank = Rank::from_c(to);
                        let rook_from = Square::from_c((File::A, rank));
                        let rook_to   = Square::from_c((File::D, rank));
                        let rook = self.get_piece(rook_from);
                        self.move_piece(rook_from, rook_to);
                        next_state.key.move_piece_sq(rook_from, rook_to, rook);
                    },
                    _ => (),
                }
            }
            PieceType::ROOK => {
                update_castling(from, us, &mut next_state.castling);
            }
            // pawns
            PieceType::PAWN => {
                match flag.v() {
                    MoveFlag::DOUBLE_PAWN_PUSH_C => {
                        // Safety: A double pawn push destination square is the definition of 
                        // an en passant square.
                        next_state.ep_capture_square = unsafe { EpCaptureSquare::try_from(to).unwrap_unchecked() };
                        next_state.key.toggle_ep_square(next_state.ep_capture_square);
                    }
                    MoveFlag::PROMOTION_KNIGHT_C..=MoveFlag::CAPTURE_PROMOTION_QUEEN_C => {
                        // Safety: We just checked, that the flag is in a valid range.
                        let promo_t = unsafe { PromoPieceType::try_from(flag).unwrap_unchecked() };
                        let promo = Piece::from_c((us, promo_t));
                        self.remove_piece(to);
                        next_state.key.toggle_piece_sq(to, moving_piece);
                        self.put_piece(to, promo);
                        next_state.key.toggle_piece_sq(to, promo);
                    }
                    _ => (),
                }
                
                next_state.plys50 = Ply { v: 0 };                
            }
            _ => ()
        }
        
        // update castling rights in the hash, if they have changed.
        if next_state.castling != self.state.get_current().castling {
            next_state.key
                .toggle_castling(self.state.get_current().castling)
                .toggle_castling(next_state.castling);
        }
        
        next_state.init(self);
        self.state.incr();

        #[inline(always)]
        const fn update_castling(sq: Square, c: Color, cr: &mut CastlingRights) {
            match c {
                Color::WHITE => match sq {
                    Square::A1 => cr.set_false(CastlingSide::QUEEN_SIDE, Color::WHITE),
                    Square::H1 => cr.set_false(CastlingSide::KING_SIDE, Color::WHITE),
                    _ => ()
                },
                Color::BLACK => match sq {
                    Square::A8 => cr.set_false(CastlingSide::QUEEN_SIDE, Color::BLACK),
                    Square::H8 => cr.set_false(CastlingSide::KING_SIDE, Color::BLACK),
                    _ => ()
                }
                _ => ()
            }
        }
    }

    pub fn unmake_move(&mut self, m: Move) {
        let us = !self.get_turn();
        let (from, to, flag) = m.into();
        
        // Safety: During the lifetime of this pointer, no other pointer
        // writes to the memory location of the popped state. 
        let popped_state = unsafe { self.state.pop_current().as_ref() };

        match flag.v() {
            // castling
            MoveFlag::KING_CASTLE_C => {
                let rank = Rank::from_c(to);
                let rook_from = Square::from_c((File::H, rank));
                let rook_to   = Square::from_c((File::F, rank));
                self.move_piece(rook_to, rook_from);
            },
            MoveFlag::QUEEN_CASTLE_C => {
                let rank = Rank::from_c(to);
                let rook_from = Square::from_c((File::A, rank));
                let rook_to   = Square::from_c((File::D, rank));
                self.move_piece(rook_to, rook_from);
            },
            // promotions
            MoveFlag::PROMOTION_KNIGHT_C..=MoveFlag::CAPTURE_PROMOTION_QUEEN_C => {
                let pawn = Piece::from_c((us, PieceType::PAWN));
                self.remove_piece(to);
                self.put_piece(to, pawn);
            }
            _ => {}
        }
        
        // move the piece
        self.move_piece(to, from);
        
        // captures
        let captured_piece = popped_state.captured_piece;
        if captured_piece != Piece::default() {
            let captured_sq = match flag {
                MoveFlag::EN_PASSANT => {
                    // Safety:
                    // If the move is an en passant, the `to` square is on the 3rd or 6th rank.
                    unsafe {
                        let target_sq = EpTargetSquare::try_from(to).unwrap_unchecked();
                        EpCaptureSquare::from((target_sq, !us)).v().unwrap_unchecked()
                    }
                }
                _ => to,
            };
            
            self.put_piece(captured_sq, captured_piece);
        }
    }

    pub fn start_position() -> Self {
        // Safety: This FEN string is valid
        unsafe {
            Position::try_from(
                &mut Fen::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            ).unwrap_unchecked()
        }
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let str: String = self.into();
        f.write_str(&str)
    }
}

impl From<&Position> for String {
    fn from(val: &Position) -> Self {
        let mut result = String::new();
        for rank in (0..=7).rev() {
            result.push_str(&(rank + 1).to_string());
            result.push(' ');
            for file in 0..=7 {
                let sq = Square::from_c((
                    File::try_from(file).unwrap(), 
                    Rank::try_from(rank).unwrap()
                ));
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

impl TryFrom<&mut Fen<'_>> for Position {
    type Error = ParseError;
    
    fn try_from(fen: &mut Fen<'_>) -> Result<Self, Self::Error> {
        let mut position = Position::default();
        let mut sq = Square::H8.v() as i8;

        // 1. Piece placement
        for char in fen.iter_token() {
            match char {
                '/' => continue,
                '1'..='8' => sq -= char.to_digit(10).ok_or(ParseError::InputOutOfRange(Box::new(char)))? as i8,        
                _ => {
                    let piece = Piece::try_from(char)?; 
                    let pos_sq = Square::try_from(sq as u8)?.flip_h();
                    position.put_piece(pos_sq, piece);
                    sq -= 1;
                }
            }
            if sq < Square::A1.v() as i8 {
                break;
            }
        }
        
        let turn = Turn::try_from(fen.iter_token().next().ok_or(ParseError::MissingInput)?)?;
        let mut state = StateInfo {
            // 2. Side to move
            turn,
            // 3. Castling ability
            castling: CastlingRights::try_from(&mut *fen)?,
            // 4. En passant target square
            ep_capture_square: EpCaptureSquare::from((EpTargetSquare::try_from(fen.iter_token())?, !turn)),
            // 5. Halfmove clock
            plys50: Ply::try_from(fen.iter_token())?,
            // 6. Fullmove counter
            ply: Ply::from((FullMoveCount::try_from(fen.iter_token())?, turn)),

            // TODO: init zobrist hash

            ..Default::default()
        };
        state.init(&position);
        position.state = StateStack::new(state);
        
        Ok(position)
    }
}