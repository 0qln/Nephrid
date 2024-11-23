use std::{cell::{LazyCell, OnceCell}, collections::{linked_list::{Cursor, CursorMut}, LinkedList}, iter::Once, ptr::NonNull};

use crate::{
    engine::{
        bitboard::Bitboard, 
        castling::CastlingRights, 
        color::Color, 
        coordinates::{File, Rank, Square}, 
        fen::Fen, 
        r#move::Move, 
        piece::{Piece, PieceType}, 
        turn::Turn, 
        zobrist
    }, misc::{ConstFrom, ParseError}
};

use super::{castling::CastlingSide, r#move::MoveFlag, piece::PromoPieceType, ply::{FullMoveCount, Ply}};

#[derive(Default, Clone)]
struct StateInfo {
    pub checkers: Bitboard,
    pub blockers: Bitboard,
    pub nstm_attacks: Bitboard,
    pub plys50: Ply,
    pub ply: Ply,
    pub ep_square: Option<Square>,
    pub castling: CastlingRights,
    pub captured_piece: Piece,
    pub key: zobrist::Hash,
    pub has_threefold_repetition: bool
}

impl StateInfo {
    pub fn init(&mut self, position: &Position) {
        let us = position.turn;
        let king = position.get_bitboard(PieceType::KING, us);
        let king_sq = king.lsb();
        let occupancy = position.get_occupancy();
        for enemy_sq in position.get_color_bb(!us) {
            let enemy = position.get_piece(enemy_sq);     

        }
        
        todo!()
    }
}

#[derive(Clone)]
struct StateStack {
    states: Vec::<StateInfo>,
    current: usize,
}

impl Default for StateStack {
    fn default() -> Self {
        let mut vec = Vec::with_capacity(32);
        vec.push(StateInfo::default());
        Self {
            states: vec,
            current: 0,
        }
    }
}

impl StateStack {
    #[inline]
    pub fn get(&self) -> &StateInfo {
        // Safety: The current index is always in range
        unsafe {
            self.states.get_unchecked(self.current)
        }
    }     
    
    #[inline]
    pub fn get_mut(&mut self) -> &mut StateInfo {
        // Safety: The current index is always in range
        unsafe {
            self.states.get_unchecked_mut(self.current)
        }
    }
    
    /// Returns the pushed state.
    #[inline]
    pub fn push(&mut self) -> NonNull<StateInfo> {
        self.current += 1;
        while self.states.len() <= self.current {
            self.states.push(StateInfo::default());
        }
        // Safety: The current index is always in range
        NonNull::from_ref(unsafe { self.states.get_unchecked(self.current) })
    }
    
    /// Returns the popped state.
    #[inline]
    pub fn pop(&mut self) -> NonNull<StateInfo> {
        // Safety: The current index is always in range
        let ret = NonNull::from_ref(unsafe { self.states.get_unchecked(self.current) });
        self.current = self.current.checked_sub(1).unwrap_or(0);
        ret
    }
}

#[derive(Clone)]
pub struct Position {
    c_bitboards: [Bitboard; 2],
    t_bitboards: [Bitboard; 7],
    pieces: [Piece; 64],
    piece_counts: [i8; 14],
    turn: Turn,
    state: StateStack,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            c_bitboards: [Bitboard::empty(); 2],
            t_bitboards: [Bitboard::empty(); 7],
            pieces: [Piece::default(); 64],
            piece_counts: [0; 14],
            turn: Turn::WHITE,
            state: StateStack::default()
        }
    }
}

impl Position {
    #[inline]
    pub fn get_bitboard(&self, piece_type: PieceType, color: Color) -> Bitboard {
        self.c_bitboards[color.v() as usize] & self.t_bitboards[piece_type.v() as usize]
    }

    #[inline]
    pub fn get_color_bb(&self, color: Color) -> Bitboard {
        self.c_bitboards[color.v() as usize]
    }

    #[inline]
    pub fn get_piece_bb(&self, piece_type: PieceType) -> Bitboard {
        self.t_bitboards[piece_type.v() as usize]
    }
    
    #[inline]
    pub fn get_occupancy(&self) -> Bitboard {
        self.get_color_bb(Color::WHITE) | self.get_color_bb(Color::BLACK)
    }
    
    #[inline]
    pub fn get_piece(&self, sq: Square) -> Piece {
        self.pieces[sq.v() as usize]
    }

    #[inline]
    pub fn get_turn(&self) -> Turn {
        self.turn
    }
    
    #[inline]
    pub fn get_ep_square(&self) -> Option<Square> {
        self.state.get().ep_square
    }
    
    #[inline]
    pub fn get_castling(&self) -> CastlingRights {
        self.state.get().castling
    }
    
    #[inline]
    pub fn get_key(&self) -> zobrist::Hash {
        self.state.get().key
    }

    #[inline]
    fn put_piece(&mut self, sq: Square, piece: Piece) {
        let target = Bitboard::from_c(sq);
        self.t_bitboards[piece.piece_type().v() as usize] |= target;
        self.c_bitboards[piece.color().v() as usize] |= target;
        self.pieces[sq.v() as usize] = piece;
        self.piece_counts[piece.piece_type().v() as usize] += 1;
    }
    
    #[inline]
    fn remove_piece(&mut self, sq: Square) {
        let target = Bitboard::from_c(sq);
        let piece = self.get_piece(sq);
        self.t_bitboards[piece.piece_type().v() as usize] ^= target;
        self.c_bitboards[piece.color().v() as usize] ^= target;
        self.pieces[sq.v() as usize] = Piece::default();
        self.piece_counts[self.get_piece(sq).piece_type().v() as usize] -= 1;
    }  
    
    #[inline]
    fn move_piece(&mut self, from: Square, to: Square) {
        let piece = self.get_piece(from);
        let from_to = Bitboard::from_c(from) ^ Bitboard::from_c(to);
        self.c_bitboards[piece.color().v() as usize] ^= from_to;
        self.t_bitboards[piece.piece_type().v() as usize] ^= from_to;
        self.pieces[from.v() as usize] = Piece::default();
        self.pieces[to.v() as usize] = piece;
    }
    
    // todo: maybe this can be sped up by passing in the color of the moving side as a const generic param.
    /// Makes a move on the board.
    pub fn make_move(&mut self, m: Move) {
        let us = self.get_turn();
        self.turn = !us;
        let (from, to, flag) = m.into();
        let moving_piece = self.get_piece(from);
        let target_piece = self.get_piece(to);
        
        // Safety: During the lifetime of this pointer, no other pointer
        // reads or writes to the memory location of the next state. 
        let next_state = unsafe { self.state.push().as_mut() };

        next_state.castling = self.state.get().castling;
        next_state.castling = self.state.get().castling;
        next_state.plys50 = self.state.get().plys50 + 1;
        next_state.ep_square = None;
        next_state.key = self.state.get().key;
        next_state.key.toggle_ep_square(self.state.get().ep_square);
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
                    // If the move is an en passant, the `to` square is 
                    // on the 3rd or 6th rank. For any of the sq values
                    // on those ranks, the formula yields a valid square.
                    unsafe {
                        // todo: test and move this logic somewhere else
                        Square::from_v(to.v() + (us.v() *  2 - 1) * 8) 
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
                       next_state.ep_square = Some(
                            // Safety:
                            // To `to` sq can only ever be on the 4th or 5th rank.
                            // For any square in on those ranks, the formula yields a valid square.
                            unsafe {
                                // todo: test and move this logic somewhere else
                                Square::from_v(to.v() + (us.v() *  2 - 1) * 8) 
                            }
                        );
                        next_state.key.toggle_ep_square(Some(to));
                    }
                    MoveFlag::PROMOTION_KNIGHT_C..MoveFlag::CAPTURE_PROMOTION_QUEEN_C => {
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
        if next_state.castling != self.state.get().castling {
            next_state.key
                .toggle_castling(self.state.get().castling)
                .toggle_castling(next_state.castling);
        }

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
        let popped_state = unsafe { self.state.pop().as_ref() };

        let captured_piece = popped_state.captured_piece;

        self.turn = us;

        // promotions
        if flag.is_promo() {
            let pawn = Piece::from_c((us, PieceType::PAWN));
            self.remove_piece(to);
            self.put_piece(to, pawn);
        }

        // castling
        match flag {
            MoveFlag::KING_CASTLE => {
                let rank = Rank::from_c(to);
                let rook_from = Square::from_c((File::H, rank));
                let rook_to   = Square::from_c((File::F, rank));
                self.move_piece(rook_to, rook_from);
            },
            MoveFlag::QUEEN_CASTLE => {
                let rank = Rank::from_c(to);
                let rook_from = Square::from_c((File::A, rank));
                let rook_to   = Square::from_c((File::D, rank));
                self.move_piece(rook_to, rook_from);
            },
            _ => ()
        }
        
        // move the piece
        self.move_piece(to, from);
        
        // captures
        if captured_piece != Piece::default() {
            let captured_sq = match flag {
                MoveFlag::EN_PASSANT => {
                    // Safety:
                    // If the move was an en passant, the `to` square is 
                    // on the 3rd or 6th rank. For any of the sq values
                    // on those ranks, the formula yields a valid square.
                    unsafe {
                        Square::from_v(to.v() + (us.v() * 2 - 1) * 8)
                    }
                },
                _ => to
            };
            
            self.put_piece(captured_sq, captured_piece);
        }
    }

    pub fn start_position() -> Self {
        // Safety: This FEN string is valid
        unsafe {
            Position::try_from(
                &mut Fen::new(&"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            ).unwrap_unchecked()
        }
    }
}

impl Into<String> for &Position {
    fn into(self) -> String {
        let mut result = String::new();
        for rank in (0..=7).rev() {
            result.push_str(&(rank + 1).to_string());
            result.push(' ');
            for file in 0..=7 {
                let sq = Square::from_c((
                    File::try_from(file).unwrap(), 
                    Rank::try_from(rank).unwrap()
                ));
                let piece = self.get_piece(sq);
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

        // Position
        for char in fen.iter_token() {
            match char {
                '/' => continue,
                '1'..='8' => sq -= char.to_digit(10).ok_or(ParseError::InputOutOfRange(Box::new(char)))? as i8,        
                _ => {
                    let piece = Piece::try_from(char)?; 
                    let pos_sq = Square::try_from(sq as u8)?.mirror();
                    position.put_piece(pos_sq, piece);
                    sq -= 1;
                }
            }
            if sq < Square::A1.v() as i8 {
                break;
            }
        }
        
        // Turn
        let char = match fen.iter_token().next() {
            None => return Err(ParseError::MissingInput),
            Some(c) => c,
        };
        position.turn = Turn::try_from(char)?;        
        
        let mut state = StateInfo {
            castling: CastlingRights::try_from(&mut *fen)?,
            ep_square: Option::<Square>::try_from(fen.iter_token())?,
            plys50: Ply::try_from(fen.iter_token())?,
            ply: Ply::from((FullMoveCount::try_from(fen.iter_token())?, position.turn)),
            ..Default::default()
        };

        // TODO: init zobrist hash
        
        // state.init(&position);
        
        Ok(position)
    }
}
