use std::cell::OnceCell;

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
pub struct PositionInfo {
    pub next: OnceCell<Box<PositionInfo>>,
    pub prev: Option<Box<PositionInfo>>,
    pub checkers: Bitboard,
    pub blockers: Bitboard,
    pub nstm_attacks: Bitboard,
    pub plys50: Ply,
    pub ply: Ply,
    pub ep_square: Option<Square>,
    pub castling_rights: CastlingRights,
    pub captured_piece: Piece,
    pub key: zobrist::Hash,
    pub has_threefold_repetition: bool
}

// impl Clone for PositionInfo {
//     fn clone(&self) -> Self {
//         Self {
//             next: ::default(),
//             prev: self.prev.clone(),
//             checkers: self.checkers,
//             blockers: self.blockers,
//             nstm_attacks: self.nstm_attacks,
//             plys50: self.plys50,
//             ply: self.ply,
//             ep_square: self.ep_square,
//             castling_rights: self.castling_rights,
//             captured_piece: self.captured_piece,
//             key: self.key,
//             has_threefold_repetition: self.has_threefold_repetition
//         }
//     }
// }

impl PositionInfo {
    pub fn init(&mut self, position: &Position) {
        let us = position.turn;
        let king = position.get_bitboard(PieceType::KING, us);
        let king_sq = king.lsb();
        let occupancy = position.get_occupancy();
        for enemy_sq in position.get_c_bitboard(!us) {
            let enemy = position.get_piece(enemy_sq);     

        }
        
        todo!()
    }
}

#[derive(Clone)]
struct Pieces([Piece; 64]);

impl Default for Pieces {
    fn default() -> Self {
        Self([Piece::default(); 64])
    }
}

#[derive(Clone, Default)]
pub struct Position {
    c_bitboards: [Bitboard; 2],
    t_bitboards: [Bitboard; 7],
    pieces: Pieces,
    piece_counts: [i8; 14],
    turn: Turn,
    state_stack: PositionInfo
}

impl Position {
    #[inline]
    pub fn get_bitboard(&self, piece_type: PieceType, color: Color) -> Bitboard {
        self.c_bitboards[color.v() as usize] & self.t_bitboards[piece_type.v() as usize]
    }
    
    #[inline]
    pub fn get_c_bitboard(&self, color: Color) -> Bitboard {
        self.c_bitboards[color.v() as usize]
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
        self.get_c_bitboard(Color::WHITE) | self.get_c_bitboard(Color::BLACK)
    }
    
    #[inline]
    pub fn get_piece(&self, sq: Square) -> Piece {
        self.pieces.0[sq.v() as usize]
    }

    #[inline]
    pub fn get_turn(&self) -> Turn {
        self.turn
    }
    
    #[inline]
    pub fn get_ep_square(&self) -> Option<Square> {
        self.state_stack.ep_square
    }
    
    #[inline]
    pub fn get_castling(&self) -> CastlingRights {
        self.state_stack.castling_rights
    }
    
    #[inline]
    pub fn put_piece(&mut self, sq: Square, piece: Piece) {
        let target = Bitboard::from_c(sq);
        self.t_bitboards[piece.piece_type().v() as usize] |= target;
        self.c_bitboards[piece.color().v() as usize] |= target;
        self.pieces.0[sq.v() as usize] = piece;
        self.piece_counts[piece.piece_type().v() as usize] += 1;
    }
    
    #[inline]
    pub fn remove_piece(&mut self, sq: Square) {
        let target = Bitboard::from_c(sq);
        let piece = self.get_piece(sq);
        self.t_bitboards[piece.piece_type().v() as usize] ^= target;
        self.c_bitboards[piece.color().v() as usize] ^= target;
        self.pieces.0[sq.v() as usize] = Piece::default();
        self.piece_counts[self.get_piece(sq).piece_type().v() as usize] -= 1;
    }  
    
    #[inline]
    pub fn move_piece(&mut self, from: Square, to: Square) {
        let piece = self.get_piece(from);
        let from_to = Bitboard::from_c(from) ^ Bitboard::from_c(to);
        self.c_bitboards[piece.color().v() as usize] ^= from_to;
        self.t_bitboards[piece.piece_type().v() as usize] ^= from_to;
        self.pieces.0[from.v() as usize] = Piece::default();
        self.pieces.0[to.v() as usize] = piece;
    }
    
    #[inline]
    fn next_state_mut(&mut self) -> &mut PositionInfo {
        self.state_stack.next.get_mut_or_init(|| Default::default())
    }

    // todo: maybe this can be sped up by passing in the color of the moving side as a const generic param.
    /// Makes a move on the board.
    pub fn make_move(&mut self, m: Move) {
        let us = self.get_turn();
        let (from, to, flag) = m.into();
        let moving_piece = self.get_piece(from);
        let target_piece = self.get_piece(to);

        self.turn = !us;
        self.next_state_mut().castling_rights = self.state_stack.castling_rights;
        
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
            
            self.next_state_mut().captured_piece = captured_piece;

            self.next_state_mut().plys50 = Ply { v: 0 };

            update_castling(captured_sq, !us, &mut self.next_state_mut().castling_rights);
            
            self.remove_piece(captured_sq);
        }
        
        self.move_piece(from, to);
        
        match moving_piece.piece_type() {
            PieceType::KING => {
                self.next_state_mut().castling_rights.set_false(CastlingSide::QUEEN_SIDE, us);
                self.next_state_mut().castling_rights.set_false(CastlingSide::KING_SIDE, us);
                let rank = Rank::from_c(to);
                match flag {
                    MoveFlag::KING_CASTLE => {
                        let rook_from = Square::from_c((File::H, rank));
                        let rook_to   = Square::from_c((File::F, rank));
                        self.move_piece(rook_from, rook_to);
                    },
                    MoveFlag::QUEEN_CASTLE => {
                        let rook_from = Square::from_c((File::A, rank));
                        let rook_to   = Square::from_c((File::D, rank));
                        self.move_piece(rook_from, rook_to);
                    },
                    _ => (),
                }
            }
            PieceType::ROOK => {
                update_castling(from, us, &mut self.next_state_mut().castling_rights);
            }
            PieceType::PAWN => {
                match flag.v() {
                    MoveFlag::DOUBLE_PAWN_PUSH_C => {
                       self.next_state_mut().ep_square = Some(
                            // Safety:
                            // To `to` sq can only ever be on the 4th or 5th rank.
                            // For any square in on those ranks, the formula yields a valid square.
                            unsafe {
                                // todo: test and move this logic somewhere else
                                Square::from_v(to.v() + (us.v() *  2 - 1) * 8) 
                            }
                        );
                    }
                    MoveFlag::PROMOTION_KNIGHT_C..MoveFlag::CAPTURE_PROMOTION_QUEEN_C => {
                        // Safety: We just checked, that the flag is in a valid range.
                        let promo = unsafe { PromoPieceType::try_from(flag).unwrap_unchecked() };
                        self.remove_piece(to);
                        self.put_piece(to, Piece::from_c((us, promo)));
                    }
                    _ => (),
                }
                
                self.next_state_mut().plys50 = Ply { v: 0 };                
            }
            _ => ()
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
        
        let mut state = PositionInfo {
            castling_rights: CastlingRights::try_from(&mut *fen)?,
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
