use crate::{
    engine::{
        bitboard::Bitboard, 
        castling::CastlingRights, 
        color::Color, 
        coordinates::{File, Rank, Square, Squares}, 
        fen::Fen, 
        r#move::Move, 
        piece::{Piece, PieceType}, 
        turn::Turn, 
        zobrist
    }, misc::{ConstFrom, ParseError}, uci::tokens::Tokenizer
};

use super::ply::{FullMoveCount, Ply};


#[derive(Default, Clone)]
pub struct PositionInfo {
    pub next: Option<Box<PositionInfo>>,
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

impl PositionInfo {
    pub fn init(&mut self, position: &Position) {
        let us = position.turn;
        let king = position.get_bitboard(PieceType::King, us);
        let king_sq = king.lsb();
        let occupancy = position.get_occupancy();
        for enemy_sq in position.c_bitboards[!us as usize].clone() {
            let enemy = position.get_piece(enemy_sq);     

        }
        
        todo!()
    }
}


#[derive(Clone)]
pub struct Position {
    c_bitboards: [Bitboard; 2],
    t_bitboards: [Bitboard; 7],
    pieces: [Piece; 64],
    piece_counts: [u8; 14],
    turn: Turn,
    state_stack: PositionInfo
}

impl Position {
    #[inline]
    pub fn get_bitboard(&self, piece_type: PieceType, color: Color) -> Bitboard {
        self.c_bitboards[color as usize] & self.t_bitboards[piece_type as usize]
    }

    #[inline]
    pub fn get_color_bb(&self, color: Color) -> Bitboard {
        self.c_bitboards[color as usize]
    }

    #[inline]
    pub fn get_piece_bb(&self, piece_type: PieceType) -> Bitboard {
        self.t_bitboards[piece_type as usize]
    }
    
    #[inline]
    pub fn get_occupancy(&self) -> Bitboard {
        self.c_bitboards[Color::White as usize] | self.c_bitboards[Color::Black as usize]
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
        self.state_stack.ep_square
    }
    
    pub fn put_piece(&mut self, sq: Square, piece: Piece) {
        let target = Bitboard::from_c(sq);
        self.t_bitboards[piece.piece_type as usize] |= target;
        self.c_bitboards[piece.color as usize] |= target;
        self.pieces[sq.v() as usize] = piece;
        self.piece_counts[piece.piece_type as usize] += 1;
    }
    
    pub fn remove_piece(&mut self, sq: Square) {
        let target = Bitboard::from_c(sq);
        let piece = self.get_piece(sq);
        self.t_bitboards[piece.piece_type as usize] ^= target;
        self.c_bitboards[piece.color as usize] ^= target;
        self.pieces[sq.v() as usize] = Piece::default();
        self.piece_counts[self.get_piece(sq).piece_type as usize] -= 1;
    }  

    pub fn make_move(&mut self, m: Move) {
        let us = self.get_turn();
        // let from = m
    }

}

impl Default for Position {
    fn default() -> Self {
        Position::try_from(
            &mut Fen::new(&"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        ).unwrap()
    }
}

impl Position {
    pub fn new() -> Self {
        Self {
            c_bitboards: [Bitboard { v: 0 }; 2],
            t_bitboards: [Bitboard { v: 0 }; 7],
            pieces: [Piece{color: Color::White, piece_type: PieceType::None}; 64],
            piece_counts: [0; 14],
            turn: Color::White,
            state_stack: PositionInfo::default()
        }
    }

    pub fn reset(&self) {
        todo!()
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
        let mut position = Position::new();
        let mut sq = Squares::H8 as i8;

        // Position
        for char in fen.iter_token() {
            match char {
                '/' => continue,
                '1'..='8' => sq -= char.to_digit(10).unwrap() as i8,        
                _ => {
                    let piece = Piece::try_from(char)?; 
                    let pos_sq = Square::try_from((sq ^ 7) as u8)?;
                    position.put_piece(pos_sq, piece);
                    sq -= 1;
                }
            }
            if sq < Squares::A1 as i8 {
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
