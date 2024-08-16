use crate::engine::{
    bitboard::Bitboard,
    color::Color,
    turn::Turn,
    piece::{Piece, PieceType},
    fen::Fen,
    castling::CastlingRights,
    coordinates::{Squares, Square, Rank, File},
    zobrist,
    r#move::Move
};

use super::ply::Ply;


#[derive(Default, Clone)]
pub struct PositionInfo {
    pub next: Option<Box<PositionInfo>>,
    pub prev: Option<Box<PositionInfo>>,
    pub checkers: Bitboard,
    pub blockers: Bitboard,
    pub nstm_attacks: Bitboard,
    pub plys50: Ply,
    pub ply: Ply,
    pub ep_square: Square,
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
    pub fn get_bitboard(&self, piece_type: PieceType, color: Color) -> Bitboard {
        self.c_bitboards[color as usize] & self.t_bitboards[piece_type as usize]
    }
    
    pub fn get_occupancy(&self) -> Bitboard {
        self.c_bitboards[Color::White as usize] | self.c_bitboards[Color::Black as usize]
    }
    
    pub fn get_piece(&self, sq: Square) -> Piece {
        self.pieces[Into::<usize>::into(sq)]
    }
    
    pub fn put_piece(&mut self, sq: Square, piece: Piece) {
        let target = Bitboard::from(sq.clone());
        self.t_bitboards[piece.piece_type as usize] |= target;
        self.c_bitboards[piece.color as usize] |= target;
        self.pieces[Into::<usize>::into(sq)] = piece;
        self.piece_counts[piece.piece_type as usize] += 1;
    }
    
    pub fn remove_piece(&mut self, sq: Square) {
        let target = Bitboard::from(sq.clone());
        let piece = self.get_piece(sq);
        self.t_bitboards[piece.piece_type as usize] ^= target;
        self.c_bitboards[piece.color as usize] ^= target;
        self.pieces[Into::<usize>::into(sq)] = Piece::default();
        self.piece_counts[self.get_piece(sq).piece_type as usize] -= 1;
    }  

    pub fn make_move(&mut self, m: Move) {
        todo!()
    }

}

impl Default for Position {
    fn default() -> Self {
        Position::try_from(
            Fen { v: [
                &"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", 
                &"w", 
                &"KQkq", 
                &"-", 
                &"0", 
                &"1"
            ]}
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
                let sq = Square::from((
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

impl<'fen> TryFrom<Fen<'fen>> for Position {
    type Error = anyhow::Error;
    
    fn try_from(fen: Fen) -> Result<Self, Self::Error> {
        let mut position = Position::new();
        let mut sq = Squares::H8 as i8;

        for char in fen.v[0].chars() {
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
        
        position.turn = fen.v[1].try_into()?;        
        
        let mut state = PositionInfo {
            castling_rights: CastlingRights::try_from(fen.v[2])?,
            ep_square: Square::try_from(fen.v[3])?,
            plys50: Ply::new(fen.v[4].parse()?, position.turn),
            ply: Ply { v: fen.v[5].parse::<u16>()? },
            has_threefold_repetition: false,
            ..Default::default()
        };

        // TODO: init zobrist hash
        
        // state.init(&position);
        
        Ok(position)
    }
}
