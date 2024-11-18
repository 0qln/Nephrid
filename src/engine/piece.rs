use std::iter::Step;

use crate::{engine::color::Color, impl_variants, misc::{ConstFrom, ParseError}};

use super::r#move::MoveFlag;

pub type TPieceType = u8;

#[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
pub struct PieceType {
    v: TPieceType
}

impl_variants! {
    TPieceType as PieceType {
        NONE,
        PAWN,
        KNIGHT,
        BISHOP,
        ROOK,
        QUEEN,
        KING,    
    }
}

impl PieceType {
    #[inline]
    pub const fn is_promo(&self) -> bool {
        self.v >= Self::KNIGHT.v && 
        self.v <= Self::QUEEN.v
    }
}

impl Step for PieceType {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        Some(end.v() as usize - start.v() as usize)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        match start.v() + count as u8 {
            ..=Self::KING_C => Some(PieceType { v: start.v() + count as u8 }),    
            _ => None
        }
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        match start.v() < count as u8 {
            false => None,
            true => Some(PieceType { v: start.v() - count as u8 })
        }
    }
}

impl TryFrom<char> for PieceType {
    type Error = ParseError;
    
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'p' => Ok(PieceType::PAWN),
            'n' => Ok(PieceType::KNIGHT),
            'b' => Ok(PieceType::BISHOP),
            'r' => Ok(PieceType::ROOK),
            'q' => Ok(PieceType::QUEEN),
            'k' => Ok(PieceType::KING),
            '.' => Ok(PieceType::NONE),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

impl Into<char> for PieceType {
    fn into(self) -> char {
        match self {
            PieceType::PAWN => 'p',
            PieceType::KNIGHT => 'n',
            PieceType::BISHOP => 'b',
            PieceType::ROOK => 'r',
            PieceType::QUEEN => 'q',
            PieceType::KING => 'k',
            PieceType::NONE => '.',
            _ => unreachable!("Invalid program state.")
        }
    }
}


#[derive(Copy, Clone, Default, PartialEq)]
pub struct PromoPieceType {
    v: TPieceType
}

impl_variants! {
    TPieceType as PromoPieceType {
        KNIGHT = PieceType::KNIGHT.v(),
        BISHOP = PieceType::BISHOP.v(),
        ROOK = PieceType::ROOK.v(),
        QUEEN = PieceType::QUEEN.v(),
    }
}

impl TryFrom<char> for PromoPieceType {
    type Error = ParseError;
    
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'n' => Ok(PromoPieceType::KNIGHT),
            'b' => Ok(PromoPieceType::BISHOP),
            'r' => Ok(PromoPieceType::ROOK),
            'q' => Ok(PromoPieceType::QUEEN),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

impl TryFrom<MoveFlag> for PromoPieceType {
    type Error = ParseError;

    fn try_from(flag: MoveFlag) -> Result<Self, Self::Error> {
        if !flag.is_promo() {
            Err(ParseError::InputOutOfRange(Box::new(flag)))?
        }
        else {
            Ok(PromoPieceType { v: (flag.v() - 2) % 4 + 2 })
        }
    }
}


pub type TPiece = u8;
      
#[derive(Copy, Clone, Default, PartialEq)]
pub struct Piece { v: TPiece }

impl Piece {
    #[inline]
    pub const fn piece_type(&self) -> PieceType {
        // Safety: Piece cannot be constructed from unchecked value.
        unsafe {
            PieceType::from_v(self.v >> 1)
        }
    }
    
    #[inline]
    pub const fn color(&self) -> Color {
        // Safety: One bit can only ever contain Color::WHITE or Color::BLACK
        unsafe {
            Color::from_v(self.v & 1)
        }
    }
    
    pub const fn v(&self) -> TPiece {
        self.v
    }
}

impl const ConstFrom<(Color, PieceType)> for Piece {
    #[inline]
    fn from_c((color, piece_type): (Color, PieceType)) -> Self {
        Piece { v: color.v() | (piece_type.v() >> 1) }
    }
}

impl const ConstFrom<(Color, PromoPieceType)> for Piece {
    #[inline]
    fn from_c((color, piece_type): (Color, PromoPieceType)) -> Self {
        Piece { v: color.v() | (piece_type.v() >> 1) }
    }
}

impl TryFrom<char> for Piece {
    type Error = ParseError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        let piece_type = PieceType::try_from(value.to_ascii_lowercase())?;
        let color = if value.is_uppercase() { Color::WHITE } else { Color::BLACK };
        Ok(Self::from_c((color, piece_type)))
    }
}

impl Into<char> for Piece {
    fn into(self) -> char {
        let mut result: char = self.piece_type().into();
        if self.color() == Color::WHITE {
            result = result.to_ascii_uppercase();
        }
        result
    }
}
