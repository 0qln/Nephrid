use core::fmt;
use std::fmt::{Debug, Display};

use thiserror::Error;

use crate::{
    core::color::{Color, colors},
    impl_variants,
    misc::{ConstFrom, InvalidValueError, ValueOutOfSetError},
    uci::tokens::Tokenizer,
};

use super::r#move::MoveFlag;

pub trait IPieceType {
    const ID: PieceType;
}

pub type TPieceType = u8;

#[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
pub struct PieceType {
    v: TPieceType,
}

impl_variants! {
    TPieceType as PieceType in piece_type {
        NONE,
        PAWN,
        KNIGHT,
        BISHOP,
        ROOK,
        QUEEN,
        KING,
    }
}

impl fmt::Debug for PieceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PieceType")
            .field("v", &Into::<char>::into(*self))
            .finish()
    }
}

impl fmt::Display for PieceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Into::<char>::into(*self))
    }
}

impl PieceType {
    #[inline]
    pub const fn is_promo(&self) -> bool {
        self.v >= piece_type::KNIGHT.v && self.v <= piece_type::QUEEN.v
    }
}

pub type PieceTypeParseError = ValueOutOfSetError<char>;

impl TryFrom<char> for PieceType {
    type Error = PieceTypeParseError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        use piece_type::*;
        match value {
            'p' => Ok(PAWN),
            'n' => Ok(KNIGHT),
            'b' => Ok(BISHOP),
            'r' => Ok(ROOK),
            'q' => Ok(QUEEN),
            'k' => Ok(KING),
            '.' => Ok(NONE),
            x => Err(Self::Error::new(x, &['p', 'n', 'n', 'r', 'q', '.'])),
        }
    }
}

impl From<PieceType> for char {
    fn from(val: PieceType) -> Self {
        use piece_type::*;
        match val {
            PAWN => 'p',
            KNIGHT => 'n',
            BISHOP => 'b',
            ROOK => 'r',
            QUEEN => 'q',
            KING => 'k',
            NONE => '.',
            _ => unreachable!("Invalid program state."),
        }
    }
}

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct PromoPieceType {
    v: PieceType,
}

impl_variants! {
    PieceType as PromoPieceType in promo_piece_type {
        KNIGHT = piece_type::KNIGHT,
        BISHOP = piece_type::BISHOP,
        ROOK = piece_type::ROOK,
        QUEEN = piece_type::QUEEN,
    }
}

impl fmt::Display for PromoPieceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.v, f)
    }
}

pub type PromoPieceParseError = ValueOutOfSetError<char>;

impl TryFrom<char> for PromoPieceType {
    type Error = PromoPieceParseError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        use promo_piece_type::*;
        match value {
            'n' => Ok(KNIGHT),
            'b' => Ok(BISHOP),
            'r' => Ok(ROOK),
            'q' => Ok(QUEEN),
            x => Err(Self::Error::new(x, &['n', 'b', 'r', 'q']).into()),
        }
    }
}

#[derive(Debug, Error)]
pub enum PromoPieceTokenizationError {
    #[error("Missing promotion piece type token.")]
    MissingToken,

    #[error("Invalid promotion piece type token: {0}")]
    InvalidToken(PromoPieceParseError),
}

impl TryFrom<&mut Tokenizer<'_>> for PromoPieceType {
    type Error = PromoPieceTokenizationError;

    fn try_from(fen: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        match fen.skip_ws().next_char() {
            Some(c) => Self::try_from(c).map_err(|e| Self::Error::InvalidToken(e)),
            None => Err(Self::Error::MissingToken),
        }
    }
}

pub type PromoPieceConvertError = InvalidValueError<MoveFlag>;

impl TryFrom<MoveFlag> for PromoPieceType {
    type Error = PromoPieceConvertError;

    fn try_from(flag: MoveFlag) -> Result<Self, Self::Error> {
        if !flag.is_promo() {
            Err(Self::Error::new(flag))
        } else {
            let v = (flag.v() - 2) % 4 + 2;
            let pt = PieceType { v };
            Ok(PromoPieceType { v: pt })
        }
    }
}

pub type TPiece = u8;

#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct Piece {
    v: TPiece,
}

impl Piece {
    #[inline]
    pub const fn piece_type(&self) -> PieceType {
        // Safety: Piece cannot be constructed from unchecked value.
        unsafe { PieceType::from_v(self.v >> 1) }
    }

    #[inline]
    pub const fn color(&self) -> Color {
        // Safety: One bit can only ever contain Color::WHITE or Color::BLACK
        unsafe { Color::from_v(self.v & 1) }
    }

    pub const fn v(&self) -> TPiece {
        self.v
    }
}

impl const ConstFrom<(Color, PieceType)> for Piece {
    #[inline]
    fn from_c((color, piece_type): (Color, PieceType)) -> Self {
        Piece {
            v: color.v() | (piece_type.v() << 1),
        }
    }
}

impl const ConstFrom<(Color, PromoPieceType)> for Piece {
    #[inline]
    fn from_c((color, piece_type): (Color, PromoPieceType)) -> Self {
        Piece {
            v: color.v() | (piece_type.v().v() << 1),
        }
    }
}

#[derive(Debug, Error)]
pub enum PieceParseError {
    #[error("Invalid piece type: {0}")]
    InvalidType(PieceTypeParseError),

    #[error("Invalid char.")]
    InvalidChar,
}

impl TryFrom<char> for Piece {
    type Error = PieceParseError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'a'..'z' => {
                let color = colors::WHITE;
                let p_type = PieceType::try_from(value).map_err(|e| Self::Error::InvalidType(e))?;
                Ok(Self::from_c((color, p_type)))
            }
            'A'..'Z' => {
                let color = colors::BLACK;
                let value = (value as u8 - b'a') as char;
                let p_type = PieceType::try_from(value).map_err(|e| Self::Error::InvalidType(e))?;
                Ok(Self::from_c((color, p_type)))
            }
            _ => Err(Self::Error::InvalidChar),
        }
    }
}

impl From<Piece> for char {
    fn from(val: Piece) -> Self {
        let mut result: char = val.piece_type().into();
        if val.color() == colors::WHITE {
            result = result.to_uppercase().next().unwrap();
        }
        result
    }
}

impl fmt::Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Piece")
            .field("v", &Into::<char>::into(*self))
            .field("piece_type", &Into::<char>::into(self.piece_type()))
            .field("color", &Into::<char>::into(self.color()))
            .finish()
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Into::<char>::into(*self))
    }
}
