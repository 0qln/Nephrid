use core::fmt;

use crate::{engine::color::Color, impl_variants, misc::{ConstFrom, ParseError}};

use super::r#move::MoveFlag;

pub trait IPieceType {
    const ID: PieceType;
}

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
        self.v >= Self::KNIGHT.v && 
        self.v <= Self::QUEEN.v
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

impl From<PieceType> for char {
    fn from(val: PieceType) -> Self {
        match val {
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


#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct PromoPieceType {
    v: PieceType
}

impl_variants! {
    PieceType as PromoPieceType {
        KNIGHT = PieceType::KNIGHT,
        BISHOP = PieceType::BISHOP,
        ROOK = PieceType::ROOK,
        QUEEN = PieceType::QUEEN,
    }
}
    
impl fmt::Display for PromoPieceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.v.fmt(f)
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
            let pt = PieceType { v: (flag.v() - 2) % 4 + 2 };
            Ok(PromoPieceType { v: pt })
        }
    }
}


#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct SlidingPieceType {
    v: TPieceType
}

impl_variants! {
    TPieceType as SlidingPieceType {
        BISHOP = PieceType::BISHOP_C,
        ROOK = PieceType::ROOK_C,
        QUEEN = PieceType::QUEEN_C,
    }
}
    
impl From<SlidingPieceType> for PieceType {
    #[inline]
    fn from(val: SlidingPieceType) -> Self {
        PieceType { v: val.v }
    }
}


#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct JumpingPieceType {
    v: TPieceType
}

impl_variants! {
    TPieceType as JumpingPieceType {
        KNIGHT = PieceType::KNIGHT_C,
        KING = PieceType::KING_C,
    }
}
    
impl From<JumpingPieceType> for PieceType {
    #[inline]
    fn from(val: JumpingPieceType) -> Self {
        PieceType { v: val.v }
    }
}


pub type TPiece = u8;
      
#[derive(Copy, Clone, Default, PartialEq, Eq)]
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
        Piece { v: color.v() | (piece_type.v() << 1) }
    }
}

impl const ConstFrom<(Color, PromoPieceType)> for Piece {
    #[inline]
    fn from_c((color, piece_type): (Color, PromoPieceType)) -> Self {
        Piece { v: color.v() | (piece_type.v().v() << 1) }
    }
}

impl TryFrom<char> for Piece {
    type Error = ParseError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        let piece_type = PieceType::try_from(value.to_lowercase().next().unwrap())?;
        let color = if value.is_uppercase() { Color::WHITE } else { Color::BLACK };
        Ok(Self::from_c((color, piece_type)))
    }
}

impl From<Piece> for char {
    fn from(val: Piece) -> Self {
        let mut result: char = val.piece_type().into();
        if val.color() == Color::WHITE {
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