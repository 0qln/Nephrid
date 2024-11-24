use crate::{
    engine::{color::Color, fen::Fen}, impl_variants, misc::ParseError
};

use super::{coordinates::File, piece::PieceType};

pub type TCastlingSide = u8;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CastlingSide {
    v: TCastlingSide
}

impl_variants! {
    TCastlingSide as CastlingSide {
        KING_SIDE = PieceType::KING.v(),
        QUEEN_SIDE = PieceType::QUEEN.v(),
    }
} 
    
impl TryFrom<File> for CastlingSide {
    type Error = ParseError;

    fn try_from(value: File) -> Result<Self, Self::Error> {
        match value {
            File::G => Ok(CastlingSide::KING_SIDE),
            File::C => Ok(CastlingSide::QUEEN_SIDE),
            x => Err(ParseError::InputOutOfRange(Box::new(x)))
        }
    }
}

#[derive(Copy, Clone, Default, PartialEq)]
pub struct CastlingRights {
    v: u8,
}

impl TryFrom<&mut Fen<'_>> for CastlingRights {
    type Error = ParseError;

    fn try_from(value: &mut Fen<'_>) -> Result<Self, Self::Error> {
        let mut result = CastlingRights::empty();
        for c in value.iter_token() {
            match c {
                'K' => result.set_true(CastlingSide::KING_SIDE, Color::WHITE),
                'Q' => result.set_true(CastlingSide::QUEEN_SIDE, Color::WHITE),
                'k' => result.set_true(CastlingSide::KING_SIDE, Color::BLACK),
                'q' => result.set_true(CastlingSide::QUEEN_SIDE, Color::BLACK),
                '-' => return Ok(result),
                x => return Err(ParseError::InputOutOfRange(Box::new(x))),
            }
        }
        Ok(result)
    }
}

impl CastlingRights {
    #[inline]
    pub const fn set_false(&mut self, side: CastlingSide, color: Color) {
        self.v &= !(1 << CastlingRights::to_index(side, color));
    }

    #[inline]
    pub const fn set_true(&mut self, side: CastlingSide, color: Color) {
        self.v |= 1 << CastlingRights::to_index(side, color);
    }
    
    #[inline]
    pub const fn is_true(&self, side: CastlingSide, color: Color) -> bool {
        self.v & (1 << CastlingRights::to_index(side, color)) != 0
    }

    #[inline]
    const fn to_index(side: CastlingSide, color: Color) -> u8 {
        assert!(
            CastlingSide::KING_SIDE.v  == 6 &&
            CastlingSide::QUEEN_SIDE.v == 5, 
            "King and queen side need to have specific values for this indexing scheme to work.");

        color.v() | (side.v & 0b10)
    }
    
    #[inline]
    pub const fn empty() -> Self {    
        CastlingRights { v: 0 }
    }
    
    #[inline]
    pub const fn v(&self) -> u8 {
        self.v
    }
}
