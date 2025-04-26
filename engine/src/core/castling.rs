use core::fmt;

use crate::{core::color::Color, impl_variants, misc::ParseError, uci::tokens::Tokenizer};

use super::{coordinates::File, piece::PieceType};

pub type TCastlingSide = u8;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CastlingSide {
    v: TCastlingSide,
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
            x => Err(ParseError::InputOutOfRange(x.to_string())),
        }
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct CastlingRights {
    v: u8,
}

impl TryFrom<&mut Tokenizer<'_>> for CastlingRights {
    type Error = ParseError;

    fn try_from(value: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        let mut result = CastlingRights::empty();
        for c in value.skip_ws().chars() {
            match c {
                'K' => result.set_true(CastlingSide::KING_SIDE, Color::WHITE),
                'Q' => result.set_true(CastlingSide::QUEEN_SIDE, Color::WHITE),
                'k' => result.set_true(CastlingSide::KING_SIDE, Color::BLACK),
                'q' => result.set_true(CastlingSide::QUEEN_SIDE, Color::BLACK),
                '-' => return Ok(result),
                x => return Err(ParseError::InputOutOfRange(x.to_string())),
            }
        }
        Ok(result)
    }
}

impl fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "-");
        }
        write!(
            f,
            "{}{}{}{}",
            if self.is_true(CastlingSide::KING_SIDE, Color::WHITE) {
                "K"
            } else {
                ""
            },
            if self.is_true(CastlingSide::QUEEN_SIDE, Color::WHITE) {
                "Q"
            } else {
                ""
            },
            if self.is_true(CastlingSide::KING_SIDE, Color::BLACK) {
                "k"
            } else {
                ""
            },
            if self.is_true(CastlingSide::QUEEN_SIDE, Color::BLACK) {
                "q"
            } else {
                ""
            },
        )
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
        // Note: King and queen side need to have specific
        // values for this indexing scheme to work.
        color.v() | (side.v & 0b10)
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.v == 0
    }

    #[inline]
    pub const fn empty() -> Self {
        CastlingRights { v: 0 }
    }

    #[inline]
    pub const fn v(&self) -> u8 {
        self.v
    }
    
    #[inline]
    pub const fn get_float(&self, side: CastlingSide, color: Color) -> f32 {
        if self.is_true(side, color) { 1.0 } else { 0.0 }
    }
    
    #[inline]
    pub fn fill_floats(&self, buf: &mut [f32; 6]) {
        buf[0] = self.get_float(CastlingSide::KING_SIDE, Color::WHITE);
        buf[1] = self.get_float(CastlingSide::QUEEN_SIDE, Color::WHITE);
        buf[2] = self.get_float(CastlingSide::KING_SIDE, Color::BLACK);
        buf[3] = self.get_float(CastlingSide::QUEEN_SIDE, Color::BLACK);
    }
}
