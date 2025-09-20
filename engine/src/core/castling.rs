use castling_side::*;
use core::fmt;

use terrors::OneOf;

use crate::{
    core::{
        color::{Color, colors::*},
        coordinates::files,
        piece::piece_type,
    },
    impl_variants,
    misc::ValueOutOfSetError,
    uci::tokens::Tokenizer,
};

use super::coordinates::File;

pub type TCastlingSide = u8;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CastlingSide {
    v: TCastlingSide,
}

impl_variants! {
    TCastlingSide as CastlingSide in castling_side {
        KING_SIDE = piece_type::KING.v(),
        QUEEN_SIDE = piece_type::QUEEN.v(),
    }
}

impl TryFrom<File> for CastlingSide {
    type Error = OneOf<(ValueOutOfSetError<File>,)>;

    fn try_from(value: File) -> Result<Self, Self::Error> {
        use castling_side::*;
        match value {
            files::G => Ok(KING_SIDE),
            files::C => Ok(QUEEN_SIDE),
            x => Err(OneOf::new(ValueOutOfSetError {
                value: x,
                expected: &[files::G, files::C],
            })),
        }
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct CastlingRights {
    v: u8,
}

impl TryFrom<&mut Tokenizer<'_>> for CastlingRights {
    type Error = OneOf<(ValueOutOfSetError<char>,)>;

    fn try_from(value: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        let mut result = CastlingRights::empty();
        for c in value.skip_ws().chars() {
            match c {
                'K' => result.set_true(KING_SIDE, WHITE),
                'Q' => result.set_true(QUEEN_SIDE, WHITE),
                'k' => result.set_true(KING_SIDE, BLACK),
                'q' => result.set_true(QUEEN_SIDE, BLACK),
                '-' => return Ok(result),
                x => {
                    return Err(OneOf::new(ValueOutOfSetError {
                        value: x,
                        expected: &['K', 'Q', 'k', 'q', '-'],
                    }));
                }
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
            if self.is_true(KING_SIDE, WHITE) { "K" } else { "" },
            if self.is_true(QUEEN_SIDE, WHITE) { "Q" } else { "" },
            if self.is_true(KING_SIDE, BLACK) { "k" } else { "" },
            if self.is_true(QUEEN_SIDE, BLACK) { "q" } else { "" },
        )
    }
}

impl CastlingRights {
    #[inline]
    pub const fn set_false(&mut self, side: CastlingSide, color: Color) {
        self.v &= !(1 << Self::to_index(side, color));
    }

    #[inline]
    pub const fn set_true(&mut self, side: CastlingSide, color: Color) {
        self.v |= 1 << Self::to_index(side, color);
    }

    #[inline]
    pub const fn is_true(&self, side: CastlingSide, color: Color) -> bool {
        self.v & (1 << Self::to_index(side, color)) != 0
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
        Self { v: 0 }
    }

    #[inline]
    pub const fn v(&self) -> u8 {
        self.v
    }
}
