use castling_sides::*;
use core::fmt;

use crate::{
    core::{
        color::{Color, colors::*},
        coordinates::{files, squares},
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
    TCastlingSide as CastlingSide in castling_sides {
        KING_SIDE = piece_type::KING.v(),
        QUEEN_SIDE = piece_type::QUEEN.v(),
    }
}

pub type CastlingSideParseError = ValueOutOfSetError<File>;

impl TryFrom<File> for CastlingSide {
    type Error = CastlingSideParseError;

    fn try_from(value: File) -> Result<Self, Self::Error> {
        use castling_sides::*;
        match value {
            files::G => Ok(KING_SIDE),
            files::C => Ok(QUEEN_SIDE),
            x => Err(Self::Error {
                value: x,
                expected: &[files::G, files::C],
            }),
        }
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct CastlingRights {
    v: u8,
}

pub type CastlingSideTokenizationError = ValueOutOfSetError<char>;

impl TryFrom<&mut Tokenizer<'_>> for CastlingRights {
    type Error = CastlingSideTokenizationError;

    fn try_from(value: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        let mut result = CastlingRights::empty();
        for c in value.chars() {
            match c {
                'K' => result.set_true(KING_SIDE, WHITE),
                'Q' => result.set_true(QUEEN_SIDE, WHITE),
                'k' => result.set_true(KING_SIDE, BLACK),
                'q' => result.set_true(QUEEN_SIDE, BLACK),
                '-' => return Ok(result),
                x => {
                    return Err(Self::Error {
                        value: x,
                        expected: &['K', 'Q', 'k', 'q', '-'],
                    });
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
    pub const fn is_empty(&self) -> bool {
        self.v == 0
    }

    #[inline]
    pub const fn v(&self) -> u8 {
        self.v
    }

    #[inline]
    pub const fn get_float(&self, side: CastlingSide, color: Color) -> f32 {
        if self.is_true(side, color) { 1.0 } else { 0.0 }
    }

    /// applies a castling mask
    #[inline]
    pub const fn apply_mask(&mut self, mask: Self) {
        self.v &= mask.v;
    }

    #[inline]
    pub const fn empty() -> Self {
        Self { v: 0 }
    }

    #[inline]
    pub const fn full() -> Self {
        Self { v: 0xFF }
    }

    #[inline]
    const fn to_index(side: CastlingSide, color: Color) -> u8 {
        // Note: King and queen side need to have specific
        // values for this indexing scheme to work.
        color.v() | (side.v & 0b10)
    }

    /// Precomputed masks for all 64 squares.
    /// Applying the mask for a square clears rights if a King or Rook
    /// moves from or is captured on that square.
    pub const MASKS: [Self; 64] = Self::init_masks();

    const fn init_masks() -> [Self; 64] {
        // Initialize all 64 squares to 0xFF (all bits 1, meaning "keep all rights")
        let mut masks = [Self::full(); 64];

        // A1 (index 0): clears White Queen-side
        masks[squares::A1.index()].set_false(QUEEN_SIDE, WHITE);

        // H1 (index 7): clears White King-side
        masks[squares::H1.index()].set_false(KING_SIDE, WHITE);

        // E1 (index 4): clears both White King-side and Queen-side
        masks[squares::E1.index()].set_false(QUEEN_SIDE, WHITE);
        masks[squares::E1.index()].set_false(KING_SIDE, WHITE);

        // A8 (index 56): clears Black Queen-side
        masks[squares::A8.index()].set_false(QUEEN_SIDE, BLACK);

        // H8 (index 63): clears Black King-side
        masks[squares::H8.index()].set_false(KING_SIDE, BLACK);

        // E8 (index 60): clears both Black King-side and Queen-side
        masks[squares::E8.index()].set_false(QUEEN_SIDE, BLACK);
        masks[squares::E8.index()].set_false(KING_SIDE, BLACK);

        masks
    }
}
