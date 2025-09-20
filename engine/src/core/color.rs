use core::fmt;
use std::ops;

use crate::{mk_variant_assert, mk_variant_utils, mk_variants, uci::tokens::Tokenizer};

#[derive(PartialEq, Eq, Copy, Clone, Default)]
pub struct Color {
    v: TColor,
}

pub type TColor = u8;

pub mod colors {
    use super::*;
    mk_variants! {
        TColor as Color {
            WHITE,
            BLACK,
        }
    }

    /// Assert that `v` is a valid variant.
    pub const fn debug_assert_variant(v: TColor) {
        debug_assert!(
            false || v == WHITE.v || v == BLACK.v,
            "v is not a valid variant."
        );
    }
}

mk_variant_utils! {TColor as Color}

impl_op!(!|c: Color| -> Color { Color { v: c.v ^ 1 } });

impl TryFrom<char> for Color {
    type Error = ParseError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'w' => Ok(Color::WHITE),
            'b' => Ok(Color::BLACK),
            x => Err(ParseError::InputOutOfRange(x.to_string())),
        }
    }
}

impl TryFrom<&mut Tokenizer<'_>> for Color {
    type Error = ParseError;

    fn try_from(fen: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        match fen.skip_ws().next_char() {
            Some(c) => Color::try_from(c),
            None => Err(ParseError::MissingValue),
        }
    }
}

impl From<Color> for char {
    fn from(val: Color) -> Self {
        match val {
            Color::WHITE => 'w',
            Color::BLACK => 'b',
            _ => unreachable!("Invalid program state."),
        }
    }
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Color")
            .field("v", &Into::<char>::into(*self))
            .finish()
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Into::<char>::into(*self))
    }
}
