use colors::*;
use core::fmt;
use std::ops;
use thiserror::Error;

use crate::{impl_variants_with_assertion, misc::ValueOutOfSetError, uci::tokens::Tokenizer};

#[derive(PartialEq, Eq, Copy, Clone, Default)]
pub struct Color {
    v: TColor,
}

pub type TColor = u8;

impl_variants_with_assertion! {
    TColor as Color in colors {
        WHITE,
        BLACK,
    }
}

impl_op!(!|c: Color| -> Color { Color { v: c.v ^ 1 } });

pub type ColorParseError = ValueOutOfSetError<char>;

impl TryFrom<char> for Color {
    type Error = ColorParseError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'w' => Ok(WHITE),
            'b' => Ok(BLACK),
            x => Err(Self::Error::new(x, &['w', 'b'])),
        }
    }
}

#[derive(Debug, Error)]
pub enum ColorTokenizationError {
    #[error("Invalid color: {0}")]
    InvalidColor(ColorParseError),
    #[error("Missing color char.")]
    MissingChar,
}

impl TryFrom<&mut Tokenizer<'_>> for Color {
    type Error = ColorTokenizationError;

    fn try_from(fen: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        match fen.skip_ws().next_char() {
            Some(c) => Self::try_from(c).map_err(|e| Self::Error::InvalidColor(e)),
            None => Err(Self::Error::MissingChar),
        }
    }
}

impl From<Color> for char {
    fn from(val: Color) -> Self {
        match val {
            WHITE => 'w',
            BLACK => 'b',
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
