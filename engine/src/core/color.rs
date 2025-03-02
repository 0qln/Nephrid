use core::fmt;
use std::ops;

use crate::{impl_variants_with_assertion, misc::ParseError, uci::tokens::Tokenizer};

#[derive(PartialEq, Eq, Copy, Clone, Default)]
pub struct Color { v: TColor }

pub type TColor = u8;

impl_variants_with_assertion! { 
    TColor as Color {
        WHITE,
        BLACK,
    } 
}

impl_op!(! | c: Color | -> Color { Color { v: c.v ^ 1 } });

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
            _ => unreachable!("Invalid program state.")
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