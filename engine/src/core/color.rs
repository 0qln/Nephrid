use colors::*;
use core::fmt;
use std::ops;

use terrors::OneOf;

use crate::{
    impl_variants_with_assertion,
    misc::{MissingTokenError, ValueOutOfSetError},
    uci::tokens::Tokenizer,
};

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

impl TryFrom<char> for Color {
    type Error = OneOf<(ValueOutOfSetError<char>,)>;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'w' => Ok(WHITE),
            'b' => Ok(BLACK),
            x => Err(OneOf::new(ValueOutOfSetError::new(x, &['w', 'b']))),
        }
    }
}

impl TryFrom<&mut Tokenizer<'_>> for Color {
    type Error = OneOf<(ValueOutOfSetError<char>, MissingTokenError)>;

    fn try_from(fen: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        match fen.skip_ws().next_char() {
            Some(c) => Color::try_from(c).map_err(OneOf::broaden),
            None => Err(OneOf::new(MissingTokenError::new("Color"))),
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
