use core::fmt;
use std::ops;

use crate::{impl_variants, misc::ParseError};

#[derive(PartialEq, Copy, Clone, Default)]
pub struct Color { v: TColor }

pub type TColor = u8;

impl_variants! { 
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
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

impl Into<char> for Color {
    fn into(self) -> char {
        match self {
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