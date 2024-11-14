use std::ops;

use crate::{impl_variants, misc::ParseError};

pub type TColor = u8;

#[derive(PartialEq, Copy, Clone, Debug, Default)]
pub struct Color {
    v: TColor
}

impl_variants! { 
    Color {
        WHITE,
        BLACK,
    } 
}

impl Color {
    #[inline]
    pub const fn v(&self) -> TColor {
        self.v
    }    

    #[inline]
    pub const unsafe fn from_v(v: TColor) -> Self {
        Color { v }
    }
}

impl_op!(! | c: Color | -> Color { Color { v: !c.v & 1 } });

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
