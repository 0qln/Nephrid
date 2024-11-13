use std::ops;

use crate::misc::ParseError;

pub type TColor = bool;

#[derive(PartialEq, Copy, Clone, Debug, Default)]
pub struct Color {
    pub v: TColor
}

impl Color {
    pub const WHITE: Color = Color { v: false };
    pub const BLACK: Color = Color { v: true };

    pub const fn new(val: TColor) -> Self {
        Color { v: val }
    }
        
}

impl_op!(! | c: Color | -> Color { Color { v: !c.v } });

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
        }
    }
}
