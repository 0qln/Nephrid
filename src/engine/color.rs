use std::ops;

use crate::misc::ParseError;

pub type TColor = bool;

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum Color {
    Black = true as isize,
    White = false as isize,
}

impl Color {
    pub fn new(val: TColor) -> Self {
        match val {
            true => Color::Black,
            false => Color::White
        }
    }
}

impl Default for Color {
    fn default() -> Self {
        Color::White
    }
}

impl ops::Not for Color {
    type Output = Color;

    fn not(self) -> Self::Output {
        match self {
            Color::White => Color::Black,            
            Color::Black => Color::White
        }
    }
}

impl TryFrom<char> for Color {
    type Error = ParseError;
    
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'w' => Ok(Color::White),
            'b' => Ok(Color::Black),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

impl Into<char> for Color {
    fn into(self) -> char {
        match self {
            Color::White => 'w',
            Color::Black => 'b',
        }
    }
}