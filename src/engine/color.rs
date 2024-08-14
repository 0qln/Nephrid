use std::ops;
use anyhow;

#[derive(PartialEq, Copy, Clone)]
pub enum Color {
    White = 0,
    Black = 1,
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
    type Error = anyhow::Error;
    
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'w' => Ok(Color::White),
            'b' => Ok(Color::Black),
            _ => Err(anyhow::Error::msg("Invalid char")),
        }
    }
}

impl TryFrom<&str> for Color {
    type Error = anyhow::Error;
    
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let first = value.chars().next().ok_or(anyhow::Error::msg("Empty string"))?;
        first.try_into()
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
