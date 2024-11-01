use crate::uci::tokens::Tokenizer;
use std::ops;

pub enum CompassRose {
    Nort = 8,
    East = 1,
    Sout = -8,
    West = -1,

    SoWe = CompassRose::Sout as isize + CompassRose::West as isize,
    NoWe = CompassRose::Nort as isize + CompassRose::West as isize,
    SoEa = CompassRose::Sout as isize + CompassRose::East as isize,
    NoEa = CompassRose::Nort as isize + CompassRose::East as isize,

    NoNoWe = 2 * CompassRose::Nort as isize + CompassRose::West as isize,
    NoNoEa = 2 * CompassRose::Nort as isize + CompassRose::East as isize,
    NoWeWe = CompassRose::Nort as isize + 2 * CompassRose::West as isize,
    NoEaEa = CompassRose::Nort as isize + 2 * CompassRose::East as isize,
    SoSoWe = 2 * CompassRose::Sout as isize + CompassRose::West as isize,
    SoSoEa = 2 * CompassRose::Sout as isize + CompassRose::East as isize,
    SoWeWe = CompassRose::Sout as isize + 2 * CompassRose::West as isize,
    SoEaEa = CompassRose::Sout as isize + 2 * CompassRose::East as isize
}

impl_op!(* |a: CompassRose, b: isize| -> isize { (a as isize) * b } );

#[derive(Debug)]
pub enum Squares {
    A1, A2, A3, A4, A5, A6, A7, A8,
    B1, B2, B3, B4, B5, B6, B7, B8,
    C1, C2, C3, C4, C5, C6, C7, C8,
    D1, D2, D3, D4, D5, D6, D7, D8,
    E1, E2, E3, E4, E5, E6, E7, E8,
    F1, F2, F3, F4, F5, F6, F7, F8,
    G1, G2, G3, G4, G5, G6, G7, G8,
    H1, H2, H3, H4, H5, H6, H7, H8,
    None
}

#[derive(Copy, Clone)]
pub struct Square { v: u8 }

impl Square {
    #[inline]
    pub const fn v(&self) -> u8 {
        self.v
    }
    
    pub const NONE: Square = Square { v: Squares::None as u8 };
}

impl Into<usize> for Square {
    #[inline]
    fn into(self) -> usize {
        self.v as usize
    }
}

impl Default for Square {
    #[inline]
    fn default() -> Self {
        Square::from(Squares::None)
    }
}

impl TryFrom<u8> for Square {
    type Error = anyhow::Error;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0..=64 => Ok(Square { v: value }),
            _ => Err(anyhow::Error::msg("Square value out of range")),
        }
    }
}

impl TryFrom<u16> for Square {
    type Error = anyhow::Error;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::try_from(value as u8)
    }
}

impl From<Squares> for Square {
    #[inline]
    fn from(value: Squares) -> Self {
        Square { v: value as u8 }
    }
}

impl From<(File, Rank)> for Square {
    #[inline]
    fn from(value: (File, Rank)) -> Self {
        Square{ v: value.0.v + value.1.v * 8u8 }
    }
}

impl TryFrom<&mut Tokenizer<'_>> for Square {
    type Error = anyhow::Error;

    #[inline]
    fn try_from(tokens: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        let file = match tokens.next() {
            Some('-') => return Ok(Square::from(Squares::None)),
            Some(c) => File::try_from(c)?,
            None => return Err(anyhow::Error::msg("Empty string")),
        };
        let rank = match tokens.next() {
            Some(c) => Rank::try_from(c)?,
            None => return Err(anyhow::Error::msg("No rank specified")),
        };
        Ok(Square::from((file, rank)))
    }
}

impl From<&str> for Square {
    #[inline]
    fn from(value: &str) -> Self {
        todo!()
    }
}


#[derive(PartialEq)]
pub struct Rank { v: u8 }

impl Rank {
    pub const _1: Rank = Rank { v: 0 }; 
    pub const _2: Rank = Rank { v: 1 }; 
    pub const _3: Rank = Rank { v: 2 }; 
    pub const _4: Rank = Rank { v: 3 }; 
    pub const _5: Rank = Rank { v: 4 }; 
    pub const _6: Rank = Rank { v: 5 }; 
    pub const _7: Rank = Rank { v: 6 }; 
    pub const _8: Rank = Rank { v: 7 };
}

impl From<Square> for Rank {
    #[inline]
    fn from(sq: Square) -> Self {
        Rank { v: sq.v / 8 }
    }
}

impl TryFrom<u8> for Rank {
    type Error = anyhow::Error;
    
    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0..=7 => Ok(Rank { v: value }),
            _ => Err(anyhow::Error::msg("Rank value out of range")),
        }
    }
}

impl TryFrom<char> for Rank {
    type Error = anyhow::Error;
    
    #[inline]
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            '1'..='8' => Ok(Rank { v: value as u8 - '1' as u8 }),
            _ => Err(anyhow::Error::msg("Invalid char")),
        }
    }
}

#[derive(PartialEq)]
pub struct File { v: u8 }

impl File {
    pub const A: File = File { v: 0 }; 
    pub const B: File = File { v: 1 }; 
    pub const C: File = File { v: 2 }; 
    pub const D: File = File { v: 3 }; 
    pub const E: File = File { v: 4 }; 
    pub const F: File = File { v: 5 }; 
    pub const G: File = File { v: 6 }; 
    pub const H: File = File { v: 7 };
}

impl Into<u8> for File {
    #[inline]
    fn into(self) -> u8 {
        self.v
    }
}

impl From<Square> for File {
    #[inline]
    fn from(sq: Square) -> Self {
        File { v: sq.v % 8 }
    }
}

impl TryFrom<u8> for File {
    type Error = anyhow::Error;
    
    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0..=7 => Ok(File { v: value }),
            _ => Err(anyhow::Error::msg("File value out of range")),
        }
    }
}

impl TryFrom<char> for File {
    type Error = anyhow::Error;
    
    #[inline]
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'a'..='h' => Ok(File { v: value as u8 - 'a' as u8 }),
            _ => Err(anyhow::Error::msg("Invalid char")),
        }
    }
}
