use crate::{impl_variants, misc::{ConstFrom, ParseError}, uci::tokens::Tokenizer};
use core::panic;
use std::ops;


#[derive(PartialEq, Copy, Clone, Debug)]
pub struct CompassRose { v: TCompassRose }

pub type TCompassRose = i8;

impl_variants! {
    TCompassRose as CompassRose {
        NORT = 8,
        EAST = 1,
        SOUT = -8,
        WEST = -1,

        SOWE = CompassRose::SOUT.v + CompassRose::WEST.v,
        NOWE = CompassRose::NORT.v + CompassRose::WEST.v,
        SOEA = CompassRose::SOUT.v + CompassRose::EAST.v,
        NOEA = CompassRose::NORT.v + CompassRose::EAST.v,

        NONOWE = 2 * CompassRose::NORT.v + CompassRose::WEST.v,
        NONOEA = 2 * CompassRose::NORT.v + CompassRose::EAST.v,
        NOWEWE = CompassRose::NORT.v + 2 * CompassRose::WEST.v,
        NOEAEA = CompassRose::NORT.v + 2 * CompassRose::EAST.v,
        SOSOWE = 2 * CompassRose::SOUT.v + CompassRose::WEST.v,
        SOSOEA = 2 * CompassRose::SOUT.v + CompassRose::EAST.v,
        SOWEWE = CompassRose::SOUT.v + 2 * CompassRose::WEST.v,
        SOEAEA = CompassRose::SOUT.v + 2 * CompassRose::EAST.v,
    }
}

impl CompassRose {
    #[inline]
    pub const fn new(v: TCompassRose) -> Self { CompassRose { v } }
    
    #[inline]
    pub const fn double(&self) -> Self { CompassRose { v: self.v * 2 } }
    
    #[inline]
    pub const fn neg(&self) -> Self { CompassRose { v: -self.v } }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Square { v: TSquare }

pub type TSquare = u8;

impl_variants! {
    TSquare as Square {
        A1, B1, C1, D1, E1, F1, G1, H1,
        A2, B2, C2, D2, E2, F2, G2, H2,
        A3, B3, C3, D3, E3, F3, G3, H3,
        A4, B4, C4, D4, E4, F4, G4, H4,
        A5, B5, C5, D5, E5, F5, G5, H5,
        A6, B6, C6, D6, E6, F6, G6, H6,
        A7, B7, C7, D7, E7, F7, G7, H7,
        A8, B8, C8, D8, E8, F8, G8, H8,
    }
}

impl_op!(<< |a: usize, b: Square| -> usize { a << b.v } );

impl Square {
    pub const MIN: TSquare = Square::A1.v;
    pub const MAX: TSquare = Square::H8.v;
    
    pub const fn mirror(self) -> Self {
        Self { v: self.v ^ 7 }
    }
}

impl TryFrom<TSquare> for Square {
    type Error = ParseError;

    #[inline]
    fn try_from(value: TSquare) -> Result<Self, Self::Error> {
        match value {
            Square::MIN..=Square::MAX => Ok(Square { v: value }),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

impl TryFrom<u16> for Square {
    type Error = ParseError;

    #[inline]
    fn try_from(value: u16) -> Result<Self, Self::Error> {
        const MIN: u16 = Square::MIN as u16;
        const MAX: u16 = Square::MAX as u16;
        match value {
            MIN..=MAX => Ok(Square { v: value as u8 }),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

impl const ConstFrom<(File, Rank)> for Square {
    #[inline]
    fn from_c(value: (File, Rank)) -> Self {
        Square{ v: value.0.v + value.1.v * 8u8 }
    }
}

impl TryFrom<&mut Tokenizer<'_>> for Option<Square> {
    type Error = ParseError;

    #[inline]
    fn try_from(tokens: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        let file = match tokens.next() {
            Some('-') => return Ok(None),
            Some(c) => File::try_from(c)?,
            None => return Err(ParseError::MissingInput),
        };
        let rank = match tokens.next() {
            Some(c) => Rank::try_from(c)?,
            None => return Err(ParseError::MissingInput),
        };
        Ok(Some(Square::from_c((file, rank))))
    }
}


#[derive(PartialEq, Debug)]
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
    
    #[inline]
    pub const fn v(&self) -> u8 {
        self.v
    }
}

impl const ConstFrom<Square> for Rank {
    #[inline]
    fn from_c(sq: Square) -> Self {
        Rank { v: sq.v / 8 }
    }
}

impl TryFrom<u8> for Rank {
    type Error = ParseError;
    
    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0..=7 => Ok(Rank { v: value }),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

impl TryFrom<char> for Rank {
    type Error = ParseError;
    
    #[inline]
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            '1'..='8' => Ok(Rank { v: value as u8 - '1' as u8 }),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

#[derive(PartialEq, Debug)]
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
    
    #[inline]
    pub const fn v(&self) -> u8 {
        self.v
    }

    pub const fn edge<const DIR: TCompassRose>() -> File {
        match CompassRose::new(DIR) {
            CompassRose::WEST => File::A,
            CompassRose::EAST => File::H,
            _ => panic!("The only two edge files are in the west and in the east."),
        }
    }
}

impl const ConstFrom<Square> for File {
    #[inline]
    fn from_c(sq: Square) -> Self {
        File { v: sq.v % 8 }
    }
}

impl TryFrom<u8> for File {
    type Error = ParseError;
    
    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0..=7 => Ok(File { v: value }),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

impl TryFrom<char> for File {
    type Error = ParseError;
    
    #[inline]
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'a'..='h' => Ok(File { v: value as u8 - 'a' as u8 }),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}


#[derive(PartialEq, Debug, Copy, Clone)]
pub struct DiagA1H8 { v: u8 }

impl DiagA1H8 {
    pub const fn v(&self) -> u8 {
        self.v
    }
}

impl const ConstFrom<Square> for DiagA1H8 {
    #[inline]
    fn from_c(sq: Square) -> Self {
        DiagA1H8 { v: 7 - Rank::from_c(sq).v + File::from_c(sq).v }
    }
}


#[derive(PartialEq, Debug)]
pub struct DiagA8H1 { v: u8 }

impl DiagA8H1 {
    pub const fn v(&self) -> u8 {
        self.v
    }
}

impl const ConstFrom<Square> for DiagA8H1 {
    #[inline]
    fn from_c(sq: Square) -> Self {
        DiagA8H1 { v: Rank::from_c(sq).v + File::from_c(sq).v }
    }
}