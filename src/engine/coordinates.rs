use crate::{misc::{ConstFrom, ParseError}, uci::tokens::Tokenizer};
use core::panic;
use std::ops;


pub type TCompassRose = isize;

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct CompassRose {
    v: TCompassRose,
}

impl CompassRose {
    pub const NORT: CompassRose = CompassRose { v: 8 };
    pub const EAST: CompassRose = CompassRose { v: 1 };
    pub const SOUT: CompassRose = CompassRose { v: -8 };
    pub const WEST: CompassRose = CompassRose { v: -1 };

    pub const SOWE: CompassRose = CompassRose { v: CompassRose::SOUT.v + CompassRose::WEST.v };
    pub const NOWE: CompassRose = CompassRose { v: CompassRose::NORT.v + CompassRose::WEST.v };
    pub const SOEA: CompassRose = CompassRose { v: CompassRose::SOUT.v + CompassRose::EAST.v };
    pub const NOEA: CompassRose = CompassRose { v: CompassRose::NORT.v + CompassRose::EAST.v };

    pub const NONOWE: CompassRose = CompassRose { v: 2 * CompassRose::NORT.v + CompassRose::WEST.v };
    pub const NONOEA: CompassRose = CompassRose { v: 2 * CompassRose::NORT.v + CompassRose::EAST.v };
    pub const NOWEWE: CompassRose = CompassRose { v: CompassRose::NORT.v + 2 * CompassRose::WEST.v };
    pub const NOEAEA: CompassRose = CompassRose { v: CompassRose::NORT.v + 2 * CompassRose::EAST.v };
    pub const SOSOWE: CompassRose = CompassRose { v: 2 * CompassRose::SOUT.v + CompassRose::WEST.v };
    pub const SOSOEA: CompassRose = CompassRose { v: 2 * CompassRose::SOUT.v + CompassRose::EAST.v };
    pub const SOWEWE: CompassRose = CompassRose { v: CompassRose::SOUT.v + 2 * CompassRose::WEST.v };
    pub const SOEAEA: CompassRose = CompassRose { v: CompassRose::SOUT.v + 2 * CompassRose::EAST.v };
    
    #[inline]
    pub const fn v(&self) -> isize { self.v }
    
    #[inline]
    pub const fn new(v: TCompassRose) -> Self { CompassRose { v } }
    
    #[inline]
    pub const fn double(&self) -> Self { CompassRose { v: self.v * 2 } }
    
    #[inline]
    pub const fn neg(&self) -> Self { CompassRose { v: -self.v } }
}

impl_op!(* |a: CompassRose, b: isize| -> CompassRose { CompassRose { v: (a.v) * b } } );
impl_op!(+ |a: CompassRose, b: isize| -> CompassRose { CompassRose { v: a.v + b } } );
impl_op!(+ |a: isize, b: CompassRose| -> CompassRose { CompassRose { v: a + b.v } } );

#[derive(Debug)]
pub enum Squares {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Square { v: u8 }

impl_op!(<< |a: usize, b: Square| -> usize { a << b.v } );

impl Square {
    pub const MIN: u8 = Squares::A1 as u8;
    pub const MAX: u8 = Squares::H8 as u8;

    #[inline]
    pub const fn v(&self) -> u8 {
        self.v
    }
    
    /// Create a square from a value in range [0, 64].
    /// This is unsafe, because the value is not checked.
    /// Only use this if you have certain knowledge of the v's range.
    #[inline]
    pub const unsafe fn from_v(v: u8) -> Self {
        Square { v }
    }
}

impl TryFrom<u8> for Square {
    type Error = ParseError;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
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

impl const ConstFrom<Squares> for Square {
    #[inline]
    fn from_c(value: Squares) -> Self {
        Square { v: value as u8 }
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