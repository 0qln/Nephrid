use crate::{
    misc::{ConstFrom, MissingTokenError, ValueOutOfRangeError, ValueOutOfSetError},
    mk_variant_utils, mk_variants,
    uci::tokens::Tokenizer,
};
use core::{fmt, panic};
use std::{iter::Step, ops};

use super::color::Color;
use squares::*;
use terrors::OneOf;

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct CompassRose {
    v: TCompassRose,
}

pub type TCompassRose = i8;

pub mod compass_rose {
    use super::*;
    mk_variants! {
        TCompassRose as CompassRose {
            NORT = 8,
            EAST = 1,
            SOUT = -8,
            WEST = -1,

            NONO = 2 * NORT.v,
            EAEA = 2 * EAST.v,
            SOSO = 2 * SOUT.v,
            WEWE = 2 * WEST.v,

            SOWE = SOUT.v + WEST.v,
            NOWE = NORT.v + WEST.v,
            SOEA = SOUT.v + EAST.v,
            NOEA = NORT.v + EAST.v,

            NONOWE = NONO.v + WEST.v,
            NONOEA = NONO.v + EAST.v,
            NOWEWE = NORT.v + WEWE.v,
            NOEAEA = NORT.v + EAEA.v,
            SOSOWE = SOSO.v + WEST.v,
            SOSOEA = SOSO.v + EAST.v,
            SOWEWE = SOUT.v + WEWE.v,
            SOEAEA = SOUT.v + EAEA.v,
        }
    }
}

mk_variant_utils! {TCompassRose as CompassRose}

impl CompassRose {
    #[inline]
    pub const fn new(v: TCompassRose) -> Self {
        CompassRose { v }
    }

    #[inline]
    pub const fn double(&self) -> Self {
        CompassRose { v: self.v * 2 }
    }

    #[inline]
    pub const fn neg(&self) -> Self {
        CompassRose { v: -self.v }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Square {
    v: TSquare,
}

pub type TSquare = u8;

pub mod squares {
    use super::*;
    mk_variants! {
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
    mk_variants! {
        TSquare as Square {
            MIN = A1.v,
            MAX = H8.v
        }
    }
}

mk_variant_utils! {TSquare as Square}

impl_op!(<< |a: usize, b: Square| -> usize { a << b.v } );
impl_op!(% |a: Square, b: u8| -> Square { Square { v: a.v % b } } );
impl_op!(-|a: Square, b: Square| -> Square { Square { v: a.v - b.v } });

impl Square {
    pub const fn flip_h(self) -> Self {
        Self { v: self.v ^ 7 }
    }

    pub const fn flip_v(self) -> Self {
        Self { v: self.v ^ 56 }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", File::from_c(*self), Rank::from_c(*self))
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}{:?}", File::from_c(*self), Rank::from_c(*self))
    }
}

impl TryFrom<TSquare> for Square {
    type Error = OneOf<(ValueOutOfRangeError<TSquare>,)>;

    #[inline]
    fn try_from(value: TSquare) -> Result<Self, Self::Error> {
        match value {
            MIN_C..=MAX_C => Ok(Square { v: value }),
            x => Err(OneOf::new(ValueOutOfRangeError::new(x, MIN_C..=MAX_C))),
        }
    }
}

// impl TryFrom<u16> for Square {
//     type Error = OneOf<(ValueOutOfRangeError<u16>,)>;

//     #[inline]
//     fn try_from(value: u16) -> Result<Self, Self::Error> {
//         match value {
//             MIN_C..=MAX_C => Ok(Square { v: value as u8 }),
//             x => Err(OneOf::new(ValueOutOfRangeError::new(x,
// MIN_C..=MAX_C))),         }
//     }
// }

impl const ConstFrom<(File, Rank)> for Square {
    #[inline]
    fn from_c(value: (File, Rank)) -> Self {
        Square { v: value.0.v + value.1.v * 8u8 }
    }
}

impl TryFrom<&mut Tokenizer<'_>> for Square {
    type Error = OneOf<(
        ValueOutOfRangeError<u16>,
        ValueOutOfRangeError<char>,
        MissingTokenError,
    )>;

    #[inline]
    fn try_from(tokens: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        let file = match tokens.next_char() {
            Some(c) => File::try_from(c).map_err(OneOf::broaden)?,
            None => return Err(OneOf::new(MissingTokenError::new("File"))),
        };
        let rank = match tokens.next_char() {
            Some(c) => Rank::try_from(c).map_err(OneOf::broaden)?,
            None => return Err(OneOf::new(MissingTokenError::new("Rank"))),
        };
        Ok(Square::from_c((file, rank)))
    }
}

impl Step for Square {
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
        Step::steps_between(&start.v, &end.v)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        Self::try_from(Step::forward_checked(start.v, count)?).ok()
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Self::try_from(Step::backward_checked(start.v, count)?).ok()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct EpTargetSquare {
    v: Option<Square>,
}

impl EpTargetSquare {
    pub const fn v(&self) -> Option<Square> {
        self.v
    }
}

impl TryFrom<Square> for EpTargetSquare {
    type Error = OneOf<(ValueOutOfSetError<Rank>,)>;

    #[inline]
    fn try_from(sq: Square) -> Result<Self, Self::Error> {
        let rank = Rank::from_c(sq);
        match rank {
            Rank::_3 | Rank::_6 => Ok(Self { v: Some(sq) }),
            x => Err(OneOf::new(ValueOutOfSetError {
                value: x,
                expected: &[Rank::_3, Rank::_6],
            })),
        }
    }
}

impl TryFrom<&mut Tokenizer<'_>> for EpTargetSquare {
    type Error = ParseError;

    #[inline]
    fn try_from(tokens: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        let file = match tokens.next_char() {
            Some('-') => return Ok(Self { v: None }),
            Some(c) => File::try_from(c)?,
            None => return Err(ParseError::MissingValue),
        };
        let rank = match tokens.next_char() {
            Some(c) => Rank::try_from(c)?,
            None => return Err(ParseError::MissingValue),
        };
        let sq = Square::from_c((file, rank));
        Self::try_from(sq)
    }
}

// the color is the color of the pawn being captured
impl From<(EpCaptureSquare, Color)> for EpTargetSquare {
    #[inline]
    fn from((sq, color): (EpCaptureSquare, Color)) -> Self {
        Self {
            v: sq.v.map(|sq| Square {
                v: (sq.v as i8 + (color.v() as i8 * 2 - 1) * 8) as u8,
            }),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct EpCaptureSquare {
    v: Option<Square>,
}

impl EpCaptureSquare {
    pub const fn v(&self) -> Option<Square> {
        self.v
    }
}

impl TryFrom<Square> for EpCaptureSquare {
    type Error = ParseError;

    #[inline]
    fn try_from(sq: Square) -> Result<Self, Self::Error> {
        let rank = Rank::from_c(sq);
        match rank {
            Rank::_4 | Rank::_5 => Ok(Self { v: Some(sq) }),
            _ => Err(ParseError::InputOutOfRange(sq.to_string())),
        }
    }
}

// the color is the color of the pawn being captured
impl From<(EpTargetSquare, Color)> for EpCaptureSquare {
    #[inline]
    fn from((sq, color): (EpTargetSquare, Color)) -> Self {
        Self {
            v: sq.v.map(|sq| Square {
                v: (sq.v as i8 - (color.v() as i8 * 2 - 1) * 8) as u8,
            }),
        }
    }
}

#[derive(PartialEq, PartialOrd, Copy, Clone)]
pub struct Rank {
    v: TRank,
}

pub type TRank = u8;

impl_op!(*|a: u8, b: Rank| -> Rank { Rank { v: a * b.v } });
impl_op!(*|a: Color, b: Rank| -> Rank { Rank { v: a.v() * b.v } });

pub mod ranks {
    use super::*;
    mk_variants! {
        TRank as Rank {
            _1, _2, _3, _4, _5, _6, _7, _8
        }
    }
}

mk_variant_utils! {TRank as Rank}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.v + 1)
    }
}

impl fmt::Debug for Rank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.v + 1)
    }
}

impl const ConstFrom<Square> for Rank {
    #[inline]
    fn from_c(sq: Square) -> Self {
        Rank { v: sq.v / 8 }
    }
}

impl From<Rank> for i8 {
    fn from(value: Rank) -> Self {
        value.v as i8
    }
}

impl TryFrom<u8> for Rank {
    type Error = ParseError;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0..=7 => Ok(Rank { v: value }),
            x => Err(ParseError::InputOutOfRange(x.to_string())),
        }
    }
}

impl TryFrom<char> for Rank {
    type Error = ParseError;

    #[inline]
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            '1'..='8' => Ok(Rank { v: value as u8 - b'1' }),
            x => Err(ParseError::InputOutOfRange(x.to_string())),
        }
    }
}

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct File {
    v: TFile,
}

pub type TFile = u8;

pub mod files {
    use super::*;
    mk_variants! {
        TFile as File {
            A, B, C, D, E, F, G, H
        }
    }
}

impl File {
    pub const fn edge<const DIR: TCompassRose>() -> File {
        match CompassRose::new(DIR) {
            CompassRose::WEST => File::A,
            CompassRose::EAST => File::H,
            _ => panic!("The only two edge files are in the west and in the east."),
        }
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Into::<char>::into(*self))
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Into::<char>::into(*self))
    }
}

impl const ConstFrom<Square> for File {
    #[inline]
    fn from_c(sq: Square) -> Self {
        File { v: sq.v % 8 }
    }
}

impl TryFrom<u8> for File {
    type Error = OneOf<(ValueOutOfRangeError<u8>,)>;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0..=7 => Ok(File { v: value }),
            x => Err(OneOf::new(ValueOutOfRangeError::new(x, 0..=7))),
        }
    }
}

impl TryFrom<char> for File {
    type Error = OneOf<(ValueOutOfRangeError<char>,)>;

    #[inline]
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'a'..='h' => Ok(File { v: value as u8 - b'a' }),
            x => Err(OneOf::new(ValueOutOfRangeError::new(x, 'a'..='h'))),
        }
    }
}

impl From<File> for char {
    fn from(val: File) -> Self {
        (val.v + b'a') as char
    }
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct DiagA1H8 {
    v: u8,
}

impl DiagA1H8 {
    pub const fn v(&self) -> u8 {
        self.v
    }
}

impl const ConstFrom<Square> for DiagA1H8 {
    #[inline]
    fn from_c(sq: Square) -> Self {
        DiagA1H8 {
            v: 7 - Rank::from_c(sq).v + File::from_c(sq).v,
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct DiagA8H1 {
    v: u8,
}

impl DiagA8H1 {
    pub const fn v(&self) -> u8 {
        self.v
    }
}

impl const ConstFrom<Square> for DiagA8H1 {
    #[inline]
    fn from_c(sq: Square) -> Self {
        DiagA8H1 {
            v: Rank::from_c(sq).v + File::from_c(sq).v,
        }
    }
}
