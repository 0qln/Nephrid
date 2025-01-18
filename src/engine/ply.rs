use crate::misc::ParseError;
use super::{fen::Fen, turn::Turn};
use std::ops;

#[derive(Default, Clone, Copy, Debug)]
pub struct FullMoveCount { pub v: u16 }

impl TryFrom<&str> for FullMoveCount {
    type Error = ParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.parse::<u16>() {
            Ok(v) => Ok(FullMoveCount { v }),
            Err(e) => return Err(ParseError::ParseIntError(e)),
        }
    }
}

impl TryFrom<&mut Fen<'_>> for FullMoveCount {
    type Error = ParseError;

    fn try_from(fen: &mut Fen<'_>) -> Result<Self, Self::Error> {
        match fen.collect_token() {
            None => return Err(ParseError::MissingInput),
            Some(tok) => FullMoveCount::try_from(tok.as_str()),
        }
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ply { pub v: u16 }

impl_op!(- |a: Ply, b: Ply| -> Ply { Ply { v: a.v - b.v } } );
impl_op!(- |a: Ply, b: u16| -> Ply { Ply { v: a.v - b } } );
impl_op!(+ |a: Ply, b: Ply| -> Ply { Ply { v: a.v + b.v } } );
impl_op!(+ |a: Ply, b: u16| -> Ply { Ply { v: a.v + b } } );

// todo: test
impl From<(FullMoveCount, Turn)> for Ply {
    fn from(value: (FullMoveCount, Turn)) -> Self {
        let (fmc, turn) = value;
        match turn {
            Turn::WHITE => Self { v: 2 * fmc.v },
            Turn::BLACK => Self { v: 2 * fmc.v + 1 },
            _ => unreachable!("Invalid program state."),
        }      
    }
}

impl TryFrom<&str> for Ply {
    type Error = ParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.parse::<u16>() {
            Ok(v) => Ok(Ply { v }),
            Err(e) => return Err(ParseError::ParseIntError(e)),
        }
    }
}

impl TryFrom<&mut Fen<'_>> for Ply {
    type Error = ParseError;

    fn try_from(fen: &mut Fen<'_>) -> Result<Self, Self::Error> {
        match fen.collect_token() {
            None => return Err(ParseError::MissingInput),
            Some(tok) => Ply::try_from(tok.as_str()),
        }
    }
}
