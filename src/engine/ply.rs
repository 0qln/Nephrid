use crate::misc::ParseError;
use super::{fen::Fen, turn::Turn};

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

#[derive(Default, Clone, Copy, Debug)]
pub struct Ply { pub v: u16 }

// todo: test
impl From<(FullMoveCount, Turn)> for Ply {
    fn from(value: (FullMoveCount, Turn)) -> Self {
        let (fmc, turn) = value;
        match turn {
            Turn::White => Self { v: 2 * fmc.v },
            Turn::Black => Self { v: 2 * fmc.v + 1 },
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