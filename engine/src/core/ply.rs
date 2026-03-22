use thiserror::Error;

use super::turn::Turn;
use crate::{core::color::colors, uci::tokens::Tokenizer};
use core::fmt;
use std::{num::ParseIntError, ops};

#[cfg(test)]
pub mod test;

#[derive(Default, Clone, Copy, Debug)]
pub struct FullMoveCount {
    pub v: u16,
}

impl fmt::Display for FullMoveCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.v)
    }
}

pub type FullMoveCountParseError = ParseIntError;

impl TryFrom<&str> for FullMoveCount {
    type Error = FullMoveCountParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.parse::<u16>() {
            Ok(v) => Ok(FullMoveCount { v }),
            Err(e) => Err(e),
        }
    }
}

impl From<Ply> for FullMoveCount {
    fn from(ply: Ply) -> Self {
        Self { v: ply.v / 2 }
    }
}

#[derive(Debug, Error)]
pub enum FullMoveCountTokenizationError {
    #[error("Invalid full move count: {0}")]
    InvalidToken(FullMoveCountParseError),

    #[error("Missing token for full move count.")]
    MissingToken,
}

impl TryFrom<&mut Tokenizer<'_>> for FullMoveCount {
    type Error = FullMoveCountTokenizationError;

    fn try_from(fen: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        match fen.next_token() {
            None => Err(Self::Error::MissingToken),
            Some(tok) => Self::try_from(tok).map_err(Self::Error::InvalidToken),
        }
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ply {
    pub v: u16,
}

impl Ply {
    /// # UCI mate score conversion
    /// Converts a sequence of plies into a UCI-compliant mate score (moves).
    pub fn to_mate_score(&self) -> i32 {
        ((self.v + 1) / 2) as i32
    }
}
impl fmt::Display for Ply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.v)
    }
}

impl_op!(-|a: Ply, b: Ply| -> Ply { Ply { v: a.v - b.v } });
impl_op!(-|a: Ply, b: u16| -> Ply { Ply { v: a.v - b } });
impl_op!(+ |a: Ply, b: Ply| -> Ply { Ply { v: a.v + b.v } } );
impl_op!(+ |a: Ply, b: u16| -> Ply { Ply { v: a.v + b } } );

impl From<(FullMoveCount, Turn)> for Ply {
    fn from(value: (FullMoveCount, Turn)) -> Self {
        let (fmc, turn) = value;
        match turn {
            colors::WHITE => Self { v: 2 * fmc.v },
            colors::BLACK => Self { v: 2 * fmc.v + 1 },
            _ => unreachable!("Invalid program state."),
        }
    }
}

pub type PlyParseError = ParseIntError;

impl TryFrom<&str> for Ply {
    type Error = PlyParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.parse::<u16>() {
            Ok(v) => Ok(Ply { v }),
            Err(e) => Err(e),
        }
    }
}

#[derive(Debug, Error)]
pub enum PlyTokenizationError {
    #[error("Invalid ply: {0}")]
    InvalidToken(PlyParseError),

    #[error("Missing token for ply.")]
    MissingToken,
}

impl TryFrom<&mut Tokenizer<'_>> for Ply {
    type Error = PlyTokenizationError;

    fn try_from(fen: &mut Tokenizer<'_>) -> Result<Self, Self::Error> {
        match fen.next_token() {
            None => Err(Self::Error::MissingToken),
            Some(tok) => Self::try_from(tok).map_err(Self::Error::InvalidToken),
        }
    }
}
