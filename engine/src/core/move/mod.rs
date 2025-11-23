use core::fmt;
use move_flags as f;
use std::ops::{Index, IndexMut};
use thiserror::Error;

use crate::{
    core::{
        castling::{CastlingSideParseError, castling_sides},
        coordinates::{File, Square, SquareTokenizationError},
        piece::{PromoPieceTokenizationError, piece_type},
        position::Position,
    },
    impl_variants,
    misc::{ConstFrom, ValueOutOfRangeError},
    uci::tokens::Tokenizer,
};

use super::{castling::CastlingSide, piece::PromoPieceType};

#[cfg(test)]
mod test;

pub struct LongAlgebraicNotation;

pub struct LongAlgebraicUciNotation<'a, 'b, 'c> {
    pub tokens: &'a mut Tokenizer<'c>,
    pub context: &'b Position,
}

impl<'a, 'b, 'c> LongAlgebraicUciNotation<'a, 'b, 'c> {
    pub const fn new(tokenizer: &'a mut Tokenizer<'c>, position: &'b Position) -> Self {
        Self {
            tokens: tokenizer,
            context: position,
        }
    }
}

pub struct StandardAlgebraicNotation;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct MoveFlag {
    v: TMoveFlag,
}

pub type TMoveFlag = u8;

impl_variants! {
    TMoveFlag as MoveFlag in move_flags {
        QUIET,
        DOUBLE_PAWN_PUSH,
        PROMOTION_KNIGHT,
        PROMOTION_BISHOP,
        PROMOTION_ROOK,
        PROMOTION_QUEEN,
        CAPTURE_PROMOTION_KNIGHT,
        CAPTURE_PROMOTION_BISHOP,
        CAPTURE_PROMOTION_ROOK,
        CAPTURE_PROMOTION_QUEEN,
        KING_CASTLE,
        QUEEN_CASTLE,
        CAPTURE,
        EN_PASSANT,
    }
}

impl fmt::Debug for MoveFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variant = match *self {
            f::QUIET => "QUIET",
            f::DOUBLE_PAWN_PUSH => "DOUBLE_PAWN_PUSH",
            f::PROMOTION_KNIGHT => "PROMOTION_KNIGHT",
            f::PROMOTION_BISHOP => "PROMOTION_BISHOP",
            f::PROMOTION_ROOK => "PROMOTION_ROOK",
            f::PROMOTION_QUEEN => "PROMOTION_QUEEN",
            f::CAPTURE_PROMOTION_KNIGHT => "CAPTURE_PROMOTION_KNIGHT",
            f::CAPTURE_PROMOTION_BISHOP => "CAPTURE_PROMOTION_BISHOP",
            f::CAPTURE_PROMOTION_ROOK => "CAPTURE_PROMOTION_ROOK",
            f::CAPTURE_PROMOTION_QUEEN => "CAPTURE_PROMOTION_QUEEN",
            f::KING_CASTLE => "KING_CASTLE",
            f::QUEEN_CASTLE => "QUEEN_CASTLE",
            f::CAPTURE => "CAPTURE",
            f::EN_PASSANT => "EN_PASSANT",
            _ => unreachable!(),
        };
        f.debug_struct("MoveFlag").field("v", &variant).finish()
    }
}

impl MoveFlag {
    #[inline]
    pub const fn is_capture(&self) -> bool {
        self.v == f::CAPTURE.v
            || self.v == f::EN_PASSANT.v
            || (self.v >= f::CAPTURE_PROMOTION_KNIGHT.v && self.v <= f::CAPTURE_PROMOTION_QUEEN.v)
    }

    #[inline]
    pub const fn is_promo(&self) -> bool {
        self.v >= f::PROMOTION_KNIGHT.v && self.v <= f::CAPTURE_PROMOTION_QUEEN.v
    }
}

impl From<(PromoPieceType, bool)> for MoveFlag {
    fn from((piece_type, captures): (PromoPieceType, bool)) -> Self {
        let mut v = piece_type.v().v();
        if captures {
            v += 4;
        }
        Self { v }
    }
}

impl TryFrom<TMoveFlag> for MoveFlag {
    type Error = ValueOutOfRangeError<TMoveFlag>;

    #[inline]
    fn try_from(value: TMoveFlag) -> Result<Self, Self::Error> {
        match value {
            0..=13 => Ok(MoveFlag { v: value }),
            x => Err(ValueOutOfRangeError::new(x, 0..=13)),
        }
    }
}

impl const ConstFrom<CastlingSide> for MoveFlag {
    fn from_c(value: CastlingSide) -> Self {
        match value {
            castling_sides::KING_SIDE => f::KING_CASTLE,
            castling_sides::QUEEN_SIDE => f::QUEEN_CASTLE,
            _ => unreachable!(),
        }
    }
}

#[derive(Default, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Move {
    v: u16,
}

impl Move {
    const SHIFT_FROM: u16 = 0;
    const SHIFT_TO: u16 = 6;
    const SHIFT_FLAG: u16 = 12;

    const MASK_FROM: u16 = 0b111111 << Move::SHIFT_FROM;
    const MASK_TO: u16 = 0b111111 << Move::SHIFT_TO;
    const MASK_FLAG: u16 = 0b1111 << Move::SHIFT_FLAG;
    const MASK_SQ: u16 = Move::MASK_FROM | Move::MASK_TO;

    #[inline]
    pub const fn null() -> Self {
        Move { v: 0 }
    }

    #[inline]
    pub const fn new(from: Square, to: Square, flag: MoveFlag) -> Self {
        Move {
            v: ((from.v() as u16) << Move::SHIFT_FROM)
                | ((to.v() as u16) << Move::SHIFT_TO)
                | ((flag.v as u16) << Move::SHIFT_FLAG),
        }
    }

    #[inline]
    pub const fn get_from(&self) -> Square {
        // Safety: 6 bits can only ever contain a value in range 0..64
        unsafe {
            let val = (self.v & Move::MASK_FROM) >> Move::SHIFT_FROM;
            Square::from_v(val as u8)
        }
    }

    #[inline]
    pub const fn get_to(&self) -> Square {
        // Safety: 6 bits can only ever contain a value in range 0..64
        unsafe {
            let val = (self.v & Move::MASK_TO) >> Move::SHIFT_TO;
            Square::from_v(val as u8)
        }
    }

    #[inline]
    pub fn get_flag(&self) -> MoveFlag {
        // Safety: The inner move flag bits are only ever set from a MoveFlag struct.
        unsafe {
            let val = (self.v & Move::MASK_FLAG) >> Move::SHIFT_FLAG;
            MoveFlag::try_from(val as u8).unwrap_unchecked()
        }
    }
}

impl From<Move> for (Square, Square, MoveFlag) {
    #[inline]
    fn from(value: Move) -> Self {
        (value.get_from(), value.get_to(), value.get_flag())
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.v == 0 {
            write!(f, "0000")
        } else if let Ok(promo) = PromoPieceType::try_from(self.get_flag()) {
            write!(f, "{}{}{}", self.get_from(), self.get_to(), promo)
        } else {
            write!(f, "{}{}", self.get_from(), self.get_to())
        }
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Move")
            .field("from", &self.get_from())
            .field("to", &self.get_to())
            .field("flag", &self.get_flag())
            .finish()
    }
}

#[derive(Error, Debug)]
pub enum MoveParseError {
    #[error("Invalid to-square: {0}")]
    InvalidToSquare(SquareTokenizationError),

    #[error("Invalid from-square: {0}")]
    InvalidFromSquare(SquareTokenizationError),

    #[error("Invalid promotion piece type: {0}")]
    InvalidPromoPieceType(PromoPieceTokenizationError),

    #[error("Expected a legal castling move, but destination file was unexpected: {0}")]
    IllegalCastling(CastlingSideParseError),
}

impl TryFrom<LongAlgebraicUciNotation<'_, '_, '_>> for Move {
    type Error = MoveParseError;

    fn try_from(move_notation: LongAlgebraicUciNotation<'_, '_, '_>) -> Result<Self, Self::Error> {
        let from = Square::try_from(&mut *move_notation.tokens)
            .map_err(MoveParseError::InvalidFromSquare)?;

        let to = Square::try_from(&mut *move_notation.tokens)
            .map_err(MoveParseError::InvalidToSquare)?;

        let moving_p = move_notation.context.get_piece(from);
        let captured_p = move_notation.context.get_piece(to);
        let abs_dist = from.v().abs_diff(to.v());
        let captures = captured_p.piece_type() != piece_type::NONE;
        let mut flag = if captures { f::CAPTURE } else { f::QUIET };

        match moving_p.piece_type() {
            piece_type::PAWN => {
                flag = match abs_dist {
                    16 => f::DOUBLE_PAWN_PUSH,
                    7 | 9 if !captures => f::EN_PASSANT,
                    _ => {
                        let promo_piece = PromoPieceType::try_from(&mut *move_notation.tokens)
                            .map_err(MoveParseError::InvalidPromoPieceType)?;

                        MoveFlag::from((promo_piece, captures))
                        // move_notation.tokens.next_char().map_or(Ok(flag), |c|
                        // { Ok() })?
                    }
                }
            }
            piece_type::KING if abs_dist == 2 => {
                let file = File::from_c(to);
                let side = CastlingSide::try_from(file).map_err(Self::Error::IllegalCastling)?;
                flag = MoveFlag::from_c(side);
            }
            _ => {}
        };

        Ok(Move::new(from, to, flag))
    }
}

impl From<Move> for usize {
    /// Converts a move to a index, such that in any given position, no two moves will have the same index and there are no gaps.
    fn from(mov: Move) -> Self {
        let from_file = File::from_c(mov.get_from());
        let promo_bias = Move::MASK_SQ + 1 + from_file.v() as u16;
        (match mov.get_flag() {
            // Promotions are special cases:
            // 1. Since promotions have multiple moves for the same from-to combination, we add a variance for different promotions.
            // 2. We need to bias, such that we don't collide with valid from-to indeces. (SQ_MASK)
            // 3. We also need to bias by the file of the from square, such that we don't collide with other promotions.
            f if f.is_promo() => promo_bias + f.v() as u16 - 2,
            // For most moves, we can just use the from and to squares.
            _ => mov.v & Move::MASK_SQ,
        }) as usize
    }
}

/// A list of moves in a single position.
/// Since the 218 is the maximum number of moves in a single position,
/// we can use a fixed length array to store the moves and by using a
/// size of 256 we can safely index into the array with a u8.
#[derive(Debug, Clone)]
pub struct MoveList([Move; 256]);

impl Default for MoveList {
    fn default() -> Self {
        Self([Move::default(); 256])
    }
}

impl Index<u8> for MoveList {
    type Output = Move;

    fn index(&self, index: u8) -> &Self::Output {
        unsafe { self.0.get_unchecked(index as usize) }
    }
}

impl IndexMut<u8> for MoveList {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        unsafe { self.0.get_unchecked_mut(index as usize) }
    }
}
