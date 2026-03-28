use core::fmt;
use move_flags as f;
use std::{
    fmt::Write,
    ops::{ControlFlow, Index, IndexMut},
};
use thiserror::Error;

use crate::{
    core::{
        bitboard::Bitboard,
        castling::{CastlingSideParseError, castling_sides},
        color::colors,
        coordinates::{EpCaptureSquare, File, Rank, Square, SquareTokenizationError},
        move_iter::{
            bishop::Bishop,
            fold_legal_moves,
            king::{self},
            knight,
            rook::Rook,
            sliding_piece::SlidingAttacks,
        },
        piece::{Piece, PromoPieceTokenizationError, piece_type},
        position::{CheckState, Position},
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

    /// If this move is a capture, returns the square on which the captured
    /// piece will be removed.
    pub fn get_capture_sq(&self) -> Option<Square> {
        let flag = self.get_flag();
        let to = self.get_to();
        match flag {
            f::EN_PASSANT => EpCaptureSquare::try_from(to).ok()?.v(),
            f if f.is_capture() => Some(to),
            _ => None,
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

    /// Convenience wrapper.
    #[inline]
    pub fn from_lan(lan: &str, context: &Position) -> Result<Move, MoveParseError> {
        let tok = &mut Tokenizer::new(lan);
        let mov = LongAlgebraicUciNotation::new(tok, context);
        Move::try_from(mov)
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
        }
        else if let Ok(promo) = PromoPieceType::try_from(self.get_flag()) {
            write!(f, "{}{}{}", self.get_from(), self.get_to(), promo)
        }
        else {
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

pub struct SAN<'a> {
    pub context: &'a Position,
    pub mov: Move,
}

impl<'a> fmt::Display for SAN<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let flag = self.mov.get_flag();

        match flag {
            move_flags::KING_CASTLE => return write!(f, "O-O"),
            move_flags::QUEEN_CASTLE => return write!(f, "O-O-O"),
            _ => (),
        };

        let from = self.mov.get_from();
        let from_bb = Bitboard::from_c(from);
        let from_file = File::from_c(from);
        let from_rank = Rank::from_c(from);
        let to = self.mov.get_to();
        let piece = self.context.get_piece(from);
        let piece_type = piece.piece_type();
        let stm = self.context.get_turn();

        if piece_type != piece_type::PAWN {
            // Convert to a white piece such that we print as uppercase
            let piece = Piece::from_c((colors::WHITE, piece_type));
            write!(f, "{piece}")?;
        }

        // 8.2.3.4: Disambiguation
        {
            let occupancy = self.context.get_occupancy();
            let pieces = self.context.get_bitboard(piece_type, stm);
            let froms = match piece_type {
                piece_type::PAWN => Bitboard::empty(), // pawn disambiguation is handled below
                piece_type::KNIGHT => knight::lookup_attacks(to),
                piece_type::BISHOP => Bishop::lookup_attacks(to, occupancy),
                piece_type::ROOK => Rook::lookup_attacks(to, occupancy),
                piece_type::QUEEN => Rook::lookup_attacks(to, occupancy),
                piece_type::KING => king::lookup_attacks(to),
                piece_type::NONE => Bitboard::empty(),
                _ => unreachable!(),
            };
            let candidates = froms & pieces;

            if candidates.pop_cnt_gt_1() {
                // Only investigate legality of the candidates once we found a pseudo legal
                // candidate.
                let legal_mask =
                    fold_legal_moves(self.context, Bitboard::empty(), |mut acc, mov| {
                        // Accumilate all from-squares, where there exists a legal move that
                        // has our to-square as destination.
                        if mov.get_to() == to {
                            acc |= Bitboard::from_c(mov.get_from())
                        }
                        ControlFlow::Continue::<(), _>(acc)
                    })
                    .continue_value()
                    .unwrap();

                let candidates = candidates & legal_mask;

                if candidates.pop_cnt_gt_1() {
                    // First, if the moving pieces can be distinguished by their originating files,
                    // the originating file letter of the moving piece is inserted immediately after
                    // the moving piece letter.
                    if candidates & Bitboard::from_c(from_file) == from_bb {
                        write!(f, "{from_file}")?;
                    }
                    // Second (when the first step fails), if the moving pieces can be distinguished
                    // by their originating ranks, the originating rank digit of the moving piece is
                    // inserted immediately after the moving piece letter.
                    else if candidates & Bitboard::from_c(from_rank) == from_bb {
                        write!(f, "{from_rank}")?;
                    }
                    // Third (when both the first and the second steps fail), the two character
                    // square coordinate of the originating square of the moving
                    // piece is inserted immediately after the moving piece
                    // letter.
                    else {
                        write!(f, "{from}")?;
                    }
                }
            }
        };

        // Captures
        if flag.is_capture() {
            if piece_type == piece_type::PAWN {
                write!(f, "{from_file}")?;
            }
            write!(f, "x")?;
        }

        // Destination Square
        write!(f, "{to}")?;

        // Promotions
        if flag.is_promo() {
            // Convert to a white piece such that we print as uppercase
            let promo = PromoPieceType::try_from(flag);
            let promo = promo.expect("We only go here if it actually is a promo");
            let promo = Piece::from_c((colors::WHITE, promo));
            write!(f, "={promo}")?;
        }

        // 8.2.3.5: Check and checkmate indication characters
        {
            // todo: find a more efficient way to check this instead of cloning
            let mut pos = self.context.clone();
            pos.make_move(self.mov);
            if pos.get_check_state() != CheckState::None {
                // If the move is a checking move, the plus sign "+" is appended as a suffix to
                // the basic SAN move notation; if the move is a checkmating move, the
                // octothorpe sign "#" is appended instead.
                f.write_char(if pos.has_legal_moves() { '+' } else { '#' })?;
            }
        }

        Ok(())
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
                    _ => match PromoPieceType::try_from(move_notation.tokens) {
                        Ok(promo_piece) => MoveFlag::from((promo_piece, captures)),
                        Err(PromoPieceTokenizationError::MissingToken) => flag,
                        Err(error) => return Err(MoveParseError::InvalidPromoPieceType(error)),
                    },
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
    /// Converts a move to a index, such that in any given position, no two
    /// moves will have the same index and there are few gaps.
    fn from(mov: Move) -> Self {
        let flag = mov.get_flag();
        let result = if flag.is_promo() {
            // Promotions are special cases:
            // 1. Since promotions have multiple moves for the same from-to combination, we
            //    add a variance for different promotions.
            // 2. We need to bias, such that we don't collide with valid from-to indeces.
            //    (SQ_MASK)
            // 3. We also need to bias by the file of the from square, such that we don't
            //    collide with other promotions.
            let min_index = Move::MASK_SQ + 1;
            let min_promo_flag = move_flags::PROMOTION_KNIGHT.v() as u16;
            let flag_off = flag.v() as u16 - min_promo_flag;
            let file_off = File::from_c(mov.get_from()).v() as u16;
            min_index + flag_off + file_off
        }
        else {
            // For most moves, we can just use the from and to squares to get a unique index
            // for any set of moves of any position.
            mov.v & Move::MASK_SQ
        };
        result as usize
    }
}

// todo
// impl TryFrom<(usize, &Position)> for Move {
//     type Error = ValueOutOfRangeError<usize>;

//     /// Inverse of the above...
//     fn try_from(value: (usize, &Position)) -> Result<Self, Self::Error> {
//         if value > Move::MASK_SQ {
//             // promo
//         } else {
//             // normal
//         }
//     }
// }

/// A list of moves in a single position.
/// Since the 218 is the maximum number of moves in a single position,
/// we can use a fixed length array to store the moves and by using a
/// size of 256 we can safely index into the array with a u8.
#[derive(Debug, Clone)]
pub struct MoveList([Move; 256]);

impl MoveList {
    /// Returns a mutable slice of the initialized moves up to `len`.
    pub fn as_mut_slice(&mut self, len: u8) -> &mut [Move] {
        // SAFETY: len is guaranteed to be <= 256 because it's a u8.
        unsafe { self.0.get_unchecked_mut(..len as usize) }
    }

    /// Returns an iterator over the initialized moves up to the first null
    /// move.
    ///
    /// NOTE: This could maybe be written more efficently. And also, just use
    /// indexing instead maybe, since we can easily prove the bounds checks
    /// in most cases.
    pub fn iter(&self) -> impl Iterator<Item = Move> {
        self.0
            .iter()
            .take_while(|&mov| *mov != Move::null())
            .copied()
    }
}

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

impl fmt::Display for MoveList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let moves = self
            .0
            .iter()
            .take_while(|mov| mov.v != 0)
            .map(|mov| mov.to_string());
        write!(f, "[{}]", moves.collect::<Vec<_>>().join(", "))
    }
}
