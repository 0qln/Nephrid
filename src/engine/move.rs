use core::fmt;

use crate::{
    engine::{
        coordinates::{ File, Square}, 
        piece::PieceType, position::Position 
    }, impl_variants, misc::{ConstFrom, ParseError}, uci::tokens::Tokenizer
};

use super::{castling::CastlingSide, piece::PromoPieceType};

pub struct LongAlgebraicNotation;

pub struct LongAlgebraicUciNotation<'a, 'b, 'c> {
    pub tokens: &'a mut Tokenizer<'c>,
    pub context: &'b Position,
}

impl<'a, 'b, 'c> LongAlgebraicUciNotation<'a, 'b, 'c> {
    pub const fn new(tokenizer: &'a mut Tokenizer<'c>, position: &'b Position) -> Self {
        Self {
            tokens: tokenizer,
            context: position
        }
    }
}

pub struct StandardAlgebraicNotation;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct MoveFlag { v: TMoveFlag }

pub type TMoveFlag = u8;

impl_variants! {
    TMoveFlag as MoveFlag {
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
            MoveFlag::QUIET => "QUIET",
            MoveFlag::DOUBLE_PAWN_PUSH => "DOUBLE_PAWN_PUSH",
            MoveFlag::PROMOTION_KNIGHT => "PROMOTION_KNIGHT",
            MoveFlag::PROMOTION_BISHOP => "PROMOTION_BISHOP",
            MoveFlag::PROMOTION_ROOK => "PROMOTION_ROOK",
            MoveFlag::PROMOTION_QUEEN => "PROMOTION_QUEEN",
            MoveFlag::CAPTURE_PROMOTION_KNIGHT => "CAPTURE_PROMOTION_KNIGHT",
            MoveFlag::CAPTURE_PROMOTION_BISHOP => "CAPTURE_PROMOTION_BISHOP",
            MoveFlag::CAPTURE_PROMOTION_ROOK => "CAPTURE_PROMOTION_ROOK",
            MoveFlag::CAPTURE_PROMOTION_QUEEN => "CAPTURE_PROMOTION_QUEEN",
            MoveFlag::KING_CASTLE => "KING_CASTLE",
            MoveFlag::QUEEN_CASTLE => "QUEEN_CASTLE",
            MoveFlag::CAPTURE => "CAPTURE",
            MoveFlag::EN_PASSANT => "EN_PASSANT",
            _ => unreachable!()
        };
        f.debug_struct("MoveFlag").field("v", &variant).finish()
    }
}
    
impl MoveFlag {
    #[inline]
    pub const fn is_capture(&self) -> bool {
        self.v == Self::CAPTURE.v ||
        self.v == Self::EN_PASSANT.v ||
        self.v >= Self::CAPTURE_PROMOTION_KNIGHT.v && self.v <= Self::CAPTURE_PROMOTION_QUEEN.v
    }
    
    #[inline]
    pub const fn is_promo(&self) -> bool {
        self.v >= Self::PROMOTION_KNIGHT.v && self.v <= Self::CAPTURE_PROMOTION_QUEEN.v
    }
}

impl From<(PromoPieceType, bool)> for MoveFlag {
    fn from((piece_type, captures): (PromoPieceType, bool)) -> Self {
        let mut v = piece_type.v().v();
        if captures { v += 4; }
        Self { v }
    }
}

impl TryFrom<TMoveFlag> for MoveFlag {
    type Error = ParseError;
    
    #[inline]
    fn try_from(value: TMoveFlag) -> Result<Self, Self::Error> {
        match value {
            0..=13 => Ok(MoveFlag { v: value }),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

impl const ConstFrom<CastlingSide> for MoveFlag {
    fn from_c(value: CastlingSide) -> Self {
        match value {
            CastlingSide::KING_SIDE => MoveFlag::KING_CASTLE,
            CastlingSide::QUEEN_SIDE => MoveFlag::QUEEN_CASTLE,
            _ => unreachable!()
        }
    }
}

#[derive(Default, Copy, Clone)]
pub struct Move { v: u16 }

impl Move {
    const SHIFT_FROM: u16 = 0;
    const SHIFT_TO: u16 = 6;
    const SHIFT_FLAG: u16 = 12;

    const MASK_FROM: u16 = 0b111111 << Move::SHIFT_FROM;
    const MASK_TO: u16 = 0b111111 << Move::SHIFT_TO;
    const MASK_FLAG: u16 = 0b1111 << Move::SHIFT_FLAG;
    
    #[inline]
    pub const fn new(from: Square, to: Square, flag: MoveFlag) -> Self {
        Move {
            v: (from.v() as u16) << Move::SHIFT_FROM
               | (to.v() as u16) << Move::SHIFT_TO
               | (flag.v as u16) << Move::SHIFT_FLAG
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
        if let Ok(promo) = PromoPieceType::try_from(self.get_flag()) {
            write!(f, "{}{}{}", self.get_from(), self.get_to(), promo)
        }
        else {
            write!(f, "{}{}", self.get_from(), self.get_to())
        }
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Move").field("v", &self.v).finish()
    }
}

impl TryFrom<LongAlgebraicUciNotation<'_, '_, '_>> for Move {
    type Error = ParseError;

    fn try_from(move_notation: LongAlgebraicUciNotation<'_, '_, '_>) -> Result<Self, Self::Error> {
        let from = Square::try_from(&mut *move_notation.tokens)?;
        let to = Square::try_from(&mut *move_notation.tokens)?;
        let moving_p = move_notation.context.get_piece(from);
        let captured_p = move_notation.context.get_piece(to);
        let abs_dist = from.v().abs_diff(to.v());
        let captures = captured_p.piece_type() != PieceType::NONE;
        let mut flag = if captures { MoveFlag::CAPTURE } else { MoveFlag::QUIET };

        match moving_p.piece_type() {
            PieceType::PAWN => {
                flag = match abs_dist {
                    16 => MoveFlag::DOUBLE_PAWN_PUSH,
                    7 | 9 if !captures => MoveFlag::EN_PASSANT,
                    _ => move_notation.tokens.next().map_or(Ok(flag),
                        |c| Ok(MoveFlag::from((PromoPieceType::try_from(c)?, captures)))
                    )?
                }
            }
            PieceType::KING if abs_dist == 2 => {
                let file = File::from_c(to);
                let side = CastlingSide::try_from(file)?;
                flag = MoveFlag::from_c(side);
            }
            _ => { }
        };

        Ok(Move::new(from, to, flag))
    }
}


