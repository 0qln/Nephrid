use crate::{
    engine::{
        coordinates::{ File, Square}, 
        piece::PieceType, position::Position 
    }, misc::{ConstFrom, ParseError}, uci::tokens::Tokenizer
};
use std::marker::PhantomData;

// todo: refactoring, clean up

pub struct LongAlgebraicNotation;
pub struct LongAlgebraicNotationUci;
pub struct StandardAlgebraicNotation;

pub struct MoveNotation<'this, 'tok, Type> {
    pub tokens: &'this mut Tokenizer<'tok>,
    pub context: &'this Position,
    r#type: PhantomData<Type>
}

impl<'a, 'b, Type> MoveNotation<'a, 'b, Type> {
    #[inline]
    pub fn new(tokens: &'a mut Tokenizer<'b>, context: &'a Position) -> Self {
        MoveNotation {
            tokens,
            context,
            r#type: PhantomData
        }
    }
}

pub type TMoveFlag = u8;

#[derive(Debug, Copy, Clone)]
pub struct MoveFlag { v: u8 }

impl MoveFlag {
    pub const QUIET: MoveFlag = MoveFlag { v: 0 };
    pub const DOUBLE_PAWN_PUSH: MoveFlag = MoveFlag { v: 1 };
    pub const PROMOTION_KNIGHT: MoveFlag = MoveFlag { v: 2 };
    pub const PROMOTION_BISHOP: MoveFlag = MoveFlag { v: 3 };
    pub const PROMOTION_ROOK: MoveFlag = MoveFlag { v: 4 };
    pub const PROMOTION_QUEEN: MoveFlag = MoveFlag { v: 5 };
    pub const CAPTURE_PROMOTION_KNIGHT: MoveFlag = MoveFlag { v: 6 };
    pub const CAPTURE_PROMOTION_BISHOP: MoveFlag = MoveFlag { v: 7 };
    pub const CAPTURE_PROMOTION_ROOK: MoveFlag = MoveFlag { v: 8 };
    pub const CAPTURE_PROMOTION_QUEEN: MoveFlag = MoveFlag { v: 9 };
    pub const KING_CASTLE: MoveFlag = MoveFlag { v: 10 };
    pub const QUEEN_CASTLE: MoveFlag = MoveFlag { v: 11 };
    pub const CAPTURE: MoveFlag = MoveFlag { v: 12 };
    pub const EN_PASSANT: MoveFlag = MoveFlag { v: 13 };
}

impl TryFrom<u8> for MoveFlag {
    type Error = ParseError;
    
    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0..=13 => Ok(MoveFlag { v: value }),
            x => Err(ParseError::InputOutOfRange(Box::new(x))),
        }
    }
}

impl From<PromotionPieceType> for MoveFlag {
    #[inline]
    fn from(value: PromotionPieceType) -> Self {
        return Self { v: value as u8 };
    }
}

impl TryFrom<u16> for MoveFlag {
    type Error = anyhow::Error;
    
    #[inline]
    fn try_from(value: u16) -> anyhow::Result<Self> {
        match value {
            0..=13 => Ok(MoveFlag { v: value as u8 }),
            _ => Err(anyhow::Error::msg("MoveFlag value out of range")),
        }
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Move { v: u16 }

impl Move {
    const SHIFT_FROM: u16 = 0;
    const SHIFT_TO: u16 = 6;
    const SHIFT_FLAG: u16 = 12;

    const MASK_FROM: u16 = 0b111111 << Move::SHIFT_FROM;
    const MASK_TO: u16 = 0b111111 << Move::SHIFT_TO;
    const MASK_FLAG: u16 = 0b1111 << Move::SHIFT_FLAG;
    
    #[inline]
    pub fn new(from: Square, to: Square, flag: MoveFlag) -> Self {
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
    pub fn get_to(&self) -> Square {
        // Safety: 6 bits can only ever contain a value in range [0, 63]
        unsafe {
            Square::try_from((self.v & Move::MASK_TO) >> Move::SHIFT_TO).unwrap_unchecked()
        }
    }
    
    #[inline]
    pub fn get_flag(&self) -> MoveFlag {
        // Safety: The inner move flag bits are only ever set from a MoveFlag struct.
        unsafe {
            MoveFlag::try_from((self.v & Move::MASK_FLAG) >> Move::SHIFT_FLAG).unwrap_unchecked()
        }
    }
}

impl TryFrom<MoveNotation<'_, '_, LongAlgebraicNotationUci>> for Move {
    type Error = ParseError;

    fn try_from(move_notation: MoveNotation<'_, '_, LongAlgebraicNotationUci>) -> Result<Self, Self::Error> {
        let from = Option::<Square>::try_from(&mut *move_notation.tokens)?.ok_or(ParseError::MissingInput)?;
        let to = Option::<Square>::try_from(&mut *move_notation.tokens)?.ok_or(ParseError::MissingInput)?;
        let moving_p = move_notation.context.get_piece(from);
        let captured_p = move_notation.context.get_piece(to);
        let abs_dist = from.v().abs_diff(to.v());
        let captures = captured_p.piece_type == PieceType::NONE;
        let mut flag = if captures { MoveFlag::CAPTURE } else { MoveFlag::QUIET };

        match moving_p.piece_type {
            PieceType::PAWN => {
                match abs_dist {
                    16 => flag = MoveFlag::DOUBLE_PAWN_PUSH,
                    7 | 9 if !captures => flag = MoveFlag::EN_PASSANT,
                    _ => if let Some(c) = move_notation.tokens.next() {
                        flag = MoveFlag::from(PromotionPieceType::try_from(c)?);
                        if captures { 
                            // Safety: 
                            //  The flag is currently set to a promotion piece [2; 5].
                            //  Adding 4 will result in a valid flag value.
                            unsafe {
                                flag = MoveFlag::try_from(flag.v + 4).unwrap_unchecked(); 
                            }
                        }
                    }
                }
            }
            PieceType::KING if abs_dist == 2 => {
                match File::from_c(to) {
                    File::G => flag = MoveFlag::KING_CASTLE,
                    File::C => flag = MoveFlag::QUEEN_CASTLE,
                    _ => { }
                }
            }
            _ => { }
        };

        Ok(Move::new(from, to, flag))
    }
}


