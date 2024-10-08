use crate::{engine::{

    coordinates::{ File, Square}, piece::PieceType, position::Position }, 
    uci::tokens::Tokenizer
};
use std::marker::PhantomData;
use super::piece::PromotionPieceType;


pub struct LongAlgebraicNotation;
pub struct LongAlgebraicNotationUci;
pub struct StandardAlgebraicNotation;

pub struct MoveNotation<'a, Type> {
    pub tokens: &'a mut Tokenizer<'a>,
    pub context: &'a Position,
    r#type: PhantomData<Type>
}

impl<'a, Type> MoveNotation<'a, Type> {
    pub fn new(tokens: &'a mut Tokenizer<'a>, context: &'a Position) -> Self {
        MoveNotation {
            tokens,
            context,
            r#type: PhantomData
        }
    }
}

pub enum MoveFlag {
    Quiet,
    DoublePawnPush,
    PromotionKnight,
    PromotionBishop,
    PromotionRook,
    PromotionQueen,
    CapturePromotionKnight,
    CapturePromotionBishop,
    CapturePromotionRook,
    CapturePromotionQueen,
    KingCastle,
    QueenCastle,
    Capture,
    EnPassant
}


pub struct Move { v: u16 }

impl Move {
    const SHIFT_FROM: u16 = 0;
    const SHIFT_TO: u16 = 6;
    const SHIFT_FLAG: u16 = 12;

    const MASK_FROM: u16 = 0b111111 << Move::SHIFT_FROM;
    const MASK_TO: u16 = 0b111111 << Move::SHIFT_TO;
    const MASK_FLAG: u16 = 0b1111 << Move::SHIFT_FLAG;
}

impl From<(Square, Square, MoveFlag)> for Move {
    fn from((from, to, flag): (Square, Square, MoveFlag)) -> Self {
        Move {
            v: (from.v as u16) << Move::SHIFT_FROM
               | (to.v as u16) << Move::SHIFT_TO
               | (flag as u16) << Move::SHIFT_FLAG
        }
    }
}

impl TryFrom<MoveNotation<'_, LongAlgebraicNotationUci>> for Move {
    type Error = anyhow::Error;

    fn try_from(move_notation: MoveNotation<'_, LongAlgebraicNotationUci>) -> Result<Self, Self::Error> {
        let from = Square::try_from(&mut *move_notation.tokens)?;
        let to = Square::try_from(&mut *move_notation.tokens)?;
        let moving_p = move_notation.context.get_piece(from);
        let captured_p = move_notation.context.get_piece(to);
        let abs_dist = from.v.abs_diff(to.v);
        let captures = captured_p.piece_type == PieceType::None;
        let mut flag = if captures { MoveFlag::Capture } else { MoveFlag::Quiet } as u16;

        match moving_p.piece_type {
            PieceType::Pawn => {
                match abs_dist {
                    16 => flag = MoveFlag::DoublePawnPush as u16,
                    7 | 9 if !captures => flag = MoveFlag::EnPassant as u16,
                    _ => if let Some(c) = move_notation.tokens.next_char_not_ws() {
                        flag = PromotionPieceType::try_from(c)? as u16;
                        if captures { flag += 4 }
                    }
                }
            }
            PieceType::King if abs_dist == 2 => {
                match File::from(to) {
                    File::G => flag = MoveFlag::KingCastle as u16,
                    File::C => flag = MoveFlag::QueenCastle as u16,
                    _ => { }
                }
            }
            _ => { }
        };

        Ok(Move {
            v: (from.v as u16) << Move::SHIFT_FROM
               | (to.v as u16) << Move::SHIFT_TO
               | (flag as u16) << Move::SHIFT_FLAG
        })
    }
}
