use crate::{
    engine::{
        coordinates::{ File, Square}, 
        piece::PieceType, position::Position 
    }, 
    uci::tokens::Tokenizer
};
use std::marker::PhantomData;
use super::piece::PromotionPieceType;


pub struct LongAlgebraicNotation;
pub struct LongAlgebraicNotationUci;
pub struct StandardAlgebraicNotation;

pub struct MoveNotation<'this, 'tok, Type> {
    pub tokens: &'this mut Tokenizer<'tok>,
    pub context: &'this Position,
    r#type: PhantomData<Type>
}

impl<'a, 'b, Type> MoveNotation<'a, 'b, Type> {
    pub fn new(tokens: &'a mut Tokenizer<'b>, context: &'a Position) -> Self {
        MoveNotation {
            tokens,
            context,
            r#type: PhantomData
        }
    }
}

#[derive(Debug, Copy, Clone)]
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


#[derive(Debug, Default)]
pub struct Move { v: u16 }

impl Move {
    const SHIFT_FROM: u16 = 0;
    const SHIFT_TO: u16 = 6;
    const SHIFT_FLAG: u16 = 12;

    const MASK_FROM: u16 = 0b111111 << Move::SHIFT_FROM;
    const MASK_TO: u16 = 0b111111 << Move::SHIFT_TO;
    const MASK_FLAG: u16 = 0b1111 << Move::SHIFT_FLAG;
    
    pub fn new(from: Square, to: Square, flag: MoveFlag) -> Self {
        Move {
            v: (from.v as u16) << Move::SHIFT_FROM
               | (to.v as u16) << Move::SHIFT_TO
               | (flag as u16) << Move::SHIFT_FLAG
        }
    }
}

impl TryFrom<MoveNotation<'_, '_, LongAlgebraicNotationUci>> for Move {
    type Error = anyhow::Error;

    fn try_from(move_notation: MoveNotation<'_, '_, LongAlgebraicNotationUci>) -> Result<Self, Self::Error> {
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
                    _ => if let Some(c) = move_notation.tokens.next() {
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


#[derive(Debug, Default)]
pub struct MoveList<'a> {
    pub v: &'a [Move],
}


