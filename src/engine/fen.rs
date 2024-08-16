use crate::uci::tokens::Tokenizer;
use std::marker::PhantomData;

pub mod parts {
    pub struct PiecesPlacement;
    pub struct SideToMove;
    pub struct CastlingAvailability; 
    pub struct EnPassantTargetSquare;
    pub struct HalfMoveClock;
    pub struct FullMoveCounter;
}

pub struct Fen<'a, Part> { 
    pub v: &'a mut Tokenizer<'a> ,
    part: PhantomData<Part>
}

impl<'a, Part> Fen<'a, Part> {
    pub fn new(v: &'a mut Tokenizer<'a>) -> Self {
        Fen { v, part: PhantomData }
    }
}
