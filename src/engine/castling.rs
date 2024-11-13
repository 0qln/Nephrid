use crate::{
    engine::{color::Color, fen::Fen, piece::PieceType},
    misc::ParseError,
};

pub enum CastlingSide {
    KingSide = PieceType::King as isize,
    QueenSide = PieceType::Queen as isize,
}

#[derive(Default, Copy, Clone)]
pub struct CastlingRights {
    v: u8,
}

impl TryFrom<&mut Fen<'_>> for CastlingRights {
    type Error = ParseError;

    fn try_from(value: &mut Fen<'_>) -> Result<Self, Self::Error> {
        let mut result = CastlingRights::default();
        for c in value.iter_token() {
            match c {
                'K' => result.set_true(CastlingSide::KingSide, Color::WHITE),
                'Q' => result.set_true(CastlingSide::QueenSide, Color::WHITE),
                'k' => result.set_true(CastlingSide::KingSide, Color::BLACK),
                'q' => result.set_true(CastlingSide::QueenSide, Color::BLACK),
                '-' => return Ok(result),
                x => return Err(ParseError::InputOutOfRange(Box::new(x))),
            }
        }
        Ok(result)
    }
}

impl CastlingRights {
    pub fn set_false(&mut self, side: CastlingSide, color: Color) {
        self.v &= !(1 << CastlingRights::to_index(side, color));
    }

    pub fn set_true(&mut self, side: CastlingSide, color: Color) {
        self.v |= 1 << CastlingRights::to_index(side, color);
    }

    fn to_index(side: CastlingSide, color: Color) -> isize {
        (color.v as isize) << 1 | (side as isize - 5)
    }
}
