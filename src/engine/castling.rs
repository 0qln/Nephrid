use crate::engine::{
    color::Color,
    piece::PieceType
};


pub enum CastlingSide {
    KingSide = PieceType::King as isize,
    QueenSide = PieceType::Queen as isize,
}


#[derive(Default, Copy, Clone)]
pub struct CastlingRights { v: u32 }

impl TryFrom<&str> for CastlingRights {
    type Error = anyhow::Error;
    
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut result = CastlingRights::default();
        for c in value.chars() {
            match c {
                'K' => result.set_true(CastlingSide::KingSide, Color::White),
                'Q' => result.set_true(CastlingSide::QueenSide, Color::White),
                'k' => result.set_true(CastlingSide::KingSide, Color::Black),
                'q' => result.set_true(CastlingSide::QueenSide, Color::Black),
                _ => return Err(anyhow::Error::msg("Invalid char")),
            }
        };
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
    
    fn to_index(side: CastlingSide, color: Color) -> u32 {
        (color as u32) << 1 | (side as u32 - 5)
    }
}
