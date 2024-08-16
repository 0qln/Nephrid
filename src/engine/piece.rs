use crate::engine::color::Color;

pub enum PromotionPieceType {
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
}

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum PieceType {
    None = 0,
    Pawn = 1,
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
    King = 6,    
}

// impl PieceType {
//     pub fn is_promotion(&self) -> bool {
//         *self >= PieceType::Knight && *self <= PieceType::Knight
//     }
// }

impl Default for PieceType {
    fn default() -> Self {
        Self::None
    }
}

impl TryFrom<char> for PromotionPieceType {
    type Error = anyhow::Error;
 
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'n' => Ok(PromotionPieceType::Knight),
            'b' => Ok(PromotionPieceType::Bishop),
            'r' => Ok(PromotionPieceType::Rook),
            'q' => Ok(PromotionPieceType::Queen),
            _ => Err(anyhow::Error::msg("Invalid char")),
        }
    }
}

impl TryFrom<char> for PieceType {
    type Error = anyhow::Error;
    
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'p' => Ok(PieceType::Pawn),
            'n' => Ok(PieceType::Knight),
            'b' => Ok(PieceType::Bishop),
            'r' => Ok(PieceType::Rook),
            'q' => Ok(PieceType::Queen),
            'k' => Ok(PieceType::King),
            '.' => Ok(PieceType::None),
            _ => Err(anyhow::Error::msg("Invalid char")),
        }
    }
}

impl Into<char> for PieceType {
    fn into(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
            PieceType::None => '.',
        }
    }
}

      
#[derive(Copy, Clone)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType
}

impl Default for Piece {
    fn default() -> Self {
        Self { 
            color: Default::default(), 
            piece_type: Default::default() 
        }
    }
}

impl TryFrom<char> for Piece {
    type Error = anyhow::Error;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        let piece_type = PieceType::try_from(value.to_ascii_lowercase())?;
        let color = match value.is_uppercase() {
            true => Color::White,
            false => Color::Black,
        };
        Ok(Self { color, piece_type })
    }
}

impl Into<char> for Piece {
    fn into(self) -> char {
        let mut result: char = self.piece_type.into();
        if self.color == Color::White {
            result = result.to_ascii_uppercase();
        }
        result
    }
}
