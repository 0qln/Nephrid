

pub struct Move(u16);

impl Move {
    const SHIFT_FROM: i32 = 0;
    const SHIFT_TO: i32 = 6;
    const SHIFT_FLAG: i32 = 12;

    const MASK_FROM: i32 = 0b111111 << Move::SHIFT_FROM;
    const MASK_TO: i32 = 0b111111 << Move::SHIFT_TO;
    const MASK_FLAG: i32 = 0b1111 << Move::SHIFT_FLAG;

    const FLAG_QUIET_MOVE: i32 = 0;
    const FLAG_DOUBLE_PAWN_PUSH: i32 = 1;
    const FLAG_PROMOTION_KNIGHT: i32 = 2;
    const FLAG_PROMOTION_BISHOP: i32 = 3;
    const FLAG_PROMOTION_ROOK: i32 = 4;
    const FLAG_PROMOTION_QUEEN: i32 = 5;
    const FLAG_CAPTURE_PROMOTION_KNIGHT: i32 = 6;
    const FLAG_CAPTURE_PROMOTION_BISHOP: i32 = 7;
    const FLAG_CAPTURE_PROMOTION_ROOK: i32 = 8;
    const FLAG_CAPTURE_PROMOTION_QUEEN: i32 = 9;
    const FLAG_KING_CASTLE: i32 = 10;
    const FLAG_QUEEN_CASTLE: i32 = 11;
    const FLAG_CAPTURE: i32 = 12;
    const FLAG_EN_PASSANT: i32 = 13;
}

impl TryFrom<String> for Move {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        todo!()        
    }
}