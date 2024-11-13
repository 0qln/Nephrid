use super::bitboard::Bitboard;

macro_rules! bb {
    ($val:expr) => {
        Bitboard { v: $val }        
    };
}

pub const PASSANTS: [Bitboard; 8] = [
    bb!(0x0202020202020202),
    bb!(0x0505050505050505),
    bb!(0x0a0a0a0a0a0a0a0a),
    bb!(0x1414141414141414),
    bb!(0x2828282828282828),
    bb!(0x5050505050505050),
    bb!(0xa0a0a0a0a0a0a0a0),
    bb!(0x4040404040404040),
];