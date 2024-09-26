use super::bitboard::Bitboard;

macro_rules! bb {
    ($val:expr) => {
        Bitboard { v: $val }        
    };

}

pub const FILES: [Bitboard; 8] = [
    bb!(0x0101010101010101),
bb!(    0x0202020202020202),
bb!(    0x0404040404040404),
bb!(    0x0808080808080808),
bb!(    0x1010101010101010),
bb!(    0x2020202020202020),
bb!(    0x4040404040404040),
bb!(    0x8080808080808080),
];

pub const PASSANTS: [Bitboard; 8] = [
bb!(            0x0202020202020202),
bb!(            0x0505050505050505),
bb!(            0x0a0a0a0a0a0a0a0a),
bb!(            0x1414141414141414),
bb!(            0x2828282828282828),
bb!(            0x5050505050505050),
bb!(            0xa0a0a0a0a0a0a0a0),
bb!(            0x4040404040404040),
];

pub const RANKS: [Bitboard; 8] = [
bb!(            0x00000000000000FF),
bb!(            0x000000000000FF00),
bb!(            0x0000000000FF0000),
bb!(            0x00000000FF000000),
bb!(            0x000000FF00000000),
bb!(            0x0000FF0000000000),
bb!(            0x00FF000000000000),
bb!(            0xFF00000000000000),
];
