use super::*;

#[test]
fn compute_attacks_corners_no_occupancy() {
    {
        let attacks = compute_attacks(Square::A1, Bitboard::empty());
        let expected = Bitboard { v: 0x1010101010101fe_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let attacks = compute_attacks(Square::A8, Bitboard::empty());
        let expected = Bitboard { v: 0xfe01010101010101_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let attacks = compute_attacks(Square::H1, Bitboard::empty());
        let expected = Bitboard { v: 0x808080808080807f_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let attacks = compute_attacks(Square::H8, Bitboard::empty());
        let expected = Bitboard { v: 0x7f80808080808080_u64 };
        assert_eq!(expected, attacks);
    }
}

#[test]
fn compute_attacks_corners_with_occupancy() {
    {
        let occupancy = Bitboard { v: 0x100000000000002_u64 };
        let attacks = compute_attacks(Square::A1, occupancy);
        let expected = Bitboard { v: 0x101010101010102_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let occupancy = Bitboard { v: 0x8001000000000000_u64 };
        let attacks = compute_attacks(Square::A8, occupancy);
        let expected = Bitboard { v: 0xfe01000000000000_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let occupancy = Bitboard { v: 0x8001_u64 };
        let attacks = compute_attacks(Square::H1, occupancy);
        let expected = Bitboard { v: 0x807f_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let occupancy = Bitboard { v: 0x4000000000000080_u64 };
        let attacks = compute_attacks(Square::H8, occupancy);
        let expected = Bitboard { v: 0x4080808080808080_u64 };
        assert_eq!(expected, attacks);
    }
}

#[test]
fn compute_attacks_center_with_occupancy() {
    let occupancy = Bitboard { v: 0x20080000_u64 };
    let attacks = compute_attacks(Square::D4, occupancy);
    let expected = Bitboard { v: 0x808080837080000_u64 };
    assert_eq!(expected, attacks);
}

#[test]
fn compute_attacks_center_no_occupancy() {
    let attacks = compute_attacks(Square::D4, Bitboard::empty());
    let expected = Bitboard { v: 0x8080808f7080808_u64 };
    assert_eq!(expected, attacks);
}