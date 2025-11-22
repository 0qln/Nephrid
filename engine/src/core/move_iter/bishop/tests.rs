use super::*;

#[test]
fn compute_attacks_sides_no_occupancy() {
    {
        let occupancy = Bitboard::empty();
        let attacks = compute_attacks(squares::A4, occupancy);
        let expected = Bitboard { v: 0x1008040200020408_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let occupancy = Bitboard::empty();
        let attacks = compute_attacks(squares::D8, occupancy);
        let expected = Bitboard { v: 0x14224180000000_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let occupancy = Bitboard::empty();
        let attacks = compute_attacks(squares::H5, occupancy);
        let expected = Bitboard { v: 0x1020400040201008_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let occupancy = Bitboard::empty();
        let attacks = compute_attacks(squares::D1, occupancy);
        let expected = Bitboard { v: 0x8041221400_u64 };
        assert_eq!(expected, attacks);
    }
}

#[test]
fn compute_attacks_sides_with_occupancy() {
    {
        let occupancy = Bitboard { v: 0x1000000000020000_u64 };
        let attacks = compute_attacks(squares::A4, occupancy);
        let expected = Bitboard { v: 0x1008040200020000_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let occupancy = Bitboard { v: 0x10000100000000_u64 };
        let attacks = compute_attacks(squares::D8, occupancy);
        let expected = Bitboard { v: 0x14020100000000_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let occupancy = Bitboard { v: 0x400000000008_u64 };
        let attacks = compute_attacks(squares::H5, occupancy);
        let expected = Bitboard { v: 0x400040201008_u64 };
        assert_eq!(expected, attacks);
    }
    {
        let occupancy = Bitboard { v: 0x1001000_u64 };
        let attacks = compute_attacks(squares::D1, occupancy);
        let expected = Bitboard { v: 16913408_u64 };
        assert_eq!(expected, attacks);
    }
}

#[test]
fn compute_attacks_center_no_occupancy() {
    let attacks = compute_attacks(squares::E4, Bitboard::empty());
    let expecteed = Bitboard { v: 0x182442800284482_u64 };
    assert_eq!(expecteed, attacks);
}

#[test]
fn compute_attacks_center_with_occupancy() {
    let occupancy = Bitboard { v: 0x80040000080080_u64 };
    let attacks = compute_attacks(squares::E4, occupancy);
    let expected = Bitboard { v: 0x80442800284080_u64 };
    assert_eq!(expected, attacks);
}
