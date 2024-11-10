use super::*;

#[test]
fn msb() {
    assert_eq!(Bitboard::empty().msb(), None);
    for i in 0..64 {
        for j in i+1..64 {
            let lo = Square::try_from(i as u8).unwrap();
            let hi = Square::try_from(j as u8).unwrap();
            let bb = Bitboard::from_c(lo) | Bitboard::from_c(hi);
            assert_eq!(bb.msb(), Some(hi));
        }
    }
}

#[test]
fn lsb() {
    assert_eq!(Bitboard::empty().lsb(), None);
    for i in 0..64 {
        for j in i+1..64 {
            let lo = Square::try_from(i as u8).unwrap();
            let hi = Square::try_from(j as u8).unwrap();
            let bb = Bitboard::from_c(lo) | Bitboard::from_c(hi);
            assert_eq!(bb.lsb(), Some(lo));
        }
    }
}

#[test]
fn split_north() {
    assert_eq!(Bitboard::split_north(Square::from_c(Squares::A1)), Bitboard { v: 0xfffffffffffffffeu64 });   
    assert_eq!(Bitboard::split_north(Square::from_c(Squares::B1)), Bitboard { v: 0xfffffffffffffffcu64 });   
    assert_eq!(Bitboard::split_north(Square::from_c(Squares::E1)), Bitboard { v: 0xfffffffe00000000u64 });   
    assert_eq!(Bitboard::split_north(Square::from_c(Squares::H7)), Bitboard { v: 0x8000000000000000u64 });   
    assert_eq!(Bitboard::split_north(Square::from_c(Squares::H8)), Bitboard { v: 0 });   
}

#[test]
fn split_south() {
    assert_eq!(Bitboard::split_south(Square::from_c(Squares::A1)), Bitboard { v: 0 });   
    assert_eq!(Bitboard::split_south(Square::from_c(Squares::B1)), Bitboard { v: 1 });   
    assert_eq!(Bitboard::split_south(Square::from_c(Squares::E1)), Bitboard { v: 0xffffffffu64 });   
    assert_eq!(Bitboard::split_south(Square::from_c(Squares::H7)), Bitboard { v: 0x3fffffffffffffffu64 });   
    assert_eq!(Bitboard::split_south(Square::from_c(Squares::H8)), Bitboard { v: 0x7fffffffffffffffu64 });   
}