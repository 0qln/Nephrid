use std::ops;
use crate::engine::{
    coordinates::{Square, Squares}
};

#[derive(Copy, Clone, Default)]
pub struct Bitboard { pub v: u64 }


impl ops::BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Bitboard { v: self.v & rhs.v }
    }
}

impl ops::BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Bitboard { v: self.v | rhs.v }
    }
}

impl ops::BitXor for Bitboard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Bitboard { v: self.v ^ rhs.v }
    }
}

impl ops::BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.v &= rhs.v;
    }
}

impl ops::BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.v |= rhs.v;
    }
}

impl ops::BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.v ^= rhs.v;
    }
}

impl ops::Shl for Bitboard {
    type Output = Self;
    
    fn shl(self, rhs: isize) -> Self::Output {
        Bitboard { v: self.v << rhs }
    }
}

impl ops::Sub for Bitboard {
    type Output = Self;
    
    fn sub(self, rhs: Self) -> Self::Output {
        Bitboard { v: self.v - rhs.v }
    }
}

impl Iterator for Bitboard {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        match self.v {
            0 => None,
            _ => Some(self.pop_lsb()),
        }
    }
}

impl Bitboard {
    pub fn lsb(&self) -> Square {
        // Safety: trailing_zeros of an u64 returns a valid square (0..=64)
        unsafe {
            Square::try_from(self.v.trailing_zeros() as u8).unwrap_unchecked()
        }
    }
 
    pub fn pop_lsb(&mut self) -> Square {
        let lsb = self.lsb();
        *self &= *self - Bitboard { v: 1 };
        lsb
    }
}

impl From<Square> for Bitboard {
    fn from(sq: Square) -> Self {
        Bitboard { v: 1 << sq.v }
    }
}
