use std::{fmt::Debug, ops};
use crate::{engine::coordinates::{
    CompassRose, File, Rank, Square
}, misc::ConstFrom};

use super::coordinates::{DiagA1H8, DiagA8H1, TCompassRose};

#[cfg(test)]
pub mod tests;

#[derive(Copy, Clone, Default, PartialEq)]
pub struct Bitboard { pub v: u64 }

impl Debug for Bitboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = f.debug_struct("Bitboard");
        for rank in (0..8).rev() {
            result.field_with(
                ['\n', '\t', char::from_digit(rank + 1, 10).unwrap()]
                    .into_iter().collect::<String>().as_str(),
                |f| {
                    for file in 0..8 {
                        let file = File::try_from(file as u8).unwrap();
                        let rank = Rank::try_from(rank as u8).unwrap();
                        let sq = Square::from_c((file, rank));
                        let bit = if self.get_bit(sq) { "x " } else { ". " };
                        f.write_str(bit)?;
                    }
                    Ok(())
                });
        }
        result.field_with("\n\t ", |f| { f.write_str("a b c d e f g h \n") });
        result.finish()
    }
}

impl_op!(+ |l: Bitboard, r: Bitboard| -> Bitboard { Bitboard { v: l.v + r.v } });
impl_op!(- |l: Bitboard, r: Bitboard| -> Bitboard { Bitboard { v: l.v - r.v } });
impl_op!(^ |l: Bitboard, r: Bitboard| -> Bitboard { Bitboard { v: l.v ^ r.v } } );
impl_op!(| |l: Bitboard, r: Bitboard| -> Bitboard { Bitboard { v: l.v | r.v } } );
impl_op!(| |l: Bitboard, r: usize| -> Bitboard { Bitboard { v: l.v | r as u64 } } );
impl_op!(& |l: Bitboard, r: Bitboard| -> Bitboard { Bitboard { v: l.v & r.v } } );
impl_op!(^= |l: &mut Bitboard, r: Bitboard| { l.v ^= r.v } );
impl_op!(|= |l: &mut Bitboard, r: Bitboard| { l.v |= r.v } );
impl_op!(&= |l: &mut Bitboard, r: Bitboard| { l.v &= r.v } );
impl_op!(! |x: Bitboard| -> Bitboard { Bitboard { v: !x.v } });

impl Iterator for Bitboard {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.pop_lsb()
    }
}

impl Bitboard {
    #[inline]
    pub const fn empty() -> Self {
        Self { v: 0 }
    }
    
    #[inline]
    pub const fn full() -> Self {
        Self { v: !0 }
    }
    
    // todo: remove when iterator is const
    #[inline]
    pub const fn next(&mut self) -> Option<Square> {
        self.pop_lsb()
    } 
   
    /// Most significant bit or None if the bitboard is empty
    #[inline]
    pub const fn msb(&self) -> Option<Square> {
        let result = self.v.leading_zeros() as u8;
        match result {
            64 => None,
            // Safety: the result is now in the range of a 
            // valid square (0..64)
            x => unsafe { Some(Square::from_v(63 - x)) },
        }
    }

    /// Least significant bit or None if the bitboard is empty
    #[inline]
    pub const fn lsb(&self) -> Option<Square> {
        let result = self.v.trailing_zeros() as u8;
        match result {
            64 => None,
            // Safety: the result is now in the range of a 
            // valid square (0..64)
            x => unsafe { Some(Square::from_v(x)) },
        }
    }
    
    /// Pop least significant bit and return it.
    #[inline]
    pub const fn pop_lsb(&mut self) -> Option<Square> {
        let lsb = self.lsb();
        self.v &= self.v.wrapping_sub(1);
        lsb
    }
    
    #[inline]
    pub const fn split_north(sq: Square) -> Self {
        Self { v: !0u64 << (sq.v() as u32) << 1 }
    }
    
    #[inline]
    pub const fn split_south(sq: Square) -> Self {
        Self { v: !0u64 >> (63 - sq.v() as u32) >> 1 }
    }

    #[inline]
    pub const fn get_bit(&self, sq: Square) -> bool {
        self.v & Self::from_c(sq).v != 0
    }
    
    #[inline]
    pub const fn shift(&self, dir: CompassRose) -> Self {
        match dir.v() >= 0 {
            true  => Bitboard { v: self.v << dir.v() },
            false => Bitboard { v: self.v >> -dir.v() },
        }
    }
    
    #[inline]
    pub const fn shift_c<const DIR: TCompassRose>(&self) -> Self {
        match DIR >= 0 {
            true  => Bitboard { v: self.v << DIR },
            false => Bitboard { v: self.v >> -DIR },
        }
    }
    
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.v == 0
    }
    
    #[inline]
    pub const fn and_c(&self, other: Bitboard) -> Self {
        Bitboard { v: self.v & other.v }
    }
}

impl const ConstFrom<Square> for Bitboard {
    #[inline]
    fn from_c(sq: Square) -> Self {
        Bitboard { v: 1u64 << sq.v() }
    }
}

impl const ConstFrom<Option<Square>> for Bitboard {
    #[inline]
    fn from_c(sq: Option<Square>) -> Self {
        match sq {
            Some(sq) => Bitboard::from_c(sq),
            None => Bitboard::empty(),
        }
    }
}

impl const ConstFrom<File> for Bitboard {
    #[inline]
    fn from_c(file: File) -> Self {
        Bitboard { v: 0x0101010101010101u64 << file.v()}
    }
}

impl const ConstFrom<Rank> for Bitboard {
    #[inline]
    fn from_c(rank: Rank) -> Self {
        Bitboard { v: 0xFFu64 << (rank.v() * 8) }
    }
}

impl const ConstFrom<DiagA1H8> for Bitboard {
    #[inline]
    fn from_c(diag: DiagA1H8) -> Self {
        const A1H8: [u64; 15] = [
            0x0100000000000000u64,
            0x0201000000000000u64,
            0x0402010000000000u64,
            0x0804020100000000u64,
            0x1008040201000000u64,
            0x2010080402010000u64,
            0x4020100804020100u64,
            0x8040201008040201u64,
            0x0080402010080402u64,
            0x0000804020100804u64,
            0x0000008040201008u64,
            0x0000000080402010u64,
            0x0000000000804020u64,
            0x0000000000008040u64,
            0x0000000000000080u64,
        ];
        Bitboard { v: A1H8[diag.v() as usize] }
    }
}

impl const ConstFrom<DiagA8H1> for Bitboard {
    #[inline]
    fn from_c(diag: DiagA8H1) -> Self {
        const A8H1: [u64; 15] = [
            0x0000000000000001u64,
            0x0000000000000102u64,
            0x0000000000010204u64,
            0x0000000001020408u64,
            0x0000000102040810u64,
            0x0000010204081020u64,
            0x0001020408102040u64,
            0x0102040810204080u64,
            0x0204081020408000u64,
            0x0408102040800000u64,
            0x0810204080000000u64,
            0x1020408000000000u64,
            0x2040800000000000u64,
            0x4080000000000000u64,
            0x8000000000000000u64,
        ];
        Bitboard { v: A8H1[diag.v() as usize] }
    }
}
        