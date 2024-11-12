use std::{fmt::Debug, ops};
use crate::{engine::coordinates::{
    CompassRose, File, Rank, Square
}, misc::ConstFrom};

use super::coordinates::{Squares, TCompassRose};

#[cfg(test)]
pub mod tests;

#[derive(Copy, Clone, Default, PartialEq)]
pub struct Bitboard { pub v: u64 }

impl Debug for Bitboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = f.debug_struct("Bitboard");
        for rank in (0..8).rev() {
            let rank_name = char::from_digit(rank + 1, 10).unwrap();
            result.field_with(
                rank_name.to_string().as_str(), 
                |f| {
                    for file in 0..8 {
                        let file = File::try_from(file as u8).unwrap();
                        let rank = Rank::try_from(rank as u8).unwrap();
                        let sq = Square::from_c((file, rank));
                        let bit = if self.get_bit(sq) { "x " } else { ". " };
                        f.write_str(bit);
                    }
                    Ok(())
                });
        }
        result.field_with(" ", |f| { f.write_str("a b c d e f g h ") });
        result.finish()
    }
}

impl_op!(+ |l: Bitboard, r: Bitboard| -> Bitboard { l + r });
impl_op!(- |l: Bitboard, r: Bitboard| -> Bitboard { l - r });
// impl_op!(<< |bb: Bitboard, v: File| -> Bitboard { bb << v } );
// impl_op!(<< |bb: Bitboard, v: Rank| -> Bitboard { bb << v } );
// impl_op!(<< |bb: Bitboard, v: isize| -> Bitboard { bb << v } );
// impl_op!(>> |bb: Bitboard, v: isize| -> Bitboard { bb >> v } );
// impl_op!(<< |bb: Bitboard, v: CompassRose| -> Bitboard { bb << v } );
// impl_op!(>> |bb: Bitboard, v: CompassRose| -> Bitboard { bb >> v } );
impl_op!(^ |l: Bitboard, r: Bitboard| -> Bitboard { l ^ r } );
impl_op!(| |l: Bitboard, r: Bitboard| -> Bitboard { l | r } );
impl_op!(| |l: Bitboard, r: usize| -> Bitboard { l | r } );
impl_op!(& |l: Bitboard, r: Bitboard| -> Bitboard { l & r } );
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
            _ => unsafe { Some(Square::from_v(result)) },
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
            _ => unsafe { Some(Square::from_v(result)) },
        }
    }
    
    /// Pop least significant bit and return it.
    #[inline]
    pub const fn pop_lsb(&mut self) -> Option<Square> {
        let lsb = self.lsb();
        self.v &= self.v.wrapping_sub(1);
        lsb
    }
    
    // todo: test
    #[inline]
    pub const fn split_north(sq: Square) -> Self {
        Self { v: !0u64 << (sq.v() as u32) << 1 }
    }
    
    // todo: test
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
    pub const fn shift_c<const Dir: TCompassRose>(&self) -> Self {
        match Dir >= 0 {
            true  => Bitboard { v: self.v << Dir },
            false => Bitboard { v: self.v >> -Dir },
        }
    }
    
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.v == 0
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

impl const ConstFrom<Squares> for Bitboard {
    #[inline]
    fn from_c(sq: Squares) -> Self {
        Bitboard::from_c(Square::from_c(sq))
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
        Bitboard { v: 0xFFu64 << rank.v() }
    }
}
