use std::ops;
use crate::{engine::coordinates::{
    CompassRose, File, Rank, Square
}, misc::ConstFrom};

#[derive(Copy, Clone, Default)]
pub struct Bitboard { pub v: u64 }

impl_op!(+ |l: Bitboard, r: Bitboard| -> Bitboard { l + r });
impl_op!(- |l: Bitboard, r: Bitboard| -> Bitboard { l - r });
impl_op!(<< |bb: Bitboard, v: File| -> Bitboard { bb << v } );
impl_op!(<< |bb: Bitboard, v: Rank| -> Bitboard { bb << v } );
impl_op!(<< |bb: Bitboard, v: isize| -> Bitboard { bb << v } );
impl_op!(>> |bb: Bitboard, v: isize| -> Bitboard { bb >> v } );
impl_op!(<< |bb: Bitboard, v: CompassRose| -> Bitboard { bb << v } );
impl_op!(>> |bb: Bitboard, v: CompassRose| -> Bitboard { bb >> v } );
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

    fn next(&mut self) -> Option<Self::Item> {
        match self.v {
            0 => None,
            _ => Some(self.pop_lsb()),
        }
    }
}

impl Bitboard {
    pub const fn empty() -> Self {
        Self { v: 0 }
    }
    
    pub const fn full() -> Self {
        Self { v: !0 }
    }
    
    // todo: remove
    pub const fn next(&mut self) -> Option<Square> {
        match self.v {
            0 => None,
            _ => Some(self.pop_lsb()),
        }
    } 
   
    // todo: test
    /// Most significant bit
    pub const fn msb(&self) -> Square {
        // Safety: trailing_zeros of an u64 returns a valid square (0..=64)
        unsafe {
            Square::from_v(self.v.leading_zeros() as u8)
        }
    }

    // todo: test
    /// Least significant bit
    pub const fn lsb(&self) -> Square {
        // Safety: trailing_zeros of an u64 returns a valid square (0..=64)
        unsafe {
            Square::from_v(self.v.trailing_zeros() as u8)
        }
    }
    
    /// Pop least significant bit and return it.
    pub const fn pop_lsb(&mut self) -> Square {
        let lsb = self.lsb();
        self.v &= self.v - 1u64;
        lsb
    }
    
    // todo: test
    pub const fn split_north(sq: Square) -> Self {
        // Safety: Square is a value 0..=64
        unsafe {
            let sq = sq.v() as u32;
            let (res, overflow) = (!0u64).overflowing_shl(sq + 1);
            let res = if overflow { 0 } else { res };
            Self { v: res }
        }
    }
    
    // todo: test
    pub const fn split_south(sq: Square) -> Self {
        Self { v: !0u64 >> sq.v() >> 1u64 }
    }
}

impl const ConstFrom<Square> for Bitboard {
    fn from_c(sq: Square) -> Self {
        Bitboard { v: 1u64 << sq.v() }
    }
}

impl const ConstFrom<File> for Bitboard {
    fn from_c(file: File) -> Self {
        Bitboard { v: 0x0101010101010101u64 << file.v()}
    }
}

impl const ConstFrom<Rank> for Bitboard {
    fn from_c(rank: Rank) -> Self {
        Bitboard { v: 0xFFu64 << rank.v() }
    }
}
