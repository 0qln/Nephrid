use std::ops;
use crate::engine::coordinates::{
    Square,
    File,
    Rank,
    CompassRose
};

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

    // todo: test
    pub fn msb(&self) -> Square {
        // Safety: trailing_zeros of an u64 returns a valid square (0..=64)
        unsafe {
            Square::try_from(self.v.leading_zeros() as u8).unwrap_unchecked()
        }
    }

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
    
    // todo: test
    pub const fn split_north(sq: Square) -> Self {
        Self { v: !0 << sq.v() << 1 }
    }
    
    // todo: test
    pub const fn split_south(sq: Square) -> Self {
        Self { v: !0 >> sq.v() >> 1 }
    }
    
    const fn from_sq(sq: Square) -> Self {
        Bitboard { v: 1 << sq.v() }
    }

    const fn from_file(file: File) -> Self {
        Bitboard { v: 0x0101010101010101u64 << file.v()}
    }

    const fn from_rank(rank: Rank) -> Self {
        Bitboard { v: 0xFFu64 << rank.v() }
    }
}

impl From<Square> for Bitboard {
    fn from(sq: Square) -> Self {
        Bitboard { v: 1 << sq.v() }
    }
}

impl From<File> for Bitboard {
    fn from(file: File) -> Self {
        Bitboard { v: 0x0101010101010101 } << file
    }
}

impl From<Rank> for Bitboard {
    fn from(rank: Rank) -> Self {
        Bitboard { v: 0xFF } << rank
    }
}
