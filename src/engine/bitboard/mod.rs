use crate::{
    engine::{
        coordinates::{CompassRose, File, Rank, Square},
        move_iter::{bishop, rook},
    },
    misc::ConstFrom,
};
use std::{fmt::Debug, ops};

use const_for::const_for;

use super::coordinates::{DiagA1H8, DiagA8H1, TCompassRose};

#[cfg(test)]
pub mod tests;

#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct Bitboard {
    pub v: u64,
}

impl Debug for Bitboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = f.debug_struct("Bitboard");
        for rank in (0..8).rev() {
            result.field_with(
                ['\n', '\t', char::from_digit(rank + 1, 10).unwrap()]
                    .into_iter()
                    .collect::<String>()
                    .as_str(),
                |f| {
                    for file in 0..8 {
                        let file = File::try_from(file as u8).unwrap();
                        let rank = Rank::try_from(rank as u8).unwrap();
                        let sq = Square::from_c((file, rank));
                        let bit = if self.get_bit(sq) { "x " } else { ". " };
                        f.write_str(bit)?;
                    }
                    Ok(())
                },
            );
        }
        result.field_with("\n\t ", |f| f.write_str("a b c d e f g h \n"));
        result.finish()
    }
}

impl_op!(+ |l: Bitboard, r: Bitboard| -> Bitboard { Bitboard { v: l.v + r.v } });
impl_op!(-|l: Bitboard, r: Bitboard| -> Bitboard { Bitboard { v: l.v - r.v } });
impl_op!(^ |l: Bitboard, r: Bitboard| -> Bitboard { Bitboard { v: l.v ^ r.v } } );
impl_op!(| |l: Bitboard, r: Bitboard| -> Bitboard { Bitboard { v: l.v | r.v } } );
impl_op!(| |l: Bitboard, r: usize| -> Bitboard { Bitboard { v: l.v | r as u64 } } );
impl_op!(&|l: Bitboard, r: Bitboard| -> Bitboard { Bitboard { v: l.v & r.v } });
impl_op!(^= |l: &mut Bitboard, r: Bitboard| { l.v ^= r.v } );
impl_op!(|= |l: &mut Bitboard, r: Bitboard| { l.v |= r.v } );
impl_op!(&= |l: &mut Bitboard, r: Bitboard| { l.v &= r.v } );
impl_op!(!|x: Bitboard| -> Bitboard { Bitboard { v: !x.v } });

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
        Self {
            v: !0u64 << (sq.v() as u32) << 1,
        }
    }

    #[inline]
    pub const fn split_south(sq: Square) -> Self {
        Self {
            v: !0u64 >> (63 - sq.v() as u32) >> 1,
        }
    }

    #[inline]
    pub const fn get_bit(&self, sq: Square) -> bool {
        self.v & Self::from_c(sq).v != 0
    }

    #[inline]
    pub const fn shift(&self, dir: CompassRose) -> Self {
        match dir.v() >= 0 {
            true => Bitboard {
                v: self.v << dir.v(),
            },
            false => Bitboard {
                v: self.v >> -dir.v(),
            },
        }
    }

    #[inline]
    pub const fn shift_c<const DIR: TCompassRose>(&self) -> Self {
        match DIR >= 0 {
            true => Bitboard { v: self.v << DIR },
            false => Bitboard { v: self.v >> -DIR },
        }
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.v == 0
    }

    #[inline]
    pub const fn and_c(&self, other: Bitboard) -> Self {
        Bitboard {
            v: self.v & other.v,
        }
    }
    
    pub const fn and_not_c(&self, other: Bitboard) -> Self {
        Bitboard {
            v: self.v & !other.v,
        }
    }

    #[inline]
    pub const fn ray(sq1: Square, sq2: Square) -> Self {
        static RAYS: [[Bitboard; 64]; 64] = {
            let mut rays = [[Bitboard::empty(); 64]; 64];
            const_for!(sq1 in Square::A1_C..(Square::H8_C+1) => {
                const_for!(sq2 in Square::A1_C..(Square::H8_C+1) => {
                    // Safety: we are only iterating over valid squares.
                    let sq1 = unsafe { Square::from_v(sq1) };
                    let sq2 = unsafe { Square::from_v(sq2) };

                    if let Some((sq1_bb, sq2_bb)) = {
                        if  Rank::from_c(sq1).v() == Rank::from_c(sq2).v() ||
                            File::from_c(sq1).v() == File::from_c(sq2).v() {
                            Some((
                                rook::compute_attacks_0_occ(sq1),
                                rook::compute_attacks_0_occ(sq2)
                            ))
                        }
                        else if
                            DiagA1H8::from_c(sq1).v() == DiagA1H8::from_c(sq2).v() ||
                            DiagA8H1::from_c(sq1).v() == DiagA8H1::from_c(sq2).v() {
                            Some((
                                bishop::compute_attacks_0_occ(sq1),
                                bishop::compute_attacks_0_occ(sq2)
                            ))
                        }
                        else {
                            None
                        }
                    } {
                        rays[sq1.v() as usize][sq2.v() as usize] = Bitboard {
                            v: (
                                Bitboard::from_c(sq1).v | 
                                Bitboard::from_c(sq2).v | 
                                (sq1_bb.v & sq2_bb.v)
                            )
                        };
                    }
                });
            });
            rays
        };

        RAYS[sq1.v() as usize][sq2.v() as usize]
    }

    #[inline]
    pub const fn between(sq1: Square, sq2: Square) -> Self {
        let ray = Self::ray(sq1, sq2);
        let (hi, lo) = if sq1.v() > sq2.v() {
            (sq1, sq2)
        } else {
            (sq2, sq1)
        };
        Bitboard {
            v: Bitboard::split_north(lo).v & Bitboard::split_south(hi).v & ray.v,
        }
    }

    pub fn pop_cnt(&self) -> u32 {
        self.v.count_ones()
    }
    
    pub const fn edges() -> Bitboard {
        Bitboard { v: !0x007E7E7E7E7E7E00_u64 }
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
        Bitboard {
            v: 0x0101010101010101u64 << file.v(),
        }
    }
}

impl const ConstFrom<Rank> for Bitboard {
    #[inline]
    fn from_c(rank: Rank) -> Self {
        Bitboard {
            v: 0xFFu64 << (rank.v() * 8),
        }
    }
}

impl const ConstFrom<CompassRose> for Bitboard {
    #[inline]
    fn from_c(dir: CompassRose) -> Self {
        match dir {
            CompassRose::NORT => Self::full().and_not_c(Self::from_c(Rank::_1)),
            CompassRose::EAST => Self::full().and_not_c(Self::from_c(File::A)),
            CompassRose::SOUT => Self::full().and_not_c(Self::from_c(Rank::_8)),
            CompassRose::WEST => Self::full().and_not_c(Self::from_c(File::H)),

            CompassRose::NONO => Self::from_c(CompassRose::NORT).and_not_c(Self::from_c(Rank::_2)),
            CompassRose::EAEA => Self::from_c(CompassRose::EAST).and_not_c(Self::from_c(File::B)),
            CompassRose::SOSO => Self::from_c(CompassRose::SOUT).and_not_c(Self::from_c(Rank::_7)),
            CompassRose::WEWE => Self::from_c(CompassRose::WEST).and_not_c(Self::from_c(File::G)),

            CompassRose::SOWE => Self::from_c(CompassRose::SOUT).and_c(Self::from_c(CompassRose::WEST)),
            CompassRose::NOWE => Self::from_c(CompassRose::NORT).and_c(Self::from_c(CompassRose::WEST)),
            CompassRose::SOEA => Self::from_c(CompassRose::SOUT).and_c(Self::from_c(CompassRose::EAST)),
            CompassRose::NOEA => Self::from_c(CompassRose::NORT).and_c(Self::from_c(CompassRose::EAST)),

            CompassRose::NONOWE => Self::from_c(CompassRose::NONO).and_c(Self::from_c(CompassRose::WEST)),
            CompassRose::NONOEA => Self::from_c(CompassRose::NONO).and_c(Self::from_c(CompassRose::EAST)),
            CompassRose::NOWEWE => Self::from_c(CompassRose::NORT).and_c(Self::from_c(CompassRose::WEWE)),
            CompassRose::NOEAEA => Self::from_c(CompassRose::NORT).and_c(Self::from_c(CompassRose::EAEA)),
            CompassRose::SOSOWE => Self::from_c(CompassRose::SOSO).and_c(Self::from_c(CompassRose::WEST)),
            CompassRose::SOSOEA => Self::from_c(CompassRose::SOSO).and_c(Self::from_c(CompassRose::EAST)),
            CompassRose::SOWEWE => Self::from_c(CompassRose::SOUT).and_c(Self::from_c(CompassRose::WEWE)),
            CompassRose::SOEAEA => Self::from_c(CompassRose::SOUT).and_c(Self::from_c(CompassRose::EAEA)),

            _ => panic!("Invalid compass rose"),
        }
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
        Bitboard {
            v: A1H8[diag.v() as usize],
        }
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
        Bitboard {
            v: A8H1[diag.v() as usize],
        }
    }
}
