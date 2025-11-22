use crate::core::{
    bitboard::Bitboard,
    color::{colors, Color},
    coordinates::{compass_rose, ranks, CompassRose, Rank},
};

#[inline]
pub(super) const fn promo_rank(c: Color) -> Rank {
    match c {
        colors::WHITE => ranks::_7,
        colors::BLACK => ranks::_2,
        _ => unreachable!(),
    }
}

#[inline]
pub(super) const fn start_rank(c: Color) -> Rank {
    match c {
        colors::WHITE => ranks::_2,
        colors::BLACK => ranks::_7,
        _ => unreachable!(),
    }
}

#[inline]
pub(super) const fn single_step(c: Color) -> CompassRose {
    match c {
        colors::WHITE => compass_rose::NORT,
        colors::BLACK => compass_rose::SOUT,
        _ => unreachable!(),
    }
}

#[inline]
pub(super) const fn double_step(c: Color) -> CompassRose {
    single_step(c).double()
}

#[inline]
pub(super) const fn forward(bb: Bitboard, dir: CompassRose) -> Bitboard {
    bb.shift(dir)
}

#[inline]
pub(super) const fn backward(bb: Bitboard, dir: CompassRose) -> Bitboard {
    bb.shift(dir.neg())
}

#[inline]
pub(super) const fn capture(c: Color, dir: CompassRose) -> CompassRose {
    CompassRose::new(dir.v() + single_step(c).v())
}
