use core::fmt;
use std::{marker::PhantomData, ops};

use crate::core::color::Perspective;

/// A penalty for `P`
pub struct Penalty<P: Perspective>(pub i32, PhantomData<P>);

impl<P: Perspective> fmt::Display for Penalty<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Penalty<{}>({})", P::COLOR, self.0) }
}

impl<P: Perspective> From<Penalty<P>> for Score<P> {
    #[inline(always)]
    fn from(val: Penalty<P>) -> Self { Score::new(-val.0) }
}

/// A bonus for `P`
#[derive(Debug, Copy, Clone)]
pub struct Score<P: Perspective>(pub i32, PhantomData<P>);

impl<P: Perspective> fmt::Display for Score<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Score<{}>({})", P::COLOR, self.0) }
}

impl<P: Perspective, Rhs: Into<Score<P>>> ops::Add<Rhs> for Score<P> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output { Self(self.0 + rhs.into().0, PhantomData) }
}

impl<P: Perspective, Rhs: Into<Score<P>>> ops::Sub<Rhs> for Score<P> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Rhs) -> Self::Output { Self(self.0 - rhs.into().0, PhantomData) }
}

impl<P: Perspective> ops::Div<i32> for Score<P> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: i32) -> Self::Output { Self(self.0 / rhs, PhantomData) }
}

impl<P: Perspective> Eq for Score<P> {}

impl<P: Perspective> Ord for Score<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.cmp(&other.0).then_with(|| self.1.cmp(&other.1)) }
}

impl<P: Perspective> PartialOrd for Score<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl<P: Perspective> PartialEq for Score<P> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<P: Perspective> Score<P> {
    pub const POS_INF: Self = Self::new(30_000);
    pub const NEG_INF: Self = Self::new(-30_000);

    #[inline(always)]
    pub const fn new(val: i32) -> Self { Self(val, PhantomData) }
}

impl<P: Perspective> Penalty<P> {
    #[inline(always)]
    pub const fn new(val: i32) -> Self { Self(val, PhantomData) }
}

// not using the `-` operator because this is not really just arithmetic
// negation, but also a perspective flip.
impl<P: Perspective> ops::Not for Score<P> {
    type Output = Score<P::Opponent>;

    /// Negate the score and flip the perspective to the opponent.
    #[inline(always)]
    fn not(self) -> Self::Output { Score::new(-self.0) }
}

impl<P: Perspective> From<Score<P>> for Cp {
    #[inline(always)]
    fn from(value: Score<P>) -> Self {
        if P::IS_WHITE {
            Cp { v: value.0 as i16 }
        }
        else {
            Cp { v: (-value.0) as i16 }
        }
    }
}

/// Centi pawns
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Cp {
    pub v: TCp,
}

pub type TCp = i16;

impl Cp {
    pub const SCALE: f32 = 350.;

    pub fn v(&self) -> TCp { self.v }

    pub fn new(v: TCp) -> Self { Self { v } }
}
