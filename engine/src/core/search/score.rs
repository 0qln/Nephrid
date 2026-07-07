use std::{fmt, marker::PhantomData, ops};

use saturating_cast::SaturatingCast;

use crate::{
    core::color::{Color, Perspective},
    impl_variants,
};

type RawScore = i32;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AnyScore {
    v: RawScore,
}

impl fmt::Display for AnyScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.v.fmt(f) }
}

impl_variants! {
    RawScore as AnyScore in scores {
        DRAW = 0,
        POS_INF = 30_000,
        NEG_INF = -30_000,
        INVALID = 0xdead_beef,
    }
}

impl AnyScore {
    pub const fn new(val: i32) -> Self { Self { v: val } }

    /// Get this score in the context of `relative_to`.
    pub const fn contextualize<P: Perspective>(self, relative_to: Color) -> Score<P> {
        if P::COLOR == relative_to {
            unsafe { self.interpret_as::<P>() }
        }
        else {
            unsafe { (-self).interpret_as::<P>() }
        }
    }

    /// # Safety
    ///
    /// The caller has to make sure that `self` is actually of perspective `P`.
    pub const unsafe fn interpret_as<P: Perspective>(self) -> Score<P> { Score::<P>::new(self) }
}

impl const ops::Neg for AnyScore {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output { Self::new(-self.v) }
}

impl const ops::Add for AnyScore {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output { Self::new(self.v + rhs.v) }
}

impl const ops::Sub for AnyScore {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output { Self::new(self.v - rhs.v) }
}

impl const ops::Div<AnyScore> for AnyScore {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: AnyScore) -> Self::Output { Self::new(self.v / rhs.v) }
}

impl const From<AnyScore> for RawScore {
    #[inline(always)]
    fn from(val: AnyScore) -> Self { val.v }
}

/// A penalty for `P`
pub struct Penalty<P: Perspective>(pub AnyScore, PhantomData<P>);

impl<P: Perspective> fmt::Display for Penalty<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Penalty<{}>({})", P::COLOR, self.0) }
}

impl<P: Perspective> From<Penalty<P>> for Score<P> {
    #[inline(always)]
    fn from(val: Penalty<P>) -> Self { Score::new(-val.0) }
}

/// A bonus for `P`
#[derive(Debug, Copy, Clone)]
pub struct Score<P: Perspective>(pub AnyScore, PhantomData<P>);

impl<P: Perspective> fmt::Display for Score<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Score<{}>({})", P::COLOR, self.0) }
}

impl<P: Perspective, Rhs: const Into<Score<P>>> const ops::Add<Rhs> for Score<P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output { Self::new(self.0 + rhs.into().0) }
}

impl<P: Perspective, Rhs: const Into<Score<P>>> const ops::Sub<Rhs> for Score<P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Rhs) -> Self::Output { Self::new(self.0 - rhs.into().0) }
}

impl<P: Perspective> const ops::Div<AnyScore> for Score<P> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: AnyScore) -> Self::Output { Self::new(self.0 / rhs) }
}

impl<P: Perspective> Eq for Score<P> {}

impl<P: Perspective> Ord for Score<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.cmp(&other.0) }
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
    #[inline(always)]
    pub const fn new(val: AnyScore) -> Self { Self(val, PhantomData) }
}

impl<P: Perspective> Penalty<P> {
    #[inline(always)]
    pub const fn new(val: AnyScore) -> Self { Self(val, PhantomData) }
}

// not using the `-` operator because this is not really just arithmetic
// negation, but also a perspective flip.
impl<P: Perspective> ops::Not for Score<P> {
    type Output = Score<P::Opponent>;

    /// Negate the score and flip the perspective to the opponent.
    #[inline(always)]
    fn not(self) -> Self::Output { Score::new(-self.0) }
}

// todo:
impl<P: Perspective> From<Score<P>> for Cp {
    #[inline(always)]
    fn from(value: Score<P>) -> Self {
        let v: TCp = value.0.v.saturating_cast();
        if P::IS_WHITE { Cp { v } } else { Cp { v: -v } }
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
