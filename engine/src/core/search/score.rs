use std::{fmt, iter, marker::PhantomData, ops};

use saturating_cast::{SaturatingCast, SaturatingElement};
use static_assertions::const_assert;

use crate::{
    core::{
        color::{Color, Perspective},
        search::ordering::MoveScore,
    },
    impl_variants,
};

pub type RawScore = i32;

#[derive(Debug, Copy)]
#[derive_const(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AnyScore {
    v: RawScore,
}

impl fmt::Display for AnyScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.v.fmt(f) }
}

impl_variants! {
    RawScore as AnyScore in scores {
        DRAW = 0,
        ZERO = 0,
        POS_INF = 30_000,
        NEG_INF = -30_000,
        NULL = 0x_C0FFEE,
    }
}

const_assert!(scores::NULL > scores::POS_INF || scores::NULL < scores::NEG_INF);

impl AnyScore {
    pub const fn new(val: i32) -> Self { Self { v: val } }

    pub const fn is_valid(&self) -> bool { *self != scores::NULL }
    pub const fn validated(&self) -> Option<AnyScore> { if *self == scores::NULL { None } else { Some(*self) } }
    pub const fn validated_mut(&mut self) -> Option<&mut AnyScore> { if *self == scores::NULL { None } else { Some(self) } }

    /// Get this score in the context of `relative_to`.
    pub const fn contextualize<P: Perspective>(self, relative_to: Color) -> Score<P> {
        if P::COLOR.v() == relative_to.v() {
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

const impl<T: const Into<RawScore>> From<T> for AnyScore {
    #[inline(always)]
    fn from(val: T) -> Self { Self::new(val.into()) }
}

const impl<T: const Into<AnyScore>> ops::AddAssign<T> for AnyScore {
    #[inline(always)]
    fn add_assign(&mut self, rhs: T) {
        debug_assert!(*self != scores::NULL, "Cannot add to a NULL score");
        self.v += rhs.into().v();
    }
}

impl SaturatingCast for AnyScore {}
impl SaturatingElement<MoveScore> for AnyScore {
    fn as_element(self) -> MoveScore {
        debug_assert_ne!(self, scores::NULL, "Cannot convert a NULL score to a MoveScore");
        (self.v().clamp(MoveScore::MIN.into(), MoveScore::MAX.into())) as MoveScore
    }
}

const impl ops::Neg for AnyScore {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output { Self::new(-self.v) }
}

const impl<Rhs: const Into<AnyScore>> ops::Add<Rhs> for AnyScore {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output { Self::new(self.v + rhs.into().v) }
}

const impl<Rhs: const Into<AnyScore>> ops::Sub<Rhs> for AnyScore {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Rhs) -> Self::Output { Self::new(self.v - rhs.into().v) }
}

const impl ops::Div<AnyScore> for AnyScore {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: AnyScore) -> Self::Output { Self::new(self.v / rhs.v) }
}

const impl<Rhs: const Into<AnyScore>> ops::Mul<Rhs> for AnyScore {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Rhs) -> Self::Output { Self::new(self.v * rhs.into().v) }
}

impl iter::Sum for AnyScore {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self { iter.fold(Self { v: 0 }, |acc, x| acc + x) }
}

/// A bonus for `P`
#[derive(Copy, Clone)]
pub struct Score<P: Perspective>(pub AnyScore, PhantomData<P>);

impl<P: Perspective> fmt::Debug for Score<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Score<{}>({})", P::COLOR, self.0) }
}

impl<P: Perspective> fmt::Display for Score<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Score<{}>({})", P::COLOR, self.0) }
}

const impl<P: Perspective, Rhs: const Into<Score<P>>> ops::Add<Rhs> for Score<P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output { Self::new(self.0 + rhs.into().0) }
}
const impl<P: Perspective> ops::Add<i32> for Score<P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i32) -> Self::Output { Self::new(self.0 + rhs) }
}

const impl<P: Perspective, Rhs: const Into<Score<P>>> ops::Sub<Rhs> for Score<P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Rhs) -> Self::Output { Self::new(self.0 - rhs.into().0) }
}
const impl<P: Perspective> ops::Sub<i32> for Score<P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self::Output { Self::new(self.0 - rhs) }
}

const impl<P: Perspective> ops::Div<AnyScore> for Score<P> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: AnyScore) -> Self::Output { Self::new(self.0 / rhs) }
}

const impl<P: Perspective> Ord for Score<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.cmp(&other.0) }
}

const impl<P: Perspective> PartialOrd for Score<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

const impl<P: Perspective> Eq for Score<P> {}
const impl<P: Perspective> PartialEq for Score<P> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<P: Perspective> Score<P> {
    #[inline(always)]
    const fn new(val: AnyScore) -> Self { Self(val, PhantomData) }

    pub const NEG_INF: Self = unsafe { scores::NEG_INF.interpret_as() };
    pub const POS_INF: Self = unsafe { scores::POS_INF.interpret_as() };
    pub const DRAW: Self = unsafe { scores::DRAW.interpret_as() };
    pub const ZERO: Self = unsafe { scores::ZERO.interpret_as() };
    pub const NULL: Self = unsafe { scores::NULL.interpret_as() };
}

// not using the `-` operator because this is not really just arithmetic
// negation, but also a perspective flip.
impl<P: Perspective> ops::Not for Score<P> {
    type Output = Score<P::Opponent>;

    /// Negate the score and flip the perspective to the opponent.
    #[inline(always)]
    fn not(self) -> Self::Output { Score::new(-self.0) }
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

impl From<AnyScore> for Cp {
    fn from(value: AnyScore) -> Self {
        let v: TCp = value.v.saturating_cast();
        Cp { v }
    }
}

impl<P: Perspective> From<Score<P>> for Cp {
    #[inline(always)]
    fn from(value: Score<P>) -> Self {
        let v: TCp = value.0.v.saturating_cast();
        if P::IS_WHITE { Cp { v } } else { Cp { v: -v } }
    }
}
