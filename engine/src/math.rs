use std::{fmt, marker::PhantomData, ops::Deref};

use crate::misc::{CheckHealth, CheckHealthResult, List};

pub fn entropy(xs: impl Iterator<Item = Probability>) -> f32 {
    -xs.filter(|x| x.v() > 0.)
        .map(|x| x.v() * x.log2())
        .sum::<f32>()
}

/// The Shannon [`entropy`] of a distribution normalized to its maximum possible
/// value (`log2(n)`), yielding a position-independent fraction in `[0, 1]`:
///
/// - `0` => fully concentrated (a single outcome holds all the probability
///   mass)
/// - `1` => uniform
///
/// A distribution of fewer than two outcomes carries no uncertainty and maps to
/// `0`.
pub fn normalized_entropy<const N: usize>(xs: &List<N, Probability>) -> NormalizedEntropy {
    let n = xs.len();
    if n <= 1 {
        return NormalizedEntropy::zero();
    }

    let h = entropy(xs.iter().copied());
    let max = (n as f32).log2();

    // `h <= log2(n)` mathematically, so the quotient lies in `[0, 1]` (up to the
    // tolerance `Bounded::new` already accounts for).
    NormalizedEntropy::new(h / max)
}

pub type NormalizedEntropy = Bounded<f32, Bounds0to1>;

pub fn avg(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len() as f32
}

pub fn variance(xs: &[f32]) -> f32 {
    let avg = avg(xs);
    xs.iter().map(|x| (x - avg).powi(2)).sum::<f32>() / xs.len() as f32
}

pub fn stddev(xs: &[f32]) -> f32 {
    variance(xs).sqrt()
}

/// Applies the softmax without allocating a new list.
pub fn softmax<const N: usize>(
    mut xs: List<N, f32>,
    temperature: f32,
    exps: &mut List<N, f32>,
) -> List<N, Probability> {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    exps.clear();
    for x in xs.iter().map(|x| ((x - max) / temperature).exp()) {
        exps.push(x);
    }

    let sum: f32 = exps.iter().sum();

    for (x, e) in xs.iter_mut().zip(exps.iter()) {
        *x = *e / sum;
    }

    // SAFETY: Probability is the same layout as an f32 and we just mathematically
    // transformed the array to be probabilities.
    unsafe { List::transmute(xs) }
}

impl<T, B: FloatBounds> Deref for Bounded<T, B> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl CheckHealth for Bounded<f32, Bounds0to1> {
    type Error = String;
    fn check_health(&self) -> CheckHealthResult<Self::Error> {
        if self.0.is_nan() {
            return Err("value was NaN".to_string());
        }
        if self.0.is_infinite() {
            return Err("value was infinite".to_string());
        }
        if self.0 < -Self::EPS || self.0 > (1. + Self::EPS) {
            return Err(format!("value {} was out of range [0; 1]", self.0));
        }
        Ok(())
    }
}

pub trait FloatBounds {
    const MIN: f32;
    const MAX: f32;
}

#[derive(Clone, Copy, Debug, PartialEq, Default, PartialOrd)]
#[repr(transparent)]
pub struct Bounded<T, B>(T, PhantomData<B>);

impl<B: FloatBounds> Bounded<f32, B> {
    /// Allowed inaccuracy
    const EPS: f32 = 1e-5;

    #[inline(always)]
    pub fn new(v: f32) -> Self {
        debug_assert!(
            v >= (B::MIN - Self::EPS) && v <= (B::MAX + Self::EPS),
            "Value must be in range [{}; {}], but was: {}",
            B::MIN,
            B::MAX,
            v
        );
        Self(v, PhantomData)
    }

    #[inline(always)]
    pub const fn new_c(v: f32) -> Self {
        Self(v, PhantomData)
    }

    #[inline(always)]
    pub const fn v(&self) -> f32 {
        self.0
    }

    /// Mixes the value with another value by a given ratio.
    #[inline(always)]
    pub fn mix(&mut self, other: Self, ratio: Ratio) {
        self.0 = (self.0 * (1. - ratio.0)) + (other.0 * ratio.0);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds0to1;

impl FloatBounds for Bounds0to1 {
    const MIN: f32 = 0.0;
    const MAX: f32 = 1.0;
}

impl Bounded<f32, Bounds0to1> {
    #[inline(always)]
    pub const fn zero() -> Probability {
        Self(0., PhantomData)
    }

    #[inline(always)]
    pub const fn one() -> Probability {
        Self(1., PhantomData)
    }

    #[inline(always)]
    pub const fn even() -> Probability {
        Self(0.5, PhantomData)
    }

    #[inline(always)]
    pub const fn inv(&self) -> Self {
        Self(1. - self.0, PhantomData)
    }
}

pub type Probability = Bounded<f32, Bounds0to1>;

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub type Ratio = Bounded<f32, Bounds0to1>;
