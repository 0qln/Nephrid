use std::{cmp::Ordering, fmt, marker::PhantomData, ops::Deref};

use crate::misc::{CheckHealth, CheckHealthResult, List};

pub fn entropy(xs: impl Iterator<Item = Probability>) -> f32 { -xs.filter(|x| x.v() > 0.).map(|x| x.v() * x.log2()).sum::<f32>() }

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

pub fn avg(xs: &[f32]) -> f32 { xs.iter().sum::<f32>() / xs.len() as f32 }

pub fn variance(xs: &[f32]) -> f32 {
    let avg = avg(xs);
    xs.iter().map(|x| (x - avg).powi(2)).sum::<f32>() / xs.len() as f32
}

pub fn stddev(xs: &[f32]) -> f32 { variance(xs).sqrt() }

/// Applies the softmax without allocating a new list.
#[cfg(not(target_feature = "avx2"))]
pub fn softmax<const N: usize>(mut xs: List<N, f32>, temperature: f32, exps: &mut List<N, f32>) -> List<N, Probability> {
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

#[cfg(target_feature = "avx2")]
pub fn softmax<const N: usize>(mut xs: List<N, f32>, temperature: f32, _exps: &mut List<N, f32>) -> List<N, Probability> {
    use wide::f32x8;

    // find max
    let mut max_vec = f32x8::splat(f32::NEG_INFINITY);
    let (xs_chunks, xs_rem) = xs.as_slice().as_chunks::<8>();
    for x_chunk in xs_chunks {
        max_vec = max_vec.max(f32x8::from(*x_chunk));
    }
    let mut max = max_vec.to_array().into_iter().fold(f32::NEG_INFINITY, f32::max);
    for x in xs_rem {
        max = max.max(*x);
    }

    // exp and sum
    let temp_vec = f32x8::splat(temperature);
    let max_vec = f32x8::splat(max);
    let mut sum_vec = f32x8::splat(0.);

    let (xs_chunks, xs_rem) = xs.as_mut_slice().as_chunks_mut::<8>();
    for x_chunk in xs_chunks.iter_mut() {
        let exp_vec = ((f32x8::from(*x_chunk) - max_vec) / temp_vec).exp();
        *x_chunk = exp_vec.to_array();
        sum_vec += exp_vec;
    }
    let mut sum: f32 = sum_vec.to_array().iter().sum();
    for x in xs_rem.iter_mut() {
        let exp = ((*x - max) / temperature).exp();
        *x = exp;
        sum += exp;
    }

    // normalize
    let sum_vec = f32x8::splat(sum);
    let (xs_chunks, xs_rem) = xs.as_mut_slice().as_chunks_mut::<8>();
    for x_chunk in xs_chunks.iter_mut() {
        let norm_vec = f32x8::from(*x_chunk) / sum_vec;
        *x_chunk = norm_vec.to_array();
    }
    for x in xs_rem.iter_mut() {
        let norm = *x / sum;
        *x = norm;
    }

    // SAFETY: Probability is the same layout as an f32 and we just mathematically
    // transformed the array to be probabilities.
    unsafe { List::transmute(xs) }
}

impl<T, B: FloatBounds> Deref for Bounded<T, B> {
    type Target = T;
    fn deref(&self) -> &Self::Target { &self.0 }
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

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Bounded<T, B>(T, PhantomData<B>);

impl<T: PartialOrd, B: PartialEq> PartialOrd for Bounded<T, B> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.0.partial_cmp(&other.0) }
}

impl Default for Bounded<f32, Bounds0to1> {
    fn default() -> Self { Self(Bounds0to1::MIN, PhantomData) }
}

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
    pub const fn new_c(v: f32) -> Self { Self(v, PhantomData) }

    #[inline(always)]
    pub const fn v(&self) -> f32 { self.0 }

    /// Mixes the value with another value by a given ratio.
    #[inline(always)]
    pub fn mix(&mut self, other: Self, ratio: Ratio) { self.0 = (self.0 * (1. - ratio.0)) + (other.0 * ratio.0); }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds0to1;

impl FloatBounds for Bounds0to1 {
    const MIN: f32 = 0.0;
    const MAX: f32 = 1.0;
}

impl Bounded<f32, Bounds0to1> {
    #[inline(always)]
    pub const fn zero() -> Probability { Self(0., PhantomData) }

    #[inline(always)]
    pub const fn one() -> Probability { Self(1., PhantomData) }

    #[inline(always)]
    pub const fn even() -> Probability { Self(0.5, PhantomData) }

    #[inline(always)]
    pub const fn inv(&self) -> Self { Self(1. - self.0, PhantomData) }
}

pub type Probability = Bounded<f32, Bounds0to1>;

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.0) }
}

pub type Ratio = Bounded<f32, Bounds0to1>;
