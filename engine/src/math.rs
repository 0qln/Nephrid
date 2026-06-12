use crate::{
    core::search::mcts::eval::{Bounded, Bounds0to1, Probability},
    misc::List,
};

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
