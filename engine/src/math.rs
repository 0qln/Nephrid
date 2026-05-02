use crate::core::search::mcts::eval::Probability;

pub fn entropy(xs: impl Iterator<Item = Probability>) -> f32 {
    -xs.filter(|x| x.v() > 0.).map(|x| x.v() * x.log2()).sum::<f32>()
}

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
