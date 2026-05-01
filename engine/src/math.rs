pub fn entropy(xs: impl Iterator<Item = f32>) -> f32 {
    -xs.filter(|&x| x > 0.).map(|x| x * x.log2()).sum::<f32>()
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
