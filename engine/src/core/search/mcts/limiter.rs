pub trait Limiter {
    /// Returns: Whether to stop searching.
    fn should_stop(&self, pos: &Position, depth: Depth) -> bool;
}

#[derive(Default, Debug)]
pub struct NoopLimiter;

impl Limiter for NoopLimiter {
    fn should_stop(&self, _pos: &Position, _depth: Depth) -> bool {
        false
    }
}
