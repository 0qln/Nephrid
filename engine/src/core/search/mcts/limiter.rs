use crate::core::Depth;
use crate::core::Limit;
use crate::core::Position;

pub trait Limiter {
    /// Returns: Whether to stop searching.
    fn should_stop(&self, params: Params) -> bool;
}

pub struct Params<'a> {
    pos: &'a Position,
    depth: Depth,
}

#[derive(Default, Debug)]
pub struct NoopLimiter;

impl Limiter for NoopLimiter {
    fn should_stop(&self, _params: Params) -> bool {
        false
    }
}

#[derive(Default, Debug)]
pub struct DefaultLimiter {
    limit: Limit,
}

impl Limiter for DefaultLimiter {
    fn should_stop(&self, _pos: &Position, depth: Depth) -> bool {
        depth > self.limit.depth || depth > Depth::MAX
    }
}
