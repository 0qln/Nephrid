use crate::core::{Depth, Limit, Position};

pub trait Limiter {
    /// Returns: Whether to stop searching.
    fn should_stop(&self, params: Params) -> bool;

    fn new(limit: &Limit) -> Self;
}

pub struct Params<'a> {
    pub pos: &'a Position,
    pub depth: Depth,
}

#[derive(Default, Clone, Debug)]
pub struct NoopLimiter;

impl Limiter for NoopLimiter {
    fn should_stop(&self, _params: Params) -> bool {
        false
    }

    fn new(_limit: &Limit) -> Self {
        Self
    }
}

#[derive(Default, Debug, Clone)]
pub struct DefaultLimiter {
    depth_limit: Depth,
}

impl Limiter for DefaultLimiter {
    fn should_stop(&self, p: Params) -> bool {
        p.depth > self.depth_limit || p.depth > Depth::MAX
    }

    fn new(limit: &Limit) -> Self {
        Self { depth_limit: limit.depth }
    }
}
