use crate::core::depth::Depth;
use crate::core::r#move::Move;

#[derive(Debug, Default, Clone)]
pub struct Target {
    pub mate: Depth,
    pub depth: Depth,
    pub search_moves: Vec<Move>,
}