use std::sync::Arc;

use crate::uci::CancellationToken;

use super::{Depth, Move, Position};

pub struct Limit {
    pub wtime: u64,
    pub btime: u64,
    pub winc: u64,
    pub binc: u64,
    pub movestogo: u16,
    pub nodes: u64,
    pub movetime: u64,
    pub active: bool,
}

impl Limit {
    pub const fn max(set_active: bool) -> Self {
        Self {
            wtime: u64::MAX,
            btime: u64::MAX,
            winc: u64::MAX,
            binc: u64::MAX,
            movestogo: 0,
            nodes: u64::MAX,
            movetime: u64::MAX,
            active: set_active,
        }
    }
}

impl Default for Limit {
    fn default() -> Self {
        Self::max(false)
    }
}

pub struct Target {
    pub mate: Depth,
    pub depth: Depth,
    pub searchmoves: Vec<Move>,
}

impl Default for Target {
    fn default() -> Self {
        Self {
            mate: Depth::NONE,
            depth: Depth::NONE,
            searchmoves: Vec::new(),
        }
    }
}

pub enum Mode {
    Normal,
    Ponder,
}

pub fn reset() {
    todo!()
}

pub fn go(position: &Position, limit: Limit, target: Target, mode: Mode, cancellation_token: CancellationToken) {
    todo!()
}
