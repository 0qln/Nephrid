use super::{Depth, Move, Position};

pub static mut MODE: Mode = Mode::Normal;
pub static mut TARGET: Target = Target::default();
pub static mut LIMIT: Limit = Limit::max(false);

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
    pub fn max(set_active: bool) -> Self {
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

pub struct Target {
    pub mate: Depth,
    pub depth: Depth,
    pub searchmoves: Vec<Move>,
}

impl Target {
    pub fn default() -> Self {
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

pub fn go(board: &mut Position) {
    todo!()
}
