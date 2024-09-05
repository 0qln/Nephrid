use crate::uci::sync::CancellationToken;

use crate::engine::{
    r#move::{MoveList},
    depth::Depth,
    position::Position,
    ply::Ply,
};

#[derive(Debug)]
pub struct Limit {
    pub wtime: u64,
    pub btime: u64,
    pub winc: u64,
    pub binc: u64,
    pub movestogo: u16,
    pub nodes: u64,
    pub movetime: u64,
}

impl Limit {
    pub fn max() -> Self {
        Self {
            wtime: u64::MAX,
            btime: u64::MAX,
            winc: u64::MAX,
            binc: u64::MAX,
            movestogo: 0,
            nodes: u64::MAX,
            movetime: u64::MAX,
        }
    }
}

impl Default for Limit {
    fn default() -> Self {
        Self {
            wtime: u64::MAX, 
            btime: u64::MAX,
            winc: 0, 
            binc: 0,
            movestogo: 0,
            nodes: u64::MAX,
            movetime: u64::MAX,
        }
    }
}

#[derive(Debug, Default)]
pub struct Target<'a> {
    pub mate: Depth,
    pub depth: Depth,
    pub search_moves: MoveList<'a>,
}

#[derive(Debug, Default)]
pub enum Mode {
    #[default]
    Normal,
    Ponder,
    Perft
}


#[derive(Debug, Default)]
pub struct Search<'a> {
    pub limit: Limit,
    pub target: Target<'a>,
    pub mode: Mode,
}

impl Search<'_> {
    pub fn reset() {
        todo!()
    }

    pub fn go(position: &mut Position, cancellation_token: CancellationToken) {
        todo!()
    }

    pub fn perft(position: &mut Position, ply: Ply, cancellation_token: CancellationToken) -> u64 {
        todo!();
        let result = 1;

        for move in position.legal_moves() {
            position.make_move(move);
            result += perft(position, , cancellation_token);
            position.unmake_move();
        }

        result;
    }

    pub fn search(position: &mut Position, cancellation_token: CancellationToken) {
        todo!()
    }
}

