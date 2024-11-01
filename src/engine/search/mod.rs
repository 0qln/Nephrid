use types::{Limit, Target};
use crate::engine::move_iter::legal_moves;
use crate::uci::sync::CancellationToken;

use crate::engine::{
    ply::Ply,
    position::Position
    ,
};
mod types;

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
        let mut result = 1;

        for m in legal_moves(&position) {
            position.make_move(m);
            result += Self::perft(position, ply - 1, cancellation_token);
            position.unmake_move();
        }

        result
    }

    pub fn search(position: &mut Position, cancellation_token: CancellationToken) {
        todo!()
    }
}

