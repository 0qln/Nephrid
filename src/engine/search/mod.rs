use types::{Limit, Target};
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

