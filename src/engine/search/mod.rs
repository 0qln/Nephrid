use mode::Mode;
use target::Target;
use limit::Limit;
use crate::uci::sync::CancellationToken;

use crate::engine::position::Position;

pub mod mode;
pub mod target;
pub mod limit;

#[derive(Debug, Default)]
pub struct Search {
    pub limit: Limit,
    pub target: Target,
    pub mode: Mode,
}

impl Search {
    pub fn reset() {
        todo!()
    }

    pub fn perft(position: &mut Position, cancellation_token: CancellationToken) -> u64 {
        let mut result = 1;

        // for m in legal_moves(&position) {
        //     position.make_move(m);
        //     result += Self::perft(position, ply - 1, cancellation_token);
        //     position.unmake_move();
        // }

        result
    }

    pub fn go(&self, mut position: Position, cancellation_token: CancellationToken) {
        todo!()
    }
}

