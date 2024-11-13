use mode::Mode;
use target::Target;
use limit::Limit;
use crate::uci::sync::CancellationToken;

use crate::engine::position::Position;

use super::depth::Depth;
use super::move_iter::legal_moves;

pub mod mode;
pub mod target;
pub mod limit;

#[derive(Debug, Default, Clone)]
pub struct Search {
    pub limit: Limit,
    pub target: Target,
    pub mode: Mode,
}

impl Search {
    pub fn reset() {
        todo!()
    }

    fn perft(&self, position: &mut Position, depth: Depth, cancellation_token: CancellationToken) -> u64 {
        let mut result = 1;
        
        if depth.v() == 0 { return result; }
        
        for m in legal_moves(position) {
            position.make_move(m);
            result += self.perft(position, depth - 1, cancellation_token.clone());
            position.unmake_move(m);
        }

        result
    }

    pub fn go(&self, position: &mut Position, cancellation_token: CancellationToken) {
        match self.mode {
            Mode::Perft => self.perft(position, self.target.depth, cancellation_token),
            _ => unimplemented!()
        };
    }
}

