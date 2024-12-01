use std::ops::ControlFlow;

use mode::Mode;
use score::Score;
use target::Target;
use limit::Limit;
use crate::uci::sync::CancellationToken;

use crate::engine::position::Position;

use super::depth::Depth;
use super::move_iter::{fold_legal_move, foreach_legal_move, legal_moves_check_single};

pub mod mode;
pub mod target;
pub mod limit;
pub mod score;

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
        
        for m in legal_moves::<false>(position, position.get_turn()) {
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
    
    fn alpha_beta(&self, position: &mut Position, depth: Depth, alpha: Score, beta: Score) -> Score {


        fold_legal_move::<false>(position, 0, |m| {
            ControlFlow::Continue(0)
                ControlFlow::Break(())
        })

    }

}

