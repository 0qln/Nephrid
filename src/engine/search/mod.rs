use std::cell::UnsafeCell;
use std::ops::ControlFlow;

use crate::uci::sync::{self, CancellationToken};
use limit::Limit;
use mode::Mode;
use score::Score;
use target::Target;

use crate::engine::position::Position;

use super::depth::Depth;
use super::move_iter::fold_legal_move;
use super::r#move::Move;

pub mod limit;
pub mod mode;
pub mod score;
pub mod target;

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

    pub fn perft(
        pos: &mut UnsafeCell<Position>,
        depth: Depth,
        cancellation_token: CancellationToken,
        f: fn(Move, u64) -> (),
    ) -> u64 {
        if depth <= Depth::MIN {
            return 1;
        }
        
        // Safety: 
        // This is safe iff unmake_move perfectly reverses the muations made by make_move.
        unsafe {
            fold_legal_move::<false, _, _, _>(&*pos.get(), 0, |acc, m| {
                pos.get_mut().make_move(m);
                let c = Self::perft(pos, depth - 1, cancellation_token.clone(), |_, _| {});
                f(m, c);
                pos.get_mut().unmake_move(m);
                ControlFlow::Continue::<(), _>(acc + c)
            }).continue_value().unwrap()
        }
    }

    pub fn go(&self, position: &mut Position, cancellation_token: CancellationToken) {
        match self.mode {
            Mode::Perft => {
                let nodes = Self::perft(&mut UnsafeCell::new(position.clone()), self.target.depth, cancellation_token, |m, c| {
                    sync::out(&format!("{m}: {c}"));
                });
                sync::out(&format!("\nNodes searched: {nodes}"));
            }
            _ => unimplemented!(),
        };
    }

    fn alpha_beta(
        &self,
        position: &mut Position,
        depth: Depth,
        alpha: Score,
        beta: Score,
    ) -> Score {
        // fold_legal_move::<false>(position, 0, |m| {
        //     ControlFlow::Continue(0)
        //         ControlFlow::Break(())
        // })

        Default::default()
    }
}
