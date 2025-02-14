use std::cell::UnsafeCell;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::uci::sync::{self, CancellationToken};
use limit::Limit;
use mode::Mode;
use target::Target;

use crate::engine::position::Position;

use super::depth::Depth;
use super::move_iter::fold_legal_moves;
use super::r#move::Move;

pub mod limit;
pub mod mode;
pub mod mcts;
pub mod target;

#[derive(Default, Debug, Clone)]
pub struct Search {
    pub limit: Limit,
    pub target: Target,
    pub mode: Mode,
    pub debug: Arc<AtomicBool>,
    tree: mcts::Tree,
}

impl Search {
    pub fn new(limit: Limit, target: Target, mode: Mode, debug: Arc<AtomicBool>) -> Self {
        Self { limit, target, mode, debug, ..Default::default() }
    }

    pub fn reset() {
        todo!()
    }

    pub fn perft(
        pos: &mut UnsafeCell<Position>,
        depth: Depth,
        cancellation_token: CancellationToken,
        f: fn(Move, u64) -> (),
    ) -> u64 {
        if cancellation_token.is_cancelled() {
            return 0;
        }

        if depth <= Depth::MIN {
            return 1;
        }
        
        // Safety: 
        // This is safe iff unmake_move perfectly reverses the muations made by make_move.
        unsafe {
            fold_legal_moves::<_, _, _>(&*pos.get(), 0, |acc, m| {
                pos.get_mut().make_move(m);
                let c = Self::perft(pos, depth - 1, cancellation_token.clone(), |_, _| {});
                f(m, c);
                pos.get_mut().unmake_move(m);
                ControlFlow::Continue::<(), _>(acc + c)
            }).continue_value().unwrap()
        }
    }
    
    pub fn mcts(
        &self,
        mut pos: Position,
        cancellation_token: CancellationToken
    ) -> Move {
        let mut tree = mcts::Tree::new(&pos);
        let mut last_best_move = None;
        
        let time_per_move = self.limit.time_per_move(&pos);
        let time_limit = Instant::now() + time_per_move;

        while {
            !cancellation_token.is_cancelled() &&
            (!self.limit.is_active || Instant::now() < time_limit)
        } {

            tree.grow(&mut pos);

            let curr_best_move = tree.best_move();
            if last_best_move != curr_best_move {
                if let Some(best_move) = curr_best_move {
                    sync::out(&format!("currmove {best_move}"));
                }
                last_best_move = curr_best_move;
            }

        }
        
        let last_best_move = last_best_move.expect("search did not complete");
        tree.advance(last_best_move);
        last_best_move
    }

    pub fn go(&self, position: &mut Position, cancellation_token: CancellationToken) {
        match self.mode {
            Mode::Perft => {
                let nodes = Self::perft(&mut UnsafeCell::new(position.clone()), self.target.depth, cancellation_token, |m, c| {
                    sync::out(&format!("{m}: {c}"));
                });
                sync::out(&format!("\nNodes searched: {nodes}"));
            }
            Mode::Normal => {
                let result = self.mcts(
                    position.clone(),
                    cancellation_token
                );
                sync::out(&format!("bestmove {result}"));
            }
            _ => unimplemented!(),
        };
    }
}
