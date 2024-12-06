use std::ops::ControlFlow;

use crate::uci::sync::{self, CancellationToken};
use limit::Limit;
use mode::Mode;
use score::Score;
use target::Target;

use crate::engine::position::Position;

use super::depth::Depth;
use super::move_iter::{
    fold_legal_move, foreach_legal_move, legal_moves_check_double, legal_moves_check_none,
    legal_moves_check_single,
};
use super::position::CheckState;
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

    fn perft(
        &self,
        pos: &mut Position,
        depth: Depth,
        cancellation_token: CancellationToken,
        f: impl Fn(Move, u64) -> (),
    ) -> u64 {
        let mut result = 1;

        println!("gfasl;ku");
        if depth <= Depth::MIN {
            return result;
        }
        
        println!("aklsdjf");

        // foreach_legal_move::<false, _, _>(pos, &mut |m| {
        //     pos.make_move(m);
        //     result += self.perft(pos, depth - 1, cancellation_token.clone());
        //     pos.unmake_move(m);
        //     ControlFlow::Continue::<()>(())
        // });

        fn do_perft(
            pos: &mut Position,
            depth: Depth,
            cancellation_token: CancellationToken,
        ) -> u64 {
            let mut result = 1;
            if depth <= Depth::MIN { return result; }
            match pos.get_check_state() {
                CheckState::None => {
                    for m in legal_moves_check_none::<false>(pos) {
                        pos.make_move(m);
                        result += do_perft(pos, depth - 1, cancellation_token.clone());
                        pos.unmake_move(m);
                    }
                }
                CheckState::Single => {
                    for m in legal_moves_check_single::<false>(pos) {
                        pos.make_move(m);
                        result += do_perft(pos, depth - 1, cancellation_token.clone());
                        pos.unmake_move(m);
                    }
                }
                CheckState::Double => {
                    for m in legal_moves_check_double::<false>(pos) {
                        pos.make_move(m);
                        result += do_perft(pos, depth - 1, cancellation_token.clone());
                        pos.unmake_move(m);
                    }
                }
            }
            result
        }

        match pos.get_check_state() {
            CheckState::None => {
                for m in legal_moves_check_none::<false>(pos) {
                    pos.make_move(m);
                    let nodes = do_perft(pos, depth - 1, cancellation_token.clone());
                    result += nodes;
                    f(m, nodes);
                    pos.unmake_move(m);
                }
            }
            CheckState::Single => {
                for m in legal_moves_check_single::<false>(pos) {
                    pos.make_move(m);
                    let nodes = do_perft(pos, depth - 1, cancellation_token.clone());
                    result += nodes;
                    f(m, nodes);
                    pos.unmake_move(m);
                }
            }
            CheckState::Double => {
                for m in legal_moves_check_double::<false>(pos) {
                    pos.make_move(m);
                    let nodes = do_perft(pos, depth - 1, cancellation_token.clone());
                    result += nodes;
                    f(m, nodes);
                    pos.unmake_move(m);
                }
            }
        }

        result
    }

    pub fn go(&self, position: &mut Position, cancellation_token: CancellationToken) {
        match self.mode {
            Mode::Perft => {
                println!("Perft depth: {}", self.target.depth.v());
                let nodes = self.perft(position, self.target.depth, cancellation_token, |m, c| {
                    sync::out(&format!("{m:?}: {c}"));
                });
                sync::out(&format!("Nodes searched: {nodes}"));
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
