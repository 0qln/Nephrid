use std::cell::UnsafeCell;
use std::ops::ControlFlow;

use crate::misc::DebugMode;
use crate::uci::sync::{self, CancellationToken};
use crate::core::position::Position;

use limit::Limit;

use super::depth::Depth;
use super::r#move::Move;
use super::move_iter::fold_legal_moves;

pub mod limit;
pub mod mcts;
pub mod mode;

pub fn perft(pos: Position, limit: Limit, ct: CancellationToken, debug: DebugMode) -> u64 {
    perft_inner(
        &mut UnsafeCell::new(pos),
        limit.depth,
        limit,
        ct.clone(),
        debug,
        |mov, count, depth: Depth, debug| {
            if debug {
                let indent = itertools::repeat_n(' ', depth.v().into()).collect::<String>();
                sync::out(&format!("{}{mov:?}: {count}", indent));
            } else {
                sync::out(&format!("{mov}: {count}"));
            }
        },
    )
}

fn perft_inner(
    pos: &mut UnsafeCell<Position>,
    depth: Depth,
    limit: Limit,
    cancellation_token: CancellationToken,
    debug: DebugMode,
    f: fn(Move, u64, Depth, bool) -> (),
) -> u64 {
    if cancellation_token.is_cancelled() {
        return 0;
    }

    if depth <= Depth::MIN {
        return 1;
    }

    // Safety:
    // This is safe iff unmake_move perfectly reverses the muations made by
    // make_move.
    unsafe {
        fold_legal_moves::<_, _, _>(&*pos.get(), 0, |acc, m| {
            pos.get_mut().make_move(m);
            let c = perft_inner(
                pos,
                depth - 1,
                limit.clone(),
                cancellation_token.clone(),
                debug.clone(),
                if debug.get() { f } else { |_, _, _, _| {} },
            );
            f(m, c, limit.depth - depth, debug.get());
            pos.get_mut().unmake_move(m);
            ControlFlow::Continue::<(), _>(acc + c)
        })
        .continue_value()
        .unwrap()
    }
}
