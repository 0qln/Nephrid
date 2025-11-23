use std::cell::UnsafeCell;
use std::ops::ControlFlow;
use std::time::Instant;

use crate::misc::DebugMode;
use crate::uci::sync::{self, CancellationToken};
use burn::prelude::Backend;
use limit::Limit;
use mcts::eval::model::Model;
use target::Target;

use crate::core::position::Position;

use super::depth::Depth;
use super::r#move::Move;
use super::move_iter::fold_legal_moves;

pub mod limit;
pub mod mcts;
pub mod mode;
pub mod target;

pub fn perft(pos: Position, target: Target, ct: CancellationToken, debug: DebugMode) -> u64 {
    perft_inner(
        &mut UnsafeCell::new(pos),
        target.depth,
        target,
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
    target: Target,
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
                target.clone(),
                cancellation_token.clone(),
                debug.clone(),
                if debug.get() { f } else { |_, _, _, _| {} },
            );
            f(m, c, target.depth - depth, debug.get());
            pos.get_mut().unmake_move(m);
            ControlFlow::Continue::<(), _>(acc + c)
        })
        .continue_value()
        .unwrap()
    }
}

pub fn mcts<B: Backend>(
    pos: Position,
    model: &Model<B>,
    limit: Limit,
    debug: DebugMode,
    ct: CancellationToken,
) -> Option<Move> {
    mcts_inner(pos, &model, limit, debug, ct)
}

fn mcts_inner<B: Backend>(
    mut pos: Position,
    model: &Model<B>,
    limit: Limit,
    _debug: DebugMode,
    ct: CancellationToken,
) -> Option<Move> {
    let mut tree = mcts::Tree::new(&pos, model);
    let mut last_best_move = None;

    let time_per_move = limit.time_per_move(&pos);
    let time_limit = Instant::now() + time_per_move;

    let mut iterations = 0;

    while !ct.is_cancelled() && (!limit.is_active || Instant::now() < time_limit) {
        tree.grow(&mut pos, &model);
        iterations += 1;

        let curr_best_move = tree.best_move();
        if last_best_move != curr_best_move {
            if let Some(mov) = curr_best_move {
                sync::out(&format!("currmove {mov}"));
                last_best_move = Some(mov);
            }
        }
    }

    println!("iterations: {iterations}");

    last_best_move
}
