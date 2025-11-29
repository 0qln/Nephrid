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

pub trait MctsStrategy {
    type Result;

    fn result(&mut self, tree: &mut mcts::Tree) -> Self::Result;
    fn step(&mut self, tree: &mut mcts::Tree);
}

#[derive(Default, Debug)]
pub struct MctsUci {
    last_best_move: Option<Move>,
}

impl MctsStrategy for MctsUci {
    type Result = Option<Move>;

    fn result(&mut self, _tree: &mut mcts::Tree) -> Self::Result {
        self.last_best_move
    }

    fn step(&mut self, tree: &mut mcts::Tree) {
        let curr_best_move = tree.best_move();
        if self.last_best_move != curr_best_move {
            if let Some(mov) = curr_best_move {
                sync::out(&format!("currmove {mov}"));
                self.last_best_move = Some(mov);
            }
        }
    }
}

/// Debugs another mcts strategy
#[derive(Default, Debug)]
pub struct MctsDebug<I: MctsStrategy> {
    inner: I,
    iterations: u64,
}

impl<I: MctsStrategy> MctsStrategy for MctsDebug<I> {
    type Result = (<I as MctsStrategy>::Result, u64);

    fn result(&mut self, tree: &mut mcts::Tree) -> Self::Result {
        (self.inner.result(tree), self.iterations)
    }

    fn step(&mut self, tree: &mut mcts::Tree) {
        self.inner.step(tree);
        self.iterations += 1;
    }
}

pub fn mcts<S: MctsStrategy + Default, B: Backend>(
    pos: Position,
    model: &Model<B>,
    limit: Limit,
    debug: DebugMode,
    ct: CancellationToken,
) -> S::Result {
    mcts_inner::<S, B>(pos, &model, limit, debug, ct, S::default())
}

fn mcts_inner<S: MctsStrategy, B: Backend>(
    mut pos: Position,
    model: &Model<B>,
    limit: Limit,
    _debug: DebugMode,
    ct: CancellationToken,
    mut strategy: S,
) -> S::Result {
    let mut tree = mcts::Tree::new(&pos, model);

    let time_per_move = limit.time_per_move(&pos);
    let time_limit = Instant::now() + time_per_move;

    while !ct.is_cancelled() && (!limit.is_active || Instant::now() < time_limit) {
        tree.grow(&mut pos, &model);
        strategy.step(&mut tree);
    }

    strategy.result(&mut tree)
}
