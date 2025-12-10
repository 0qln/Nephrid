use std::cell::UnsafeCell;
use std::ops::ControlFlow;
use std::time::Instant;

use crate::misc::DebugMode;
use crate::uci::sync::{self, CancellationToken};
use itertools::Itertools;
use limit::Limit;

use crate::core::position::Position;

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

pub trait MctsStrategy {
    type Result;
    type Step;

    fn result(&mut self, tree: &mut mcts::Tree) -> Self::Result;
    fn step(&mut self, tree: &mut mcts::Tree) -> Self::Step;
}

#[derive(Default, Debug)]
pub struct MctsFindBest {
    last_best_move: Option<Move>,
}

impl MctsStrategy for MctsFindBest {
    type Result = Option<Move>;
    type Step = Option<Move>;

    fn result(&mut self, _tree: &mut mcts::Tree) -> Self::Result {
        self.last_best_move
    }

    fn step(&mut self, tree: &mut mcts::Tree) -> Self::Step {
        let curr_best_move = tree.best_move();
        if self.last_best_move != curr_best_move {
            if let Some(mov) = curr_best_move {
                self.last_best_move = Some(mov);
                return Some(mov);
            }
        }
        return None;
    }
}

#[derive(Default, Debug)]
pub struct MctsUci {
    find_best: MctsFindBest,
}

impl MctsStrategy for MctsUci {
    type Result = <MctsFindBest as MctsStrategy>::Result;
    type Step = <MctsFindBest as MctsStrategy>::Step;

    fn result(&mut self, tree: &mut mcts::Tree) -> Self::Result {
        self.find_best.result(tree)
    }

    fn step(&mut self, tree: &mut mcts::Tree) -> Self::Step {
        let step = self.find_best.step(tree);
        let pv = tree.principal_variation();
        if let Some(mov) = step {
            sync::out(&format!("currmove {mov}"));
            sync::out(&format!(
                "info pv {}",
                pv.iter().map(|x| x.mov().to_string()).join(" ")
            ));
        }
        step
    }
}

/// Debugs another mcts strategy
#[derive(Default, Debug)]
pub struct MctsDebug<I: MctsStrategy> {
    inner: I,
    iteration: u64,
}

impl<I: MctsStrategy> MctsStrategy for MctsDebug<I> {
    type Result = (<I as MctsStrategy>::Result, u64);
    type Step = (<I as MctsStrategy>::Step, u64);

    fn result(&mut self, tree: &mut mcts::Tree) -> Self::Result {
        (self.inner.result(tree), self.iteration)
    }

    fn step(&mut self, tree: &mut mcts::Tree) -> Self::Step {
        let step = (self.inner.step(tree), self.iteration);
        self.iteration += 1;
        step
    }
}

#[derive(Default, Debug)]
pub struct MctsLimiter {
    limit: Limit,
}

impl mcts::Limiter for MctsLimiter {
    fn should_stop(&self, _pos: &Position, depth: Depth) -> bool {
        depth > self.limit.depth || depth > Depth::MAX
    }
}

pub fn mcts<S: MctsStrategy + Default, E: mcts::Evaluator>(
    pos: Position,
    tree: &mut mcts::Tree,
    model: &mut E,
    limit: Limit,
    debug: DebugMode,
    ct: CancellationToken,
) -> S::Result {
    mcts_inner::<S, E>(pos, model, limit, debug, ct, S::default())
}

fn mcts_inner<S: MctsStrategy, E: mcts::Evaluator>(
    mut pos: Position,
    model: &mut E,
    limit: Limit,
    _debug: DebugMode,
    ct: CancellationToken,
    mut strategy: S,
) -> S::Result {
    let limiter = MctsLimiter { limit: limit.clone() };

    let time_per_move = limit.time_per_move(&pos);
    let time_limit = Instant::now() + time_per_move;

    while !ct.is_cancelled() && (!limit.is_active || Instant::now() < time_limit) {
        tree.grow(&mut pos, model, &limiter);
        strategy.step(tree);
    }

    strategy.result(tree)
}
