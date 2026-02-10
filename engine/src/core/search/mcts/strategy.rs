use std::time::{Duration, Instant};

use crate::{
    core::{Move, search::mcts::Tree},
    uci::sync,
};

pub trait MctsStrategy {
    type Result;
    type Step;

    fn start(&mut self, tree: &mut Tree);
    fn result(&mut self, tree: &mut Tree) -> Self::Result;
    fn step(&mut self, tree: &mut Tree) -> Self::Step;
}

#[derive(Default, Debug)]
pub struct MctsFindBest {
    last_best_move: Option<Move>,
}

impl MctsStrategy for MctsFindBest {
    type Result = Option<Move>;
    type Step = Option<Move>;

    fn result(&mut self, _tree: &mut Tree) -> Self::Result {
        self.last_best_move
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        let curr_best_move = tree.best_move();
        if self.last_best_move != curr_best_move
            && let Some(mov) = curr_best_move
        {
            self.last_best_move = Some(mov);
            return Some(mov);
        }
        None
    }

    fn start(&mut self, _tree: &mut Tree) {
        ()
    }
}

#[derive(Default, Debug)]
pub struct MctsUci {
    find_best: MctsFindBest,
    search_start: Option<Instant>,
}

impl MctsUci {
    pub fn search_time(&self) -> Option<Duration> {
        Some(Instant::now() - self.search_start?)
    }
}

impl MctsStrategy for MctsUci {
    type Result = <MctsFindBest as MctsStrategy>::Result;
    type Step = <MctsFindBest as MctsStrategy>::Step;

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        let result = self.find_best.result(tree);
        if let Some(best_move) = result {
            sync::out(&format!("bestmove {best_move}"));
        }
        result
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        let step = self.find_best.step(tree);
        if let Some(mov) = step {
            sync::out(&format!("currmove {mov}"));
            sync::out(&format!(
                "info nps {} pv {}",
                self.search_time()
                    .map(|t| tree.size() as u128 * 1_000_000_000 / t.as_nanos())
                    .unwrap_or_default(),
                tree.principal_variation()
            ));
        }
        step
    }

    fn start(&mut self, _tree: &mut Tree) {
        self.search_start = Some(Instant::now());
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

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        (self.inner.result(tree), self.iteration)
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        let step = (self.inner.step(tree), self.iteration);
        self.iteration += 1;
        step
    }

    fn start(&mut self, _tree: &mut Tree) {
        ()
    }
}
