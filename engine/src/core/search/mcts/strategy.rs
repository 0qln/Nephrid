use crate::core::search::mcts::Tree;
use crate::core::Move;
use crate::uci::sync;

pub trait MctsStrategy {
    type Result;
    type Step;

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

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        self.find_best.result(tree)
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
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

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        (self.inner.result(tree), self.iteration)
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        let step = (self.inner.step(tree), self.iteration);
        self.iteration += 1;
        step
    }
}
