use core::fmt;
use std::time::{Duration, Instant};

use crate::{
    core::{
        Move,
        depth::Depth,
        ply::Ply,
        search::mcts::{
            Tree,
            eval::Cp,
            node::{self, WinRate, node_state::Evaluated},
        },
    },
    uci::sync,
};

pub trait MctsStrategy {
    type Result;
    type Step;

    fn start(&mut self, tree: &mut Tree);
    fn result(&mut self, tree: &mut Tree) -> Self::Result;
    fn step(&mut self, tree: &mut Tree) -> Self::Step;
    fn should_stop(&mut self, _tree: &Tree) -> bool {
        false
    }
}

#[derive(Default, Debug)]
pub struct MctsFindBest {
    last_best_move: Option<Move>,
}

impl MctsStrategy for MctsFindBest {
    type Result = Option<Move>;
    type Step = Option<Move>;

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        self.last_best_move.or_else(|| tree.best_move())
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

    fn start(&mut self, _tree: &mut Tree) {}
}

#[derive(Debug)]
pub struct UciCp(Cp);

impl fmt::Display for UciCp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cp {}", self.0)
    }
}

#[derive(Debug)]
pub enum UciScore {
    Mate(i32),
    Centipawns(UciCp),
    LowerBound(UciCp),
    UpperBound(UciCp),
}

impl fmt::Display for UciScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mate(mate) => write!(f, "score mate {mate}"),
            Self::Centipawns(cp) => write!(f, "score {cp}"),
            Self::LowerBound(cp) => write!(f, "score {cp} lowerbound"),
            Self::UpperBound(cp) => write!(f, "score {cp} upperbound"),
        }
    }
}

#[derive(Default, Debug)]
pub struct UciNodes(usize);

impl fmt::Display for UciNodes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "nodes {}", self.0)
    }
}

#[derive(Default, Debug)]
pub struct UciNps(u128);

impl fmt::Display for UciNps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "nps {}", self.0)
    }
}

#[derive(Default, Debug)]
pub struct UciPondermove(Move);

impl fmt::Display for UciPondermove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ponder {}", self.0)
    }
}

#[derive(Default, Debug)]
pub struct UciSeldepth(Depth);

impl fmt::Display for UciSeldepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seldepth {}", self.0)
    }
}

pub struct UciPv(node::Path);

impl fmt::Display for UciPv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pv {}", self.0)
    }
}

#[derive(Default, Debug)]
pub struct UciSearchtime(Duration);

impl fmt::Display for UciSearchtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "time {}", self.0.as_millis())
    }
}

#[derive(Default, Debug)]
pub struct UciCurrmove(Move);

impl fmt::Display for UciCurrmove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "currmove {}", self.0)
    }
}

pub enum UciArg<T: fmt::Display> {
    None,
    Some(T),
}

impl<T: fmt::Display> fmt::Display for UciArg<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Self::Some(arg) = &self {
            write!(f, " {}", arg)
        }
        else {
            Ok(())
        }
    }
}

impl<T: fmt::Display> From<Option<T>> for UciArg<T> {
    fn from(opt: Option<T>) -> Self {
        if let Some(arg) = opt {
            Self::Some(arg)
        }
        else {
            Self::None
        }
    }
}

#[derive(Default, Debug)]
pub struct MctsUci {
    find_best: MctsFindBest,
    search_start: Option<Instant>,
    last_uci_out: Option<Instant>,
}

impl MctsUci {
    pub fn search_time(&self) -> Option<UciSearchtime> {
        Some(UciSearchtime(Instant::now() - self.search_start?))
    }

    /// Number of nodes per second since start of the search, given the current
    /// number of nodes in the search tree.
    pub fn nps(&self, num_nodes: usize) -> Option<UciNps> {
        self.search_time()
            .map(|t| num_nodes as u128 * 1_000_000_000 / t.0.as_nanos())
            .map(UciNps)
    }

    fn determine_score(&self, tree: &Tree, pv_len: usize) -> Option<UciScore> {
        let root = tree.get_root();
        let root_value = root.borrow().value();

        let mate_in_plies = Ply { v: pv_len as u16 };

        if root_value.is_proven_loss() {
            // we are root node, proven loss for parent node means we are winning
            // => don't alter sign.
            Some(UciScore::Mate(mate_in_plies.to_mate_score()))
        }
        else if root_value.is_proven_win() {
            // we are root node, proven win for parent node means we are losing
            // => set sign negative.
            Some(UciScore::Mate(-mate_in_plies.to_mate_score()))
        }
        else if let Some(evaluated) = root.clone().try_into::<Evaluated>() {
            // (invert bc it's relative to parent node)
            let win_rate = -WinRate::from(evaluated);
            let cp = UciCp(Cp::from(win_rate));
            Some(UciScore::Centipawns(cp))
        }
        else {
            None
        }
    }

    /// # uci_info
    ///
    /// Send the [UCI info command](https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf#file-uci-protocol-specification-txt-L248).
    fn uci_info(&self, tree: &Tree, mov: Move) {
        let tree_size = tree.size();
        let pv = tree.principal_variation();

        let currmove = UciArg::Some(UciCurrmove(mov));
        let score = UciArg::from(self.determine_score(tree, pv.len()));
        let nodes = UciArg::Some(UciNodes(tree_size));
        let nps = UciArg::from(self.nps(tree_size));
        let depth = UciArg::<Depth>::None; // Some(format!("depth {}", tree.mindepth()));
        let seldepth = UciArg::<UciSeldepth>::None; //UciArg::Some(UciSeldepth(tree.maxheight().into()));
        let pv = UciArg::Some(UciPv(pv));
        let time = UciArg::from(self.search_time());
        let string = UciArg::<String>::None;

        sync::out(&format!(
            "info{currmove}{score}{nodes}{nps}{depth}{seldepth}{time}{pv}{string}"
        ));
    }

    /// # uci_bestmove
    ///
    /// Send the [UCI bestmove command](https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf#file-uci-protocol-specification-txt-L207).
    fn uci_bestmove(&self, tree: &Tree, mov: Move) {
        let pv = tree.principal_variation();
        let best_move = UciArg::Some(mov);
        let ponder_move = UciArg::from(pv.0.get(1).map(|b| UciPondermove(b.mov())));

        sync::out(&format!("bestmove{best_move}{ponder_move}"));
    }
}

impl MctsStrategy for MctsUci {
    type Result = <MctsFindBest as MctsStrategy>::Result;
    type Step = <MctsFindBest as MctsStrategy>::Step;

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        let result = self.find_best.result(tree);
        if let Some(mov) = result {
            self.uci_info(tree, mov);
            self.uci_bestmove(tree, mov);
        }
        result
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        let step = self.find_best.step(tree);
        let now = Instant::now();
        let last_out = self.last_uci_out;
        if let Some(mov) = self.find_best.last_best_move
            && last_out.is_none_or(|x| now - x > Duration::from_millis(500))
        {
            self.uci_info(tree, mov);
            self.last_uci_out = Some(now);
        }
        step
    }

    fn should_stop(&mut self, tree: &Tree) -> bool {
        // stop if we have a proven win or loss at the root
        let root = tree.get_root();
        let root_value = root.borrow().value();
        root_value.is_proven_win() || root_value.is_proven_loss()
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

    fn start(&mut self, _tree: &mut Tree) {}
}
