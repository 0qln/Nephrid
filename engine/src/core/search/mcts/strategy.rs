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
            write!(f, "{} ", arg)
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
        Some(Instant::now() - self.search_start?).map(UciSearchtime)
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
    
    /// # uci_bestmove
    ///
    /// Send the UCI info command.
    ///
    ///
    /// ## [UCI-Spec](https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf#file-uci-protocol-specification-txt-L248)
    ///
    ///	the engine wants to send information to the GUI. This should be done whenever one of the info has changed.
    ///	The engine can send only selected infos or multiple infos with one info command,
    ///	e.g. "info currmove e2e4 currmovenumber 1" or
    ///	     "info depth 12 nodes 123456 nps 100000".
    ///	Also all infos belonging to the pv should be sent together
    ///	e.g. "info depth 2 score cp 214 time 1242 nodes 2124 nps 34928 pv e2e4 e7e5 g1f3"
    ///	I suggest to start sending "currmove", "currmovenumber", "currline" and "refutation" only after one second
    ///	to avoid too much traffic.
    ///	Additional info:
    ///	* depth <x>
    ///		search depth in plies
    ///	* seldepth <x>
    ///		selective search depth in plies,
    ///		if the engine sends seldepth there must also be a "depth" present in the same string.
    ///	* time <x>
    ///		the time searched in ms, this should be sent together with the pv.
    ///	* nodes <x>
    ///		x nodes searched, the engine should send this info regularly
    ///	* pv <move1> ... <movei>
    ///		the best line found
    ///	* multipv <num>
    ///		this for the multi pv mode.
    ///		for the best move/pv add "multipv 1" in the string when you send the pv.
    ///		in k-best mode always send all k variants in k strings together.
    ///	* score
    ///		* cp <x>
    ///			the score from the engine's point of view in centipawns.
    ///		* mate <y>
    ///			mate in y moves, not plies.
    ///			If the engine is getting mated use negative values for y.
    ///		* lowerbound
    ///	      the score is just a lower bound.
    ///		* upperbound
    ///		   the score is just an upper bound.
    ///	* currmove <move>
    ///		currently searching this move
    ///	* currmovenumber <x>
    ///		currently searching move number x, for the first move x should be 1 not 0.
    ///	* hashfull <x>
    ///		the hash is x permill full, the engine should send this info regularly
    ///	* nps <x>
    ///		x nodes per second searched, the engine should send this info regularly
    ///	* tbhits <x>
    ///		x positions where found in the endgame table bases
    ///	* sbhits <x>
    ///		x positions where found in the shredder endgame databases
    ///	* cpuload <x>
    ///		the cpu usage of the engine is x permill.
    ///	* string <str>
    ///		any string str which will be displayed be the engine,
    ///		if there is a string command the rest of the line will be interpreted as <str>.
    ///	* refutation <move1> <move2> ... <movei>
    ///	   move <move1> is refuted by the line <move2> ... <movei>, i can be any number >= 1.
    ///	   Example: after move d1h5 is searched, the engine can send
    ///	   "info refutation d1h5 g6h5"
    ///	   if g6h5 is the best answer after d1h5 or if g6h5 refutes the move d1h5.
    ///	   if there is no refutation for d1h5 found, the engine should just send
    ///	   "info refutation d1h5"
    ///		The engine should only send this if the option "UCI_ShowRefutations" is set to true.
    ///	* currline <cpunr> <move1> ... <movei>
    ///	   this is the current line the engine is calculating. <cpunr> is the number of the cpu if
    ///	   the engine is running on more than one cpu. <cpunr> = 1,2,3....
    ///	   if the engine is just using one cpu, <cpunr> can be omitted.
    ///	   If <cpunr> is greater than 1, always send all k lines in k strings together.
    ///		The engine should only send this if the option "UCI_ShowCurrLine" is set to true.
    fn uci_info(&self, tree: &Tree, mov: Move) {
        let tree_size = tree.size();
        let pv = tree.principal_variation();

        let currmove = UciArg::Some(UciCurrmove(mov));
        let score = UciArg::from(self.determine_score(tree, pv.len()));
        let nodes = UciArg::Some(UciNodes(tree_size));
        let nps = UciArg::from(self.nps(tree_size));
        let depth = UciArg::<Depth>::None; // Some(format!("depth {}", tree.mindepth()));
        let seldepth = UciArg::Some(UciSeldepth(tree.maxheight().into()));
        let pv = UciArg::Some(UciPv(pv));
        let time = UciArg::from(self.search_time());
        let string = UciArg::<String>::None;

        sync::out(&format!(
            "info {currmove}{score}{nodes}{nps}{depth}{seldepth}{time}{pv}{string}"
        ));
    }
    
    /// # uci_bestmove
    ///
    /// Send the UCI bestmove command.
    ///
    ///
    /// ## [UCI-Spec](https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf#file-uci-protocol-specification-txt-L207)
    ///
    /// * bestmove <move1> [ ponder <move2> ]
    ///	the engine has stopped searching and found the move <move> best in this position.
    ///	the engine can send the move it likes to ponder on. The engine must not start pondering automatically.
    ///	this command must always be sent if the engine stops searching, also in pondering mode if there is a
    ///	"stop" command, so for every "go" command a "bestmove" command is needed!
    ///	Directly before that the engine should send a final "info" command with the final search information,
    ///	the the GUI has the complete statistics about the last search.
    fn uci_bestmove(&self, tree: &Tree, mov: Move) {
        let pv = tree.principal_variation();
        let ponder_move = UciArg::from(pv.0.get(1).map(|b| UciPondermove(b.mov())));

        sync::out(&format!("bestmove {mov} {ponder_move}"));
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
            && last_out.is_none_or(|x| now - x > Duration::from_millis(500)) {
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
