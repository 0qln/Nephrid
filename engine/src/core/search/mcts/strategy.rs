use std::time::{Duration, Instant};

use crate::{
    core::{
        Move,
        chrono::{ChronoParams, TimeMan},
        params::IParams,
        ply::Ply,
        position::Position,
        search::{
            PonderToken,
            limit::UciLimit,
            mcts::{
                Tree,
                node::{WinRate, node_state::Evaluated},
            },
            score::Cp,
            strat::*,
        },
    },
    misc::{CancellationToken, DebugMode},
};

pub trait MctsStrategy {
    type Result;
    type Step;

    fn start(&mut self, tree: &mut Tree, pos: &Position);
    fn result(&mut self, tree: &mut Tree) -> Self::Result;
    fn step(&mut self, tree: &mut Tree) -> Self::Step;
    fn should_stop(&mut self, tree: &Tree) -> bool;
}

#[derive(Default, Debug)]
pub struct MctsFindBest {
    last_best_move: Option<Move>,
}

impl MctsStrategy for MctsFindBest {
    type Result = Option<Move>;
    type Step = Option<Move>;

    fn result(&mut self, tree: &mut Tree) -> Self::Result { self.last_best_move.or_else(|| tree.maybe_best_move(tree.root())) }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        let curr_best_move = tree.maybe_best_move(tree.root());
        if self.last_best_move != curr_best_move
            && let Some(mov) = curr_best_move
        {
            self.last_best_move = Some(mov);
            return Some(mov);
        }
        None
    }

    fn start(&mut self, _tree: &mut Tree, _pos: &Position) {}

    fn should_stop(&mut self, _tree: &Tree) -> bool { false }
}

#[derive(Debug)]
pub struct MctsUci<X: IParams> {
    find_best: MctsFindBest,
    last_uci_out: Option<Instant>,

    // search control
    ct: CancellationToken,
    pt: Option<PonderToken>,
    debug: DebugMode,

    // runtime tracking
    time_man: TimeMan<X>,
    nodes_begin: u64,
    terminal_nodes_begin: u64,
    iterations: u64,
    is_not_pondering: bool,

    // configuration
    limit: UciLimit,
}

impl<X: IParams> MctsUci<X>
where
    X::Ref: ChronoParams,
{
    pub fn new(limit: UciLimit, debug: DebugMode, ct: CancellationToken, pt: Option<PonderToken>, params: X::Ref) -> Self {
        Self {
            limit,
            debug,
            ct,
            pt,
            time_man: TimeMan::new(params),
            find_best: Default::default(),
            last_uci_out: None,
            nodes_begin: 0,
            terminal_nodes_begin: 0,
            iterations: 0,
            is_not_pondering: false,
        }
    }

    pub fn search_time(&self) -> Option<UciSearchtime> { Some(UciSearchtime(self.time_man.elapsed_search_time()?)) }

    /// Number of nodes per second since start of the search, given the current
    /// number of nodes in the search tree.
    pub fn nps(&self, num_nodes: u64) -> Option<UciNps> { self.search_time().map(|t| UciNps::from_nodes_and_time(num_nodes, t.0)) }

    /// Determine score in centipawns / mate-in-x, etc.
    /// Returns `None` if the root node is not evaluated or unproven.
    pub fn determine_score(&self, tree: &Tree, pv_len: usize) -> Option<UciScore> {
        let root = tree.node(tree.root());
        let root_value = root.value();

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
        else if let Some(evaluated) = tree.try_node::<Evaluated>(tree.root()) {
            // (invert bc it's relative to parent node)
            // todo: i think it would be more accurate to take the winrate of the best move
            // that we can make...
            let win_rate = WinRate::from(evaluated).inv();
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
        let pv = tree.principal_line();
        let new_nodes = tree_size as u64 - self.nodes_begin;

        let currmove = UciArg::Some(UciCurrmove(mov));
        let score = UciArg::from(self.determine_score(tree, pv.len()));
        let nodes = UciArg::Some(UciNodes(tree_size));
        let nps = UciArg::from(self.nps(new_nodes));
        let depth = UciArg::Some(UciDepth(tree.compute_minheight().into()));
        let seldepth = UciArg::Some(UciSeldepth(tree.maxheight().into()));
        let pv = UciArg::Some(UciPv(&pv));
        let time = UciArg::from(self.search_time());
        let string = UciArg::<String>::None;

        println!("info{currmove}{score}{nodes}{nps}{depth}{seldepth}{time}{pv}{string}");
    }

    /// # uci_bestmove
    ///
    /// Send the [UCI bestmove command](https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf#file-uci-protocol-specification-txt-L207).
    fn uci_bestmove(&self, tree: &Tree, mov: Move) {
        let pv = tree.principal_line();
        let best_move = UciArg::Some(mov);
        let ponder_move = UciArg::from(pv.0.get(1).map(|b| UciPondermove(b.mov())));

        println!("bestmove{best_move}{ponder_move}");
    }

    fn output_frequency(&self) -> Duration {
        if self.debug.get() {
            Duration::from_millis(200)
        }
        else {
            Duration::from_millis(500)
        }
    }

    pub fn limit(&self) -> &UciLimit { &self.limit }
}

impl<X: IParams> MctsStrategy for MctsUci<X>
where
    X::Ref: ChronoParams,
{
    type Result = <MctsFindBest as MctsStrategy>::Result;
    type Step = <MctsFindBest as MctsStrategy>::Step;

    fn start(&mut self, tree: &mut Tree, pos: &Position) {
        self.nodes_begin = tree.size() as u64;
        self.terminal_nodes_begin = tree.terminal_nodes() as u64;
        self.iterations = 0;

        self.time_man.init_limits(&self.limit, pos);
        self.is_not_pondering = self.pt.is_none();
    }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        self.iterations += 1;
        let step = self.find_best.step(tree);
        let now = Instant::now();
        let last_out = self.last_uci_out;
        if let Some(mov) = self.find_best.last_best_move
            && last_out.is_none_or(|x| now - x > self.output_frequency())
        {
            self.uci_info(tree, mov);
            self.last_uci_out = Some(now);
        }
        step
    }

    fn should_stop(&mut self, tree: &Tree) -> bool {
        // 1. User typed "stop" (GUI interrupt)
        // We ALWAYS respect this, whether pondering or not.
        if self.ct.is_cancelled() {
            return true;
        }

        // 2. Ponder Hit transition
        if let Some(ponder_tok) = &self.pt
            && !self.is_not_pondering
            && !ponder_tok.should_ponder()
        {
            // transition to normal search and set time limits.
            #[allow(unreachable_code)] // todo: implement this
            self.time_man.init_limits(&self.limit, todo!());
            self.is_not_pondering = true;
        }

        // 3. IF WE ARE STILL PONDERING, NEVER STOP ON OUR OWN.
        // We ignore mates and limits until the GUI tells us otherwise.
        if !self.is_not_pondering {
            return false;
        }

        // --- Everything below this line ONLY applies during a normal search ---

        // 4. Proven win/loss at root
        let root = tree.node(tree.root());
        let root_value = root.value();
        if root_value.is_proven_win() || root_value.is_proven_loss() {
            return true;
        }

        // 5. Standard time/node limits
        if self.limit.is_active() && (self.limit.is_reached(tree.size() as u64 - self.nodes_begin, self.iterations) || self.time_man.reached_limit())
        {
            return true;
        }

        false
    }

    fn result(&mut self, tree: &mut Tree) -> Self::Result {
        let result = self.find_best.result(tree);
        if let Some(mov) = result {
            self.uci_info(tree, mov);
            self.uci_bestmove(tree, mov);
        }
        result
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

    fn result(&mut self, tree: &mut Tree) -> Self::Result { (self.inner.result(tree), self.iteration) }

    fn step(&mut self, tree: &mut Tree) -> Self::Step {
        let step = (self.inner.step(tree), self.iteration);
        self.iteration += 1;
        step
    }

    fn start(&mut self, tree: &mut Tree, pos: &Position) { self.inner.start(tree, pos); }

    fn should_stop(&mut self, tree: &Tree) -> bool { self.inner.should_stop(tree) }
}
