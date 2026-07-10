use crate::core::{
    chrono::ChronoParams,
    eval::StaticEvaluator,
    params::IParams,
    position::PieceInfoObserver,
    search::{id::IdParams, mcts::search::MctsParams, quiesce::QSearchParams, score::Cp, tt::TranspositionTable},
};
use thiserror::Error;

use crate::{
    core::{
        Game, Move,
        config::Configuration,
        move_iter::opt,
        search::{
            limit::UciLimit,
            mcts::{
                MctsConfig, MctsParts,
                node::{
                    Tree, WinRate,
                    node_state::{Evaluated, Switch},
                },
                select::Selector,
                strategy::MctsUci,
            },
        },
    },
    misc::CancellationToken,
};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Sender, channel},
    },
    thread,
};

use crate::{
    core::{
        position::Position,
        search::{mcts::mcts, perft::perft},
    },
    misc::DebugMode,
};

pub mod id;
pub mod limit;
pub mod mcts;
pub mod mode;
pub mod ordering;
pub mod perft;
pub mod quiesce;
pub mod score;
pub mod strat;
pub mod tt;

pub struct SearchThread {
    pub tx: Sender<Command>,
}

#[derive(Error, Debug)]
pub enum ExecError {
    #[error("Uninitialized state")]
    UninitState(),
    #[error("Bad config: {0}")]
    BadConfig(String),
    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

pub trait SearchWorker {
    type X: IParams;

    fn new() -> Self;
    fn exec(&mut self, cmd: Command) -> Result<(), ExecError>;
}

/// Iterative deepening worker.
pub struct IdWorker<E: StaticEvaluator, X: IParams> {
    // todo: don't store the construction information here but the tt and timeman itself
    tt: TranspositionTable<id::TTEntry>,
    params: X::Ref,
    eval: E,
}

impl<E: StaticEvaluator + Default, X: IParams + Default> SearchWorker for IdWorker<E, X>
where
    X::Ref: IdParams + ChronoParams + QSearchParams,
{
    type X = X;

    fn new() -> Self {
        Self {
            tt: TranspositionTable::new(0),
            params: <Self::X as Default>::default().shared(),
            eval: E::default(),
        }
    }

    fn exec(&mut self, cmd: Command) -> Result<(), ExecError> {
        match cmd {
            Command::Perft(mut pos, limit, ct, debug, captures_only) => {
                if captures_only {
                    perft::<opt::Captures>(&mut pos, &limit, ct, debug);
                }
                else {
                    perft::<opt::AllLegal>(&mut pos, &limit, ct, debug);
                }
                Ok(())
            }
            Command::Normal(mut pos, limit, ct, debug) => {
                // todo: initiating the nnue before every search works for now, but we can
                // probably just do it on the fly in AdvanceState...
                self.eval.observe_forward().on_init(pos.piece_info());

                let best_move = id::go(&mut pos, limit, &debug, ct, &mut self.tt, &mut self.eval, self.params.clone());

                if let Some(mov) = best_move {
                    println!("bestmove {mov}");
                }

                Ok(())
            }
            Command::Ponder(_pos, _limit, _ct, _dbg, _ponder) => {
                todo!()
            }
            Command::AdvanceState(_) => {
                // no need to update the tt, the depth will be the same
                Ok(())
            }
            Command::RollbackAndAdvance(_) => {
                // no need to update the tt, the depth will be the same
                Ok(())
            }
            Command::Configure(config) => {
                let cfg = || config.lock().map_err(|e| ExecError::BadConfig(format!("Config cannot be locked: {e}")));

                self.tt = TranspositionTable::new_of_size(cfg()?.hash());
                self.params = Self::X::try_from_config(cfg()?).map_err(|e| ExecError::BadConfig(format!("Bad configuration: {e}")))?;
                self.eval = E::try_from_config(cfg()?).map_err(|e| ExecError::BadConfig(format!("Bad configuration: {e}")))?;

                Ok(())
            }
            Command::ResetState => Ok(()),
            Command::Debug => todo!(),
            Command::IsReady => {
                println!("readyok");
                Ok(())
            }
            Command::PrintPv(_) => todo!(),
        }
    }
}

/// Monte Carlo Tree Search worker.
pub struct MctsWorker<const MPV: usize, C: MctsConfig, X: IParams> {
    mcts_parts: Option<C::Parts>,
    mcts_state: mcts::SearchState,
    backup_tree: Option<Tree>,
    params: X::Ref,
}

impl<const MPV: usize, C, X: IParams + Default> SearchWorker for MctsWorker<MPV, C, X>
where
    C: MctsConfig<Strat = MctsUci>,
    X::Ref: MctsParams,
{
    type X = X;

    fn new() -> Self {
        let mcts_state = mcts::SearchState::default();
        let mcts_parts = None;
        let backup_tree = None;
        Self {
            mcts_parts,
            mcts_state,
            backup_tree,
            params: <Self::X as Default>::default().shared(),
        }
    }

    fn exec(&mut self, cmd: Command) -> Result<(), ExecError> {
        match cmd {
            Command::PrintPv(pos) => {
                let pv = self.mcts_state.tree.principal_line();
                let continuation = pv.0.into_iter().map(|b| b.mov());
                let game = Game::from_moves(pos, continuation, &mut ());
                let pgn = game.to_pgn();
                println!("Principal Variation:\n{pgn}");
                Ok(())
            }
            Command::IsReady => {
                println!("readyok");
                Ok(())
            }
            Command::Perft(mut pos, limit, ct, debug, captures_only) => {
                if captures_only {
                    perft::<opt::Captures>(&mut pos, &limit, ct, debug);
                }
                else {
                    perft::<opt::AllLegal>(&mut pos, &limit, ct, debug);
                }
                Ok(())
            }
            Command::Normal(mut pos, limit, ct, debug) => {
                let parts = self.mcts_parts.as_ref().ok_or(ExecError::UninitState())?;
                let state = &mut self.mcts_state;
                let strat = &mut C::Strat::new(limit, debug, ct, None);
                let params = self.params.clone();

                let result = mcts::<MPV, C, _, X>(&mut pos, parts, state, strat, params);

                if result.is_none() {
                    todo!("Log error or something: got no result from mcts search.")
                };

                Ok(())
            }
            Command::Ponder(mut pos, limit, ct, debug, pt) => {
                let parts = self.mcts_parts.as_ref().ok_or(ExecError::UninitState())?;
                let state = &mut self.mcts_state;
                let strat = &mut C::Strat::new(limit, debug, ct, Some(pt));

                let result = mcts::<MPV, C, _, X>(&mut pos, parts, state, strat, self.params.clone());

                if result.is_none() {
                    todo!("Log error or something: got no result from mcts search.")
                };

                Ok(())
            }
            Command::Configure(config) => {
                let cfg = &config.lock().map_err(|e| ExecError::BadConfig(format!("Config cannot be locked: {e}")))?;

                let mut parts = <C::Parts as TryFrom<&Configuration>>::try_from(cfg).map_err(|e| ExecError::BadConfig(e.to_string()))?;

                parts.warmup(MPV).map_err(ExecError::BadConfig)?;

                self.mcts_parts = Some(parts);

                Ok(())
            }
            Command::AdvanceState(mov) => {
                self.backup_tree = Some(self.mcts_state.tree.clone());
                self.mcts_state.advance_to(mov);
                Ok(())
            }
            Command::RollbackAndAdvance(mov) => {
                if let Some(backup) = self.backup_tree.take() {
                    self.mcts_state.tree = backup;
                    self.mcts_state.advance_to(mov);
                }
                else {
                    self.mcts_state.tree = Tree::default();
                }
                Ok(())
            }
            Command::ResetState => {
                self.mcts_state.tree = Tree::default();
                Ok(())
            }
            Command::Debug => {
                let tree = &self.mcts_state.tree;
                let root = tree.node(tree.root());
                let root_evaluated = tree.try_node::<Evaluated>(tree.root());
                println!(
                    "({}) ----  v {: >8.2}/{: <8} w {} cp {}",
                    root.state(),
                    root.value(),
                    root.visits(),
                    root_evaluated.map(|x| (WinRate::from(x).inv()).to_string()).unwrap_or("/".to_string()),
                    root_evaluated
                        .map(|x| Cp::from(WinRate::from(x).inv()).to_string())
                        .unwrap_or("/".to_string())
                );
                match tree.node_switch(tree.root()) {
                    Switch::Leaf(_node) => {}
                    Switch::Branching(node) => {
                        for branch in tree.branches(node) {
                            let node = tree.node(branch.node());
                            let state = node.state();
                            println!("* ({}) {}", state, branch.mov());
                        }
                    }
                    Switch::Terminal(_node) => {}
                    Switch::Evaluated(root_id) => {
                        let root_best_move = tree.maybe_best_move(tree.root());
                        for branch_id in tree.branch_ids(root_id) {
                            let branch = tree.branch(branch_id);
                            let node = tree.node(branch.node());
                            let mov = branch.mov();
                            let state = node.state();
                            let parts = self.mcts_parts.as_ref().unwrap();
                            let selector = <C as MctsConfig>::Parts::selector(parts);
                            let exploration = selector.exploration(tree, branch_id, root_id);
                            let exploitation = selector.exploitation(tree, branch_id, root_id);
                            let score = exploration + exploitation;
                            println!(
                                "{} {: >9} {: <5} v {: >8.2}/{: <8} p {:.3} ~ {}",
                                if root_best_move == Some(mov) { '*' } else { '-' },
                                state.to_string(),
                                mov.to_string(),
                                node.value(),
                                node.visits(),
                                branch.policy(),
                                score
                            );
                        }
                    }
                }
                println!("---");
                Ok(())
            }
        }
    }
}

pub fn init<W: SearchWorker>(default_config: Arc<Mutex<Configuration>>) -> SearchThread {
    let (tx, rx) = channel::<Command>();
    thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let mut worker = W::new();
            loop {
                let cmd = rx.recv().expect("Should be able to receive data");
                let result = worker.exec(cmd);
                if let Err(e) = result {
                    println!("Error executing command: {e}");
                }
            }
        })
        .expect("Failed to spawn search thread.");

    _ = tx.send(Command::Configure(default_config));

    SearchThread { tx }
}

#[derive(Debug, Clone)]
pub enum Command {
    Perft(Position, UciLimit, CancellationToken, DebugMode, bool),
    Normal(Position, UciLimit, CancellationToken, DebugMode),
    Ponder(Position, UciLimit, CancellationToken, DebugMode, PonderToken),
    AdvanceState(Move),
    RollbackAndAdvance(Move),
    Configure(Arc<Mutex<Configuration>>),
    ResetState,
    Debug,
    IsReady,
    PrintPv(Position),
}

#[derive(Debug, Clone)]
pub struct PonderToken(Arc<AtomicBool>);

impl Default for PonderToken {
    fn default() -> Self { Self(Arc::new(AtomicBool::new(false))) }
}

impl PonderToken {
    pub fn should_ponder(&self) -> bool { !self.0.load(Ordering::Relaxed) }

    pub fn stop_ponder(&self) { self.0.store(true, Ordering::Relaxed) }

    pub fn start_ponder(&self) { self.0.store(false, Ordering::Relaxed) }
}
