use thiserror::Error;

use crate::{
    core::{
        Move,
        config::Configuration,
        search::{
            limit::UciLimit,
            mcts::{
                MctsConfig, MctsParts,
                eval::Cp,
                node::{
                    Tree, WinRate,
                    node_state::{Evaluated, Switch},
                },
                select::Selector,
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
        search::{
            mcts::{mcts, strategy::MctsUci},
            perft::perft,
        },
    },
    misc::DebugMode,
};

pub mod limit;
pub mod mcts;
pub mod mode;
pub mod perft;

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

pub trait SearchWorker: Default {
    fn exec(&mut self, cmd: Command) -> Result<(), ExecError>;
}

/// Iterative deepening worker.
pub struct IdWorker; // todo

/// Monte Carlo Tree Search worker.
pub struct MctsWorker<const MPV: usize, C: MctsConfig> {
    mcts_parts: Option<C::Parts>,
    mcts_state: mcts::SearchState,
    backup_tree: Option<Tree>,
}

impl<const MPV: usize, C: MctsConfig> Default for MctsWorker<MPV, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MPV: usize, C: MctsConfig> MctsWorker<MPV, C> {
    pub fn new() -> Self {
        let mcts_state = mcts::SearchState::default();
        let mcts_parts = None;
        let backup_tree = None;
        Self {
            mcts_parts,
            mcts_state,
            backup_tree,
        }
    }
}

impl<const MPV: usize, C: MctsConfig<Strat = MctsUci>> SearchWorker for MctsWorker<MPV, C> {
    fn exec(&mut self, cmd: Command) -> Result<(), ExecError> {
        match cmd {
            Command::Perft(mut pos, limit, ct, debug) => {
                perft::<true>(&mut pos, &limit, ct, debug);
                Ok(())
            }
            Command::Normal(mut pos, limit, ct, debug) => {
                let parts = self.mcts_parts.as_ref().ok_or(ExecError::UninitState())?;
                let state = &mut self.mcts_state;
                let strat = &mut C::Strat::new(limit, debug, ct, None);

                let result = mcts::<MPV, C, _>(&mut pos, parts, state, strat);

                if result.is_none() {
                    todo!("Log error or something: got no result from mcts search.")
                };

                Ok(())
            }
            Command::Ponder(mut pos, limit, ct, debug, pt) => {
                let parts = self.mcts_parts.as_ref().ok_or(ExecError::UninitState())?;
                let state = &mut self.mcts_state;
                let strat = &mut C::Strat::new(limit, debug, ct, Some(pt));

                let result = mcts::<MPV, C, _>(&mut pos, parts, state, strat);

                if result.is_none() {
                    todo!("Log error or something: got no result from mcts search.")
                };

                Ok(())
            }
            Command::Configure(config) => {
                let config_lock = config.lock();
                let cfg = match config_lock {
                    Ok(ref cfg) => cfg,
                    Err(_) => &Configuration::default(),
                };

                // #[allow(clippy::unnecessary_fallible_conversions)]
                let parts = <C::Parts as TryFrom<&Configuration>>::try_from(cfg)
                    .map_err(|e| ExecError::BadConfig(e.to_string()))?;

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
                    root_evaluated
                        .map(|x| (-WinRate::from(x)).to_string())
                        .unwrap_or("/".to_string()),
                    root_evaluated
                        .map(|x| Cp::from(-WinRate::from(x)).to_string())
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
                            println!(
                                "{} {: >9} {: <5} v {: >8.2}/{: <8} p {:.3} ~ {}",
                                if root_best_move == Some(mov) { '*' } else { '-' },
                                state.to_string(),
                                mov.to_string(),
                                node.value(),
                                node.visits(),
                                branch.policy(),
                                selector.score(tree, branch_id, root_id)
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

pub fn init<W: SearchWorker>() -> SearchThread {
    let (tx, rx) = channel::<Command>();
    thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let mut worker = W::default();
            loop {
                let cmd = rx.recv().expect("Should be able to receive data");
                let result = worker.exec(cmd);
                if let Err(e) = result {
                    println!("Error executing command: {e}");
                }
            }
        })
        .expect("Failed to spawn search thread.");

    _ = tx.send(Command::Configure(Arc::new(Mutex::new(
        Configuration::default(),
    ))));

    SearchThread { tx }
}

#[derive(Debug, Clone)]
pub enum Command {
    Perft(Position, UciLimit, CancellationToken, DebugMode),
    Normal(Position, UciLimit, CancellationToken, DebugMode),
    Ponder(
        Position,
        UciLimit,
        CancellationToken,
        DebugMode,
        PonderToken,
    ),
    AdvanceState(Move),
    RollbackAndAdvance(Move),
    Configure(Arc<Mutex<Configuration>>),
    ResetState,
    Debug,
}

#[derive(Debug, Clone)]
pub struct PonderToken(Arc<AtomicBool>);

impl Default for PonderToken {
    fn default() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }
}

impl PonderToken {
    pub fn should_ponder(&self) -> bool {
        !self.0.load(Ordering::Relaxed)
    }

    pub fn stop_ponder(&self) {
        self.0.store(true, Ordering::Relaxed)
    }

    pub fn start_ponder(&self) {
        self.0.store(false, Ordering::Relaxed)
    }
}
