use thiserror::Error;

use crate::{
    core::{
        Move,
        config::Configuration,
        search::mcts::{
            MctsParts,
            eval::Cp,
            node::{
                Tree, WinRate,
                node_state::{Evaluated, NodeSwitch},
            },
            select::Selector,
        },
    },
    uci::sync,
};
use std::{
    error::Error,
    sync::{
        Arc, Mutex, atomic::{AtomicBool, Ordering}, mpsc::{Sender, channel}
    },
    thread,
};

use crate::{
    core::{
        position::Position,
        search::{
            limit::Limit,
            mcts::{mcts, strategy::MctsUci},
            perft::perft,
        },
    },
    misc::DebugMode,
    uci::sync::CancellationToken,
};

pub mod limit;
pub mod mcts;
pub mod mode;
pub mod perft;

pub struct Thread {
    pub tx: Sender<Command>,
}

pub struct Worker {
    mcts_parts: Option<mcts::config::mcts::Parts>,
    mcts_state: mcts::SearchState,
    backup_tree: Option<Tree>
}

#[derive(Error, Debug)]
pub enum ExecError {
    #[error("Uninitialized state")]
    UninitState(),
    #[error("Bad config: {0}")]
    BadConfig(Box<dyn Error>),
    #[error("Runtime error: {0}")]
    RuntimeError(Box<dyn Error>),
}

impl Default for Worker {
    fn default() -> Self {
        Self::new()
    }
}

impl Worker {
    pub fn new() -> Self {
        let mcts_state = mcts::SearchState::default();
        let mcts_parts = None;
        let backup_tree = None;
        Self { mcts_parts, mcts_state, backup_tree }
    }

    pub fn exec(&mut self, cmd: Command) -> Result<(), ExecError> {
        match cmd {
            Command::Perft(pos, limit, ct, debug) => {
                perft(pos, limit, ct, debug);
                Ok(())
            }
            Command::Normal(mut pos, limit, ct, debug) => {
                let parts = self.mcts_parts.as_ref().ok_or(ExecError::UninitState())?;
                let state = &mut self.mcts_state;

                let result = mcts(&mut pos, parts, state, limit, debug, ct, None, MctsUci::default());

                if result.is_none() {
                    todo!("Log error or something: got no result from mcts search.")
                };

                Ok(())
            }
            Command::Ponder(mut pos, limit, ct, debug, ponder) => {
                let parts = self.mcts_parts.as_ref().ok_or(ExecError::UninitState())?;
                let state = &mut self.mcts_state;

                let result = mcts(&mut pos, parts, state, limit, debug, ct, Some(ponder), MctsUci::default());

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

                #[allow(clippy::unnecessary_fallible_conversions)]
                let parts = <&mcts::config::mcts::Parts as MctsParts>::Instance::try_from(cfg)
                    .map_err(|e| ExecError::BadConfig(Box::new(e)))?;

                self.mcts_parts = Some(parts);
                Ok(())
            }
            Command::AdvanceState(mov) => {
                self.backup_tree = Some(self.mcts_state.tree.clone());
                self.mcts_state.tree.advance_to(|b| b.mov() == mov);
                Ok(())
            }
            Command::RollbackAndAdvance(mov) => {
                if let Some(backup) = self.backup_tree.take() {
                    self.mcts_state.tree = backup;
                    self.mcts_state.tree.advance_to(|b| b.mov() == mov);
                } else {
                    self.mcts_state.tree = Tree::default();
                }
                Ok(())
            }
            Command::ResetState => {
                self.mcts_state.tree = Tree::default();
                Ok(())
            }
            Command::MctsDebugTree => {
                let tree = &self.mcts_state.tree;
                let root = tree.get_root();
                println!(
                    "({}) ----  v {: >8.2}/{: <8} w {} cp {}",
                    root.state(),
                    root.clone().borrow().value(),
                    root.clone().borrow().visits(),
                    root.clone()
                        .try_into::<Evaluated>()
                        .map(|x| (-WinRate::from(x)).to_string())
                        .unwrap_or("/".to_string()),
                    root.clone()
                        .try_into::<Evaluated>()
                        .map(|x| Cp::from(-WinRate::from(x)).to_string())
                        .unwrap_or("/".to_string())
                );
                match root.into_ct() {
                    NodeSwitch::Leaf(_node) => {}
                    NodeSwitch::Branching(node) => {
                        for branch in node.borrow().branches() {
                            let node = branch.node();
                            let state = node.state();
                            println!("* ({}) {}", state, branch.mov());
                        }
                    }
                    NodeSwitch::Terminal(_node) => {}
                    NodeSwitch::Evaluated(node) => {
                        let root_visits = node.borrow().visits();
                        for branch in node.borrow().branches() {
                            let mov = branch.mov();
                            let node = branch.node();
                            let state = node.state();
                            let node = node.borrow();
                            let parts: &mcts::config::mcts::Parts =
                                self.mcts_parts.as_ref().unwrap();
                            let selector = MctsParts::<{ mcts::config::MPV }>::selector(&parts);
                            println!(
                                "{} {: >9} {: <5} v {: >8.2}/{: <8} p {:.3} ~ {}",
                                if tree.best_move() == Some(mov) { '*' } else { '-' },
                                state.to_string(),
                                mov.to_string(),
                                node.value(),
                                node.visits(),
                                branch.policy(),
                                selector.score(branch, root_visits)
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

pub fn init() -> Thread {
    let (tx, rx) = channel::<Command>();
    thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let mut worker = Worker::new();
            loop {
                let cmd = rx.recv().expect("Should be able to receive data");
                let result = worker.exec(cmd);
                if let Err(e) = result {
                    sync::out(&format!("Error executing command: {e}"));
                }
            }
        })
        .expect("Failed to spawn search thread.");

    _ = tx.send(Command::Configure(Arc::new(Mutex::new(
        Configuration::default(),
    ))));

    Thread { tx }
}

#[derive(Debug, Clone)]
pub enum Command {
    Perft(Position, Limit, CancellationToken, DebugMode),
    Normal(Position, Limit, CancellationToken, DebugMode),
    Ponder(Position, Limit, CancellationToken, DebugMode, PonderToken),
    AdvanceState(Move),
    RollbackAndAdvance(Move),
    Configure(Arc<Mutex<Configuration>>),
    ResetState,
    MctsDebugTree,
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