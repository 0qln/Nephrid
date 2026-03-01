use bumpalo::Bump;
use thiserror::Error;

use crate::{
    core::{
        Move,
        config::Configuration,
        search::mcts::{MctsParts, node::Tree},
    },
    uci::sync,
};
use std::{
    error::Error,
    sync::{
        Arc, Mutex,
        mpsc::{Sender, channel},
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

pub struct Worker<'a> {
    mcts_parts: Option<mcts::config::mcts::Parts>,
    mcts_state: mcts::SearchState<'a>,
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

impl<'a> Worker<'a> {
    pub fn new_in(bump: &'a Bump) -> Self {
        let mcts_state = mcts::SearchState::new_in(&bump);
        let mcts_parts = None;
        Self { mcts_parts, mcts_state }
    }

    pub fn exec(&mut self, bump: &'a mut Bump, cmd: Command) -> Result<(), ExecError> {
        match cmd {
            Command::Perft(pos, limit, ct, debug) => {
                perft(pos, limit, ct, debug);
                Ok(())
            }
            Command::Normal(pos, limit, ct, debug) => {
                let parts = self.mcts_parts.as_ref().ok_or(ExecError::UninitState())?;
                let state = &mut self.mcts_state;

                let result = mcts(
                    &pos,
                    parts,
                    state,
                    bump,
                    limit,
                    debug,
                    ct,
                    MctsUci::default(),
                );

                match result {
                    None => todo!("Log error or something"),
                    Some(_) => {}
                };

                Ok(())
            }
            Command::Ponder => {
                unimplemented!("todo");
            }
            Command::Configure(config) => {
                let config_lock = config.lock();
                let cfg = match config_lock {
                    Ok(ref cfg) => cfg,
                    Err(_) => &Configuration::default(),
                };
                let parts = <&mcts::config::mcts::Parts as MctsParts>::Instance::try_from(cfg)
                    .map_err(|e| ExecError::BadConfig(Box::new(e)))?;
                self.mcts_parts = Some(parts);
                Ok(())
            }
            Command::AdvanceState(mov) => {
                *bump = Bump::new();
                let new_tree = self
                    .mcts_state
                    .tree
                    .into_advance_to(bump, |b| b.mov() == mov)
                    .expect("oops");

                self.mcts_state.tree = new_tree;

                Ok(())
            }
            Command::ResetState => {
                *bump = Bump::new();
                self.mcts_state.tree = Tree::new_in(bump);
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
    AdvanceState(Move),
    Configure(Arc<Mutex<Configuration>>),
    Ponder,
    ResetState,
}
