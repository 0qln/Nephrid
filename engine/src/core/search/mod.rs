use crate::core::{
    Move,
    config::Configuration,
    search::mcts::{MctsParts, node::Tree},
};
use std::{
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

pub fn init() -> Thread {
    let (tx, rx) = channel::<Command>();
    thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let mut mcts_state = mcts::SearchState::default();
            loop {
                let cmd = rx.recv().expect("Should be able to receive data");
                match cmd {
                    Command::Perft(pos, limit, ct, debug, _config) => {
                        perft(pos, limit, ct, debug);
                    }
                    Command::Normal(pos, limit, ct, debug, config) => {
                        let config_lock = config.lock();
                        let cfg = match config_lock {
                            Ok(ref cfg) => cfg,
                            Err(_) => &Configuration::default(),
                        };

                        let mut mcts_parts = <mcts::config::mcts::Parts as MctsParts>::new(cfg);

                        let strat = MctsUci::default();
                        let result = mcts(
                            &pos,
                            &mut mcts_parts,
                            &mut mcts_state,
                            limit,
                            debug,
                            ct,
                            strat,
                        );
                        result.expect("");
                    }
                    Command::Ponder => {
                        unimplemented!("todo");
                    }
                    Command::AdvanceState(mov) => {
                        mcts_state.tree.advance_to(|b| b.mov() == mov);
                    }
                    Command::ResetState => {
                        mcts_state.tree = Tree::default();
                    }
                }
            }
        })
        .expect("Failed to spawn search thread.");
    Thread { tx }
}

#[derive(Debug, Clone)]
pub enum Command {
    Perft(
        Position,
        Limit,
        CancellationToken,
        DebugMode,
        Arc<Mutex<Configuration>>,
    ),
    Normal(
        Position,
        Limit,
        CancellationToken,
        DebugMode,
        Arc<Mutex<Configuration>>,
    ),
    AdvanceState(Move),
    Ponder,
    ResetState,
}
