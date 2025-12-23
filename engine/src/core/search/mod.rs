use crate::core::Move;
use crate::core::search::mcts::node::Tree;
use std::{
    sync::mpsc::{Sender, channel},
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
            // let mcts_parts = NNState::<mcts::config::Backend>::default();
            let mcts_parts = mcts::config::mcts::Parts::default();
            loop {
                let cmd = rx.recv().expect("Should be able to receive data");
                match cmd {
                    Command::Perft(pos, limit, ct, debug) => {
                        perft(pos, limit, ct, debug);
                    }
                    Command::Normal(pos, limit, ct, debug) => {
                        let strat = MctsUci::default();
                        let result =
                            mcts(&pos, &mcts_parts, &mut mcts_state, limit, debug, ct, strat);
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
    Perft(Position, Limit, CancellationToken, DebugMode),
    Normal(Position, Limit, CancellationToken, DebugMode),
    AdvanceState(Move),
    Ponder,
    ResetState,
}
