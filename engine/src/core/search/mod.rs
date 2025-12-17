use crate::core::Move;
use std::{
    sync::mpsc::{Sender, channel},
    thread,
};

use crate::{
    core::{
        position::Position,
        search::{
            limit::Limit,
            mcts::{MctsState, mcts, strategy::MctsUci},
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
    thread::spawn(move || {
        let mut mcts_state = MctsState::default();
        loop {
            let cmd = rx.recv().expect("Should be able to receive data");
            match cmd {
                Command::Perft(pos, limit, ct, debug) => {
                    perft(pos, limit, ct, debug);
                }
                Command::Normal(pos, limit, ct, debug) => {
                    let (result, state) =
                        mcts(pos, mcts_state, limit, debug, ct, MctsUci::default());
                    result.expect("");
                    mcts_state = state;
                }
                Command::Ponder => {
                    unimplemented!("todo");
                }
                Command::AdvanceState(mov) => {
                    mcts_state.tree.advance_to(|b| b.mov() == mov);
                }
                Command::ResetState => {
                    unimplemented!("todo")
                }
            }
        }
    });
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
