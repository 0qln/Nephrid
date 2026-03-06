use search::{limit::Limit, mode::Mode};
use std::{
    sync::{Arc, Mutex},
    thread, time,
};

use self::r#move::LongAlgebraicUciNotation;
use crate::{
    core::{
        config::Configuration,
        depth::Depth,
        r#move::Move,
        position::{FenImport, PgnExport, Position},
        search::{Command, Thread},
    },
    misc::{DebugMode, trim_newline},
    uci::{
        sync::{self, CancellationToken, UciError},
        tokens::Tokenizer,
    },
};
use std::{error::Error, process};

pub mod bitboard;
pub mod castling;
pub mod color;
pub mod config;
pub mod coordinates;
pub mod depth;
pub mod r#move;
pub mod move_iter;
pub mod piece;
pub mod ply;
pub mod position;
pub mod search;
pub mod turn;
pub mod zobrist;

#[derive(Debug, Default, Clone)]
pub struct Game {
    /// The moves that have been made.
    history: Vec<Move>,

    /// The currently set position of the engine. Changing this during a search
    /// should not immediatly affect the search tree.
    position: Position,
}

impl Game {
    pub fn new(position: Position) -> Self {
        Self { position, ..Default::default() }
    }

    pub fn moves(&self) -> &[Move] {
        &self.history
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn position_mut(&mut self) -> &mut Position {
        &mut self.position
    }

    pub fn push_move(&mut self, mov: Move) {
        self.history.push(mov);
        self.position.make_move(mov);
    }

    pub fn to_pgn(&self) -> PgnExport {
        PgnExport::from_current_pos(self.position.clone(), &self.history[..])
    }
}

/// Stores relevant information of the chess engine.
pub struct Engine {
    /// Current engine configuration.
    config: Arc<Mutex<Configuration>>,

    /// Search thread
    search_t: Thread,

    /// Whether the engine runs in debug mode.
    debug: DebugMode,

    /// Current game state.
    game: Game,

    /// Cached position source string to improve position decoding speed.
    _pos_src: String,
}

impl Engine {
    pub fn new() -> Self {
        let search_t = search::init();
        Self {
            config: Default::default(),
            search_t,
            debug: Default::default(),
            game: Default::default(),
            _pos_src: Default::default(),
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

pub fn execute_uci(
    engine: &mut Engine,
    mut command: String,
    cancellation_token: CancellationToken,
) -> Result<(), Box<dyn Error>> {
    trim_newline(&mut command);
    let mut tokenizer = Tokenizer::new(command.as_str());
    match tokenizer.next_token() {
        Some("d") | Some("print") => {
            let pos = &engine.game.position();
            let str = if engine.debug.get() {
                format!("{pos:?}")
            }
            else {
                format!("{pos}")
            };

            sync::out(&str);

            Ok(())
        }
        Some("pgn") => {
            let pgn = engine.game.to_pgn();
            let str = format!("{pgn}");

            sync::out(&str);

            Ok(())
        }
        Some("quit") => {
            process::exit(0);
        }
        Some("stop") => {
            // signal the thread the finish safely.
            cancellation_token.cancel();

            Ok(())
        }
        Some("go") => {
            let token = cancellation_token.clone();
            let position = engine.game.position().clone();
            let debug = engine.debug.clone();

            macro_rules! collect_and_parse {
                ($field:expr) => {{
                    let token = tokenizer.next_token();
                    $field = token.map_or(Ok(Default::default()), |s| s.parse())?;
                }};
            }

            let (mode, limit) = {
                let mut mode = Mode::default();
                let mut limit = Limit::default();

                while let Some(token) = tokenizer.next_token() {
                    match token {
                        "perft" => mode = Mode::Perft,
                        "ponder" => mode = Mode::Ponder,
                        "wtime" => collect_and_parse!(limit.wtime),
                        "btime" => collect_and_parse!(limit.btime),
                        "winc" => collect_and_parse!(limit.winc),
                        "binc" => collect_and_parse!(limit.binc),
                        "movestogo" => collect_and_parse!(limit.movestogo),
                        "depth" => collect_and_parse!(limit.depth),
                        "iterations" => collect_and_parse!(limit.iterations),
                        "nodes" => collect_and_parse!(limit.nodes),
                        "mate" => collect_and_parse!(limit.mate),
                        "movetime" => collect_and_parse!(limit.movetime),
                        "infinite" => limit.is_active = false,
                        "searchmoves" => {
                            // interpret all remaining arguments as moves.
                            while let Some(token) = tokenizer.next_token() {
                                let mut token = Tokenizer::new(token);
                                let mov = LongAlgebraicUciNotation::new(&mut token, &position);
                                limit.search_moves.push(Move::try_from(mov)?);
                            }
                            break;
                        }
                        // to be compatible with stockfish syntax
                        // e.g. "go 7" is interpreted as "go depth 7"
                        depth if let Ok(depth) = Depth::try_from(depth) => limit.depth = depth,
                        o => return Err(UciError::UnknownOption(o.to_owned()).into()),
                    };
                }

                (mode, limit)
            };

            let cmd = match mode {
                Mode::Normal => Command::Normal(position, limit, token, debug),
                Mode::Ponder => Command::Ponder,
                Mode::Perft => Command::Perft(position, limit, token, debug),
            };

            engine.search_t.tx.send(cmd)?;

            Ok(())
        }
        Some("position") => {
            let process_move = |engine: &mut Engine, mov| -> Result<(), Box<dyn Error>> {
                // decode move
                let mov = Move::from_lan(mov, engine.game.position())?;

                // advance the position
                engine.game.push_move(mov);

                // also advance the mcts game tree
                let game_tree_caching = engine
                    .config
                    .lock()
                    .map(|c| c.game_tree_caching())
                    .unwrap_or(false);

                engine.search_t.tx.send(match game_tree_caching {
                    true => Command::AdvanceState(mov),
                    false => Command::ResetState,
                })?;

                Ok(())
            };

            let cached = command.len() > engine._pos_src.len()
                && !engine._pos_src.is_empty()
                && command[..engine._pos_src.len()] == engine._pos_src;

            // First, try to simply update the current position with new moves.
            if cached {
                let new_moves = &command[engine._pos_src.len()..];
                for tok in Tokenizer::new(new_moves).tokens() {
                    if tok == "moves" {
                        continue;
                    }
                    process_move(engine, tok)?;
                }
            }
            // Otherwise build a new position
            else {
                engine.game = Game::new(match tokenizer.next_token() {
                    Some("fen") => Position::try_from(FenImport(&mut tokenizer))?,
                    Some("startpos") => Position::start_position(),
                    None => return Err(UciError::MissingArgument("value").into()),
                    Some(x) => {
                        return Err(UciError::InvalidValue(
                            x.to_string(),
                            vec!["fen".to_string(), "startpos".to_string()],
                        )
                        .into());
                    }
                });

                if tokenizer.next_token() == Some("moves") {
                    for tok in tokenizer.tokens() {
                        process_move(engine, tok)?;
                    }
                }
            }

            engine._pos_src = command;

            Ok(())
        }
        Some("ucinewgame") => {
            engine._pos_src = "".to_string();

            // also advance the mcts game tree
            engine.search_t.tx.send(Command::ResetState)?;

            Ok(())
        }
        Some("uci") => {
            // Id response
            sync::out("id name Nephrid");
            sync::out("id author 0qln");
            // Option response
            engine.config.lock().expect("Config dead :(").print_uci();
            // Uciok response
            sync::out("uciok");
            Ok(())
        }
        Some("setoption") => {
            // collect name
            let mut name = String::new();
            while let Some(token) = tokenizer.next_token() {
                match token {
                    "name" => continue,
                    "value" => break,
                    part => {
                        name.push_str(part);
                        name.push(' ');
                    }
                };
            }

            let name = name.trim();

            if name.is_empty() {
                return Err(UciError::MissingArgument("name").into());
            }

            // collect value
            let mut new_value = String::new();
            while let Some(token) = tokenizer.next_token() {
                new_value.push_str(token);
                new_value.push(' ');
            }

            let new_value = new_value.trim();

            engine
                .config
                .lock()
                .expect("Config dead :(")
                .set(name, new_value)?;

            // update search thread config.
            let cfg = engine.config.clone();
            let cmd = Command::Configure(cfg);
            engine.search_t.tx.send(cmd)?;

            Ok(())
        }
        Some("debug") => {
            let debug = match tokenizer.next_token() {
                Some("on") => true,
                Some("off") => false,
                Some(x) => {
                    return Err(UciError::InvalidValue(
                        x.to_string(),
                        vec!["on".to_string(), "off".to_string()],
                    )
                    .into());
                }
                None => return Err(UciError::MissingArgument("value").into()),
            };
            engine.debug.set(debug);
            Ok(())
        }
        Some("isready") => {
            sync::out("readyok");
            Ok(())
        }
        Some("perf") => {
            execute_uci(engine, "go".to_owned(), cancellation_token.clone())?;
            execute_uci(engine, "go".to_owned(), cancellation_token.clone())?;

            let dur = tokenizer
                .next_token()
                .map(&str::parse::<u64>)
                .map(Result::ok)
                .flatten()
                .unwrap_or(5);
            let dur = time::Duration::from_secs(dur);
            thread::sleep(dur);

            execute_uci(engine, "quit".to_owned(), cancellation_token)
        }
        Some(unknown) => Err(UciError::InvalidCommand(unknown.to_string()).into()),
        None => Ok(()),
    }
}
