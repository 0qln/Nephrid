use search::mode::Mode;
use std::{
    sync::{Arc, Mutex},
    thread, time,
};
use thiserror::Error;

use crate::{
    core::{
        config::Configuration,
        depth::Depth,
        r#move::{Move, SanParseError},
        position::{
            EpdLineImport, EpdLineParseError, EpdOp, FenExport, FenImport, FenParseError,
            PgnImport, PgnImportError, Position, ReducedPgn,
        },
        search::{Command, PonderToken, SearchThread, SearchWorker, limit::UciLimit},
    },
    misc::{CancellationToken, DebugMode, trim_newline},
    uci::{UciError, tokens::Tokenizer},
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

#[derive(Debug, Error)]
pub enum EpdImportError {
    #[error("EPD line parsing failed: {0}")]
    ParseEpd(#[from] EpdLineParseError),

    #[error("SAN move parsing failed: {0}")]
    ParseSan(#[from] SanParseError),
}

impl Game {
    pub fn from_moves(position: Position, moves: impl Iterator<Item = Move>) -> Self {
        let mut game = Self::from_position(position);
        for mov in moves {
            game.push_move(mov);
        }
        game
    }

    pub fn from_history(position: Position, history: Vec<Move>) -> Self {
        Self { position, history }
    }

    pub fn from_position(position: Position) -> Self {
        Self { position, ..Default::default() }
    }

    pub fn from_fen(fen: FenImport<'_, '_>) -> Result<Self, FenParseError> {
        Ok(Self::from_position(Position::try_from(fen)?))
    }

    pub fn from_epd(epd: EpdLineImport<'_, '_>) -> Result<Self, EpdImportError> {
        let (pos, ops) = epd.try_into()?;
        let mut game = Self::from_position(pos);

        if let Some(op) = ops.iter().find(|op| matches!(op.0.as_ref(), "sm")) {
            let EpdOp(_, mov) = op;
            let mov = Move::from_san(mov, &game.position)?;
            game.push_move(mov);
        }

        Ok(game)
    }

    pub fn from_pgn(pgn: PgnImport<'_, '_>) -> Result<Self, PgnImportError> {
        let pgn = ReducedPgn::try_from(pgn.0)?;

        let position = pgn.start_position()?;
        let mut game = Self::from_position(position);
        for san in pgn.moves() {
            let mov = Move::from_san(san, &game.position)?;
            game.push_move(mov);
        }
        Ok(game)
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

    pub fn to_pgn(&self) -> ReducedPgn {
        ReducedPgn::from_current_pos(self.position.clone(), &self.history[..])
    }
}

/// Stores relevant information of the chess engine.
pub struct Engine {
    /// Current engine configuration.
    config: Arc<Mutex<Configuration>>,

    /// Search thread
    search_t: SearchThread,

    /// A token to trigger the transition from pondering to normal search.
    ponder_token: PonderToken,

    /// Whether the engine runs in debug mode.
    debug: DebugMode,

    /// Current game state.
    game: Game,

    /// Cached position source string to improve position decoding speed.
    _pos_src: String,
}

impl Engine {
    pub fn new<Searcher: SearchWorker>() -> Self {
        Self {
            config: Default::default(),
            search_t: search::init::<Searcher>(),
            debug: Default::default(),
            game: Default::default(),
            _pos_src: Default::default(),
            ponder_token: Default::default(),
        }
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
            if engine.debug.get() {
                println!("{pos:?}")
            }
            else {
                println!("{pos}")
            };

            Ok(())
        }
        Some("search") => match tokenizer.next_token() {
            Some("d") => Ok(engine.search_t.tx.send(Command::Debug)?),
            Some(unknown) => Err(UciError::InvalidCommand(unknown.to_string()).into()),
            None => unimplemented!(),
        },
        Some("pgn") => {
            println!("{}", engine.game.to_pgn());
            Ok(())
        }
        Some("fen") => {
            println!("{}", FenExport(&engine.game.position));
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
                let mut limit = UciLimit::default();

                let config = engine.config.lock().expect("Config dead :(");
                limit.lag_buf = config.gui_lag();

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
                                let mov = Move::from_lan(token, &position)?;
                                limit.search_moves.push(mov);
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
                Mode::Ponder => {
                    let ponder_hit = engine.ponder_token.clone();
                    ponder_hit.start_ponder();
                    Command::Ponder(position, limit, token, debug, ponder_hit)
                }
                Mode::Perft => Command::Perft(position, limit, token, debug),
            };

            engine.search_t.tx.send(cmd)?;

            Ok(())
        }
        Some("ponderhit") => {
            engine.ponder_token.stop_ponder();
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

            // First, try to simply update the current position with new appended moves.
            if cached {
                let new_moves = &command[engine._pos_src.len()..];
                for tok in Tokenizer::new(new_moves).tokens() {
                    if tok == "moves" {
                        continue;
                    }
                    process_move(engine, tok)?;
                }
            }
            // Otherwise, we have a diverging position string.
            else {
                // 1. Parse the new base position into a temporary game state
                let mut new_game = match tokenizer.next_token() {
                    Some("pgn") => Game::from_pgn(PgnImport(&mut tokenizer))?,
                    Some("epd") => Game::from_epd(EpdLineImport(&mut tokenizer))?,
                    Some("fen") => Game::from_fen(FenImport(&mut tokenizer))?,
                    Some("startpos") => Game::from_position(Position::start_position()),
                    None => return Err(UciError::MissingArgument("value").into()),
                    Some(x) => {
                        return Err(UciError::InvalidValue(
                            x.to_string(),
                            vec!["fen".to_string(), "startpos".to_string()],
                        )
                        .into());
                    }
                };

                // 2. Parse all the new moves into our temporary game state
                if tokenizer.next_token() == Some("moves") {
                    for tok in tokenizer.tokens() {
                        let mov = Move::from_lan(tok, new_game.position())?;
                        new_game.push_move(mov);
                    }
                }

                // 3. Compare the histories to detect a 1-ply divergence (Ponder Miss)
                let old_moves = engine.game.moves();
                let new_moves = new_game.moves();

                let is_ponder_miss = !old_moves.is_empty()
                    && new_moves.len() == old_moves.len()
                    && old_moves[..old_moves.len() - 1] == new_moves[..new_moves.len() - 1]
                    && old_moves.last() != new_moves.last();

                let game_tree_caching = engine
                    .config
                    .lock()
                    .map(|c| c.game_tree_caching())
                    .unwrap_or(false);

                if is_ponder_miss && game_tree_caching {
                    // Ponder Miss detected! Restore the 1-ply backup and advance down the actual
                    // move.
                    let actual_move = *new_moves.last().unwrap();
                    engine
                        .search_t
                        .tx
                        .send(Command::RollbackAndAdvance(actual_move))?;
                }
                else {
                    // Completely new game or caching disabled: safely reset the tree entirely.
                    engine.search_t.tx.send(Command::ResetState)?;
                }

                // 4. Officially update the engine's game state
                engine.game = new_game;
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
            println!("id name Nephrid");
            println!("id author 0qln");

            // Option response
            engine.config.lock().expect("Config dead :(").print_uci();

            // Uciok response
            println!("uciok");

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
            println!("readyok");
            Ok(())
        }
        Some("perf") => {
            execute_uci(engine, "go".to_owned(), cancellation_token.clone())?;
            execute_uci(engine, "go".to_owned(), cancellation_token.clone())?;

            let dur = tokenizer
                .next_token()
                .map(&str::parse::<u64>)
                .and_then(Result::ok)
                .unwrap_or(5);
            let dur = time::Duration::from_secs(dur);
            thread::sleep(dur);

            execute_uci(engine, "quit".to_owned(), cancellation_token)
        }
        Some(unknown) => Err(UciError::InvalidCommand(unknown.to_string()).into()),
        None => Ok(()),
    }
}
