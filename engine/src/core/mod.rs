use burn_cuda::{Cuda, CudaDevice};
use itertools::Either;
use search::{limit::Limit, mode::Mode};

use self::r#move::LongAlgebraicUciNotation;
use crate::{
    core::{
        config::{ConfigOptionType, Configuration},
        depth::Depth,
        r#move::Move,
        position::Position,
        search::MctsUci,
    },
    misc::trim_newline,
};
use crate::{
    misc::DebugMode,
    uci::{
        sync::{self, CancellationToken, UciError},
        tokens::Tokenizer,
    },
};
use std::{
    error::Error,
    process,
    thread::{self, JoinHandle},
};

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

// todo:
// instead of storing the gametree in between moves, try using bump-allocation to allocate all the
// nodes. Maybe the speed up is better than storing the compute? (We can't do both, since with bump
// allocation we either would have to move the subtree to a new `Bump`, or we would just not be
// able to deallocate the unused nodes. If our search is *that* slow that we aren't even using that
// much memory for the Tree, maybe just risc having a huge memory leak for each `ucinewgame` then
// :3 idk)
//
/// # The search state.
///
/// Either we have ownership of a search-tree, or we have the join handle of the thread that
/// will give us back the ownership of the search-tree.
///
/// (An option because maybe we just started something else like perft or some sht)
pub type SearchState = Option<Either<mcts::Tree, JoinHandle<mcts::Tree>>>;

/// Stores relevant information of the chess engine.
#[derive(Default)]
pub struct Engine {
    /// Current engine configuration.
    config: Configuration,

    /// Search state of the engine
    search_state: SearchState,

    /// Whether the engine runs in debug mode.
    debug: DebugMode,

    /// The currently set position of the engine. Changing this during a search should not
    /// immediatly affect the search tree.
    position: Position,

    /// Cached position source string to improve position decoding speed.
    _pos_src: String,
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
            let pos = &engine.position;
            let str = if engine.debug.get() {
                format!("{pos:?}")
            } else {
                format!("{pos}")
            };

            sync::out(&str);

            Ok(Either::Left(()))
        }
        Some("quit") => {
            process::exit(0);
        }
        Some("stop") => {
            // signal the thread the finish safely.
            cancellation_token.cancel();

            if let Some(search_state) = engine.search_state {
                // wait for the engine to finish and take back ownership of the search tree.
                let tree = match search_state {
                    Left(tree) => tree,
                    Right(join_handle) => join_handle.join()?,
                };

                engine.search_state = Some(Left(tree));
            }

            Ok(())
        }
        Some("go") => {
            let token = cancellation_token.clone();
            let position = engine.position.clone();
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
                        "nodes" => collect_and_parse!(limit.nodes),
                        "mate" => collect_and_parse!(limit.mate),
                        "movetime" => collect_and_parse!(limit.movetime),
                        "infinite" => limit.is_active = false,
                        "searchmoves" => {
                            // interpret all remaining arguments as moves.
                            while let Some(token) = tokenizer.next_token() {
                                let mov = LongAlgebraicUciNotation::new(&mut tokenizer, &position);
                                limit.search_moves.push(Move::try_from(mov)?);
                            }
                            break;
                        }
                        // to be compatible with stockfish
                        depth if Depth::try_from(depth).is_ok() => {
                            limit.depth = depth.try_into().unwrap()
                        }
                        o => return Err(UciError::UnknownOption(o)),
                    };
                }

                (mode, limit)
            };

            // todo: <Send> the self.tree to the thread. Maybe copy since we can't garantee that
            // the thread will finish and return ownership? Or just capture the <thread> and safe
            // it in the <self as Engine> state machine thingy and regain ownership when we cancel
            // the thread, e.g. with a UCI "stop" command?

            let thread = thread::spawn(move || {
                match mode {
                    Mode::Perft => {
                        let nodes = search::perft(position.clone(), limit, token, debug);
                        sync::out(&format!("\nNodes searched: {nodes}"));
                    }
                    Mode::Normal => {
                        // todo: don't hardcode this...
                        type Backend = Cuda<f32>;
                        let device = CudaDevice::default();
                        let model = ModelConfig::new().init::<Backend>(&device);
                        let model = EvalModel::new(model, &device);

                        let result =
                            search::mcts::<MctsUci, _>(position, &mut model, limit, debug, token)
                                .expect("search did not complete");

                        sync::out(&format!("bestmove {result}"));
                    }
                    _ => unimplemented!(),
                };
            });

            Ok(Either::Right(thread))
        }
        Some("position") => {
            if command.len() > engine._pos_src.len()
                && !engine._pos_src.is_empty()
                && command[..engine._pos_src.len()] == engine._pos_src
            {
                let new_moves = &command[engine._pos_src.len()..];
                for tok in Tokenizer::new(new_moves).tokens() {
                    if tok == "moves" {
                        continue;
                    }
                    let tok = &mut Tokenizer::new(tok);
                    let mov = LongAlgebraicUciNotation::new(tok, &engine.position);
                    let mov = Move::try_from(mov)?;
                    engine.position.make_move(mov);

                    if Some(Either::Left(search_state)) = engine.search_state {
                        search_state.advance(mov);
                    } else {
                        println!("search state is out of sync!!!");
                    }
                }
                engine._pos_src = command;
                return Ok(Either::Left(()));
            }
            match tokenizer.next_token() {
                Some("fen") => engine.position = Position::try_from(&mut tokenizer)?,
                Some("startpos") => engine.position = Position::start_position(),
                None => return Err(UciError::MissingArgument("value").into()),
                Some(x) => {
                    return Err(UciError::InvalidValue(
                        x.to_string(),
                        vec!["fen".to_string(), "startpos".to_string()],
                    )
                    .into());
                }
            };
            if tokenizer.next_token() == Some("moves") {
                for tok in tokenizer.tokens() {
                    let tok = &mut Tokenizer::new(tok);
                    let mov = LongAlgebraicUciNotation::new(tok, &engine.position);
                    engine.position.make_move(Move::try_from(mov)?);
                }
            }
            engine._pos_src = command;
            Ok(())
        }
        Some("ucinewgame") => {
            engine._pos_src = "".to_string();
            engine.search_state = None;
            Ok(())
        }
        Some("uci") => {
            // Id response
            sync::out("id name Nephrid");
            sync::out("id author 0qln");
            // Option response
            for option in &engine.config.0 {
                sync::out(&option.to_string());
            }
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
                        // Reintroduce spaces between parts
                        if !name.is_empty() {
                            name.push(' ')
                        }
                    }
                };
            }

            if name.is_empty() {
                return Err(UciError::MissingArgument("name").into());
            }

            // collect value
            let mut new_value = String::new();
            while let Some(token) = tokenizer.next_token() {
                new_value.push_str(token);
                // Reintroduce spaces between parts
                if !new_value.is_empty() {
                    new_value.push(' ')
                }
            }

            let new_value = new_value.trim();

            let name = name.trim();
            match engine.config.find_mut(name) {
                None => Err(UciError::UnknownOption(name))?,
                Some(option) => match &mut option.cfg_type {
                    ConfigOptionType::Check { value, .. } => {
                        if new_value.is_empty() {
                            return Err(UciError::MissingArgument("value").into());
                        }
                        *value = new_value.parse()?;
                    }
                    ConfigOptionType::Spin { min, max, value, .. } => {
                        if new_value.is_empty() {
                            return Err(UciError::MissingArgument("value").into());
                        }
                        let parsed = new_value.parse()?;
                        if parsed < *min || parsed > *max {
                            return Err(UciError::InputOutOfRange(
                                new_value.to_string(),
                                min.to_string(),
                                max.to_string(),
                            )
                            .into());
                        }
                        *value = parsed;
                    }
                    ConfigOptionType::Combo { options, value, .. } => {
                        let new_value = new_value.to_string();
                        if !options.0.contains(&new_value) {
                            return Err(UciError::InvalidValue(new_value, options.clone().0).into());
                        }
                        *value = new_value;
                    }
                    ConfigOptionType::Button { callback } => callback(),
                    ConfigOptionType::String(value) => *value = new_value.into(),
                },
            };
            Ok(Either::Left(()))
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
            Ok(Either::Left(()))
        }
        Some("isready") => {
            sync::out("readyok");
            Ok(Either::Left(()))
        }
        Some(unknown) => Err(UciError::InvalidCommand(unknown.to_string()).into()),
        None => Ok(Either::Left(())),
    }
}
