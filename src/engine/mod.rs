use search::{limit::Limit, mode::Mode, target::Target, Search};

use self::r#move::LongAlgebraicUciNotation;
use crate::engine::{
    config::{ConfigOptionType, Configuration},
    depth::Depth,
    fen::Fen,
    position::Position,
    r#move::Move,
};
use crate::{
    misc::ParseError,
    uci::{
        sync::{self, CancellationToken, UciError},
        tokens::Tokenizer,
    },
};
use std::{
    default,
    error::Error,
    process,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

pub mod bitboard;
pub mod castling;
pub mod color;
pub mod config;
pub mod coordinates;
pub mod depth;
pub mod fen;
pub mod r#move;
pub mod move_iter;
pub mod piece;
pub mod ply;
pub mod position;
pub mod search;
pub mod turn;
pub mod zobrist;

#[derive(Default)]
pub struct Engine {
    config: Configuration,
    debug: Arc<AtomicBool>,
    position: Position,
}

// mod uci
pub fn execute_uci(
    engine: &mut Engine,
    tokenizer: &mut Tokenizer<'_>,
    cancellation_token: CancellationToken,
) -> Result<(), Box<dyn Error>> {
    match tokenizer.collect_token().as_deref() {
        Some("d") => {
            let pos: String = (&engine.position).into();
            sync::out(&pos);
            Ok(())
        }
        Some("quit") => {
            process::exit(0);
        }
        Some("stop") => {
            cancellation_token.cancel();
            Ok(())
        }
        Some("go") => {
            let token = cancellation_token.clone();
            let mut position = engine.position.clone();
            let debug = engine.debug.clone();

            macro_rules! collect_and_parse {
                ($field:expr) => {{
                    let token = tokenizer.collect_token();
                    $field = token.map_or(Ok(Default::default()), |s| s.parse())?;
                }};
            }

            let mut mode = Mode::default();
            let mut limit = Limit::default();
            let mut target = Target::default();

            while let Some(token) = tokenizer.collect_token().as_deref() {
                match token {
                    "perft" => mode = Mode::Perft,
                    "ponder" => mode = Mode::Ponder,
                    "wtime" => collect_and_parse!(limit.wtime),
                    "btime" => collect_and_parse!(limit.btime),
                    "winc" => collect_and_parse!(limit.winc),
                    "binc" => collect_and_parse!(limit.binc),
                    "movestogo" => collect_and_parse!(limit.movestogo),
                    "depth" => collect_and_parse!(target.depth),
                    "nodes" => collect_and_parse!(limit.nodes),
                    "mate" => collect_and_parse!(target.mate),
                    "movetime" => collect_and_parse!(limit.movetime),
                    "infinite" => limit.is_active = false,
                    // to be compatible with stockfish
                    depth if Depth::try_from(depth).is_ok() => {
                        target.depth = depth.try_into().unwrap()
                    }
                    /*searchmoves*/
                    _ => {
                        let move_notation = LongAlgebraicUciNotation::new(tokenizer, &position);
                        match Move::try_from(move_notation) {
                            Ok(m) => target.search_moves.push(m),
                            Err(e) => sync::out(&format!("Error: {e}")),
                        }
                    }
                };
            }

            thread::spawn(move || {
                let search = Search::new(limit, target, mode, debug);
                search.go(&mut position, token);
            });
            Ok(())
        }
        Some("position") => {
            match tokenizer.collect_token().as_deref() {
                Some("fen") => engine.position = Position::try_from(&mut *tokenizer)?,
                Some("startpos") => engine.position = Position::start_position(),
                None => return Err(UciError::MissingArgument("value").into()),
                Some(x) => {
                    return Err(UciError::InvalidValue(
                        x.to_string(),
                        vec!["fen".to_string(), "startpos".to_string()],
                    )
                    .into())
                }
            };
            if tokenizer.collect_token().as_deref() == Some("moves") {
                while tokenizer.goto_next_token() {
                    let mov = LongAlgebraicUciNotation::new(&mut *tokenizer, &engine.position);
                    engine.position.make_move(Move::try_from(mov)?);
                }
            }
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
            while let Some(token) = tokenizer.collect_token().as_deref() {
                match token {
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
            while let Some(token) = tokenizer.collect_token().as_deref() {
                new_value.push_str(token);
                // Reintroduce spaces between parts
                if !new_value.is_empty() {
                    new_value.push(' ')
                }
            }

            match engine.config.find_mut(name.as_str()) {
                None => Err(UciError::UnknownOption)?,
                Some(option) => match &mut option.cfg_type {
                    ConfigOptionType::Check { value, .. } => {
                        if new_value.is_empty() {
                            return Err(UciError::MissingArgument("value").into());
                        }
                        *value = new_value.parse()?;
                    }
                    ConfigOptionType::Spin {
                        min, max, value, ..
                    } => {
                        if new_value.is_empty() {
                            return Err(UciError::MissingArgument("value").into());
                        }
                        let parsed = new_value.parse()?;
                        if parsed < *min || parsed > *max {
                            return Err(UciError::InputOutOfRange(
                                new_value,
                                min.to_string(),
                                max.to_string(),
                            )
                            .into());
                        }
                        *value = parsed;
                    }
                    ConfigOptionType::Combo { options, value, .. } => {
                        if !options.0.contains(&new_value) {
                            return Err(UciError::InvalidValue(new_value, options.clone().0).into());
                        }
                        *value = new_value;
                    }
                    ConfigOptionType::Button { callback } => callback(),
                    ConfigOptionType::String(value) => *value = new_value.into(),
                },
            };
            Ok(())
        }
        Some("ucinewgame") => {
            engine.position = Position::start_position();
            Ok(())
        }
        Some("debug") => {
            let debug = match tokenizer.collect_token().as_deref() {
                Some("on") => true,
                Some("off") => false,
                Some(x) => {
                    return Err(UciError::InvalidValue(
                        x.to_string(),
                        vec!["on".to_string(), "off".to_string()],
                    )
                    .into())
                }
                None => return Err(UciError::MissingArgument("value").into()),
            };
            Ok(engine.debug.store(debug, Ordering::Relaxed))
        }
        Some("isready") => Ok(sync::out(&"readyok")),
        Some(unknown) => Err(UciError::InvalidCommand(unknown.to_string()).into()),
        None => Ok(()),
    }
}
