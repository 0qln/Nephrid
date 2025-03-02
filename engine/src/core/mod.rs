use search::{Search, limit::Limit, mode::Mode, target::Target};

use self::r#move::LongAlgebraicUciNotation;
use crate::uci::{
    sync::{self, CancellationToken, UciError},
    tokens::Tokenizer,
};
use crate::{
    core::{
        config::{ConfigOptionType, Configuration},
        depth::Depth,
        r#move::Move,
        position::Position,
    },
    misc::trim_newline,
};
use std::{
    error::Error,
    iter, process,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
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

#[derive(Default)]
pub struct Engine {
    config: Configuration,
    debug: Arc<AtomicBool>,
    position: Position,
    pos_src: String,
}

pub fn execute_uci(
    engine: &mut Engine,
    mut command: String,
    cancellation_token: CancellationToken,
) -> Result<(), Box<dyn Error>> {
    trim_newline(&mut command);
    let mut tokenizer = Tokenizer::new(command.as_str());
    match tokenizer.next_token().as_deref() {
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
                    let token = tokenizer.next_token();
                    $field = token.map_or(Ok(Default::default()), |s| s.parse())?;
                }};
            }

            let mut mode = Mode::default();
            let mut limit = Limit::default();
            let mut target = Target::default();

            while let Some(token) = tokenizer.next_token().as_deref() {
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
                        let move_notation =
                            LongAlgebraicUciNotation::new(&mut tokenizer, &position);
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
            if {
                command.len() > engine.pos_src.len()
                    && !engine.pos_src.is_empty()
                    && &command[..engine.pos_src.len()] == engine.pos_src
            } {
                let new_moves = &command[engine.pos_src.len()..];
                for tok in Tokenizer::new(new_moves).tokens() {
                    if tok == "moves" {
                        continue;
                    }
                    let tok = &mut Tokenizer::new(tok);
                    let mov = LongAlgebraicUciNotation::new(tok, &engine.position);
                    engine.position.make_move(Move::try_from(mov)?);
                }
                return Ok(engine.pos_src = command);
            }
            match tokenizer.next_token().as_deref() {
                Some("fen") => engine.position = Position::try_from(&mut tokenizer)?,
                Some("startpos") => engine.position = Position::start_position(),
                None => return Err(UciError::MissingArgument("value").into()),
                Some(x) => {
                    return Err(UciError::InvalidValue(x.to_string(), vec![
                        "fen".to_string(),
                        "startpos".to_string(),
                    ])
                    .into());
                }
            };
            if tokenizer.next_token().as_deref() == Some("moves") {
                for tok in tokenizer.tokens() {
                    let tok = &mut Tokenizer::new(tok);
                    let mov = LongAlgebraicUciNotation::new(tok, &engine.position);
                    engine.position.make_move(Move::try_from(mov)?);
                }
            }
            Ok(engine.pos_src = command)
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
            while let Some(token) = tokenizer.next_token().as_deref() {
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
            while let Some(token) = tokenizer.next_token().as_deref() {
                new_value.push_str(token);
                // Reintroduce spaces between parts
                if !new_value.is_empty() {
                    new_value.push(' ')
                }
            }

            let new_value = new_value.trim();

            match engine.config.find_mut(name.trim()) {
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
            Ok(())
        }
        Some("ucinewgame") => Ok(engine.pos_src = "".to_string()),
        Some("debug") => {
            let debug = match tokenizer.next_token().as_deref() {
                Some("on") => true,
                Some("off") => false,
                Some(x) => {
                    return Err(UciError::InvalidValue(x.to_string(), vec![
                        "on".to_string(),
                        "off".to_string(),
                    ])
                    .into());
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
