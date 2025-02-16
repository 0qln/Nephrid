use search::{limit::Limit, Search};

use crate::uci::{
    sync::{self, CancellationToken},
    tokens::Tokenizer,
};
use crate::engine::{
    config::{ConfigOptionType, Configuration},
    depth::Depth,
    fen::Fen,
    position::Position,
    r#move::Move,
};
use std::{default, process, sync::{atomic::{AtomicBool, Ordering}, Arc}, thread};
use self::r#move::LongAlgebraicUciNotation;

pub mod search;
pub mod zobrist;
pub mod move_iter;
pub mod color;
pub mod piece;
pub mod depth;
pub mod turn;
pub mod fen;
pub mod bitboard;
pub mod position;
pub mod coordinates;
pub mod config;
pub mod castling;
pub mod r#move;
pub mod ply;


#[derive(Default)]
pub struct Engine {
    config: Configuration,
    position: Position,
    search: Search,
}

// mod uci
pub fn execute_uci(engine: &mut Engine, tokenizer: &mut Tokenizer<'_>, cancellation_token: CancellationToken) {
    match tokenizer.collect_token().as_deref() {
        Some("d") => {
            let pos: String = (&engine.position).into();
            sync::out(&pos);
        }
        Some("quit")=> {
            process::exit(0);
        }
        Some("stop") => {
            cancellation_token.cancel();
        }
        Some("go") => {
            let mut search = engine.search.clone();

            macro_rules! collect_and_parse {
                ($field:expr, $default:expr) => {{
                    let token = tokenizer.collect_token();
                    $field = token.map_or($default, |s| s.parse().unwrap_or($default));
                }};
            }

            engine.search.limit = Limit::default();

            while let Some(token) = tokenizer.collect_token().as_deref() {
                match token {
                    "perft" => search.mode = search::mode::Mode::Perft,
                    "ponder" => search.mode = search::mode::Mode::Ponder,
                    "wtime" => collect_and_parse!(search.limit.wtime, 0),
                    "btime" => collect_and_parse!(search.limit.btime, 0),
                    "winc" => collect_and_parse!(search.limit.winc, 0),
                    "binc" => collect_and_parse!(search.limit.binc, 0),
                    "movestogo" => collect_and_parse!(search.limit.movestogo, 0),
                    "depth" => collect_and_parse!(search.target.depth, Depth::MIN),
                    "nodes" => collect_and_parse!(search.limit.nodes, 0),
                    "mate" => collect_and_parse!(search.target.mate, Depth::MIN),
                    "movetime" => collect_and_parse!(search.limit.movetime, 0),
                    "infinite" => search.limit.is_active = false,
                    // to be compatible with stockfish
                    depth if Depth::try_from(depth).is_ok() => search.target.depth = depth.try_into().unwrap(),
                    /*searchmoves*/ | _ => {
                        let move_notation = LongAlgebraicUciNotation::new(tokenizer, &engine.position);
                        match Move::try_from(move_notation) {
                            Ok(m) => search.target.search_moves.push(m),
                            Err(e) => sync::out(&format!("Error: {e}"))
                        }
                    },
                };
            }

            let token = cancellation_token.clone();
            let mut position = engine.position.clone();

            thread::spawn(move || { search.go(&mut position, token); }); 
        }
        Some("position") => {
            match tokenizer.collect_token().as_deref() {
                Some("fen") => {
                    let fen: &mut Fen = tokenizer;
                    match Position::try_from(fen) {
                        Ok(pos) => engine.position = pos,
                        Err(e) => sync::out(&format!("Error: {e}"))
                    }
                },
                Some("startpos") => engine.position = Position::start_position(),
                None => sync::out("Error: Missing arguments"),
                Some(x) => sync::out(&format!("Error: Invalid argument: {}", x))
            };
            if tokenizer.collect_token().as_deref() == Some("moves") {
                while tokenizer.goto_next_token() {
                    let move_notation = LongAlgebraicUciNotation::new(&mut *tokenizer, &engine.position);
                    match Move::try_from(move_notation) {
                        Ok(m) => engine.position.make_move(m),
                        Err(e) => sync::out(&format!("Error: {e}"))
                    };
                }
            }
        }
        Some("uci") => {
            // Id response
            sync::out("id name Nephrid");
            sync::out("id author 0qln");
            // Option response
            for option in &engine.config.0 {
                let opt_str: String = option.into(); 
                sync::out(&opt_str);                    
            }
            // Uciok response
            sync::out("uciok")
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
                None => sync::out(&format!("Unknown option: '{name}'")),
                Some(option) => {
                    match &mut option.cfg_type {
                        ConfigOptionType::Check { value, .. } => {
                            match new_value.parse() {
                                Ok(parsed) => *value = parsed,
                                Err(_) => sync::out("Error: Missing or invalid value")
                            }
                        },
                        ConfigOptionType::Spin { min, max, value, .. } => {
                            match new_value.parse() {
                                Ok(parsed) if *value >= *min && *value <= *max => *value = parsed,
                                Ok(_) => sync::out("Error: Value out of range"),
                                Err(_) => sync::out("Error: Missing or invalid value")
                            }
                        },
                        ConfigOptionType::Combo { options, value, .. } => {
                            match !new_value.is_empty() {
                                true if options.contains(&new_value) => *value = String::from(&new_value),
                                true => sync::out("Error: Invalid value"),
                                false => sync::out("Error: Missing value"),
                            }
                        },
                        ConfigOptionType::Button { callback } => {
                            callback()
                        },
                        ConfigOptionType::String(value) => {
                            match new_value.is_empty() {
                                true => *value = Some(String::from("<empty>")),
                                false => *value = Some(new_value)
                            }
                        },
                    }
                },
            }
        }
        Some("ucinewgame") => {
            engine.position = Position::start_position();
        }
        Some("debug") => {
            let debug = match tokenizer.collect_token().as_deref() {
                Some("on") => true,
                Some("off") => false,
                Some(x) => return sync::out(&format!("Error: Invalid argument: {}", x)),
                None => return sync::out("Error: Missing arguments"),
            };
            
            engine.search.debug.store(debug, Ordering::Relaxed)
        }
        Some("isready") => {
            sync::out(&"readyok")
        }
        Some(unknown) => { 
            sync::out(&format!("Unknown UCI command: '{unknown}'")) 
        }
        None => {

        }
    }    
}
