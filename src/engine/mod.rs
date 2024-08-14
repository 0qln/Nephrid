use std::{
    sync::Arc,
    thread
};
use crate::{uci, CancellationToken};
use crate::engine::{
    depth::Depth,
    position::{Position, PositionInfo},
    config::{ConfigOptionType, Configuration, ConfigOption},
    r#move::Move,
};

pub mod search;
pub mod zobrist;
pub mod transposition_table;
pub mod move_gen;
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


pub fn d(position: Option<Position>) {
    let pos: String = position.into();
    uci::out(&pos)
}

pub fn execute_uci(
    input: &String, 
    config: &mut Configuration, 
    cancellation_token: &CancellationToken,
    position: Option<&Position>) 
{
    let mut tokenizer: uci::Tokenizer = uci::tokenize(input.as_str());
    match tokenizer
        .collect_until_ws()
        .unwrap_or(String::new())
        .as_str()
    {
        "d" => {
            let pos: String = (*position).into();
            uci::out(&pos)
        }
        "quit"=> {
            cancellation_token.cancel()
        }
        "go" => {
            search::reset();

            let mut mode = search::Mode::Normal;

            let mut limit = search::Limit {
                wtime: u64::MAX, btime: u64::MAX,
                winc: 0, binc: 0,
                movestogo: 0,
                nodes: u64::MAX,
                movetime: u64::MAX,
                active: true
            };

            let mut target = search::Target {
                mate: Depth::MAX,
                depth: Depth::MAX,
                searchmoves: Vec::new()
            };

            macro_rules! collect_and_parse {
                ($tokenizer:expr, $field:expr, $default:expr) => {
                    $field = $tokenizer
                        .collect_until_ws()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or($default)
                };
            }

            while let Some(token) = tokenizer.collect_until_ws() {
                match token.as_str() {
                    "ponder" => mode = search::Mode::Ponder,
                    "wtime" => collect_and_parse!(tokenizer, limit.wtime, 0),
                    "btime" => collect_and_parse!(tokenizer, limit.btime, 0),
                    "winc" => collect_and_parse!(tokenizer, limit.winc, 0),
                    "binc" => collect_and_parse!(tokenizer, limit.binc, 0),
                    "movestogo" => collect_and_parse!(tokenizer, limit.movestogo, 0),
                    "depth" => collect_and_parse!(tokenizer, target.depth, Depth::MIN),
                    "nodes" => collect_and_parse!(tokenizer, limit.nodes, 0),
                    "mate" => collect_and_parse!(tokenizer, target.mate, Depth::MIN),
                    "movetime" => collect_and_parse!(tokenizer, limit.movetime, 0),
                    "infinite" => limit.active = false,
                    _ => {
                        match Move::try_from(token) {
                            Ok(m) => target.searchmoves.push(m),
                            Err(e) => uci::out(&format!("Error: {e}"))
                        }
                    },
                };
            }

            // TODO: use config to set up search params
            
            let token = cancellation_token.clone();

            let x = 1;
            consume(x);
            consume(x);

            match position {
                Some(p) => thread::spawn(move || {
                    search::go(p, limit, target, mode, token);
                }),
                None => thread::spawn(move || { })
            };
        },
        "uci" => {
            // Id response
            uci::out(&"id name Nephrid");
            uci::out(&"id author 0qln");
            // Option response
            for option in &config.0 {
                let opt_str: String = option.into(); 
                uci::out(&opt_str);                    
            }
            // Uciok response
            uci::out("uciok")
        },
        "setoption" => {
            // Skip to name
            tokenizer.collect_until_ws();

            // collect name
            let mut name = String::new();
            while let Some(token) = tokenizer.collect_until_ws() {
                match token.as_str() {
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
            while let Some(token) = tokenizer.collect_until_ws() {
                new_value.push_str(&token);
                // Reintroduce spaces between parts
                if !new_value.is_empty() {
                    new_value.push(' ')
                }
            }

            match config.find_mut(name.as_str()) {
                None => uci::out(&format!("Unknown option: '{name}'")),
                Some(option) => {
                    match &mut option.cfg_type {
                        ConfigOptionType::Check { value, .. } => {
                            match new_value.parse() {
                                Ok(parsed) => *value = parsed,
                                Err(_) => uci::out("Error: Missing or invalid value")
                            }
                        },
                        ConfigOptionType::Spin { min, max, value, .. } => {
                            match new_value.parse() {
                                Ok(parsed) if *value >= *min && *value <= *max => *value = parsed,
                                Ok(_) => uci::out("Error: Value out of range"),
                                Err(_) => uci::out("Error: Missing or invalid value")
                            }
                        },
                        ConfigOptionType::Combo { options, value, .. } => {
                            match !new_value.is_empty() {
                                true if options.contains(&new_value) => *value = String::from(&new_value),
                                true => uci::out("Error: Invalid value"),
                                false => uci::out("Error: Missing value"),
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
        },
        "ucinewgame" => {
            position.reset();
        },
        unknown => uci::out(&format!("Unknown UCI command: '{unknown}'")),
    }
}


