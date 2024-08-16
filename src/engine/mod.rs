use crate::{uci, CancellationToken};
use crate::engine::{
    depth::Depth,
    position::{Position, PositionInfo},
    config::{ConfigOptionType, Configuration, ConfigOption},
    r#move::Move,
    fen::Fen,
};
use std::thread;
use anyhow::Error;

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


pub struct Engine<'token> {
    pub config: Configuration,
    pub position: Position,
    pub cancellation_token: &'token CancellationToken,    
}

impl Engine<'_> {
    pub fn execute_uci(&mut self, input: &String) {
        let mut tokenizer: uci::Tokenizer = uci::tokenize(input.as_str());
        match tokenizer
            .collect_until_ws()
            .unwrap_or(String::new())
            .as_str()
        {
            "d" => {
                let pos: String = (&self.position).into();
                uci::out(&pos);
            }
            "quit"=> {
                self.cancellation_token.cancel()
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
                
                let token = self.cancellation_token.clone();
                let position = self.position.clone();

                thread::spawn(move || {
                    search::go(&position, limit, target, mode, token);
                });
            },
            "position" => {
                match tokenizer.collect_until_ws() {
                    Some(token) => {
                        if token == "fen" {
                            let p1 = tokenizer.collect_until_ws().unwrap();
                            let p2 = tokenizer.collect_until_ws().unwrap();
                            let p3 = tokenizer.collect_until_ws().unwrap();
                            let p4 = tokenizer.collect_until_ws().unwrap();
                            let p5 = tokenizer.collect_until_ws().unwrap();
                            let p6 = tokenizer.collect_until_ws().unwrap();
                            let fen = Fen { v: [ p1.as_str(), p2.as_str(), p3.as_str(), p4.as_str(), p5.as_str(), p6.as_str() ] };
                            match Position::try_from(fen) {
                                Ok(pos) => self.position = pos,
                                Err(e) => uci::out(&format!("Error: {e}"))
                            }
                        }   
                    },
                    None => uci::out("Error: Missing arguments")
                }
                if tokenizer.collect_until_ws() == Some("moves".to_string()) {
                    while let Some(token) = tokenizer.collect_until_ws() {
                        match Move::try_from(token) {
                            Ok(m) => self.position.make_move(m),
                            Err(e) => uci::out(&format!("Error: {e}"))
                        }
                    }
                }
            }
            "uci" => {
                // Id response
                uci::out(&"id name Nephrid");
                uci::out(&"id author 0qln");
                // Option response
                for option in &self.config.0 {
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

                match self.config.find_mut(name.as_str()) {
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
                self.position.reset()
            },
            unknown => { 
                uci::out(&format!("Unknown UCI command: '{unknown}'")) 
            },
        }    
    }
}



