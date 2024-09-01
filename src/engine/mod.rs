use crate::uci::{
    sync::{ CancellationToken, self },
    tokens::Tokenizer,
};
use crate::engine::{
    depth::Depth,
    position::Position,
    config::{ConfigOptionType, Configuration},
    r#move::Move,
    fen::Fen,
};
use std::thread;

use self::r#move::{LongAlgebraicNotationUci, MoveNotation};

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


#[derive(Default)]
pub struct Engine {
    pub config: Configuration,
    pub position: Position,
    pub cancellation_token: CancellationToken,    
}

impl Engine {
    pub fn execute_uci(&mut self, tokenizer: &mut Tokenizer<'_>) {
        match tokenizer.collect_token().as_deref()
        {
            Some("d") => {
                let pos: String = (&self.position).into();
                sync::out(&pos);
            }
            Some("quit")=> {
                self.cancellation_token.cancel()
            }
            Some("go") => {
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
                    ($tokenizer:expr, $field:expr, $default:expr) => {{
                        // TODO: clean this up
                        let token = $tokenizer.collect_token();
                        let token = token.as_deref();
                        if token.is_none() { $field = $default; return ;}
                        let token = token.unwrap();
                        $field = token.parse().unwrap_or($default);
                    }};
                }

                while let Some(token) = tokenizer.collect_token().as_deref() {
                    match token {
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
                            let move_notation = MoveNotation::<LongAlgebraicNotationUci>::new(
                                &mut *tokenizer,
                                &self.position
                            );
                            match Move::try_from(move_notation) {
                                Ok(m) => target.searchmoves.push(m),
                                Err(e) => sync::out(&format!("Error: {e}"))
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
            }
            Some("position") => {
                match tokenizer.collect_token().as_deref() {
                    Some("fen") => {
                        let fen: &mut Fen = tokenizer;
                        match Position::try_from(fen) {
                            Ok(pos) => self.position = pos,
                            Err(e) => sync::out(&format!("Error: {e}"))
                        }
                    },
                    None => sync::out("Error: Missing arguments"),
                    Some(x) => sync::out(&format!("Error: Invalid argument: {}", x))
                };
                if tokenizer.collect_token().as_deref() == Some("moves") {
                    while tokenizer.goto_next_token() {
                        let move_notation = MoveNotation::<LongAlgebraicNotationUci>::new(&mut *tokenizer, &self.position);
                        match Move::try_from(move_notation) {
                            Ok(m) => self.position.make_move(m),
                            Err(e) => sync::out(&format!("Error: {e}"))
                        };
                    }
                }
            }
            Some("uci") => {
                // Id response
                sync::out(&"id name Nephrid");
                sync::out(&"id author 0qln");
                // Option response
                for option in &self.config.0 {
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
                    new_value.push_str(&token);
                    // Reintroduce spaces between parts
                    if !new_value.is_empty() {
                        new_value.push(' ')
                    }
                }

                match self.config.find_mut(name.as_str()) {
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
                self.position.reset()
            }
            Some(unknown) => { 
                sync::out(&format!("Unknown UCI command: '{unknown}'")) 
            }
            None => {

            }
        }    
    }
}



