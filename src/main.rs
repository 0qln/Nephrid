use std::{error::Error, fmt, io::{self, stdin, Stdout}, iter::Peekable, path::Display, str::Chars};
use std::thread;
use std::io::Write;

fn main() {

    assert_eq!(Square::from(Squares::A1), Square::from((0, 0)));

    let input_stream = stdin();
    let mut engine = Engine::default();
    loop {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => {
                engine.execute_uci(input);
            }
            Err(err) => println!("Error: {err}"),
        }
    }
}

struct Tokenizer<'a>(Peekable<Chars<'a>>);

impl<'a> Tokenizer<'a> {

    pub fn new(input: &'a str) -> Self {
        Tokenizer(input.chars().peekable())
    }

    fn collect_ws(&mut self) {
        while self.0.next_if(|&c| c.is_whitespace()).is_some() {}
    }

    fn collect_until_ws(&mut self) -> Option<String> {
        self.collect_ws();
        let mut buffer = String::new();
        while let Some(c) = self.0.next_if(|&c| !c.is_whitespace()) {
            buffer.push(c);
        }

        if buffer.is_empty() { None } else { Some(buffer) }
    }
    
    pub fn collect_until_ws_while<'f, F>(&mut self, predicate: &'f F) -> impl Iterator<Item = String> + '_
    where
        F: FnMut(&str) -> bool + 'f,
    {
        std::iter::from_fn(move || {
            let token = self.collect_until_ws()?;
            if predicate(&token) {
                Some(token)
            } else {
                None
            }
        })
    }

    fn collect_bool(&mut self) -> bool {
        self.collect_until_ws().is_some_and(|s| s == "true")
    }

    // fn collect_move_LAN(&mut self, first_char: char, context: Position) -> Move {
    //
    // }
    //
    // fn collect_move_SAN(&mut self, first_char: char, context: Position) -> Move {
    //
    // }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        return  self.collect_until_ws();
    }
}

fn tokenize(input: &str) -> Tokenizer {
    Tokenizer(input.chars().peekable())
}

pub enum UciCommand {
    Uci,
    Debug(bool),
    IsReady,
    SetOption {
        name: String,
        value: Option<String>,
    },
    Register,
    UciNewGame,
    Position {
        fen: String,
        moves: Vec<String>,
    },
    Go {
        ponder: bool,
        limit: Option<SearchLimit>,
        target: Option<SearchTarget>
    },
    Stop,
    PonderHit,
    Quit,
}

impl UciCommand {

    // fn parse_setoption(tokenizer: &mut Tokenizer) -> Self {
    //     
    // }
    //
    // fn parse_go(tokenizer: &mut Tokenizer) -> Self {
    //     // let mut ponder = false;
    //     // let mut searchmoves = Vec::new();
    //     // let mut wtime = 0;
    //     // let mut btime = 0;
    //     // let mut winc = 0;
    //     // let mut binc = 0;
    //     // let mut movestogo = 0;
    //     // let mut depth = 0;
    //     // let mut nodes = 0;
    //     // let mut mate = 0;
    //     // let mut movetime = 0;
    //     // let mut infinite = false;
    //     
    //     // let mut result = Self::Go {  ponder:false, searchmoves: Vec::new(), wtime: u64::MAX, btime: u64::MAX, winc: 0, binc: 0, movestogo: 0, u64.max, u64.max, u64.max, 0, false };
    //
    //     
    //     
    //     // result
    //
    //     // Self::Go {
    //     //     ponder,
    //     //     searchmoves,
    //     //     wtime,
    //     //     btime,
    //     //     winc,
    //     //     binc,
    //     //     movestogo,
    //     //     depth,
    //     //     nodes,
    //     //     mate,
    //     //     movetime,
    //     //     infinite,
    //     // }
    // }
    //
    // fn parse_position(tokenizer: &mut Tokenizer) -> Self {
    //     todo!()
    // }

}

pub fn uci_out(msg: &str) {
    let stdout = io::stdout();
    let _ = writeln!(&mut stdout.lock(), "{}", msg);
}

// pub enum UciResponse {
//     Id(String, String),
//     UciOk,
//     ReadyOk,
//     BestMove(String, Option<String>),
//     CopyProtection(bool),
//     Registration(bool),
//     Info(InfoResponse),
//     Option(),
//     Error(String)
// }

// impl UciResponse {
//     fn send(&mut self) {
//         match self {
//             UciResponse::Error(error) => {  }
//         }
//     }
//     
//     fn err(error: &str) {
//         UciResponse::Error(String::from(error)).send();
//     }
// }
//
// pub enum InfoResponse {
//     CurrLine(Option<u8>, Box<Vec<Move>>),
// }

pub enum ConfigOptionType {
    Check {
        default: bool,
        value: bool,
    },
    Spin {
        default: i32,
        min: i32,
        max: i32,
        value: i32,
    },
    Combo {
        default: String,
        options: Vec<String>,
        value: String
    },
    Button {
        callback: Box<dyn Fn()>,
    },
    String(Option<String>), // Some -> value, None -> "<empty>"
}

impl From<&ConfigOptionType> for String {
    fn from(val: &ConfigOptionType) -> Self {
        let mut result = String::new();
        
        macro_rules! push_field {
            ($name:expr, $field:expr) => {
                result.push_str($name);
                result.push_str(" ");
                result.push_str(&$field.to_string());
                result.push_str(" ");
            };
        }

        match val {
            ConfigOptionType::Check { default, value } => {
                result.push_str("check ");
                push_field!("default", default);
                push_field!("value", value);
            },
            ConfigOptionType::Spin { default, min, max, value } => {
                result.push_str("spin ");
                push_field!("default", default);
                push_field!("min", min);
                push_field!("max", max);
                push_field!("value", value);
            },
            ConfigOptionType::Combo { default, options, value } => {
                result.push_str("combo ");
                push_field!("default", default);
                for opt in options {
                    push_field!("var", opt);
                }
                push_field!("value", value);
            },
            ConfigOptionType::Button { callback: _ } => {
                result.push_str("button ");
            },
            ConfigOptionType::String(maybe) => {
                result.push_str("string");
                push_field!("value", &match maybe {
                    Some(str) => str.clone(),
                    None => String::from("<empty>")
                });
            },
        };
        result
    }
}

pub struct ConfigOption {
    cfg_name: String,
    cfg_type: ConfigOptionType,
}

impl From<&ConfigOption> for String {
    fn from(option: &ConfigOption) -> String {
        let type_str: String = (&option.cfg_type).into();
        let mut result = String::from("option");
        result.push_str(" name ");
        result.push_str(&option.cfg_name);
        result.push_str(" type ");
        result.push_str(&type_str);
        result
    }
}

pub struct Board {
    
}
impl Board {
    fn reset(&self) {
        todo!()
    }
}

pub struct SearchLimit {
    wtime: u64, btime: u64,
    winc: u64, binc: u64,
    movestogo: u16,
    nodes: u64,
    movetime: u64,
    active: bool
}

pub struct SearchTarget {
    mate: u8,
    depth: u8,
    searchmoves: Vec<Move>,
}

pub enum SearchMode {
    Normal,
    Ponder
}

pub struct Search {
}

impl Search {

    pub fn reset(&mut self) {
        
        todo!()
        
    }

    pub fn go(&mut self, mode: SearchMode, limit: SearchLimit, target: SearchTarget, board: &mut Board) {
        
        todo!()    
        
    }

}

pub struct Engine {
    config: Vec<ConfigOption>,
    search: Search,
    board: Board,
}

impl Engine {
    pub fn default() -> Self {
        Self {
            config: vec![
                ConfigOption {
                    cfg_name: String::from("Hash"),
                    cfg_type: ConfigOptionType::Spin {
                        default: 16,
                        min: 1,
                        max: 32 * 1024 * 1024,
                        value: 16,
                    },
                },
                ConfigOption {
                    cfg_name: String::from("Threads"),
                    cfg_type: ConfigOptionType::Spin {
                        default: 1,
                        min: 1,
                        max: 1,
                        value: 1,
                    },
                },
                ConfigOption {
                    cfg_name: String::from("Clear Hash"),
                    cfg_type: ConfigOptionType::Button {
                        callback: Box::from(|| {
                            todo!("clear hash");
                        }),
                    },
                },
            ],
            search: Search{},
            board: Board {},
        }
    }
    
    fn execute_uci(&mut self, input: String) {
        let mut tokenizer: Tokenizer = tokenize(input.as_str());
        match tokenizer
            .collect_until_ws()
            .unwrap_or(String::new())
            .as_str()
        {
            "go" => {
                self.search.reset();
                let mut mode = SearchMode::Normal;
                let mut limit = SearchLimit {
                    wtime: u64::MAX, btime: u64::MAX,
                    winc: 0, binc: 0,
                    movestogo: 0,
                    nodes: u64::MAX,
                    movetime: u64::MAX,
                    active: true
                };
                let mut target = SearchTarget {
                    mate: u8::MAX,
                    depth: u8::MAX ,
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
                        "ponder" => mode = SearchMode::Ponder,
                        "wtime" => collect_and_parse!(tokenizer, limit.wtime, 0),
                        "btime" => collect_and_parse!(tokenizer, limit.btime, 0),
                        "winc" => collect_and_parse!(tokenizer, limit.winc, 0),
                        "binc" => collect_and_parse!(tokenizer, limit.binc, 0),
                        "movestogo" => collect_and_parse!(tokenizer, limit.movestogo, 0),
                        "depth" => collect_and_parse!(tokenizer, target.depth, 0),
                        "nodes" => collect_and_parse!(tokenizer, limit.nodes, 0),
                        "mate" => collect_and_parse!(tokenizer, target.mate, 0),
                        "movetime" => collect_and_parse!(tokenizer, limit.movetime, 0),
                        "infinite" => limit.active = false,
                        _ => {
                            match Move::try_from(token) {
                                Ok(m) => target.searchmoves.push(m),
                                Err(e) => uci_out(&format!("Error: {e}"))
                            }
                        },
                    };
                }
                self.search.go(mode, limit, target, &mut self.board);
            },
            "uci" => {
                // Id response
                uci_out(&"id name Nephrid");
                uci_out(&"id author 0qln");
                // Option response
                for option in &self.config {
                    let opt_str: String = option.into(); 
                    uci_out(&opt_str);                    
                }
                // Uciok response
                uci_out("uciok")
            },
            // "debug" => UciCommand::Debug(tokenizer.collect_bool()),
            // "isready" => UciCommand::IsReady,
            // "position" => UciCommand::parse_position(&mut tokenizer),
            // "register" => UciCommand::Register,
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

                match self.config.iter_mut().find(|opt| opt.cfg_name == name) {
                    Some(option) => {
                        match &mut option.cfg_type {
                                ConfigOptionType::Check { value, .. } => {
                                    match new_value.parse() {
                                        Ok(parsed) => *value = parsed,
                                        Err(_) => uci_out("Error: Missing or invalid value")
                                    }
                                },
                                ConfigOptionType::Spin { min, max, value, .. } => {
                                    match new_value.parse() {
                                        Ok(parsed) if *value >= *min && *value <= *max => *value = parsed,
                                        Ok(_) => uci_out("Error: Value out of range"),
                                        Err(_) => uci_out("Error: Missing or invalid value")
                                    }
                                },
                                ConfigOptionType::Combo { options, value, ..} => {
                                    match !new_value.is_empty() {
                                        true if options.contains(&new_value) => *value = String::from(&new_value),
                                        true => uci_out("Error: Invalid value"),
                                        false => uci_out("Error: Missing value"),
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
                    None => uci_out(&format!("Unknown option: '{name}'"))
                }
            },
            "ucinewgame" => {
                self.board.reset();
            },
            unknown => uci_out(&format!("Unknown UCI command: '{unknown}'")),
        }
    }
}

pub struct Position {}

#[derive(Debug)]
#[repr(u8)]
pub enum Squares {
    A1, A2, A3, A4, A5, A6, A7, A8,
    B1, B2, B3, B4, B5, B6, B7, B8,
    C1, C2, C3, C4, C5, C6, C7, C8,
    D1, D2, D3, D4, D5, D6, D7, D8,
    E1, E2, E3, E4, E5, E6, E7, E8,
    F1, F2, F3, F4, F5, F6, F7, F8,
    G1, G2, G3, G4, G5, G6, G7, G8,
    H1, H2, H3, H4, H5, H6, H7, H8,
}

#[derive(Debug, PartialEq)]
pub struct Square(u8);

impl Square {

}

impl From<Squares> for Square {
    fn from(value: Squares) -> Self {
        Square { 0: value as u8 }
    }
}

impl From<(File, Rank)> for Square {
    fn from(value: (File, Rank)) -> Self {
        Square{0: value.0 * 8 + value.1}
    }
}

impl From<Box<&str>> for Square {
    fn from(value: Box<&str>) -> Self {
        let mut chars = value.chars();
        let mut sq: Square = Square { 0: 0 };
        if let Some(rank) = chars.next() {
            sq.0 += ((rank as u8) - ('1' as u8)) * 8;
        }
        if let Some(file) = chars.next() {
            sq.0 += ((file as u8) - ('a' as u8));
        }
        sq
    }
}

pub type Rank = u8;

pub type File = u8;


pub struct Move(u16);

impl Move {
    const SHIFT_FROM: i32 = 0;
    const SHIFT_TO: i32 = 6;
    const SHIFT_FLAG: i32 = 12;

    const MASK_FROM: i32 = 0b111111 << Move::SHIFT_FROM;
    const MASK_TO: i32 = 0b111111 << Move::SHIFT_TO;
    const MASK_FLAG: i32 = 0b1111 << Move::SHIFT_FLAG;

    const FLAG_QUIET_MOVE: i32 = 0;
    const FLAG_DOUBLE_PAWN_PUSH: i32 = 1;
    const FLAG_PROMOTION_KNIGHT: i32 = 2;
    const FLAG_PROMOTION_BISHOP: i32 = 3;
    const FLAG_PROMOTION_ROOK: i32 = 4;
    const FLAG_PROMOTION_QUEEN: i32 = 5;
    const FLAG_CAPTURE_PROMOTION_KNIGHT: i32 = 6;
    const FLAG_CAPTURE_PROMOTION_BISHOP: i32 = 7;
    const FLAG_CAPTURE_PROMOTION_ROOK: i32 = 8;
    const FLAG_CAPTURE_PROMOTION_QUEEN: i32 = 9;
    const FLAG_KING_CASTLE: i32 = 10;
    const FLAG_QUEEN_CASTLE: i32 = 11;
    const FLAG_CAPTURE: i32 = 12;
    const FLAG_EN_PASSANT: i32 = 13;
    

}

#[derive(Debug)]
pub enum MoveError {
    InvalidFormat,
}
impl Error for MoveError {}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            MoveError::InvalidFormat => write!(f, "Invalid move format"),
        }
    }
}

// impl From<String> for Move {
//     fn from(value: String) -> Self {
//         todo!()
//     }
// }

impl TryFrom<String> for Move {
    type Error = MoveError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        todo!()        
    }
}
