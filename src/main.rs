use std::{io::stdin, iter::Peekable, str::Chars};

fn main() {

    assert_eq!(Square::from(Squares::A1), Square::from((0, 0)));

    let input_stream = stdin();
    let engine = Engine::default();
    loop {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => {
                let cmd = UciCommand::parse(input);
                engine.execute(cmd);
            }
            Err(err) => println!("Error: {err}"),
        }
    }
}

struct Tokenizer<'a>(Peekable<Chars<'a>>);

impl<'a> Tokenizer<'a> {
    fn collect_ws(&mut self) {
        while let Some(&c) = self.0.peek() {
            if !c.is_whitespace() {
                break;
            }
            self.0.next();
        }
    }

    fn collect_until_ws(&mut self) -> Option<String> {
        self.collect_ws();
        let mut buffer = String::new();
        while let Some(&c) = self.0.peek() {
            if c.is_whitespace() {
                break;
            }
            buffer.push(c);
            self.0.next();
        }
        if buffer.is_empty() {
            None
        } else {
            Some(buffer)
        }
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

// impl<'a> Iterator for Tokenizer<'a> {
//     type Item = String;
//
//     fn next(&mut self) -> Option<String> {
//         let c = self.0.next();
//     }
// }

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
        searchmoves: Vec<String>,
        wtime: u64,
        btime: u64,
        winc: u64,
        binc: u64,
        movestogo: u16,
        depth: u8,
        nodes: u64,
        mate: u64,
        movetime: u64,
        infinite: bool,
    },
    Stop,
    PonderHit,
    Quit,
    Unknown(Box<str>),
}

impl UciCommand {
    pub fn parse(input: String) -> Self {
        let mut tokenizer: Tokenizer = tokenize(input.as_str());
        match tokenizer
            .collect_until_ws()
            .unwrap_or(String::new())
            .as_str()
        {
            "go" => UciCommand::parse_go(&mut tokenizer),
            "uci" => UciCommand::Uci,
            "debug" => UciCommand::Debug(tokenizer.collect_bool()),
            "isready" => UciCommand::IsReady,
            "position" => UciCommand::parse_position(&mut tokenizer),
            "register" => UciCommand::Register,
            "setoption" => UciCommand::parse_setoption(&mut tokenizer),
            "ucinewgame" => UciCommand::UciNewGame,
            unknown => UciCommand::Unknown(Box::from(unknown)),
        }
    }

    fn parse_setoption(tokenizer: &mut Tokenizer) -> Self {
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
        let mut value = String::new();
        while let Some(token) = tokenizer.collect_until_ws() {
            value.push_str(&token);
            // Reintroduce spaces between parts
            if !value.is_empty() {
                value.push(' ')
            }
        }

        Self::SetOption {
            name,
            value: if value.is_empty() { None } else { Some(value) },
        }
    }

    fn parse_go(tokenizer: &mut Tokenizer) -> Self {
        let mut ponder = false;
        let mut searchmoves = Vec::new();
        let mut wtime = 0;
        let mut btime = 0;
        let mut winc = 0;
        let mut binc = 0;
        let mut movestogo = 0;
        let mut depth = 0;
        let mut nodes = 0;
        let mut mate = 0;
        let mut movetime = 0;
        let mut infinite = false;

        while let Some(token) = tokenizer.collect_until_ws() {
            match token.as_str() {
                "ponder" => ponder = true,
                "wtime" => {
                    wtime = tokenizer
                        .collect_until_ws()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0)
                }
                "btime" => {
                    btime = tokenizer
                        .collect_until_ws()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0)
                }
                "winc" => {
                    winc = tokenizer
                        .collect_until_ws()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0)
                }
                "binc" => {
                    binc = tokenizer
                        .collect_until_ws()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0)
                }
                "movestogo" => {
                    movestogo = tokenizer
                        .collect_until_ws()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0)
                }
                "depth" => {
                    depth = tokenizer
                        .collect_until_ws()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0)
                }
                "nodes" => {
                    nodes = tokenizer
                        .collect_until_ws()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0)
                }
                "mate" => {
                    mate = tokenizer
                        .collect_until_ws()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0)
                }
                "movetime" => {
                    movetime = tokenizer
                        .collect_until_ws()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0)
                }
                "infinite" => infinite = true,
                _ => searchmoves.push(token),
            };
        }

        Self::Go {
            ponder,
            searchmoves,
            wtime,
            btime,
            winc,
            binc,
            movestogo,
            depth,
            nodes,
            mate,
            movetime,
            infinite,
        }
    }

    fn parse_position(tokenizer: &mut Tokenizer) -> Self {
        todo!()
    }
}

pub enum UciResponse {
    Id(String, String),
    UciOk,
    ReadyOk,
    BestMove(String, Option<String>),
    CopyProtection(bool),
    Registration(bool),
    Info(InfoResponse),
    Option(),
}

impl UciResponse {
    // TODO: implement with mutex
    fn send(response: UciResponse) {}
}

pub enum InfoResponse {
    CurrLine(Option<u8>, Box<Vec<Move>>),
}

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
    },
    Button {
        callback: Box<dyn Fn()>,
    },
    String(Option<String>), // Some -> "value", None -> "<empty>"
}

pub struct ConfigOption {
    cfg_name: String,
    cfg_type: ConfigOptionType,
}

pub struct Engine {
    config: Vec<ConfigOption>,
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
                        max: 33554432,
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
        }
    }

    pub fn execute(&self, command: UciCommand) {}
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
    
    // pub fn new() -> Self {
    //     Self(
    //         
    //     )
    // }

}

impl From<String> for Move {
    fn from(value: String) -> Self {
        todo!()
    }
}
