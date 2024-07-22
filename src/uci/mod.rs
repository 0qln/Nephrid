use std::{io, iter::Peekable, str::Chars};

pub struct Tokenizer<'a>(Peekable<Chars<'a>>);

impl<'a> Tokenizer<'a> {

    pub fn new(input: &'a str) -> Self {
        Tokenizer(input.chars().peekable())
    }

    pub fn collect_ws(&mut self) {
        while self.0.next_if(|&c| c.is_whitespace()).is_some() {}
    }

    pub fn collect_until_ws(&mut self) -> Option<String> {
        self.collect_ws();
        let mut buffer = String::new();
        while let Some(c) = self.0.next_if(|&c| !c.is_whitespace()) {
            buffer.push(c);
        }

        if buffer.is_empty() { None } else { Some(buffer) }
    }
    
    // pub fn collect_until_ws_while<'f, F>(&mut self, predicate: &'f F) -> impl Iterator<Item = String> + '_
    // where
    //     F: FnMut(&str) -> bool + 'f,
    // {
    //     std::iter::from_fn(move || {
    //         let token = self.collect_until_ws()?;
    //         if predicate(&token) {
    //             Some(token)
    //         } else {
    //             None
    //         }
    //     })
    // }

    pub fn collect_bool(&mut self) -> bool {
        self.collect_until_ws().is_some_and(|s| s == "true")
    }

    // pub fn collect_move_LAN(&mut self, first_char: char, context: Position) -> Move {
    //
    // }
    //
    // pub fn collect_move_SAN(&mut self, first_char: char, context: Position) -> Move {
    //
    // }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        return  self.collect_until_ws();
    }
}

pub fn tokenize(input: &str) -> Tokenizer {
    Tokenizer(input.chars().peekable())
}

/// Thread safe CLI output.
pub fn out(msg: &str) {
    let stdout = io::stdout();
    let _ = writeln!(&mut stdout.lock(), "{}", msg);
}

