use std::iter::Peekable;
use std::str::Chars;

pub struct Tokenizer<'chars>(Peekable<Chars<'chars>>);

impl<'chars> Tokenizer<'chars> {

    pub fn new(input: &'chars str) -> Self {
        Tokenizer(input.chars().peekable())
    }

    pub fn collect_ws(&mut self) {
        while self.0.next_if(|&c| c.is_whitespace()).is_some() {}
    }

    pub fn collect_until_ws(&mut self) -> Option<String> {
        self.collect_ws();
        let mut buffer = String::new();
        while let Some(c) = self.next_char_not_ws() {
            buffer.push(c);
        }

        if buffer.is_empty() { None } else { Some(buffer) }
    }
   
    pub fn collect_bool(&mut self) -> bool {
        self.collect_until_ws().is_some_and(|s| s == "true")
    }

    pub fn next_char(&mut self) -> Option<char> {
        self.0.next()
    }

    pub fn next_char_not_ws(&mut self) -> Option<char> {
        self.0.next_if(|&c| !c.is_whitespace())
    }
}

pub fn tokenize(input: &str) -> Tokenizer {
    Tokenizer(input.chars().peekable())
}
