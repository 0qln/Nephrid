use std::iter::Peekable;
use std::str::Chars;

pub struct Tokenizer<'chr> { 
    v: Peekable<Chars<'chr>>, 
}

impl Iterator for Tokenizer<'_> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.v.next_if(|&c| !c.is_whitespace())
    }
} 

impl<'chr> Tokenizer<'chr> 
{
    pub fn new(v: &'chr str) -> Self {
        Self { v: v.chars().peekable() }
    }

    fn skip_ws(&mut self) {
        while self.v.next_if(|&c| c.is_whitespace()).is_some() {}
    }

    pub fn collect_token(&mut self) -> Option<String> {
        self.skip_ws();
        self.v.peek()?;
        Some(self.collect::<String>())
    }

    pub fn iter_token(&mut self) -> &mut Self {
        self.skip_ws();
        self
    }

    pub fn goto_next_token(&mut self) -> bool {
        self.skip_ws();
        self.v.peek().is_some()
    }
}

