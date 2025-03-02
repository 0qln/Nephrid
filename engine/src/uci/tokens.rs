use std::iter::{Enumerate, Peekable};
use std::str::Chars;

pub struct Tokenizer<'a> {
    src: &'a str,
    seq: Peekable<Enumerate<Chars<'a>>>,
}

impl<'a> Tokenizer<'a> {
    pub fn new(v: &'a str) -> Self {
        Self {
            src: v,
            seq: v.chars().enumerate().peekable(),
        }
    }

    pub fn skip_ws(&mut self) -> &mut Self {
        while self.seq.next_if(|&c| c.1.is_whitespace()).is_some() {}
        self
    }

    pub fn next_token(&mut self) -> Option<&'a str> {
        let start = self.skip_ws().seq.peek()?.0;
        let end = self.chars_with_index().last().map_or(start, |c| c.0);
        Some(&self.src[start..=end])
    }

    pub fn tokens(&mut self) -> TokenIterator<'_, 'a> {
        TokenIterator(self)
    }

    pub fn next_char(&mut self) -> Option<char> {
        self.next_char_with_index().map(|c| c.1)
    }

    pub fn chars(&mut self) -> CharIterator<'_, 'a> {
        CharIterator(self)
    }
    
    pub fn next_char_with_index(&mut self) -> Option<(usize, char)> {
        self.seq.next_if(|&c| !c.1.is_whitespace())
    }
    
    pub fn chars_with_index(&mut self) -> CharsWithIndexIterator<'_, 'a> {
        CharsWithIndexIterator(self)
    }
}

pub struct CharIterator<'a, 'b>(&'a mut Tokenizer<'b>);

impl Iterator for CharIterator<'_, '_> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next_char()
    }
}

pub struct CharsWithIndexIterator<'a, 'b>(&'a mut Tokenizer<'b>);

impl Iterator for CharsWithIndexIterator<'_, '_> {
    type Item = (usize, char);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next_char_with_index()
    }
}

pub struct TokenIterator<'a, 'b>(&'a mut Tokenizer<'b>);

impl<'b> Iterator for TokenIterator<'_, 'b> {
    type Item = &'b str;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next_token()
    }
}
