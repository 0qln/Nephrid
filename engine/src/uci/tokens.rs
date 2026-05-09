use std::{
    iter::{Enumerate, Peekable},
    str::Chars,
};

pub struct Peeked<'a, 'b>(&'a mut Tokenizer<'b>, &'a str);

impl<'a, 'b> Peeked<'a, 'b> {
    pub fn val(&self) -> &'a str {
        self.1
    }

    pub fn consume(self) {
        for _ in 0..self.1.len() {
            self.0.seq.next().expect(
                "Tokenizer state is inconsistent: peeked token length exceeds remaining \
                 characters.",
            );
        }
    }
}

#[derive(Clone)]
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
        let token = &self.src[start..=end];
        Some(token)
    }

    pub fn peek_next_token<'b>(&'b mut self) -> Option<Peeked<'b, 'a>> {
        let start = self.skip_ws().seq.peek()?.0;
        let end = self.src[start..]
            .chars()
            .enumerate()
            .take_while(|c| !c.1.is_whitespace())
            .last()
            .unwrap()
            .0;
        let token = &self.src[start..=(start + end)];
        Some(Peeked(self, token))
    }

    pub fn tokens(&mut self) -> TokenIterator<'_, 'a> {
        TokenIterator(self)
    }

    pub fn next_char(&mut self) -> Option<char> {
        self.next_char_with_index().map(|c| c.1)
    }

    pub fn consume_next_char(&mut self) {
        self.next_char_with_index();
    }

    pub fn has_next_char(&mut self) -> bool {
        self.peek_next_char().is_some()
    }

    pub fn peek_next_char(&mut self) -> Option<char> {
        self.peek_next_char_with_index().map(|c| c.1)
    }

    pub fn chars(&mut self) -> CharIterator<'_, 'a> {
        CharIterator(self)
    }

    pub fn next_char_with_index(&mut self) -> Option<(usize, char)> {
        self.seq.next_if(|&c| !c.1.is_whitespace())
    }

    pub fn peek_next_char_with_index(&mut self) -> Option<&(usize, char)> {
        self.seq.peek()
    }

    pub fn chars_with_index(&mut self) -> impl Iterator<Item = (usize, char)> {
        pub struct CharsWithIndexIterator<'a, 'b>(&'a mut Tokenizer<'b>);

        impl Iterator for CharsWithIndexIterator<'_, '_> {
            type Item = (usize, char);

            fn next(&mut self) -> Option<Self::Item> {
                self.0.next_char_with_index()
            }
        }

        CharsWithIndexIterator(self)
    }

    /// Unconditionally consumes and returns the next character.
    pub fn consume_char(&mut self) -> Option<char> {
        self.seq.next().map(|(_, c)| c)
    }

    /// Reads characters into a slice until the delimiter is found, consuming
    /// the delimiter.
    pub fn take_until(&mut self, delimiter: char) -> Option<&'a str> {
        let start = self.seq.peek()?.0;

        for (idx, c) in self.seq.by_ref() {
            if c == delimiter {
                return Some(&self.src[start..idx]);
            }
        }
        None
    }
}

pub struct CharIterator<'a, 'b>(&'a mut Tokenizer<'b>);

impl Iterator for CharIterator<'_, '_> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next_char()
    }
}

pub struct TokenIterator<'a, 'b>(&'a mut Tokenizer<'b>);

impl<'b> Iterator for TokenIterator<'_, 'b> {
    type Item = &'b str;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next_token()
    }
}
