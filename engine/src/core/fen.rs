use crate::uci::tokens::Tokenizer;

pub type Fen<'a> = Tokenizer<'a>;

// pub struct Fen<'fen, 'tok> { 
//     v: &'fen mut Tokenizer<'tok>
// }

// impl<'fen, 'tok> Fen<'fen, 'tok> {
//     pub fn new(v: &'fen mut Tokenizer<'tok>) -> Self {
//         Self { v }
//     }
//
//     pub fn next_part<'a, 'b>(&'a mut self) -> TokenIterator<'b, collecting::UntilWs> {
//     }
// }

