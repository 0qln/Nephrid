use super::{r#move::Move, position::Position};

pub mod pawn;
pub mod knight;
pub mod bishop;
pub mod rook;
pub mod queen;
pub mod king;

pub fn gen_plegals<'a>(pos: &'a Position) -> impl Iterator<Item = Move> + 'a {
    let moves: &[Move] = &[

    ];
}

pub fn legal_moves<'a>(pos: &'a Position) -> impl Iterator<Item = Move> + 'a {
    todo!()
}