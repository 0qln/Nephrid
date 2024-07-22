use super::{Position, Move};

pub struct MoveIter;

impl MoveIter {

}

impl Iterator for MoveIter {
    type Item = Move;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

pub fn legal(position: &mut Position) -> impl IntoIterator<Item=Move, IntoIter=MoveIter> {
    MoveIter
}

pub fn captures(position: &mut Position) {

}

