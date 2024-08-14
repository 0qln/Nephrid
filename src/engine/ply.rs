use crate::engine::color::Color;


#[derive(Default)]
pub struct Ply { pub v: u16 }

impl Ply {
    pub fn new(fmc: u16, turn: Color) -> Self {
        match turn {
            Color::White => Self { v: 2 * fmc - 1 },
            Color::Black => Self { v: 2 * fmc },
        }       
    }
}
