use crate::engine::color::Color;
use std::ops;

#[derive(Default, Clone)]
pub struct Ply { v: u16 }

impl Ply {
    pub fn new(fmc: u16, turn: Color) -> Self {
        match turn {
            Color::White => Self { v: 2 * fmc },
            Color::Black => Self { v: 2 * fmc + 1 },
        }       
    }
}

impl From<u16> for Ply {
    fn from(value: u16) -> Self {
        Self { v: value }
    }
}

impl_op!(+ |l: Ply, r: Ply| -> Ply { l + r });
impl_op!(- |l: Ply, r: Ply| -> Ply { l - r });
impl_op!(- |l: Ply, r: u32| -> Ply { l - r });