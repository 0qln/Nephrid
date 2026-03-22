use std::{cmp::max, fmt, num::ParseIntError, ops, str::FromStr};

use crate::core::{ply::Ply, search::mcts::node::Height};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Depth {
    v: u8,
}

impl fmt::Display for Depth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.v)
    }
}

impl_op!(-|a: Depth, b: u8| -> Depth { Depth { v: a.v - b } });
impl_op!(+|a: Depth, b: u8| -> Depth { Depth { v: a.v + b } });
impl_op!(+|a: Depth, b: Depth| -> Depth { Depth { v: a.v + b.v } });
impl_op!(+=|a: &mut Depth, b: u8| { a.v += b });
impl_op!(-|a: Depth, b: Depth| -> Depth { Depth { v: a.v - b.v } });

impl TryFrom<&str> for Depth {
    type Error = ParseIntError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.parse() {
            Ok(depth) => Ok(Depth { v: depth }),
            Err(e) => Err(e),
        }
    }
}

impl FromStr for Depth {
    type Err = ParseIntError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Depth::try_from(value)
    }
}

impl Depth {
    pub const ROOT: Depth = Depth { v: 0 };
    pub const MAX: Depth = Depth { v: 250 };
    pub const NONE: Depth = Depth { v: 255 };

    pub const fn v(&self) -> u8 {
        self.v
    }

    pub fn new(depth: u8) -> Depth {
        Depth { v: depth }
    }
}

impl Default for Depth {
    fn default() -> Self {
        Depth::NONE
    }
}

impl From<Ply> for Depth {
    fn from(ply: Ply) -> Self {
        Depth { v: ply.v as u8 }
    }
}

impl From<Height> for Depth {
    fn from(value: Height) -> Self {
        let v = value.0.saturating_add(1);
        let cap = Self::MAX.v as u16;
        Depth { v: max(v, cap) as u8 }
    }
}
