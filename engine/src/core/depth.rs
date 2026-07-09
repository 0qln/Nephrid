use std::{cmp::min, fmt, iter::Step, num::ParseIntError, ops, str::FromStr};

use crate::core::{ply::Ply, search::mcts::node::Height};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Depth {
    v: u8,
}

impl Step for Depth {
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) { Step::steps_between(&start.v, &end.v) }

    fn forward_checked(start: Self, count: usize) -> Option<Self> { Self::try_from(Step::forward_checked(start.v, count)?).ok() }

    fn backward_checked(start: Self, count: usize) -> Option<Self> { Self::try_from(Step::backward_checked(start.v, count)?).ok() }
}

impl fmt::Display for Depth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.v) }
}

impl_op!(-|a: Depth, b: u8| -> Depth { Depth { v: a.v - b } });
impl_op!(+|a: Depth, b: u8| -> Depth { Depth { v: a.v + b } });
impl_op!(+|a: Depth, b: Depth| -> Depth { Depth { v: a.v + b.v } });
impl_op!(+=|a: &mut Depth, b: u8| { a.v += b });
impl_op!(-=|a: &mut Depth, b: u8| { a.v -= b });
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

    fn from_str(value: &str) -> Result<Self, Self::Err> { Depth::try_from(value) }
}

impl Depth {
    pub const ROOT: Depth = Depth { v: 0 };
    pub const MAX: Depth = Depth { v: 250 };
    pub const NONE: Depth = Depth { v: 255 };
    pub const QS: Depth = Depth { v: 254 };

    pub const fn v(&self) -> u8 { self.v }

    pub const fn index(&self) -> usize { self.v as usize }

    pub const fn new(depth: u8) -> Depth { Depth { v: depth } }

    pub const fn saturating_sub(&self, rhs: u8) -> Depth { Self { v: self.v.saturating_sub(rhs) } }
    pub const fn div_floor(&self, rhs: u8) -> Depth { Self { v: self.v.div_floor(rhs) } }
}

impl TryFrom<u8> for Depth {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if value <= Self::MAX.v {
            Ok(Depth { v: value })
        }
        else {
            Err(())
        }
    }
}

impl Default for Depth {
    fn default() -> Self { Depth::NONE }
}

impl From<Ply> for Depth {
    fn from(ply: Ply) -> Self { Depth { v: ply.v as u8 } }
}

impl From<Height> for Depth {
    fn from(value: Height) -> Self {
        let v = value.0.saturating_sub(1);
        let cap = Self::MAX.v as u16;
        Depth { v: min(v, cap) as u8 }
    }
}
