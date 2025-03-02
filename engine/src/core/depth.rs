use std::ops;
use std::{num::ParseIntError, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Depth {
    v: u8,
}

impl_op!(-|a: Depth, b: u8| -> Depth { Depth { v: a.v - b } });

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
    pub const MIN: Depth = Depth { v: 0 };
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
