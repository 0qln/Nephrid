use core::fmt;
use std::time::Duration;

use crate::core::{depth::Depth, r#move::Move, search::score::Cp};

#[derive(Debug)]
pub struct UciCp(pub Cp);

impl fmt::Display for UciCp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "cp {}", self.0) }
}

#[derive(Debug)]
pub enum UciScore {
    Mate(i32),
    Centipawns(UciCp),
    LowerBound(UciCp),
    UpperBound(UciCp),
}

impl fmt::Display for UciScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mate(mate) => write!(f, "score mate {mate}"),
            Self::Centipawns(cp) => write!(f, "score {cp}"),
            Self::LowerBound(cp) => write!(f, "score {cp} lowerbound"),
            Self::UpperBound(cp) => write!(f, "score {cp} upperbound"),
        }
    }
}

#[derive(Default, Debug)]
pub struct UciNodes(pub usize);

impl fmt::Display for UciNodes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "nodes {}", self.0) }
}

#[derive(Default, Debug)]
pub struct UciNps(pub u128);

impl UciNps {
    pub fn from_nodes_and_time(nodes: u64, time: Duration) -> Self {
        let nps = if time.as_nanos() > 0 {
            nodes as u128 * 1_000_000_000 / time.as_nanos()
        }
        else {
            0
        };
        Self(nps)
    }
}

impl fmt::Display for UciNps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "nps {}", self.0) }
}

#[derive(Default, Debug)]
pub struct UciPondermove(pub Move);

impl fmt::Display for UciPondermove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "ponder {}", self.0) }
}

#[derive(Default, Debug)]
pub struct UciDepth(pub Depth);

impl fmt::Display for UciDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "depth {}", self.0) }
}

#[derive(Default, Debug)]
pub struct UciSeldepth(pub Depth);

impl fmt::Display for UciSeldepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "seldepth {}", self.0) }
}

pub struct UciPv<'a, Path>(pub &'a Path);

impl<P> fmt::Display for UciPv<'_, P>
where
    for<'a> &'a P: IntoIterator<Item = Move>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pv")?;
        for mov in self.0 {
            write!(f, " {mov}")?;
        }
        Ok(())
    }
}

#[derive(Default, Debug)]
pub struct UciSearchtime(pub Duration);

impl fmt::Display for UciSearchtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "time {}", self.0.as_millis()) }
}

#[derive(Default, Debug)]
pub struct UciCurrmove(pub Move);

impl fmt::Display for UciCurrmove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "currmove {}", self.0) }
}

pub enum UciArg<T: fmt::Display> {
    None,
    Some(T),
}

impl<T: fmt::Display> fmt::Display for UciArg<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Self::Some(arg) = &self {
            write!(f, " {}", arg)
        }
        else {
            Ok(())
        }
    }
}

impl<T: fmt::Display> From<Option<T>> for UciArg<T> {
    fn from(opt: Option<T>) -> Self {
        if let Some(arg) = opt {
            Self::Some(arg)
        }
        else {
            Self::None
        }
    }
}
