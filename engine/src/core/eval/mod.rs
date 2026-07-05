use core::fmt;
use std::ops::Deref;

use crate::core::{
    color::{Color, Perspective}, config::Configuration, coordinates::EpTargetSquare, eval::hce::TaperValue, position::{PieceInfo, PieceInfoObserver}, search::score::Score, turn::Turn
};

pub mod hce;
pub mod nnue;

pub trait StaticEvaluator: Sized {
    fn eval<P: Perspective>(&self, pos: &PieceInfo, turn: Turn, ep_sq: EpTargetSquare, phase: TaperValue) -> Score<P>;

    fn try_from_config<C: Deref<Target = Configuration>>(config: C) -> Result<Self, impl fmt::Display>;

    #[allow(static_mut_refs)]
    fn observe(&mut self) -> &mut impl PieceInfoObserver {
        static mut NULL_OBSERVER: () = ();
        unsafe { &mut NULL_OBSERVER }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GameResult {
    Win { relative_to: Color },
    Draw,
}
