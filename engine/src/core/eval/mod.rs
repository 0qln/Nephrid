use crate::core::{
    color::{Color, Perspective},
    coordinates::EpTargetSquare,
    eval::hce::TaperValue,
    position::PieceInfo,
    search::score::Score,
    turn::Turn,
};

pub mod hce;
pub mod nnue;

pub trait StaticEvaluator {
    fn eval<P: Perspective>(&self, pos: &PieceInfo, turn: Turn, ep_sq: EpTargetSquare, phase: TaperValue) -> Score<P>;
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GameResult {
    Win { relative_to: Color },
    Draw,
}
