use crate::core::{
    color::{Color, Perspective},
    coordinates::EpTargetSquare,
    eval::hce::TaperValue,
    position::{PieceInfo, PieceInfoObserver},
    search::score::Score,
    turn::Turn,
};

pub mod hce;
pub mod nnue;

pub trait StaticEvaluator {
    fn eval<P: Perspective>(&self, pos: &PieceInfo, turn: Turn, ep_sq: EpTargetSquare, phase: TaperValue) -> Score<P>;

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
