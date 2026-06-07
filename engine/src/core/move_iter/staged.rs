use std::ops::ControlFlow;

use crate::{
    core::{
        r#move::{MAX_UNREACHABLE_MOVES, Move},
        move_iter::{self, fold_moves},
        position::Position,
        search::ordering::{MoveScorer, ScoredMove},
    },
    misc::List,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtStage {
    HashMove,
    // todo: promotions
    // GoodCaptures,
    // Killers,
    // BadCaptures,
    // Quiets,
    AllLegal,
    Done,
}

pub const trait Stage {
    fn stage() -> RtStage;
}
pub mod stages {
    use super::*;

    pub struct HashMove;
    impl Stage for HashMove {
        fn stage() -> RtStage {
            RtStage::HashMove
        }
    }

    pub struct AllLegal;
    impl Stage for AllLegal {
        fn stage() -> RtStage {
            RtStage::AllLegal
        }
    }
}

impl RtStage {
    fn next(&mut self) {
        *self = match self {
            RtStage::HashMove => RtStage::AllLegal,
            RtStage::AllLegal => RtStage::Done,
            RtStage::Done => panic!("No more stages"),
            // Stage::HashMove => Stage::GoodCaptures,
            // Stage::GoodCaptures => Stage::Killers,
            // Stage::Killers => Stage::BadCaptures,
            // Stage::BadCaptures => Stage::Quiets,
            // Stage::Quiets => panic!("No more stages"),
        }
    }
}

#[derive(Debug)]
pub struct MoveGenerator {
    stage: RtStage,
    hash_move: Move,
}

pub struct MoveGenExhausted;

impl MoveGenerator {
    pub fn new_with_hashmove(hash_move: Move) -> Self {
        Self {
            stage: RtStage::HashMove,
            hash_move,
        }
    }

    pub fn new_in_stage(stage: RtStage) -> Self {
        Self { stage, hash_move: Move::null() }
    }

    // todo: it would be preferred to not have the move iter options here as a
    // generic argument. this is currently required for qsearch to pass it's
    // custom gen options, but might not be required any more if we in the
    // future introduce stages to generate good captures / bad captures /
    // promotions etc.
    //
    /// Pushes the next stage of moves into the list.
    pub fn next<O: move_iter::Options>(
        &mut self,
        pos: &Position,
        scorer: &impl MoveScorer,
        list: &mut List<{ MAX_UNREACHABLE_MOVES }, ScoredMove>,
    ) -> Result<(), MoveGenExhausted> {
        match self.stage {
            RtStage::HashMove => {
                if pos.is_legal(self.hash_move) && self.hash_move != Move::null() {
                    list.push(ScoredMove::new(
                        self.hash_move,
                        scorer.score::<stages::HashMove>(pos, self.hash_move),
                    ));
                }
            }
            RtStage::AllLegal => {
                let start = list.len();

                _ = fold_moves::<O, _, _, _>(pos, (), |_, m| {
                    // todo: this could be slow... maybe check this somewhere else?
                    if m != self.hash_move {
                        list.push(ScoredMove::new(m, 0));
                    }
                    ControlFlow::Continue::<(), _>(())
                });

                // todo: it would make more sense to perform the sorting in the ordering
                // module... but idk cause here we have the info for sorting efficiently...

                // generate the see score outside of the move generation and the sorting, such
                // that it isn't computed for each comparison and we don't distrurb cache
                // locality.
                for &mut ScoredMove { mov, ref mut score } in list.as_mut_subslice(start..) {
                    *score = scorer.score::<stages::AllLegal>(pos, mov);
                }
            }
            RtStage::Done => {
                return Err(MoveGenExhausted);
            }
        }

        self.stage.next();

        Ok(())
    }
}
