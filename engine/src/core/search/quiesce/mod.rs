use crate::core::{
    color::{Color, Perspective},
    depth::Depth,
    eval::{
        StaticEvaluator,
        hce::{TaperValue, piece_score},
    },
    r#move::Move,
    piece::{PromoPieceType, piece_type},
    position::{CheckState, Position},
    search::{
        data::{ReplacementStrategy, TTIsValid, TTKey, TTMove, TTStaticEval, TranspositionTable},
        id::RbSet,
        ordering::{self, MovePicker, MoveScore, RtStage, Stage},
        score::{AnyScore, Score, scores},
    },
};

pub const trait QSearchParams {
    fn futility_margin(&self) -> AnyScore;
    fn delta_pruning_threshold(&self) -> TaperValue;
}

pub type TT<Data, Strat: ReplacementStrategy<Data = Data>> = TranspositionTable<Data, Strat>;

pub struct QSearcher<'a, Entry, Replace> {
    tt: &'a mut TT<Entry, Replace>,
}

impl<'a, E, R> QSearcher<'a, E, R> {
    pub fn new(tt: &'a mut TT<E, R>) -> Self { Self { tt } }
}

impl<'a, E: TTKey + TTMove + TTIsValid + TTStaticEval + Default + Clone, R: ReplacementStrategy<Data = E>> QSearcher<'a, E, R> {
    /// # Q-Search
    ///
    /// Make the position quiet.
    ///
    /// [q-search](https://www.chessprogramming.org/Quiescence_Search)
    pub fn go<P: Perspective>(
        &mut self,
        pos: &mut Position,
        mut alpha: Score<P>,
        beta: Score<P>,
        params: impl QSearchParams + Clone,
        eval: &mut impl StaticEvaluator,
        depth: Depth,
    ) -> Score<P> {
        let mut best_value = Score::new(scores::NEG_INF);

        let in_check = pos.get_check_state() != CheckState::None;
        let stm = P::COLOR;
        let piece_info = pos.piece_info();
        let phase = TaperValue::from_position(piece_info);
        let key = pos.get_key();

        let mut static_eval = |this: &mut Self| {
            let tt_score = this.tt.raw_mut(key).and_then(|e| e.static_eval_mut().validated_mut());

            // try fetching from tt in case there's already a valid entry here, don't overwrite
            if let Some(valid_score) = tt_score {
                // Safety: we can interpret it as a Score<P> without worrying about perspective for
                // most positions, unless there is a zobrist hash key collision.
                return unsafe { valid_score.interpret_as() }
            }

            // else compute
            let score = eval.eval(piece_info, stm, pos.get_ep_target_square(), phase);

            this.tt.get_mut(key);

            score
        };

        if depth == Depth::new(0) {

            return static_eval();
        }
        let tt_move = tt_entry.and_then(|entry| entry.tt_move()).unwrap_or(Move::null());

        // stand pad if not in check
        if !in_check {
            best_value = static_eval();

            if best_value >= beta {
                return best_value;
            }
            if best_value > alpha {
                alpha = best_value;
            }
        }

        // move gen
        let scorer = MoveScorer { color: P::COLOR, phase };
        let mut move_picker = MovePicker::new_with_max_stage(
            tt_move,
            RbSet::<Move, 2>::default(),
            if in_check {
                RtStage::Done
            }
            else {
                RtStage::YieldBadCaptures
            },
        );

        // recurse
        while let Some(m) = move_picker.next_for::<P>(pos, &scorer) {
            // if !pos.is_legal_for::<P>(m) {
            //     continue;
            // }

            // delta pruning
            if !in_check && phase < params.delta_pruning_threshold() {
                let value_bonus = PromoPieceType::try_from(m.get_flag())
                    .ok()
                    .map(|promo| piece_score(promo.into()) - piece_score(piece_type::PAWN))
                    .unwrap_or(0);

                let captured_value = m
                    .get_capture_sq()
                    .map(|capt_sq| piece_score(pos.get_piece(capt_sq).piece_type()))
                    .unwrap_or(0);

                let futility_margin = params.futility_margin();
                let futility_score = captured_value + value_bonus + futility_margin;

                if best_value + Score::new(futility_score) < alpha {
                    continue;
                }
            }

            pos.make_move_for::<P>(m, eval.observe());

            let score = !self.go(pos, !beta, !alpha, params.clone(), eval, depth - 1);

            pos.unmake_move_for::<P>(m, eval.observe());

            if score >= beta {
                return score;
            }
            if score > best_value {
                best_value = score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        best_value
    }
}

struct MoveScorer {
    color: Color,
    phase: TaperValue,
}
impl ordering::MoveScorer for MoveScorer {
    fn score<S: Stage>(&self, pos: &Position, mov: Move) -> MoveScore {
        match S::stage() {
            ordering::RtStage::YieldHashMove => {
                todo!("we don't yet have a hashmove in qsearch")
            }
            ordering::RtStage::GenerateCapturesAndPromos | ordering::RtStage::YieldGoodCapturesAndPromos | ordering::RtStage::YieldBadCaptures => {
                let pieces = pos.piece_info();

                let (from, to, _) = mov.into();
                let piece = pieces.get_piece(from);

                ordering::see(pieces, mov, self.color) + ordering::psqt(self.phase, piece.piece_type(), from, to, mov.get_flag(), self.color)
            }
            ordering::RtStage::YieldKillers => todo!("we don't yet have killers in qsearch"),
            ordering::RtStage::GenerateQuiets | ordering::RtStage::YieldQuiets => {
                debug_assert!(
                    pos.get_check_state() != CheckState::None,
                    "we should never be generating quiets in qsearch if we're not in check"
                );
                let pieces = pos.piece_info();
                let (from, to, _) = mov.into();
                let piece = pieces.get_piece(from);
                ordering::psqt(self.phase, piece.piece_type(), from, to, mov.get_flag(), self.color)
            }
            ordering::RtStage::Done => todo!("why are we scoring Done??"),
        }
    }
}
