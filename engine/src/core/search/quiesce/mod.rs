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
        data::{ReplacementStrategy, TTKey, TTMove, TTStaticEval, TranspositionTable},
        id::RbSet,
        ordering::{self, MovePicker, MoveScore, RtStage, Stage},
        score::{AnyScore, Score, scores},
    },
};

pub const trait QSearchParams {
    fn futility_margin(&self) -> AnyScore;
    fn delta_pruning_threshold(&self) -> TaperValue;
}

pub type TT<Data, Strat> = TranspositionTable<Data, Strat>;

pub struct QSearcher<'a, Entry, Replace> {
    tt: &'a mut TT<Entry, Replace>,
}

impl<'a, E, R> QSearcher<'a, E, R> {
    pub fn new(tt: &'a mut TT<E, R>) -> Self { Self { tt } }
}

impl<'a, E: TTKey + TTMove + TTStaticEval + Clone, R: ReplacementStrategy<Data = E>> QSearcher<'a, E, R> {
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
        let piece_info = pos.piece_info();
        let phase = TaperValue::from_position(piece_info);
        let key = pos.get_key();

        let mut static_eval = Score::<P>::new(scores::NULL);
        let mut lazy_static_eval = |this: &mut Self, pos: &Position| {
            // is it already computed? if so, return it.
            if static_eval.0.validated().is_some() {
                return static_eval;
            }

            let mut compute = || eval.eval(pos.piece_info(), P::COLOR, pos.get_ep_target_square(), phase);

            // check the tt
            if let Some(tt_entry) = this.tt.raw_mut(key) {
                let tt_score = tt_entry.static_eval_mut();
                // if the tt contains a valid static_eval, return it.
                if tt_score.validated().is_some() {
                    static_eval = unsafe { tt_score.interpret_as() };
                }
                // else compute it
                else {
                    static_eval = compute();
                    *tt_score = static_eval.0;
                }
            }
            // if theres a foreign tt entry blocking our current key, just compute and don't store.
            else {
                static_eval = compute();
            }

            static_eval
        };

        if depth == Depth::new(0) {
            return lazy_static_eval(self, pos);
        }

        let tt_entry = self.tt.raw_mut(key);
        let tt_move = tt_entry.map(|entry| entry.mov()).unwrap_or(Move::null());

        // stand pad if not in check
        if !in_check {
            best_value = lazy_static_eval(self, pos);

            if best_value >= beta {
                return best_value;
            }
            if best_value > alpha {
                alpha = best_value;
            }
        }

        // move gen
        let tt_move_flag = tt_move.get_flag();
        let scorer = MoveScorer { color: P::COLOR, phase, tt_move };
        let mut move_picker = MovePicker::new_with_max_stage(
            // don't search tt_move if we don't search only captures.
            if in_check || !(tt_move_flag.is_capture() || tt_move_flag.is_promo()) {
                Move::null()
            }
            else {
                tt_move
            },
            // todo: killers if were in check (looking at quiets)?
            RbSet::<Move, 2>::default(),
            // if in check, we only want to search captures and promos, otherwise we want to search all moves.
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
                    .unwrap_or(scores::DRAW);

                let captured_value = m
                    .get_capture_sq()
                    .map(|capt_sq| piece_score(pos.get_piece(capt_sq).piece_type()))
                    .unwrap_or(scores::DRAW);

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
    tt_move: Move,
}
impl ordering::MoveScorer for MoveScorer {
    fn score<S: Stage>(&self, pos: &Position, mov: Move) -> MoveScore {
        match S::stage() {
            ordering::RtStage::YieldHashMove => {
                debug_assert!(mov == self.tt_move, "hashmove stage should only yield the tt move");
                0
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
