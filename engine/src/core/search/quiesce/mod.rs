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
        data::{ReplacementStrategy, TTBound, TTDepth, TTKey, TTMove, TTScore, TTStaticEval, TranspositionTable},
        id::{Bound, RbSet},
        ordering::{self, MovePicker, MoveScore, RtStage, Stage},
        score::{AnyScore, Score, scores},
        tree::NodeType,
    },
    zobrist,
};

pub const trait QSearchParams {
    fn futility_margin(&self) -> AnyScore;
    fn delta_pruning_threshold(&self) -> TaperValue;
}

pub type TT<Data, Strat> = TranspositionTable<Data, Strat>;

pub struct QSearcher<'a, Entry, Replace> {
    tt: &'a mut TT<Entry, Replace>,
    phase: TaperValue,
}

impl<'a, E, R> QSearcher<'a, E, R> {
    pub fn new(tt: &'a mut TT<E, R>, phase: TaperValue) -> Self { Self { tt, phase } }
}

impl<'a, E: From<TTEntry> + TTKey + TTBound + TTScore + TTMove + TTDepth + TTStaticEval + Clone, R: ReplacementStrategy<Data = E>>
    QSearcher<'a, E, R>
{
    /// # Q-Search
    ///
    /// Make the position quiet.
    ///
    /// [q-search](https://www.chessprogramming.org/Quiescence_Search)
    pub fn go<P: Perspective, T: NodeType>(
        &mut self,
        pos: &mut Position,
        mut alpha: Score<P>,
        beta: Score<P>,
        params: impl QSearchParams + Clone,
        eval: &mut impl StaticEvaluator,
        depth: Depth,
    ) -> Score<P> {
        let mut best_score = Score::NEG_INF;

        let in_check = pos.get_check_state() != CheckState::None;
        let key = pos.get_key();

        let mut static_eval = Score::<P>::NULL;
        let mut lazy_static_eval = |this: &mut Self, pos: &Position| {
            // is it already computed? if so, return it.
            if static_eval.0.validated().is_some() {
                return static_eval;
            }

            let mut compute = || eval.eval(pos.piece_info(), P::COLOR, pos.get_ep_target_square(), this.phase);

            // check the tt
            if let Some(tt_entry) = this.tt.get_mut(key) {
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

        let tt_entry = self.tt.get_mut(key);
        let tt_move = tt_entry.as_ref().map(|e| e.mov()).unwrap_or(Move::null());

        // tt cutoff
        // {
        //     if T::KIND == NodeKind::Cut
        //         && let Some(entry) = tt_entry
        //         && ((entry.bound() == Bound::Exact)
        //             || (entry.bound() == Bound::Lower && entry.score() >= beta.0)
        //             || (entry.bound() == Bound::Upper && entry.score() <= alpha.0))
        //     {
        //         return Score::new(entry.score());
        //     }
        // }

        // stand pad if not in check
        if !in_check {
            best_score = lazy_static_eval(self, pos);

            if best_score >= beta {
                return best_score;
            }
            if best_score > alpha {
                alpha = best_score;
            }
        }

        // move gen
        let tt_move_flag = tt_move.get_flag();
        let scorer = MoveScorer {
            color: P::COLOR,
            phase: self.phase,
            tt_move,
        };
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
        let mut best_move = Move::null();
        while let Some(m) = move_picker.next_for::<P>(pos, &scorer) {
            // if !pos.is_legal_for::<P>(m) {
            //     continue;
            // }

            // delta pruning
            if !in_check && self.phase < params.delta_pruning_threshold() {
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
                // Safety: the score was constructed relative to `P`
                let futility_score = unsafe { futility_score.interpret_as() };

                if best_score + futility_score < alpha {
                    continue;
                }
            }

            eval.forward();
            let phase_before = self.phase;
            pos.make_move_for::<P>(m, &mut (&mut self.phase, eval.observe_forward()));

            let score = !self.go::<P::Opponent, T>(pos, !beta, !alpha, params.clone(), eval, depth - 1);

            pos.unmake_move_for::<P>(m, eval.observe_backward());
            eval.backward();
            self.phase = phase_before;

            if score > best_score {
                best_score = score;
                best_move = m;
            }
            if score > alpha {
                alpha = score;

                if score >= beta {
                    // fail high
                    break;
                }
            }
        }

        self.tt.try_insert(TTEntry {
            key,
            depth: Depth::NONE,
            score: scores::NULL, // best_score.0,
            static_eval: static_eval.0,
            bound: Bound::None, // Bound::from_scores(beta - 1, beta, best_score),
            mov: best_move,
        });

        best_score
    }
}

pub struct TTEntry {
    key: zobrist::Hash,
    depth: Depth,
    score: AnyScore,
    static_eval: AnyScore,
    bound: Bound,
    mov: Move,
}

impl const TTMove for TTEntry {
    fn mov(&self) -> Move { self.mov }
}

impl const TTKey for TTEntry {
    fn key(&self) -> zobrist::Hash { self.key }
}

impl const TTDepth for TTEntry {
    fn depth(&self) -> Depth { self.depth }
}

impl const TTStaticEval for TTEntry {
    fn static_eval(&self) -> AnyScore { self.static_eval }
    fn static_eval_mut(&mut self) -> &mut AnyScore { &mut self.static_eval }
}

impl const TTBound for TTEntry {
    fn bound(&self) -> Bound { self.bound }
}

impl const TTScore for TTEntry {
    fn score(&self) -> AnyScore { self.score }
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
