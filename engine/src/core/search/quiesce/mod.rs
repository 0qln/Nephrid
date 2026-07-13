use crate::core::{
    color::{Color, Perspective},
    depth::Depth,
    eval::{
        StaticEvaluator,
        hce::{TaperValue, piece_score, tapered_psqt},
    },
    r#move::Move,
    piece::{PromoPieceType, piece_type},
    ply::Ply,
    position::{CheckState, Position},
    search::{
        data::{ReplacementStrategy, TTBound, TTDepth, TTKey, TTMove, TTScore, TTStaticEval, TranspositionTable},
        id::{self, Bound},
        ordering::{self, MovePicker, MoveScore, RtStage, Stage},
        score::{AnyScore, Score, scores},
        tree::NodeType,
    },
    zobrist,
};

pub const trait QSearchParams {
    fn futility_margin(&self) -> AnyScore;
    fn delta_pruning_threshold(&self) -> TaperValue;
    fn movecount_pruning_factor(&self) -> AnyScore;
    fn phase_pruning_factor(&self) -> AnyScore;
}

pub type TT<Data, Strat> = TranspositionTable<Data, Strat>;

/// # Q-Search
///
/// Make the position quiet.
///
/// [q-search](https://www.chessprogramming.org/Quiescence_Search)
pub struct QSearcher<'a, Entry, Replace> {
    tt: &'a mut TT<Entry, Replace>,
    ss: &'a mut id::SS,
    root_ply: Ply,
}

impl<'a, E, R> QSearcher<'a, E, R> {
    pub fn new(tt: &'a mut TT<E, R>, ss: &'a mut id::SS, root_ply: Ply) -> Self { Self { tt, ss, root_ply } }
}

impl<'a, E: From<TTEntry> + TTKey + TTBound + TTScore + TTMove + TTDepth + TTStaticEval + Clone, R: ReplacementStrategy<Data = E>>
    QSearcher<'a, E, R>
{
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
        let rel_ply: Depth = (pos.ply() - self.root_ply).into();
        let &id::SearchEntry { phase, .. } = self.ss.get(rel_ply);

        let mut static_eval = Score::<P>::NULL;
        let mut lazy_static_eval = |this: &mut Self, pos: &Position| {
            // is it already computed? if so, return it.
            if static_eval.0.is_valid() {
                return static_eval;
            }

            static_eval = if let Some(entry) = this.tt.get_mut(key) {
                let score_ref = entry.static_eval_mut();
                if let Some(score) = score_ref.validated_mut() {
                    // Safety: unless we've had a hash collision, this score is for the same
                    // position
                    unsafe { score.interpret_as() }
                }
                else {
                    let score = eval.eval(pos.piece_info(), P::COLOR, pos.get_ep_target_square(), phase);
                    *score_ref = score.0;
                    score
                }
            }
            else {
                eval.eval(pos.piece_info(), P::COLOR, pos.get_ep_target_square(), phase)
            };

            static_eval
        };

        if depth == Depth::new(0) {
            // todo: return the tt score if it is valid for a more accurate eval than
            // static?
            return lazy_static_eval(self, pos);
        }

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
            // todo: return the tt score if it is valid for a more accurate eval than
            // static?
            best_score = lazy_static_eval(self, pos);

            if best_score >= beta {
                return best_score;
            }
            if best_score > alpha {
                alpha = best_score;
            }
        }

        // move gen
        let hash_move = if let Some(entry) = self.tt.get(key) {
            let tt_move = entry.mov();
            let tt_move_flag = tt_move.get_flag();
            if in_check || tt_move_flag.is_capture() || tt_move_flag.is_promo() {
                tt_move
            }
            else {
                Move::null()
            }
        }
        else {
            Move::null()
        };
        let scorer = MoveScorer {
            color: P::COLOR,
            phase,
            tt_move: hash_move,
        };
        let mut move_picker = MovePicker::new_with_max_stage(
            hash_move,
            // todo: killers if were in check (looking at quiets)?
            id::Killers::default(),
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
        let mut curr = 0;
        while let Some(m) = move_picker.next_for::<P>(pos, &scorer) {
            let (from, to, flag) = m.into();

            // if !pos.is_legal_for::<P>(m) {
            //     continue;
            // }

            // delta pruning
            if !in_check && phase < params.delta_pruning_threshold() {
                let value_bonus = PromoPieceType::try_from(flag)
                    .ok()
                    .map(|promo| {
                        let promo_pt = promo.into();
                        let lost = piece_score(piece_type::PAWN) + tapered_psqt(phase, piece_type::PAWN, from, P::COLOR);
                        let gained = piece_score(promo_pt) + tapered_psqt(phase, promo_pt, to, P::COLOR);
                        gained - lost
                    })
                    .unwrap_or(scores::ZERO);

                let captured_value = m
                    .get_capture_sq()
                    .map(|capt_sq| {
                        let pt = pos.get_piece(capt_sq).piece_type();
                        piece_score(pt) + tapered_psqt(phase, pt, capt_sq, P::Opponent::COLOR)
                    })
                    .unwrap_or(scores::ZERO);

                let futility_margin = params.futility_margin();

                // should allow for more aggressive futility pruning at moves that were regarded
                // less important by the move ordering
                let move_count_margin = params.movecount_pruning_factor() * curr;

                // the endgame contains
                let phase_margin = AnyScore::new(0);

                // Safety: the score was constructed relative to `P`
                let futility_score = captured_value + value_bonus + futility_margin + move_count_margin + phase_margin;
                let futility_score = unsafe { futility_score.interpret_as() };

                if best_score + futility_score < alpha {
                    continue;
                }
            }

            self.ss.propagate_forward(rel_ply, |s, next_s| next_s.phase = s.phase);
            eval.forward();
            pos.make_move_for::<P>(m, &mut (self.ss.get_mut(rel_ply + 1).phase, eval.observe_forward()));

            let score = !self.go::<P::Opponent, T>(pos, !beta, !alpha, params.clone(), eval, depth - 1);

            pos.unmake_move_for::<P>(m, eval.observe_backward());
            eval.backward();

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

            curr += 1;
        }

        self.tt.try_insert(TTEntry {
            key,
            depth: Depth::NONE,
            score: best_score.0,
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
