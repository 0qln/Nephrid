use core::fmt;
use std::{
    cmp::{Reverse, max, min},
    convert::Infallible,
    ops::{ControlFlow, Deref},
    path::PathBuf,
    str::FromStr,
    time::{Duration, Instant},
};

use saturating_cast::SaturatingCast;

use crate::{
    core::{
        chrono::{ChronoParams, TimeMan},
        color::{
            Color, Perspective, colors,
            perspectives::{self},
        },
        config::Configuration,
        coordinates::EpTargetSquare,
        depth::Depth,
        eval::{
            GameResult, StaticEvaluator,
            hce::{self, TaperValue, bishop_pair, hygge_king, king_safety, material, mobility, passed_pawns},
            nnue::{self, AccumulatorStack, EagerAccUpdates},
        },
        r#move::{MAX_LEGAL_MOVES, Move, MoveList},
        move_iter::{fold_moves, opt::AllLegal},
        params::IParams,
        piece::piece_type,
        ply::Ply,
        position::{CheckState, PieceInfo, PieceInfoObserver, Position},
        search::{
            data::{self, Line, PieceHistories, RbSet, SearchStack, TTBound, TTDepth, TTKey, TTMove, TTScore, TTStaticEval, TranspositionTable},
            limit::UciLimit,
            mcts::eval::Quality,
            ordering::{self, MovePicker, MoveScore, MoveScorer, RtStage, ScoredMove, Stage},
            quiesce::{self, QSearchParams, QSearcher},
            score::{AnyScore, Cp, Score, scores},
            strat::{UciArg, UciCp, UciCurrmove, UciDepth, UciNodes, UciNps, UciPv, UciScore, UciSearchtime, UciSeldepth},
            tree::{NodeKind, NodeType, node_types::*},
        },
        turn::Turn,
        zobrist,
    },
    math::{self, NormalizedEntropy, interpolate_i32, lmr_u8},
    misc::{CancellationToken, DebugMode, List},
};

#[cfg(test)] pub mod test;

/// Softmax temperature applied to root-move qualities when computing the
/// normalized root entropy used as a soft stopping target. Qualities are
/// `tanh`-squashed into `[-1, 1]`, so a sub-1 temperature is needed for a
/// confident position to produce a peaked (low-entropy) distribution.
const ROOT_ENTROPY_TEMP: f32 = 0.3;

#[derive(Default)]
pub struct HceEvaluator;

impl StaticEvaluator for HceEvaluator {
    fn eval<P: Perspective>(&mut self, pos: &PieceInfo, turn: Turn, ep_sq: EpTargetSquare, phase: TaperValue) -> Score<P> {
        fn static_value<P: Perspective>(pos: &PieceInfo, ep_sq: EpTargetSquare, phase: TaperValue, turn: Turn) -> Score<P> {
            material::<P>(pos)
                + mobility::<P>(pos, phase)
                + hce::psqt::<P>(pos, phase)
                + bishop_pair::<P>(pos)
                + king_safety::<P>(pos, ep_sq, turn, phase)
                + passed_pawns::<P>(pos, ep_sq, turn)
                + hygge_king::<P>(pos, phase)
        }

        let (ep_w, ep_b) = if P::COLOR == colors::WHITE {
            (ep_sq, EpTargetSquare::none())
        }
        else {
            (EpTargetSquare::none(), ep_sq)
        };
        let w_q = static_value::<P>(pos, ep_w, phase, turn);
        let b_q = static_value::<P::Opponent>(pos, ep_b, phase, turn);
        w_q + !b_q
    }

    fn try_from_config<C: Deref<Target = Configuration>>(_cfg: C) -> Result<Self, Infallible> { Ok(Self) }
}

pub struct NnueEvaluator {
    accs: AccumulatorStack<EagerAccUpdates>,
    curr: Depth,
}

impl NnueEvaluator {
    fn new() -> Self {
        Self {
            accs: AccumulatorStack::default(),
            curr: Depth::ROOT,
        }
    }
}

impl Default for NnueEvaluator {
    fn default() -> Self { Self::new() }
}

impl StaticEvaluator for NnueEvaluator {
    fn eval<P: Perspective>(&mut self, _: &PieceInfo, _: Turn, _: EpTargetSquare, _: TaperValue) -> Score<P> {
        let nnue = nnue::get_nnue();
        let accs = self.accs.get_accs_mut(self.curr);
        let (stm_acc, nstm_acc) = accs.get_mut_for::<P>();
        let nnue_eval = nnue.forward(stm_acc, nstm_acc);
        // Safety: We picked `P` as side-to-move above.
        unsafe { nnue_eval.interpret_as() }
    }

    fn forward(&mut self) {
        let old = self.curr;
        let new = old + 1;
        self.curr = new;

        self.accs.propagate(old, new);
    }

    fn backward(&mut self) { self.curr -= 1; }

    fn observe_forward(&mut self) -> &mut impl PieceInfoObserver { self.accs.get_accs_mut(self.curr) }
    // observer backward does nothing, since we just pop to the latest state.

    fn try_from_config<C: Deref<Target = Configuration>>(cfg: C) -> Result<Self, impl fmt::Display> {
        let nnue_str = cfg.nnue_path();
        let nnue_bytes = if nnue_str.is_empty() {
            nnue::DEFAULT_NNUE
        }
        else {
            let nnue_path = PathBuf::from_str(nnue_str).expect("Infallible was returned??");
            nnue::read_net_bytes(&nnue_path).map_err(|e| format!("Bad nnue file: {e}"))?
        };
        nnue::set_nnue(nnue_bytes).map_err(|e| format!("Unhealthy nnue: {e}"))?;

        Ok::<_, String>(Self::new())
    }
}

#[allow(dead_code)]
struct HceThreatener;

impl HceThreatener {
    /// Finds the biggest incoming threat to `P`, giving a score for
    /// `P::Opponent`.
    #[allow(dead_code)]
    fn threat<P: Perspective>(&self, pos: &Position) -> Score<P::Opponent> {
        const QUEEN_SCORE: AnyScore = hce::piece_score(piece_type::QUEEN);
        const ROOK_SCORE: AnyScore = hce::piece_score(piece_type::ROOK);

        let mut max_threat = Score::<P::Opponent>::ZERO;

        // todo: only generate captures, promos, and checks.
        // todo: or just track this in the make_move unmake_move functions.
        let moves = pos.collect_legals_for::<P::Opponent, _>(MoveList::new());
        for &mov in moves.iter() {
            match pos.does_check(mov) {
                CheckState::None => {}
                CheckState::Single => return unsafe { QUEEN_SCORE.interpret_as() },
                CheckState::Double => return unsafe { (QUEEN_SCORE + ROOK_SCORE).interpret_as() },
            }

            if mov.get_flag().is_capture() {
                let see: AnyScore = ordering::see(pos.piece_info(), mov, P::Opponent::COLOR).into();
                let see_threat = unsafe { see.interpret_as::<P::Opponent>() };
                max_threat = max(max_threat, see_threat);
            }
        }

        max_threat
    }
}

pub struct SearchStats {
    pub nodes: u64,
    pub iterations: u64,

    /// Time taken for last iteration.
    pub iter_time: Duration,

    /// Entropy of the last completed iteration.
    pub root_entropy: NormalizedEntropy,

    /// Number of times the best move has been the best move in a row.
    pub root_movestreak: u32,
}

impl Default for SearchStats {
    fn default() -> Self {
        Self {
            nodes: 0,
            iterations: 0,
            iter_time: Duration::ZERO,
            // Max uncertainty until a policy is computed
            root_entropy: NormalizedEntropy::one(),
            root_movestreak: 0,
        }
    }
}

pub const trait IdParams {
    fn nmp_reduction(&self) -> Depth;
    fn nmp_phase_threshold(&self) -> TaperValue;
    fn nmp_depth_factor(&self) -> u8;
    fn nmp_phase_factor(&self) -> u32;
    fn nmp_margin(&self) -> AnyScore;
    fn nmp_depth_margin(&self) -> i32;
}

#[allow(clippy::too_many_arguments)]
pub fn go<X: IParams>(
    pos: &mut Position,
    limit: UciLimit,
    timeman: &mut TimeMan<X>,
    debug: &DebugMode,
    ct: CancellationToken,
    tt: &mut TT,
    hh: &mut HH,
    eval: &mut impl StaticEvaluator,
    params: X::Ref,
) -> Option<Move>
where
    X::Ref: ChronoParams + QSearchParams + ScorerParams + IdParams + Clone + fmt::Debug,
{
    let depth_lim = min(Depth::MAX, limit.depth);

    eval.observe_forward().on_init(pos.piece_info());

    if debug.get() {
        println!("info string Starting ID Search with Params: {params:?}");
    }

    let mut searcher = Searcher::<_, X>::new(pos, limit, timeman, ct, tt, hh, eval, params.clone());
    let mut stats = SearchStats::default();
    let mut best_move = None;
    let mut last_best_move;

    for depth in (Depth::ROOT + 1)..=depth_lim {
        let iter_start = Instant::now();

        let best_score = searcher.search_root(pos, &mut stats, depth);

        // make sure to break before messing up the order of the previous iteration with
        // the incomplete results from this iteration.
        if searcher.aborted {
            break;
        }

        last_best_move = searcher.root_best_move();

        searcher.sort_root();

        best_move = searcher.root_best_move();
        if let Some(best_move) = best_move
            && let Some(search_time) = searcher.timeman.elapsed_search_time()
        {
            uci_info(depth, &stats, Cp::from(best_score), best_move, search_time, searcher.pv());
        }

        // update stats
        let iter_end = Instant::now();
        stats.iter_time = iter_end - iter_start;
        stats.iterations += 1;
        stats.root_entropy = {
            let root_logits = searcher.root_logits();
            let root_policy = math::softmax(root_logits, ROOT_ENTROPY_TEMP, &mut List::new());
            math::normalized_entropy(&root_policy)
        };
        stats.root_movestreak = if best_move == last_best_move {
            stats.root_movestreak + 1
        }
        else {
            0
        };

        let timeman = &mut searcher.timeman;

        timeman.hint_time_target(stats.iter_time);
        timeman.hint_movestreak_target(stats.root_movestreak);

        if searcher.should_stop(&stats) || searcher.timeman.reached_target() {
            break;
        }
    }

    best_move
}

#[derive(Debug)]
struct RootStats {
    mov: ScoredMove,
}

impl RootStats {
    #[inline]
    fn new(m: Move, score: MoveScore) -> Self { Self { mov: ScoredMove::new(m, score) } }

    #[inline]
    fn mov(&self) -> Move { self.mov.mov() }

    #[inline]
    fn scored_move(&self) -> &ScoredMove { &self.mov }

    #[inline]
    fn score(&self) -> MoveScore { self.mov.score() }

    #[inline]
    fn set_score(&mut self, score: MoveScore) { self.mov.set_score(score); }
}

struct Searcher<'a, 'b, E: StaticEvaluator, X: IParams> {
    root_stats: List<{ MAX_LEGAL_MOVES }, RootStats>,
    root_ply: Ply,
    limit: UciLimit,
    timeman: &'a mut TimeMan<X>,
    ct: CancellationToken,
    aborted: bool,
    ss: SS,
    tt: &'a mut TT,
    hh: &'a mut HH,
    eval: &'b mut E,
    params: X::Ref,
    #[cfg(feature = "id-nmp")]
    in_nmp_verify: bool,
}

impl<'a, 'b, E: StaticEvaluator, X: IParams> Searcher<'a, 'b, E, X>
where
    X::Ref: QSearchParams + IdParams + ScorerParams + ChronoParams + Clone,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        pos: &Position,
        limit: UciLimit,
        timeman: &'a mut TimeMan<X>,
        ct: CancellationToken,
        tt: &'a mut TT,
        hh: &'a mut HH,
        eval: &'b mut E,
        params: X::Ref,
    ) -> Self {
        let mut stats = List::<{ MAX_LEGAL_MOVES }, RootStats>::new();
        _ = fold_moves::<AllLegal, _, _, _>(pos, (), |_, m| {
            stats.push(RootStats::new(m, 0));
            ControlFlow::Continue::<(), ()>(())
        });

        Self {
            root_stats: stats,
            root_ply: pos.ply(),
            limit,
            timeman,
            ct,
            aborted: false,
            ss: SS::from(vec![SearchEntry {
                phase: TaperValue::from_position(pos.piece_info()),
                ..Default::default()
            }]),
            tt,
            hh,
            eval,
            params,
            #[cfg(feature = "id-nmp")]
            in_nmp_verify: false,
        }
    }

    fn sort_root(&mut self) { self.root_stats.as_mut_slice().sort_by_key(|mov| Reverse(mov.score())); }

    fn root_best_move(&self) -> Option<Move> { self.root_stats.get(0).map(|x| x.mov()) }

    fn pv(&self) -> &Line { &self.ss.get(Depth::ROOT).line }

    fn root_logits(&self) -> List<{ MAX_LEGAL_MOVES }, f32> {
        let mut root_logits = List::<{ MAX_LEGAL_MOVES }, f32>::new();
        self.root_stats
            .iter()
            .map(|x| Quality::from(Cp::new(x.score())).v())
            .collect_into(&mut root_logits);
        root_logits
    }

    fn should_stop(&self, stats: &SearchStats) -> bool {
        let nodes = stats.nodes;
        let iters = stats.iterations;

        // user requested stop
        if self.ct.is_cancelled() {
            return true;
        }

        // time manager says we should stop or limit has been reached
        if self.limit.is_active() && (self.timeman.reached_limit() || self.limit.is_reached(nodes, iters)) {
            return true;
        }

        // otherwise keep searching
        false
    }

    /// returns the score relative to the current player
    fn search_root(&mut self, pos: &mut Position, stats: &mut SearchStats, depth: Depth) -> AnyScore {
        fn alpha<P: Perspective>() -> Score<P> { Score::NEG_INF }
        fn beta<P: Perspective>() -> Score<P> { Score::POS_INF }
        match pos.get_turn() {
            colors::WHITE => self.search::<perspectives::White, Root>(pos, stats, depth, alpha(), beta()).0,
            colors::BLACK => self.search::<perspectives::Black, Root>(pos, stats, depth, alpha(), beta()).0,
            _ => unreachable!(),
        }
    }

    fn scorer_for<P: Perspective>(&mut self, tt_move: Move, killers: Killers, phase: TaperValue) -> Scorer<'_, X> {
        Scorer {
            tt_move,
            killers,
            hh: self.hh,
            color: P::COLOR,
            phase,
            params: self.params.clone(),
        }
    }

    /// returns the score relative to `P`
    /// `T`: The expected [NodeType] of this node.
    fn search<P: Perspective, T: NodeType>(
        &mut self,
        pos: &mut Position,
        stats: &mut SearchStats,
        depth: Depth,
        mut alpha: Score<P>,
        beta: Score<P>,
    ) -> Score<P> {
        #[cfg(feature = "id-fhr")]
        let threatener = &HceThreatener;

        debug_assert!(alpha < beta);

        // incremment stats
        stats.nodes += 1;

        // check if stop is requested or we have reached a limit
        if stats.nodes.is_multiple_of(4096) && self.should_stop(stats) {
            self.aborted = true;
            return Score::NEG_INF;
        }

        // check if game is over
        if let Some(result) = pos.game_result() {
            return match result {
                GameResult::Win { .. } => Score::NEG_INF,
                GameResult::Draw => Score::DRAW,
            };
        }

        let rel_ply: Depth = (pos.ply() - self.root_ply).into();
        let &SearchEntry { phase, killers, .. } = self.ss.get(rel_ply);
        let pline = for<'l> |ss: &'l mut SS| -> &'l mut Box<Line> { &mut ss.get_mut(rel_ply).line };
        let line_and_pline = for<'l> |ss: &'l mut SS| -> (&'l Box<Line>, &'l mut Box<Line>) {
            let [line, pline] = unsafe { ss.get_disjoint_unchecked_mut([rel_ply + 1, rel_ply]) };
            (&line.line, &mut pline.line)
        };

        pline(&mut self.ss).clear();

        // qsearch at the leaf nodes
        if depth == Depth::ROOT || rel_ply >= Depth::MAX {
            return QSearcher::new(pos, self.tt, &mut self.ss, self.root_ply).go::<P, T>(
                pos,
                alpha,
                beta,
                self.params.clone(),
                self.eval,
                Depth::new(100),
            );
        }

        let kind = T::KIND;
        let key = pos.get_key();
        let orig_alpha = alpha;

        let tt_entry = self.tt.get(key).cloned();

        // tt-cutoff
        if kind != NodeKind::Root
            && let Some(ref entry) = tt_entry
            && entry.depth >= depth
            && ((entry.bound == Bound::Exact)
                || (entry.bound == Bound::Lower && entry.score >= beta.0)
                || (entry.bound == Bound::Upper && entry.score <= alpha.0))
        {
            // Safety: unless we've had a hash collision, this score is for the same
            // position and thus for the same player.
            return unsafe { entry.score.interpret_as() };
        }

        // todo: these 'is it already computed? if so, return it.' are not required.
        // they are just for convenience when compiling with different
        // features... find a clean way to solve this or make sure the compiler
        // can understand when they will already be computed...

        #[cfg(any(feature = "id-fhr", feature = "id-nmp"))]
        let mut static_eval = Score::NULL;
        #[cfg(not(any(feature = "id-fhr", feature = "id-nmp")))]
        let static_eval = Score::<P>::NULL;

        #[cfg(any(feature = "id-fhr", feature = "id-nmp"))]
        let mut lazy_static_eval = |this: &mut Self, pos: &Position| {
            // is it already computed? if so, return it.
            if static_eval.0.is_valid() {
                return static_eval;
            }

            let eval = tt_entry
                .as_ref()
                // Safety: unless we've had a hash collision, this score is for the same position
                .map(|e| unsafe { e.static_eval.interpret_as() })
                .unwrap_or_else(|| this.eval.eval(pos.piece_info(), P::COLOR, pos.get_ep_target_square(), phase));

            static_eval = eval;

            eval
        };

        #[cfg(feature = "id-fhr")]
        let mut threat = Score::<P::Opponent>::NULL;

        #[cfg(feature = "id-fhr")]
        let mut lazy_threat_score = |pos: &Position| {
            // is it already computed? if so, return it.
            if threat.0.is_valid() {
                return threat;
            }

            let score = tt_entry
                .as_ref()
                // Safety: unless we've had a hash collision, this score is for the same position
                .map(|e| unsafe { e.threat.interpret_as() })
                .unwrap_or_else(|| threatener.threat::<P>(pos));

            threat = score;

            score
        };

        // null move pruning
        #[cfg(feature = "id-nmp")]
        {
            let nmp_margin = unsafe { self.params.nmp_margin().interpret_as() };

            let nmp_depth_margin = unsafe { AnyScore::from(depth.v() as i32 * self.params.nmp_depth_margin()).interpret_as() };

            let nmp_r: Depth = self.params.nmp_reduction()
                // scale the reduction up based on depth
                + depth.div_floor(self.params.nmp_depth_factor());
            // todo: test this idea
            // // scale the reduction down based on phase (we want deeper searches in the
            // endgame)
            // - Depth::new(phase.v().div_floor(params.nmp_phase_factor()) as u8); // todo:
            //   honestly phase could just be a u8

            let is_in_check = pos.get_check_state() != CheckState::None;

            if kind == NodeKind::Cut
                // are we in verification search?
                && !self.in_nmp_verify
                // don't underflow depth
                && depth > nmp_r
                // don't allow nmp when node is in check
                && !is_in_check
                // don't do nmp in endgames, where zugzwang is more likely
                && phase < self.params.nmp_phase_threshold() && pos.has_non_pawn_material::<P>()
                // don't bother attempting to improve beta with a tempo down when our static eval is not
                // even better than beta
                && lazy_static_eval(self, pos) >= beta - nmp_margin - nmp_depth_margin
            {
                let nmp_depth = depth - nmp_r - 1;

                pos.make_null_move();

                let nm_score = !self.search::<P::Opponent, All>(pos, stats, nmp_depth, !beta, !beta + 1);

                pos.unmake_null_move();

                if nm_score >= beta {
                    // verification search
                    self.in_nmp_verify = true;
                    let verification_score = self.search::<P, All>(pos, stats, nmp_depth, beta - 1, beta);
                    self.in_nmp_verify = false;

                    if verification_score >= beta {
                        return verification_score;
                    }
                }
            }
        }

        // move gen
        let tt_move = tt_entry.as_ref().map(|e| e.mov).unwrap_or(Move::null());
        let mut move_picker = if kind == NodeKind::Root {
            MovePicker::from_scored(self.root_stats.iter().map(|m| m.scored_move()).cloned())
        }
        else {
            MovePicker::new(tt_move, killers)
        };

        // fail-high reductions
        let fhr_reduct = cfg_select! {
            feature = "id-fhr" => {{
                let in_check = pos.get_check_state() != CheckState::None;

                if kind == NodeKind::Cut && !in_check {
                    // the quiet score of this position is the static score minus threat score (the
                    // best threat that the opponent can do).
                    let q_score = lazy_static_eval(self, pos) + !lazy_threat_score(pos);

                    // if the quiet score
                    if q_score >= beta { 1 } else { 0 }
                }
                else {
                    0
                }
            }}
            _ => 0
        };

        let mut best_score = Score::NEG_INF;
        let mut best_move = Move::null();
        let mut curr = 0;
        let mut hh_searched_quiets = MoveList::new();

        // todo: take killers by ref
        while let Some(m) = move_picker.next_for::<P>(pos, &self.scorer_for::<P>(tt_move, killers, phase)) {
            let (from, to, flag) = m.into();
            let moving_piece = pos.get_piece(from);
            let moving_pt = moving_piece.piece_type();

            // make the move
            self.ss.propagate_forward(rel_ply, |s, next_s| next_s.phase = s.phase);
            self.eval.forward();
            pos.make_move_for::<P>(m, &mut (&mut self.ss.get_mut(rel_ply + 1).phase, self.eval.observe_forward()));

            // depth
            let (mut depth_ext, mut depth_reduct) = (0, 0);

            let gives_check = pos.get_check_state() != CheckState::None;

            // check extensions
            if gives_check {
                depth_ext += 1;
            }

            // late move reductions
            if depth >= Depth::new(3) && curr > 1 {
                depth_reduct += lmr_u8(depth.v(), curr as u8);
            }

            // recurse
            let score = {
                let new_depth = depth - 1 + depth_ext;
                let full_depth = new_depth.saturating_sub(fhr_reduct);
                let reduced_depth = new_depth.saturating_sub(depth_reduct + fhr_reduct);

                if curr == 0 {
                    // search with a full window to get an exact score.
                    !self.search::<P::Opponent, Pv>(pos, stats, full_depth, !beta, !alpha)
                }
                else {
                    // assume that our move ordering is good the first move will be the best one.
                    // to prove that this move cannot improve our first move, perform a zero window
                    // search with [a,a+1] (~ [-(a-1),-a]). we don't care by how much this move is
                    // able to improve alpha since we assume that it cannot.
                    let mut zws_score = !self.search::<P::Opponent, Cut>(
                        pos,
                        stats,
                        // scout with a reduced depth
                        reduced_depth,
                        !(alpha + 1),
                        !alpha,
                    );

                    // if the reduced depth search fails high, we must verify that it is actually
                    // good and do a full depth re-search.
                    if zws_score > alpha && reduced_depth != full_depth {
                        zws_score = !self.search::<P::Opponent, Cut>(
                            pos,
                            stats,
                            full_depth,
                            // still zero-window.
                            !(alpha + 1),
                            !alpha,
                        );
                    }

                    // if zws_score is a lower_bound, we have to research to get an exact score.
                    if zws_score > alpha
                        // don't bother researching if this move will cause a fail-high anyway
                        && zws_score < beta
                    {
                        // new lower_bound, since it was able to beat alpha
                        let (alpha, beta) = (!beta, !zws_score);
                        match kind {
                            NodeKind::Cut => !self.search::<P::Opponent, All>(pos, stats, full_depth, alpha, beta),
                            _ => !self.search::<P::Opponent, Cut>(pos, stats, full_depth, alpha, beta),
                        }
                    }
                    else {
                        zws_score
                    }
                }
            };

            // unmake the move
            pos.unmake_move_for::<P>(m, self.eval.observe_backward());
            self.eval.backward();

            // check for cancellation
            if self.aborted {
                return Score::DRAW;
            }

            // update root moves
            if kind == NodeKind::Root {
                // store the score for the root moves, such that we can use it for sorting in
                // the next iteration.
                // todo: don't just clamp, mate values will get lost etc.
                self.root_stats.as_mut_slice()[curr].set_score(score.0.v().saturating_cast());
            }

            if score > best_score {
                best_score = score;
                best_move = m;
            }

            if score > alpha {
                // update alpha
                alpha = score;

                // update pv
                let (line, pline) = line_and_pline(&mut self.ss);
                pline.clear();
                pline.push(m);
                pline.extend_from_slice(1.., line.as_slice());

                if score >= beta {
                    // mark quiet moves, fail-high as killer moves
                    if !flag.is_capture() && !flag.is_promo() {
                        // update killers
                        if m != tt_move {
                            self.ss.get_mut(rel_ply).killers._push(m);
                        }

                        // update hh
                        {
                            let hh_bonus = MoveScore::from(depth.v()).pow(2);

                            // penalty history heuristic that were expected but
                            // failed to cause a cutoff
                            for &searched_quiet in hh_searched_quiets.as_slice() {
                                let (from, to, _) = searched_quiet.into();
                                let moving_pt = pos.get_piece(from).piece_type();
                                self.hh.update_for::<P::Opponent>(moving_pt, to, -hh_bonus);
                            }

                            // reward history heuristic
                            self.hh.update_for::<P>(moving_pt, to, hh_bonus);
                        }
                    }

                    // fail high
                    break;
                }
            }
            else {
                // fail low
            }

            curr += 1;

            // push any move whose statistic can be used to estimate a quiet moves score.
            // that includes killers and the hashmove.
            if !flag.is_capture() && !flag.is_promo() {
                hh_searched_quiets.push(m);
            }
        }

        self.tt.try_insert(TTEntry {
            key,
            depth,
            score: best_score.0,
            static_eval: static_eval.0,
            #[cfg(feature = "id-fhr")]
            threat: threat.0,
            bound: Bound::from_scores(orig_alpha, beta, best_score),
            mov: best_move,
        });

        best_score
    }
}

pub type HH = PieceHistories;

pub type TT = TranspositionTable<TTEntry, TTReplace>;

pub type SS = SearchStack<SearchEntry>;

pub type Killers = RbSet<Move, 2>;
impl Copy for Killers {}

#[derive(Clone)]
pub struct TTEntry {
    key: zobrist::Hash,
    depth: Depth,
    score: AnyScore,
    static_eval: AnyScore,
    #[cfg(feature = "id-fhr")]
    threat: AnyScore,
    bound: Bound,
    mov: Move,
}

impl From<quiesce::TTEntry> for TTEntry {
    fn from(e: quiesce::TTEntry) -> Self {
        // todo: make sure there is no actual moving happening here and the compiler
        // should just inline the construction in the qsearch function from the
        // quiesce::TTEntry here.
        Self {
            key: e.key(),
            depth: e.depth(),
            score: e.score(),
            static_eval: e.static_eval(),
            #[cfg(feature = "id-fhr")]
            threat: scores::NULL,
            bound: e.bound(),
            mov: e.mov(),
        }
    }
}

const impl TTKey for TTEntry {
    fn key(&self) -> zobrist::Hash { self.key }
}

const impl TTScore for TTEntry {
    fn score(&self) -> AnyScore { self.score }
}

const impl TTMove for TTEntry {
    fn mov(&self) -> Move { self.mov }
}

const impl TTDepth for TTEntry {
    fn depth(&self) -> Depth { self.depth }
}

const impl TTBound for TTEntry {
    fn bound(&self) -> Bound { self.bound }
}

const impl data::TTStaticEval for TTEntry {
    fn static_eval(&self) -> AnyScore { self.static_eval }
    fn static_eval_mut(&mut self) -> &mut AnyScore { &mut self.static_eval }
}

const impl Default for TTEntry {
    fn default() -> Self {
        Self {
            key: zobrist::Hash::default(),
            depth: Depth::NONE,
            score: scores::NULL,
            static_eval: scores::NULL,
            #[cfg(feature = "id-fhr")]
            threat: scores::NULL,
            bound: Bound::None,
            mov: Move::null(),
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct TTReplace;

impl data::ReplacementStrategy for TTReplace {
    type Data = TTEntry;

    fn should_replace(old: &TTEntry, new: &TTEntry) -> bool {
        if old.depth == Depth::NONE {
            return true;
        }

        if new.depth == Depth::NONE {
            return false;
        }

        if new.depth > old.depth {
            return true;
        }

        if new.depth < old.depth {
            return false;
        }

        if new.bound == Bound::Exact && old.bound != Bound::Exact {
            return true;
        }

        false
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    None,
    Exact,
    Lower,
    Upper,
}

impl Bound {
    pub fn from_scores<P: Perspective>(alpha: Score<P>, beta: Score<P>, score: Score<P>) -> Self {
        if score <= alpha {
            Self::Upper
        }
        else if score >= beta {
            Self::Lower
        }
        else {
            Self::Exact
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct SearchEntry {
    pub killers: Killers,
    pub phase: TaperValue,
    pub line: Box<Line>,
}

pub const trait ScorerParams {
    fn hh_weight(&self) -> i32;
    fn total_weight(&self) -> i32 { 128 }
}

pub struct Scorer<'a, X: IParams> {
    pub tt_move: Move,
    pub killers: Killers,
    pub hh: &'a PieceHistories,
    pub color: Color,
    pub phase: TaperValue,
    pub params: X::Ref,
}

impl<X: IParams> MoveScorer for Scorer<'_, X>
where
    X::Ref: ScorerParams,
{
    #[inline(always)]
    fn score<S: Stage>(&self, pos: &Position, mov: Move) -> MoveScore {
        match S::stage() {
            // no need to score hashmove, there's only 1
            RtStage::YieldHashMove => {
                debug_assert!(mov == self.tt_move, "hashmove stage should only yield the tt move");
                0
            }

            // captures and promos, ordered by see value.
            RtStage::GenerateCapturesAndPromos | RtStage::YieldGoodCapturesAndPromos | RtStage::YieldBadCaptures => {
                // todo: currently see evaluates the promo values, but we don't need a whole
                // see for quiet promos, maybe that can be optimized...
                ordering::see(pos.piece_info(), mov, self.color)
            }

            // score killer moves by their age
            RtStage::YieldKillers => {
                let age = self.killers._position(&mov);
                debug_assert!(age.is_some(), "move in killer stage should be a killer move");

                // Safety: assert above
                let age = unsafe { age.unwrap_unchecked() };

                -(age as MoveScore)
            }

            // score quiet moves by psqt diff or history heuristic
            RtStage::GenerateQuiets | RtStage::YieldQuiets => {
                let (from, to, flag) = mov.into();
                let pieces = pos.piece_info();
                let piece = pieces.get_piece(from);
                let piece_type = piece.piece_type();

                let hh_score = self.hh.get(self.color, piece_type, to);
                let psqt_score = ordering::psqt(self.phase, piece_type, from, to, flag, self.color);

                // todo: interpolate by depth?
                // todo: interpolate by game phase?

                let hh_weight = self.params.hh_weight();
                let total_weight = self.params.total_weight();

                interpolate_i32(psqt_score as i32, hh_score as i32, hh_weight, total_weight) as MoveScore
            }

            RtStage::Done => 0,
        }
    }
}

fn uci_info(depth: Depth, stats: &SearchStats, best_score: Cp, best_move: Move, search_time: Duration, pv: &Line) {
    let depth = UciArg::Some(UciDepth(depth));
    let seldepth = UciArg::<UciSeldepth>::None; // TODO
    let score = UciArg::Some(UciScore::Centipawns(UciCp(best_score)));
    let nodes = UciArg::Some(UciNodes(stats.nodes as usize));
    let nps = UciArg::Some(UciNps::from_nodes_and_time(stats.nodes, search_time));
    let currmove = UciArg::Some(UciCurrmove(best_move));
    let time = UciArg::Some(UciSearchtime(search_time));
    let pv = UciArg::Some(UciPv(pv));
    let string = UciArg::<String>::None;

    println!("info{currmove}{score}{nodes}{nps}{depth}{seldepth}{time}{pv}{string}");
}
