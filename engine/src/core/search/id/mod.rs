use core::fmt;
use std::{
    cmp::{Reverse, max, min},
    convert::Infallible,
    hint::assert_unchecked,
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
        piece::piece_type,
        ply::Ply,
        position::{CheckState, PieceInfo, PieceInfoObserver, Position},
        search::{
            data::{self, TTBound, TTDepth, TTKey, TTMove, TTScore, TranspositionTable},
            limit::UciLimit,
            mcts::eval::Quality,
            ordering::{self, MovePicker, MoveScore, MoveScorer, RtStage, ScoredMove, Stage},
            quiesce::{QSearchParams, qsearch},
            score::{AnyScore, Cp, Score, scores},
            strat::{UciArg, UciCp, UciCurrmove, UciDepth, UciNodes, UciNps, UciPv, UciScore, UciSearchtime, UciSeldepth},
            tree::{NodeKind, NodeType, node_types::*},
        },
        turn::Turn,
        zobrist,
    },
    math::{self, NormalizedEntropy},
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
        Score::new(nnue_eval)
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
        let mut max_threat = Score::<P::Opponent>::new(scores::DRAW);

        // todo: only generate captures, promos, and checks.
        let moves = pos.collect_legals_for::<P::Opponent, _>(MoveList::new());
        for &mov in moves.iter() {
            match pos.does_check(mov) {
                CheckState::None => {}
                CheckState::Single => return Score::new(hce::piece_score(piece_type::QUEEN)),
                CheckState::Double => return Score::new(hce::piece_score(piece_type::QUEEN) + hce::piece_score(piece_type::ROOK)),
            }

            if mov.get_flag().is_capture() {
                let see = ordering::see(pos.piece_info(), mov, P::Opponent::COLOR);
                max_threat = max(max_threat, Score::from(see));
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
}

#[allow(clippy::too_many_arguments)]
pub fn go(
    pos: &mut Position,
    limit: UciLimit,
    timeman: &mut TimeMan,
    debug: &DebugMode,
    ct: CancellationToken,
    tt: &mut TT,
    eval: &mut impl StaticEvaluator,
    params: impl ChronoParams + QSearchParams + IdParams + Clone + fmt::Debug,
) -> Option<Move> {
    let depth_lim = min(Depth::MAX, limit.depth);

    eval.observe_forward().on_init(pos.piece_info());

    if debug.get() {
        println!("info string Starting ID Search with Params: {params:?}");
    }

    let mut searcher = Searcher::new(pos, limit, timeman, ct, tt, eval);
    let mut stats = SearchStats::default();
    let mut best_move = None;
    let mut last_best_move;

    for depth in (Depth::ROOT + 1)..=depth_lim {
        let best_score = searcher.search_root(params.clone(), pos, &mut stats, depth);

        // make sure to break before messing up the order of the previous iteration with
        // the incomplete results from this iteration.
        if searcher.aborted {
            break;
        }

        last_best_move = searcher.root_best_move();

        searcher.sort_root();

        best_move = searcher.root_best_move();
        if let Some(best_move) = best_move {
            let search_time = Instant::now() - searcher.timeman.time_start();
            uci_info(depth, &stats, Cp::from(best_score), best_move, search_time);
        }

        // update stats
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
        timeman.hint_time_target(timeman.time_limit().map(|x| x - stats.iter_time));
        timeman.hint_movestreak_target(Some(params.movestreak_target()));
        timeman.set_curr_movestreak(stats.root_movestreak);

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

struct Searcher<'a, 'b, E: StaticEvaluator> {
    root_stats: List<{ MAX_LEGAL_MOVES }, RootStats>,
    root_ply: Ply,
    limit: UciLimit,
    timeman: &'a mut TimeMan,
    ct: CancellationToken,
    aborted: bool,
    ss: SearchStack<SearchEntry>,
    tt: &'a mut TT,
    eval: &'b mut E,
    phase: TaperValue,
}

impl<'a, 'b, E: StaticEvaluator> Searcher<'a, 'b, E> {
    fn new(pos: &Position, limit: UciLimit, timeman: &'a mut TimeMan, ct: CancellationToken, tt: &'a mut TT, eval: &'b mut E) -> Self {
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
            ss: SearchStack::new(),
            tt,
            eval,
            phase: TaperValue::from_position(pos.piece_info()),
        }
    }

    fn sort_root(&mut self) { self.root_stats.as_mut_slice().sort_by_key(|mov| Reverse(mov.score())); }

    fn root_best_move(&self) -> Option<Move> { self.root_stats.get(0).map(|x| x.mov()) }

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
    fn search_root(&mut self, params: impl QSearchParams + IdParams + Clone, pos: &mut Position, stats: &mut SearchStats, depth: Depth) -> AnyScore {
        fn alpha<P: Perspective>() -> Score<P> { Score::new(scores::NEG_INF) }
        fn beta<P: Perspective>() -> Score<P> { Score::new(scores::POS_INF) }
        match pos.get_turn() {
            colors::WHITE => self.search::<perspectives::White, Root>(params, pos, stats, depth, alpha(), beta()).0,
            colors::BLACK => self.search::<perspectives::Black, Root>(params, pos, stats, depth, alpha(), beta()).0,
            _ => unreachable!(),
        }
    }

    /// returns the score relative to `P`
    fn search<P: Perspective, T: NodeType>(
        &mut self,
        params: impl QSearchParams + IdParams + Clone,
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
            return scores::NEG_INF.into();
        }

        // check if game is over
        if let Some(result) = pos.game_result() {
            return match result {
                GameResult::Win { .. } => scores::NEG_INF.into(),
                GameResult::Draw => scores::DRAW.into(),
            };
        }

        let rel_ply: Depth = (pos.ply() - self.root_ply).into();

        // qsearch at the leaf nodes
        if depth == Depth::ROOT || rel_ply >= Depth::MAX {
            return qsearch::<P>(pos, self.phase, alpha, beta, params, self.eval, Depth::new(100));
        }

        let kind = T::KIND;
        let is_root_node = kind == NodeKind::Root;
        let key = pos.get_key();
        let orig_alpha = alpha;
        let killers = self.ss.get(rel_ply).killers.clone();

        // tt-cutoff
        {
            let tt_entry = self.tt.get(key);
            if !is_root_node
                && let Some(entry) = tt_entry
                && entry.depth >= depth
                && ((entry.bound == Bound::Exact)
                    || (entry.bound == Bound::Lower && entry.score >= beta.0)
                    || (entry.bound == Bound::Upper && entry.score <= alpha.0))
            {
                return Score::new(entry.score);
            }
        }

        // todo: these 'is it already computed? if so, return it.' are not required.
        // they are just for convenience when compiling with different
        // features... find a clean way to solve this or make sure the compiler
        // can understand when they will already be computed...

        // this node's static eval.
        #[cfg(any(feature = "id-fhr", feature = "id-nmp"))]
        let mut static_eval = Score::<P>::new(scores::NULL);

        #[cfg(any(feature = "id-fhr", feature = "id-nmp"))]
        let mut lazy_static_eval = |this: &mut Self, pos: &Position| {
            // is it already computed? if so, return it.
            if static_eval.0.is_valid() {
                return static_eval;
            }

            let mut compute = || this.eval.eval(pos.piece_info(), P::COLOR, pos.get_ep_target_square(), this.phase);

            // check the tt
            if let Some(tt_entry) = this.tt.raw_mut(key) {
                use crate::core::search::data::TTStaticEval;
                let tt_score = tt_entry.static_eval_mut();
                // if the tt contains a valid static_eval, return it.
                if tt_score.is_valid() {
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

        #[cfg(feature = "id-fhr")]
        let mut threat = Score::<P::Opponent>::new(scores::NULL);

        #[cfg(feature = "id-fhr")]
        let mut lazy_threat_score = |this: &mut Self, pos: &Position| {
            // is it already computed? if so, return it.
            if threat.0.is_valid() {
                return threat;
            }

            let compute = || threatener.threat::<P>(pos);

            // check the tt
            if let Some(tt_entry) = this.tt.raw_mut(key) {
                let tt_threat = &mut tt_entry.threat;
                // if the tt contains a valid static_eval, return it.
                if tt_threat.is_valid() {
                    threat = unsafe { tt_threat.interpret_as() };
                }
                // else compute it
                else {
                    threat = compute();
                    *tt_threat = threat.0;
                }
            }
            // if theres a foreign tt entry blocking our current key, just compute and don't store.
            else {
                threat = compute();
            }

            threat
        };

        // null move pruning
        #[cfg(feature = "id-nmp")]
        {
            let nmp_r: Depth = params.nmp_reduction()
                // scale the reduction up based on depth
                + depth.div_floor(params.nmp_depth_factor());
            // todo: test this idea
            // // scale the reduction down based on phase (we want deeper searches in the
            // endgame)
            // - Depth::new(phase.v().div_floor(params.nmp_phase_factor()) as u8); // todo:
            //   honestly phase could just be a u8
            let is_in_check = pos.get_check_state() != CheckState::None;
            if kind == NodeKind::Cut
                // don't underflow depth
                && depth > nmp_r
                // don't allow nmp when node is in check
                && !is_in_check
                // don't do nmp in endgames, where zugzwang is more likely
                && self.phase < params.nmp_phase_threshold() && pos.has_non_pawn_material::<P>()
                // don't bother attempting to improve beta with a tempo down when our static eval is not
                // even better than beta
                && lazy_static_eval(self, pos) >= beta - Score::new(params.nmp_margin()) - Score::from(depth.v() * 15)
            {
                pos.make_null_move();

                let nm_score = !self.search::<P::Opponent, Normal>(
                    params.clone(),
                    pos,
                    stats,
                    // scout with a reduced depth
                    depth - nmp_r - 1,
                    !(alpha + 1),
                    !alpha,
                );

                pos.unmake_null_move();

                // todo: verification search?

                if nm_score >= beta {
                    return nm_score;
                }
            }
        }

        // move gen
        let tt_entry = self.tt.get(key);
        let tt_move = tt_entry.map(|e| e.mov).unwrap_or(Move::null());
        let mut move_picker = if is_root_node {
            MovePicker::from_scored(self.root_stats.iter().map(|m| m.scored_move()).cloned())
        }
        else {
            MovePicker::new(tt_move, killers.clone())
        };

        let scorer = Scorer {
            tt_move,
            killers,
            color: P::COLOR,
            phase: self.phase,
        };

        // fail-high reductions
        let fhr_reduct = cfg_select! {
            feature = "id-fhr" => {{
                let in_check = pos.get_check_state() != CheckState::None;

                if kind == NodeKind::Cut && !in_check {
                    // the quiet score of this position is the static score minus threat score (the
                    // best threat that the opponent can do).
                    let q_score = lazy_static_eval(self, pos) + !lazy_threat_score(self, pos);

                    // if the quiet score
                    if q_score >= beta { 1 } else { 0 }
                }
                else {
                    0
                }
            }}
            _ => 0
        };

        let mut best_score = Score::new(scores::NEG_INF);
        let mut best_move = Move::null();
        let mut curr = 0;
        while let Some(m) = move_picker.next_for::<P>(pos, &scorer) {
            // generating plegals and then filtering hasn't show to be faster, maybe
            // optimize this more and then try again.
            // if !pos.is_legal_for::<P>(m) {
            //     continue;
            // }

            // make the move
            let phase_before = self.phase;
            self.eval.forward();
            pos.make_move_for::<P>(m, &mut (&mut self.phase, self.eval.observe_forward()));

            // depth
            let (mut depth_ext, mut depth_reduct) = (0, 0);

            let gives_check = pos.get_check_state() != CheckState::None;

            // check extensions
            if gives_check {
                depth_ext += 1;
            }

            // late move reductions
            #[allow(clippy::approx_constant)]
            if depth >= Depth::new(3) && curr > 1 {
                let d = depth.v() as f32;
                let m = curr as f32;
                let lmr = 0.99 + f32::ln(d) * f32::ln(m) / 3.14;
                depth_reduct += lmr as u8;
            }

            // recurse
            let score = {
                let new_depth = depth - 1 + depth_ext;
                let full_depth = new_depth.saturating_sub(fhr_reduct);
                let reduced_depth = new_depth.saturating_sub(depth_reduct + fhr_reduct);

                if curr == 0 {
                    // search with a full window to get an exact score.
                    !self.search::<P::Opponent, Normal>(params.clone(), pos, stats, full_depth, !beta, !alpha)
                }
                else {
                    // assume that our move ordering is good the first move will be the best one.
                    // to prove that this move cannot improve our first move, perform a zero window
                    // search with [a,a+1] (~ [-(a-1),-a]). we don't care by how much this move is
                    // able to improve alpha since we assume that it cannot.
                    let mut zws_score = !self.search::<P::Opponent, Cut>(
                        params.clone(),
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
                            params.clone(),
                            pos,
                            stats,
                            // research at full depth
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
                        !self.search::<P::Opponent, Normal>(
                            params.clone(),
                            pos,
                            stats,
                            full_depth,
                            !beta,
                            !zws_score, // new lower_bound, since it was able to beat alpha
                        )
                    }
                    else {
                        zws_score
                    }
                }
            };

            // unmake the move
            pos.unmake_move_for::<P>(m, self.eval.observe_backward());
            self.eval.backward();
            self.phase = phase_before;

            // check for cancellation
            if self.aborted {
                return Score::new(scores::DRAW);
            }

            // update root moves
            if is_root_node {
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
                alpha = score;

                if score >= beta {
                    // mark quiet moves, fail-high as killer moves
                    if !m.get_flag().is_capture() && m != tt_move {
                        self.ss.get_mut(rel_ply).killers._push(m);
                    }

                    // fail high
                    break;
                }
            }
            else {
                // fail low
            }

            curr += 1;
        }

        self.tt.try_insert(TTEntry {
            key,
            depth,
            score: best_score.0,
            #[cfg(any(feature = "id-fhr", feature = "id-nmp"))]
            static_eval: static_eval.0,
            #[cfg(feature = "id-fhr")]
            threat: threat.0,
            bound: Bound::from_scores(orig_alpha, beta, best_score),
            mov: best_move,
        });

        best_score
    }
}

pub type TT = TranspositionTable<TTEntry, TTReplace>;

#[derive(Clone)]
pub struct TTEntry {
    key: zobrist::Hash,
    depth: Depth,
    score: AnyScore,
    #[cfg(any(feature = "id-fhr", feature = "id-nmp"))]
    static_eval: AnyScore,
    #[cfg(feature = "id-fhr")]
    threat: AnyScore,
    bound: Bound,
    mov: Move,
}

impl const TTKey for TTEntry {
    fn key(&self) -> zobrist::Hash { self.key }
}

impl const TTScore for TTEntry {
    fn score(&self) -> AnyScore { self.score }
}

impl const TTMove for TTEntry {
    fn mov(&self) -> Move { self.mov }
}

impl const TTDepth for TTEntry {
    fn depth(&self) -> Depth { self.depth }
}

impl const TTBound for TTEntry {
    fn bound(&self) -> Bound { self.bound }
}

#[cfg(any(feature = "id-fhr", feature = "id-nmp"))]
impl const data::TTStaticEval for TTEntry {
    fn static_eval(&self) -> AnyScore { self.static_eval }
    fn static_eval_mut(&mut self) -> &mut AnyScore { &mut self.static_eval }
}

impl const Default for TTEntry {
    fn default() -> Self {
        Self {
            key: zobrist::Hash::default(),
            depth: Depth::NONE,
            score: scores::NULL,
            #[cfg(any(feature = "id-fhr", feature = "id-nmp"))]
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

    fn should_replace(_: &TTEntry, _: &TTEntry) -> bool { true }
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

pub struct SearchStack<T> {
    entries: Vec<T>,
}

impl<T: Clone + Default> Default for SearchStack<T> {
    fn default() -> Self { Self::new() }
}

impl<T: Clone + Default> SearchStack<T> {
    pub fn new() -> Self {
        Self {
            entries: vec![T::default(); Depth::MAX.v() as usize + 1],
        }
    }
}

impl<T> SearchStack<T> {
    #[inline(always)]
    pub fn propagate(&mut self, old: Depth, new: Depth, mut f: impl FnMut(&T, &mut T)) {
        let old_idx = old.v() as usize;
        let new_idx = new.v() as usize;
        unsafe {
            let [parent, child] = self.entries.get_disjoint_unchecked_mut([old_idx, new_idx]);
            f(parent, child);
        }
    }

    pub fn get_mut(&mut self, ply: Depth) -> &mut T {
        let idx = ply.v() as usize;

        // todo: qsearch might exceed depth max ??

        // Safety: entries is atleast Depth::MAX + 1
        unsafe { self.entries.get_unchecked_mut(idx) }
    }

    pub fn get(&self, ply: Depth) -> &T {
        let idx = ply.v() as usize;

        // Safety: entries is atleast Depth::MAX + 1
        unsafe { self.entries.get_unchecked(idx) }
    }
}

#[derive(Default, Clone, Debug)]
pub struct SearchEntry {
    killers: RbSet<Move, 2>,
}

/// A Ring Buffer Set of size `N`.
///
/// Maintains up to `N` unique elements. When an element is pushed:
/// - If it already exists, it is promoted to the front (index 0), and the
///   elements before its old position are shifted down.
/// - If it is new, all elements are shifted down, evicting the oldest.
///
/// # Examples
///
/// ```
/// # use engine::core::search::id::RbSet;
///
/// let mut killers = RbSet::<i32, 3>::new();
///
/// killers.push(10);
/// killers.push(20);
/// killers.push(30);
/// assert_eq!(killers, RbSet::from([30, 20, 10]));
///
/// // Pushing an existing element moves it to the front (Promotes it)
/// killers.push(20);
/// assert_eq!(killers, RbSet::from([20, 30, 10]));
///
/// // Pushing a new element evicts the oldest (10 drops off)
/// killers.push(40);
/// assert_eq!(killers, RbSet::from([40, 20, 30]));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RbSet<T, const N: usize> {
    items: [T; N],
}

impl<T: const Default, const N: usize> const Default for RbSet<T, N> {
    #[inline]
    fn default() -> Self {
        Self {
            items: [const { T::default() }; N],
        }
    }
}

impl<T: const Default + Copy + Eq, const N: usize> const From<[T; N]> for RbSet<T, N> {
    fn from(items: [T; N]) -> Self { Self { items } }
}

impl<T: const Default + Copy + Eq, const N: usize> RbSet<T, N> {
    #[inline(always)]
    pub const fn new() -> Self { Self { items: [T::default(); N] } }

    // todo: make sure this is unrolled for our N=2/3
    // todo: this is O(n) but i don't  think this matters for our n=2 lmao
    #[inline(always)]
    pub fn push(&mut self, item: T) {
        let pos = self.position(&item).unwrap_or(N - 1);

        // Safety: pos is either the index of the item in the set, or the last index if
        // the item is not
        unsafe {
            assert_unchecked(pos < N);
        }

        for i in (1..=pos).rev() {
            self.items[i] = self.items[i - 1];
        }

        self.items[0] = item;
    }

    #[inline(always)]
    pub fn position(&self, item: &T) -> Option<usize> { self.items.iter().position(|x| x == item) }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] { &self.items }
}

// todo: benchmark that this is actually faster...
/// spezialized version of the const generic impls.
impl<T: Default + Copy + Eq> RbSet<T, 2> {
    #[inline(always)]
    pub fn _push(&mut self, item: T) {
        if self.items[0] != item {
            self.items[1] = self.items[0];
            self.items[0] = item;
        }
    }

    #[inline(always)]
    pub fn _position(&self, item: &T) -> Option<usize> {
        if self.items[0] == *item {
            Some(0)
        }
        else if self.items[1] == *item {
            Some(1)
        }
        else {
            None
        }
    }

    #[inline(always)]
    pub fn _is_empty(&self) -> bool { self.items[0] == T::default() && self.items[1] == T::default() }
}

pub struct Scorer {
    pub tt_move: Move,
    pub killers: RbSet<Move, 2>,
    pub color: Color,
    pub phase: TaperValue,
}

impl MoveScorer for Scorer {
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

            // score quiet moves by psqt diff
            RtStage::GenerateQuiets | RtStage::YieldQuiets => {
                let (from, to, _) = mov.into();
                let pieces = pos.piece_info();
                let piece = pieces.get_piece(from);
                let piece_type = piece.piece_type();

                ordering::psqt(self.phase, piece_type, from, to, mov.get_flag(), self.color)
            }

            RtStage::Done => 0,
        }
    }
}

fn uci_info(depth: Depth, stats: &SearchStats, best_score: Cp, best_move: Move, search_time: Duration) {
    let depth = UciArg::Some(UciDepth(depth));
    let seldepth = UciArg::<UciSeldepth>::None; // TODO
    let score = UciArg::Some(UciScore::Centipawns(UciCp(best_score)));
    let nodes = UciArg::Some(UciNodes(stats.nodes as usize));
    let nps = UciArg::Some(UciNps::from_nodes_and_time(stats.nodes, search_time));
    let currmove = UciArg::Some(UciCurrmove(best_move));
    let time = UciArg::Some(UciSearchtime(search_time));
    let pv = UciArg::<UciPv<MoveList>>::None; // TODO
    let string = UciArg::<String>::None;

    println!("info{currmove}{score}{nodes}{nps}{depth}{seldepth}{time}{pv}{string}");
}
