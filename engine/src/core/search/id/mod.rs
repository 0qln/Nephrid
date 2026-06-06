use std::{cmp::Reverse, hint::assert_unchecked, ops::ControlFlow, time::Instant};

use crate::{
    core::{
        color::{
            Perspective, colors,
            perspectives::{self},
        },
        depth::Depth,
        r#move::{MAX_LEGAL_MOVES, Move, MoveList},
        move_iter::fold_legal_moves,
        params::HceParams,
        ply::Ply,
        position::{CheckState, Position},
        search::{
            id::node_types::*,
            limit::UciLimit,
            mcts::eval::{
                GameResult,
                hce::{self, PolicyInput, TaperValue, see},
            },
            score::{Cp, Score},
            strat::{
                UciArg, UciCp, UciCurrmove, UciDepth, UciNodes, UciNps, UciPv, UciScore,
                UciSearchtime, UciSeldepth,
            },
            tt::{self, TranspositionTable},
        },
        zobrist,
    },
    misc::{CancellationToken, DebugMode, List},
};

#[derive(Default)]
struct SearchStats {
    nodes: u64,
    iterations: u64,
}

pub fn go(
    pos: &mut Position,
    limit: UciLimit,
    _debug: &DebugMode,
    ct: CancellationToken,
) -> Option<Move> {
    let mut searcher = Searcher::new(pos, limit, ct);
    let mut stats = SearchStats::default();
    let mut best_move = None;

    let mut depth = Depth::ROOT + 1;

    // todo: cap at Depth::MAX
    while !searcher.should_stop(&stats) {
        let best_score = searcher.search_root(pos, &mut stats, depth);

        // make sure to break before messing up the order of the previous iteration with
        // the incomplete results from this iteration.
        if searcher.aborted {
            break;
        }

        searcher.sort_root();

        best_move = searcher.root_best_move();
        if let Some(best_move) = best_move {
            uci_info(depth, &stats, Cp { v: best_score as i16 }, best_move);
        }

        // update stats
        stats.iterations += 1;

        // increment depth for next iteration
        depth += 1;
    }

    best_move
}

struct RootStats {
    score: i32,
    mov: Move,
}

struct Searcher {
    root_stats: List<{ MAX_LEGAL_MOVES }, RootStats>,
    root_ply: Ply,
    limit: UciLimit,
    time_limit: Instant,
    ct: CancellationToken,
    aborted: bool,
    tt: TranspositionTable<TTEntry>,
    ss: SearchStack,
}

impl Searcher {
    fn new(pos: &Position, limit: UciLimit, ct: CancellationToken) -> Self {
        let mut stats = List::<{ MAX_LEGAL_MOVES }, RootStats>::new();
        _ = fold_legal_moves::<_, _, _>(pos, (), |_, m| {
            stats.push(RootStats { mov: m, score: 0 });
            ControlFlow::Continue::<(), ()>(())
        });

        let search_start = Instant::now();
        let time_per_move = limit.time_per_move(pos);
        let time_limit = search_start + time_per_move;

        Self {
            root_stats: stats,
            root_ply: pos.ply(),
            limit,
            time_limit,
            ct,
            aborted: false,
            tt: TranspositionTable::new(1 << 20), // TODO: make this configurable
            ss: SearchStack::new(),
        }
    }

    fn sort_root(&mut self) {
        self.root_stats
            .as_mut_slice()
            .sort_by_key(|mov| Reverse(mov.score));
    }

    fn root_best_move(&self) -> Option<Move> {
        self.root_stats.get(0).map(|x| x.mov)
    }

    fn should_stop(&self, stats: &SearchStats) -> bool {
        let now = Instant::now();
        let nodes = stats.nodes;
        let iters = stats.iterations;

        // user requested stop
        if self.ct.is_cancelled() {
            return true;
        }

        // limit has been reached
        if self.limit.is_active() && self.limit.is_reached(nodes, now, self.time_limit, iters) {
            return true;
        }

        // otherwise keep searching
        false
    }

    /// returns the score relative to the current player
    fn search_root(&mut self, pos: &mut Position, stats: &mut SearchStats, depth: Depth) -> i32 {
        match pos.get_turn() {
            colors::WHITE => {
                self.search::<perspectives::White, Root>(
                    pos,
                    stats,
                    depth,
                    Score::NEG_INF,
                    Score::POS_INF,
                )
                .0
            }
            colors::BLACK => {
                self.search::<perspectives::Black, Root>(
                    pos,
                    stats,
                    depth,
                    Score::NEG_INF,
                    Score::POS_INF,
                )
                .0
            }
            _ => unreachable!(),
        }
    }

    // todo: remove the dependency on mcts::hce and write a custom one.
    //
    /// returns the score relative to `P`
    fn search<P: Perspective, T: NodeType>(
        &mut self,
        pos: &mut Position,
        stats: &mut SearchStats,
        depth: Depth,
        mut alpha: Score<P>,
        beta: Score<P>,
    ) -> Score<P> {
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
                GameResult::Draw => Score::new(0),
            };
        }

        // qsearch at the leaf nodes
        if depth == Depth::ROOT {
            return hce::qsearch(pos, alpha, beta, HceParams);
        }

        // vars
        let rel_ply: Depth = (pos.ply() - self.root_ply).into();
        let pieces = pos.piece_info();
        let phase = TaperValue::from_position(pieces);
        let is_root = T::IS_ROOT;
        let stm = P::COLOR;
        let key = pos.get_key();
        let orig_alpha = alpha;
        let tt_entry = self.tt.get(key);

        // tt-cutoff
        if !is_root
            && let Some(entry) = tt_entry
            && entry.depth >= depth
            && ((entry.bound == Bound::Exact)
                || (entry.bound == Bound::Lower && entry.score >= beta.0)
                || (entry.bound == Bound::Upper && entry.score <= alpha.0))
        {
            return Score::new(entry.score);
        }

        let tt_move = tt_entry.map(|e| e.mov);

        // move gen
        let mut moves = MoveList::new();
        if is_root {
            // if root node and if we have info on them, they are already ordered by the
            // previous iteration. othewise this is the first iteration, so order them by
            // generic move ordering (the branch below).
            for m in self.root_stats.iter() {
                moves.push(m.mov);
            }
        }
        else {
            // generic move ordering
            let mut move_list = List::<{ MAX_LEGAL_MOVES }, (Move, i32)>::new();
            _ = fold_legal_moves::<_, _, _>(pos, (), |_, m| {
                move_list.push((m, 0));
                ControlFlow::Continue::<(), ()>(())
            });

            let killers = &self.ss.entry(rel_ply).killers;

            // generate the see score outside of the move generation and the sorting, such
            // that it isn't computed for each comparison and we don't distrurb cache
            // locality.
            for &mut (m, ref mut score) in move_list.as_mut_slice() {
                *score = if Some(m) == tt_move {
                    300_000
                }
                else {
                    // todo: currently see for quiet moves evaluates promotion values etc. there is
                    // probably a better way to order promotions than using see. then we can skip
                    // see for quiets all together...

                    let is_capture = m.get_flag().is_capture();

                    if is_capture {
                        let see = see(pieces, m, P::COLOR);
                        if see >= 0 {
                            // good captures (210_000..)
                            210_000 + see
                        }
                        else {
                            // bad captures (100_000..)
                            100_000 + see
                        }
                    }
                    // T1 killers (..200_000)
                    else if let Some(age) = killers._position(&m) {
                        200_000 - (age as i32 * 10_000)
                    }
                    else {
                        let (from, to, _) = m.into();
                        let piece = pieces.get_piece(from);
                        let piece_type = piece.piece_type();
                        let see = see(pieces, m, P::COLOR);

                        see + PolicyInput::psqt(phase, piece_type, from, to, stm)
                    }
                };
            }

            // sort by score descending
            move_list
                .as_mut_slice()
                .sort_unstable_by_key(|&(_, score)| Reverse(score));

            for &(m, _) in move_list.as_slice() {
                moves.push(m);
            }
        };

        let mut best_score = Score::NEG_INF;
        let mut best_move = Move::null();
        for (i, m) in moves.iter().copied().enumerate() {
            // make the move
            pos.make_move_for::<P>(m);

            // depth
            let mut depth_extension = 0;
            let mut depth_reduction = 0;
            let gives_check = pos.get_check_state() != CheckState::None;

            // check extensions
            if gives_check {
                depth_extension += 1;
            }

            // late move reductions
            #[allow(clippy::approx_constant)]
            if depth >= Depth::new(3) && i > 1 {
                let d = depth.v() as f32;
                let m = i as f32;
                let lmr = 0.99 + f32::ln(d) * f32::ln(m) / 3.14;
                depth_reduction += lmr as u8;
            }

            // recurse
            let score = {
                let new_depth = depth - 1 + depth_extension;

                if i == 0 {
                    // search with a full window to get an exact score.
                    !self.search::<P::Opponent, Normal>(pos, stats, new_depth, !beta, !alpha)
                }
                else {
                    // assume that our move ordering is good the first move will be the best one.
                    // to prove that this move cannot improve our first move, perform a zero window
                    // search with [a,a+1] (~ [-(a-1),-a]). we don't care by how much this move is
                    // able to improve alpha since we assume that it cannot.
                    let mut zws_score = !self.search::<P::Opponent, Normal>(
                        pos,
                        stats,
                        // scout with a reduced depth
                        new_depth - depth_reduction,
                        !(alpha + Score::new(1)),
                        !alpha,
                    );

                    // if the reduced depth search fails high, we must verify that it is actually
                    // good and do a full depth re-search.
                    if zws_score > alpha && depth_reduction > 0 {
                        zws_score = !self.search::<P::Opponent, Normal>(
                            pos,
                            stats,
                            // research at full depth
                            new_depth,
                            // still zero-window.
                            !(alpha + Score::new(1)),
                            !alpha,
                        );
                    }

                    // if zws_score is a lower_bound, we have to research to get an exact score.
                    if zws_score > alpha
                        // don't bother researching if this move will cause a fail-high anyway
                        && zws_score < beta
                    {
                        !self.search::<P::Opponent, Normal>(
                            pos, stats, new_depth, !beta,
                            // new lower_bound, since it was able to beat alpha
                            !zws_score,
                        )
                    }
                    else {
                        zws_score
                    }
                }
            };

            // unmake the move
            pos.unmake_move_for::<P>(m);

            if self.aborted {
                return Score::new(0);
            }

            if is_root {
                // store the score for the root moves, such that we can use it for sorting in
                // the next iteration.
                self.root_stats.as_mut_slice()[i].score = score.0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
            }

            if score > alpha {
                alpha = score;

                if score >= beta {
                    // mark quiet moves, fail-high as killer moves
                    if !m.get_flag().is_capture() && Some(m) != tt_move {
                        self.ss.entry_mut(rel_ply).killers._push(m);
                    }

                    // fail high
                    break;
                }
            }
            else {
                // fail low
            }
        }

        self.tt.insert(TTEntry {
            key,
            depth,
            score: best_score.0,
            bound: Bound::from_scores(orig_alpha, beta, best_score),
            mov: best_move,
        });

        best_score
    }
}

trait NodeType {
    const IS_ROOT: bool;
}

mod node_types {
    use super::NodeType;

    pub struct Root;
    impl NodeType for Root {
        const IS_ROOT: bool = true;
    }

    pub struct Normal;
    impl NodeType for Normal {
        const IS_ROOT: bool = false;
    }
}

#[derive(Clone)]
pub struct TTEntry {
    key: zobrist::Hash,
    depth: Depth,
    score: i32,
    bound: Bound,
    mov: Move,
}

impl tt::ZKey for TTEntry {
    fn key(&self) -> zobrist::Hash {
        self.key
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Bound {
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

#[derive(Default)]
struct SearchStack {
    entries: Vec<SearchEntry>,
}

impl SearchStack {
    pub fn new() -> Self {
        Self {
            entries: vec![SearchEntry::default(); Depth::MAX.v() as usize + 1],
        }
    }

    pub fn entry_mut(&mut self, ply: Depth) -> &mut SearchEntry {
        let idx = ply.v() as usize;
        // Safety: entries is atleast Depth::MAX + 1
        unsafe { self.entries.get_unchecked_mut(idx) }
    }

    pub fn entry(&self, ply: Depth) -> &SearchEntry {
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
    fn from(items: [T; N]) -> Self {
        Self { items }
    }
}

impl<T: const Default + Copy + Eq, const N: usize> RbSet<T, N> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self { items: [T::default(); N] }
    }

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
    pub fn position(&self, item: &T) -> Option<usize> {
        self.items.iter().position(|x| x == item)
    }
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
}

fn uci_info(depth: Depth, stats: &SearchStats, best_score: Cp, best_move: Move) {
    let depth = UciArg::Some(UciDepth(depth));
    let seldepth = UciArg::<UciSeldepth>::None; // TODO
    let score = UciArg::Some(UciScore::Centipawns(UciCp(best_score)));
    let nodes = UciArg::Some(UciNodes(stats.nodes as usize));
    let nps = UciArg::<UciNps>::None; // TODO
    let currmove = UciArg::Some(UciCurrmove(best_move));
    let time = UciArg::<UciSearchtime>::None; // TODO
    let pv = UciArg::<UciPv<MoveList>>::None; // TODO
    let string = UciArg::<String>::None;

    println!("info{currmove}{score}{nodes}{nps}{depth}{seldepth}{time}{pv}{string}");
}
