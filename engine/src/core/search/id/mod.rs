use std::{cmp::Reverse, ops::ControlFlow, time::Instant};

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
        position::Position,
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
    limit: UciLimit,
    time_limit: Instant,
    ct: CancellationToken,
    aborted: bool,
    tt: TranspositionTable<TTEntry>,
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
            limit,
            time_limit,
            ct,
            aborted: false,
            tt: TranspositionTable::new(1 << 20), // TODO: make this configurable
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

        // start of assuming this is a loss
        let mut best_score = Score::NEG_INF;

        // vars
        let piece_info = pos.piece_info();
        let phase = TaperValue::from_position(piece_info);
        let is_root = T::IS_ROOT;
        let stm = P::COLOR;
        let key = pos.get_key();
        let orig_alpha = alpha;

        // tt-cutoff
        if !is_root && let Some(entry) = self.tt.get(key) {
            if entry.depth >= depth
                && ((entry.bound == Bound::Exact)
                    || (entry.bound == Bound::Lower && entry.score >= beta.0)
                    || (entry.bound == Bound::Upper && entry.score <= alpha.0))
            {
                return Score::new(entry.score);
            }
        }

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

            // generate the see score outside of the move generation and the sorting, such
            // that it isn't computed for each comparison and we don't distrurb cache
            // locality.
            for &mut (m, ref mut score) in move_list.as_mut_slice() {
                let (from, to, _) = m.into();
                let piece = pos.get_piece(from);

                *score = see(pos.piece_info(), m, P::COLOR)
                    + PolicyInput::psqt(phase, piece.piece_type(), from, to, stm);
            }

            // move ordering
            move_list
                .as_mut_slice()
                .sort_unstable_by_key(|&(_, score)| Reverse(score));

            for &(m, _) in move_list.as_slice() {
                moves.push(m);
            }
        };

        for (i, m) in moves.iter().copied().enumerate() {
            // make the move
            pos.make_move_for::<P>(m);

            // recurse
            let score = !self.search::<P::Opponent, Normal>(pos, stats, depth - 1, !beta, !alpha);

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

            if score >= beta {
                best_score = score;
                break;
            }
            if score > best_score {
                best_score = score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        self.tt.insert(TTEntry {
            key,
            depth,
            score: best_score.0,
            bound: Bound::from_scores(orig_alpha, beta, best_score),
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
