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
        },
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
    ct: &CancellationToken,
) -> Option<Move> {
    let mut searcher = Searcher::from_pos(&pos);
    let mut stats = SearchStats::default();
    let mut best_move = None;

    let search_start = Instant::now();
    let time_per_move = limit.time_per_move(pos);
    let time_limit = search_start + time_per_move;

    let should_stop = |stats: &SearchStats| {
        let now = Instant::now();
        let nodes = stats.nodes;
        let iters = stats.iterations;

        // user requested stop
        ct.is_cancelled()

        // limit has been reached
            || (limit.is_active() && limit.is_reached(nodes, now, time_limit, iters))
    };

    let mut depth = Depth::ROOT + 1;

    while !should_stop(&stats) {
        searcher.search_root(pos, &mut stats, depth);
        searcher.sort_root();

        best_move = searcher.root_best_move();
        if let Some(best_move) = best_move {
            // todo cleanup
            let best_score = searcher.root_stats.get(0).unwrap().score;
            uci_info(depth, &stats, Cp { v: best_score as i16 }, best_move);
        }

        depth += 1;
    }

    best_move
}

struct RootStats {
    score: i32,
    mov: Move,
}

#[derive(Default)]
struct Searcher {
    root_stats: List<{ MAX_LEGAL_MOVES }, RootStats>,
}

impl Searcher {
    fn from_pos(pos: &Position) -> Self {
        let mut stats = List::<{ MAX_LEGAL_MOVES }, RootStats>::new();
        _ = fold_legal_moves::<_, _, _>(pos, (), |_, m| {
            stats.push(RootStats { mov: m, score: 0 });
            ControlFlow::Continue::<(), ()>(())
        });

        Self { root_stats: stats }
    }

    fn sort_root(&mut self) {
        self.root_stats
            .as_mut_slice()
            .sort_by_key(|mov| Reverse(mov.score));
    }

    fn root_best_move(&self) -> Option<Move> {
        self.root_stats.get(0).map(|x| x.mov)
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

        for m in moves.iter().copied() {
            // make the move
            pos.make_move_for::<P>(m);

            // recurse
            let score = !self.search::<P::Opponent, Normal>(pos, stats, depth - 1, !beta, !alpha);

            // unmake the move
            pos.unmake_move_for::<P>(m);

            // todo: there has to be a better way to do this
            if is_root {
                // store the score for the root moves, such that we can use it for sorting in
                // the next iteration.
                let idx = self
                    .root_stats
                    .iter()
                    .position(|x| x.mov == m)
                    .expect("move should be in root_moves");
                self.root_stats.as_mut_slice()[idx].score = score.0;
            }

            if score >= beta {
                return score;
            }
            if score > best_score {
                best_score = score;
            }
            if score > alpha {
                alpha = score;
            }
        }

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
