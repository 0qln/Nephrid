use ghost_cell::{GhostCell, GhostToken};

use crate::{
    core::{
        Depth, Limit, Move,
        r#move::MoveList,
        move_iter::{fold_legal_moves, fold_legal_moves_g, fold_legals},
        position::Position,
    },
    misc::DebugMode,
    uci::sync::{self, CancellationToken},
};
use std::ops::ControlFlow;

#[cfg(test)]
pub mod test;

pub fn perft<const Q: bool>(
    pos: &mut Position,
    limit: &Limit,
    ct: CancellationToken,
    debug: DebugMode,
) -> u64 {
    perft_inner_collect(
        pos,
        limit.depth,
        limit,
        &ct,
        &debug,
        |mov, count, depth: Depth, debug| {
            if debug {
                let indent = itertools::repeat_n(' ', depth.v().into()).collect::<String>();
                sync::out(&format!("{}{mov:?}: {count}", indent));
            }
            else {
                sync::out(&format!("{mov}: {count}"));
            }
        },
        |pos| {
            let mut list = MoveList::default();
            let n = fold_legals::<Q, _, _, _>(pos, 0_u8, |curr, m| {
                list[curr] = m;
                ControlFlow::Continue::<(), _>(curr + 1)
            })
            .continue_value()
            .unwrap();
            (list, n)
        },
    )
}

pub fn perft_iter<const Q: bool>(
    pos: Position,
    limit: &Limit,
    ct: CancellationToken,
    debug: DebugMode,
) -> u64 {
    GhostToken::new(move |mut tok| {
        let pos = GhostCell::new(pos);
        perft_inner_iter(
            &pos,
            &mut tok,
            limit.depth,
            limit,
            &ct,
            &debug,
            |mov, count, depth: Depth, debug| {
                if debug {
                    let indent = itertools::repeat_n(' ', depth.v().into()).collect::<String>();
                    sync::out(&format!("{}{mov:?}: {count}", indent));
                }
                else {
                    sync::out(&format!("{mov}: {count}"));
                }
            },
        )
    })
}

#[allow(unused)]
fn perft_inner_iter<'brand>(
    pos: &GhostCell<'brand, Position>,
    pos_tok: &mut GhostToken<'brand>,
    depth: Depth,
    limit: &Limit,
    cancellation_token: &CancellationToken,
    debug: &DebugMode,
    f: fn(Move, u64, Depth, bool) -> (),
) -> u64 {
    if cancellation_token.is_cancelled() {
        return 0;
    }

    if depth <= Depth::ROOT {
        return 1;
    }

    fold_legal_moves_g::<_, _, _>(pos, pos_tok, 0, |acc, m, pos_tok| {
        {
            let pos_mut = pos.borrow_mut(pos_tok);
            pos_mut.make_move(m);
        }
        let c = perft_inner_iter(
            pos,
            pos_tok,
            depth - 1,
            limit,
            cancellation_token,
            debug,
            if debug.get() { f } else { |_, _, _, _| {} },
        );
        f(m, c, limit.depth - depth, debug.get());

        {
            let pos_mut = pos.borrow_mut(pos_tok);
            pos_mut.unmake_move(m);
        }

        ControlFlow::Continue::<(), _>(acc + c)
    })
    .continue_value()
    .unwrap()
}

pub fn perft_inner_collect(
    pos: &mut Position,
    depth: Depth,
    limit: &Limit,
    cancellation_token: &CancellationToken,
    debug: &DebugMode,
    f: fn(Move, u64, Depth, bool) -> (),
    moves: fn(&Position) -> (MoveList, u8),
) -> u64 {
    if cancellation_token.is_cancelled() {
        return 0;
    }

    if depth <= Depth::ROOT {
        return 1;
    }

    let (move_list, n_moves) = moves(pos);

    let mut acc = 0;
    for i in 0..n_moves {
        let m = move_list[i];

        pos.make_move(m);
        let c = perft_inner_collect(
            pos,
            depth - 1,
            limit,
            cancellation_token,
            debug,
            if debug.get() { f } else { |_, _, _, _| {} },
            moves,
        );
        f(m, c, limit.depth - depth, debug.get());
        pos.unmake_move(m);
        acc += c;
    }
    acc
}
