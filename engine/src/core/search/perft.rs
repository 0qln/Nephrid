use crate::{
    core::{
        Depth, Limit, Move,
        color::{Perspective, colors, perspectives},
        r#move::MoveList,
        move_iter::{fold_legal_moves, fold_legals},
        position::Position,
    },
    misc::DebugMode,
    uci::sync::{self, CancellationToken},
};
use std::{cell::UnsafeCell, ops::ControlFlow};

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
        |pos, list| {
            _ = fold_legals::<Q, _, _, _>(pos, (), |_, m| {
                list.push(m);
                ControlFlow::Continue::<(), _>(())
            });
        },
    )
}

#[allow(unused)]
fn perft_inner_iter(
    pos: &mut UnsafeCell<Position>,
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

    // this is actually not sound :-)
    // you'd have to create and destroy the &*pos.get() pointer each time when you
    // want to read inside fold_legal_moves instead of holding it across the
    // pos.get_mut() pointer aliases.
    // see: https://doc.rust-lang.org/nightly/core/cell/struct.UnsafeCell.html#aliasing-rules
    //
    // considering the fact that this is barely any faster than just generating the
    // moves up front and the iterating over them, this should not be used.
    unsafe {
        fold_legal_moves::<_, _, _>(&*pos.get(), 0, |acc, m| {
            pos.get_mut().make_move(m);
            let c = perft_inner_iter(
                pos,
                depth - 1,
                limit,
                cancellation_token,
                debug,
                if debug.get() { f } else { |_, _, _, _| {} },
            );
            f(m, c, limit.depth - depth, debug.get());
            pos.get_mut().unmake_move(m);
            ControlFlow::Continue::<(), _>(acc + c)
        })
        .continue_value()
        .unwrap()
    }
}

pub fn perft_inner_collect(
    pos: &mut Position,
    depth: Depth,
    limit: &Limit,
    cancellation_token: &CancellationToken,
    debug: &DebugMode,
    f: fn(Move, u64, Depth, bool) -> (),
    mut moves: impl FnMut(&Position, &mut MoveList),
) -> u64 {
    match pos.get_turn() {
        colors::WHITE => perft_inner_collect_for::<perspectives::White>(
            pos,
            depth,
            limit,
            cancellation_token,
            debug,
            f,
            &mut moves,
        ),
        colors::BLACK => perft_inner_collect_for::<perspectives::Black>(
            pos,
            depth,
            limit,
            cancellation_token,
            debug,
            f,
            &mut moves,
        ),
        _ => unreachable!("Invalid program state."),
    }
}

pub fn perft_inner_collect_for<P: Perspective>(
    pos: &mut Position,
    depth: Depth,
    limit: &Limit,
    cancellation_token: &CancellationToken,
    debug: &DebugMode,
    f: fn(Move, u64, Depth, bool) -> (),
    moves: &mut impl FnMut(&Position, &mut MoveList),
) -> u64 {
    if cancellation_token.is_cancelled() {
        return 0;
    }

    if depth <= Depth::ROOT {
        return 1;
    }

    let mut move_list = MoveList::new();
    moves(pos, &mut move_list);

    let mut acc = 0;
    for &m in move_list.iter() {
        pos.make_move_for::<P>(m);
        let c = perft_inner_collect_for::<P::Opponent>(
            pos,
            depth - 1,
            limit,
            cancellation_token,
            debug,
            if debug.get() { f } else { |_, _, _, _| {} },
            moves,
        );
        f(m, c, limit.depth - depth, debug.get());
        pos.unmake_move_for::<P>(m);
        acc += c;
    }
    acc
}
