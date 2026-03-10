use crate::{
    core::{Depth, Limit, Move, r#move::MoveList, move_iter::fold_legal_moves, position::Position},
    misc::DebugMode,
    uci::sync::{self, CancellationToken},
};
use std::{cell::UnsafeCell, ops::ControlFlow};

pub fn perft(mut pos: Position, limit: Limit, ct: CancellationToken, debug: DebugMode) -> u64 {
    perft_inner_collect(
        &mut pos,
        limit.depth,
        &limit,
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
}

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

    if depth <= Depth::MIN {
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

fn perft_inner_collect(
    pos: &mut Position,
    depth: Depth,
    limit: &Limit,
    cancellation_token: &CancellationToken,
    debug: &DebugMode,
    f: fn(Move, u64, Depth, bool) -> (),
) -> u64 {
    if cancellation_token.is_cancelled() {
        return 0;
    }

    if depth <= Depth::MIN {
        return 1;
    }

    let mut move_list = MoveList::default();
    let n_moves = fold_legal_moves::<_, _, _>(&pos, 0_u8, |curr, m| {
        move_list[curr] = m;
        ControlFlow::Continue::<(), _>(curr + 1)
    })
    .continue_value()
    .unwrap();

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
        );
        f(m, c, limit.depth - depth, debug.get());
        pos.unmake_move(m);
        acc += c;
    }
    acc
}
