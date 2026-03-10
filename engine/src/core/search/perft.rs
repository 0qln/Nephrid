use crate::{
    core::{Depth, Limit, Move, r#move::MoveList, move_iter::fold_legal_moves, position::Position},
    misc::DebugMode,
    uci::sync::{self, CancellationToken},
};
use std::{cell::UnsafeCell, ops::ControlFlow};

pub fn perft(pos: Position, limit: Limit, ct: CancellationToken, debug: DebugMode) -> u64 {
    perft_inner(
        &mut UnsafeCell::new(pos),
        limit.depth,
        &limit,
        &ct,
        &debug,
        |mov, count, depth: Depth, debug| {
            if debug {
                let indent = itertools::repeat_n(' ', depth.v().into()).collect::<String>();
                sync::out(&format!("{}{mov:?}: {count}", indent));
            } else {
                sync::out(&format!("{mov}: {count}"));
            }
        },
    )
}

fn perft_inner(
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

    // todo:
    // this is actually not sound :-)
    // you'd have to create and destroy the &*pos.get() pointer each time when you
    // want to read inside fold_legal_moves instead of holding it across the
    // pos.get_mut() pointer aliases.
    // see: https://doc.rust-lang.org/nightly/core/cell/struct.UnsafeCell.html#aliasing-rules
    unsafe {
        fold_legal_moves::<_, _, _>(&*pos.get(), 0, |acc, m| {
            pos.get_mut().make_move(m);
            let c = perft_inner(
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
