use crate::{
    core::{
        Depth, Move, UciLimit,
        color::{Perspective, colors, perspectives},
        r#move::MoveList,
        move_iter::fold_legals,
        position::Position,
    },
    misc::{CancellationToken, DebugMode},
};
use std::ops::ControlFlow;

#[cfg(test)]
pub mod test;

pub fn perft<const Q: bool>(
    pos: &mut Position,
    limit: &UciLimit,
    ct: CancellationToken,
    debug: DebugMode,
) -> u64 {
    let nodes = perft_inner_collect(
        pos,
        limit.depth,
        limit,
        &ct,
        &debug,
        |mov, count, depth, debug| {
            if debug {
                let indent = itertools::repeat_n(' ', depth.v().into()).collect::<String>();
                println!("{}{mov:?}: {count}", indent);
            }
            else {
                println!("{mov}: {count}");
            }
        },
        |pos, list| {
            _ = fold_legals::<Q, _, _, _>(pos, (), |_, m| {
                list.push(m);
                ControlFlow::Continue::<(), _>(())
            });
        },
    );

    println!("\nNodes searched: {nodes}\n");

    nodes
}

pub fn perft_inner_collect(
    pos: &mut Position,
    depth: Depth,
    limit: &UciLimit,
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
    limit: &UciLimit,
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
