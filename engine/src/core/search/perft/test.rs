use std::{collections::HashSet, ops::ControlFlow};

use itertools::Itertools;

use crate::{
    core::{
        depth::Depth,
        r#move::MoveList,
        move_iter::{fold_legals, sliding_piece::magics},
        position::{FenExport, Position},
        search::{limit::Limit, perft::perft_inner_collect},
        zobrist,
    },
    misc::DebugMode,
    uci::sync::CancellationToken,
};

// fn compare_capture_filtering(mut pos: Position, depth: Depth) {
//     let limit = Limit { depth, ..Default::default() };
//     let ct = CancellationToken::default();
//     let debug = DebugMode::default();

//     let filtered = super::perft_filter_captures(&mut pos, &limit, ct.clone(),
// debug.clone());     let generated = super::perft::<false>(&mut pos, &limit,
// ct.clone(), debug.clone());

//     assert_eq!(filtered, generated);
// }

fn compare_capture_filtering_find_error(mut pos: Position, depth: Depth) {
    let limit = Limit { depth, ..Default::default() };
    let ct = CancellationToken::default();
    let debug = DebugMode::default();

    perft_inner_collect(
        &mut pos,
        limit.depth,
        &limit,
        &ct,
        &debug,
        |_, _, _, _| {},
        move |pos| {
            let mut list_skipped = MoveList::default();
            let n_skipped = fold_legals::<false, _, _, _>(&pos, 0_u8, |curr, m| {
                list_skipped[curr] = m;
                ControlFlow::Continue::<(), _>(curr + 1)
            })
            .continue_value()
            .unwrap();

            let mut list_filtered = MoveList::default();
            let n_filtered = fold_legals::<true, _, _, _>(&pos, 0_u8, |curr, m| {
                if m.get_flag().is_capture() {
                    list_filtered[curr] = m;
                    ControlFlow::Continue::<(), _>(curr + 1)
                }
                else {
                    ControlFlow::Continue::<(), _>(curr)
                }
            })
            .continue_value()
            .unwrap();

            assert_eq!(
                n_skipped,
                n_filtered,
                "Move count mismatch in position: {} \nExpected: {} \nGot: {} \nDiff: {:?}",
                FenExport(pos),
                &list_filtered,
                &list_skipped,
                {
                    let expected = list_filtered.iter().collect::<HashSet<_>>();
                    let result = list_skipped.iter().collect::<HashSet<_>>();
                    expected
                        .symmetric_difference(&result)
                        .into_iter()
                        .cloned()
                        .collect_vec()
                }
            );

            (list_filtered, n_filtered)
        },
    );
}

#[test]
pub fn compare_capture_filtering_test_1() {
    magics::init();
    zobrist::init();

    compare_capture_filtering_find_error(
        Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .unwrap(),
        Depth::new(8),
    )
}

#[test]
pub fn compare_capture_filtering_test_2() {
    magics::init();
    zobrist::init();

    compare_capture_filtering_find_error(
        Position::from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8").unwrap(),
        Depth::new(20),
    )
}

#[test]
pub fn compare_capture_filtering_test_3() {
    magics::init();
    zobrist::init();

    compare_capture_filtering_find_error(
        Position::from_fen(
            "r1bqk2r/1pp1bppp/p1n1pn2/3p4/3P4/P1N1PN2/1PP1BPPP/R1BQK2R w KQkq - 0 1",
        )
        .unwrap(),
        Depth::new(9),
    )
}
