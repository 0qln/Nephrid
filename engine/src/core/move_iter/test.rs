use std::{collections::HashSet, ops::ControlFlow};

use itertools::Itertools;

use crate::{
    core::{
        depth::Depth,
        r#move::{Move, MoveList},
        move_iter::{
            Options, fold_moves,
            opt::{self, Checks, Threats},
            sliding_piece::magics,
        },
        position::{CheckState, FenExport, Position},
        search::{self, limit::UciLimit},
        zobrist,
    },
    misc::{CancellationToken, DebugMode},
};

fn test_pos(fen: &str, depth: Depth, expected: u64) {
    magics::init();
    zobrist::init();

    let mut pos = Position::from_fen(fen).unwrap();
    let limit = UciLimit { depth, ..Default::default() };
    let debug = DebugMode::default();
    let ct = CancellationToken::new();
    let result = search::perft::perft::<opt::AllLegal>(&mut pos, &limit, ct, debug);
    assert_eq!(expected, result);
}

#[test]
fn test_legal_moves_0() { test_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", Depth::new(5), 4865609); }

#[test]
fn test_legal_moves_1() {
    test_pos(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        Depth::new(5),
        193690690,
    );
}

#[test]
fn test_legal_moves_2() { test_pos("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", Depth::new(6), 11030083); }

#[test]
fn test_legal_moves_3() {
    test_pos(
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        Depth::new(5),
        15833292,
    );
}

#[test]
fn test_legal_moves_4() { test_pos("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ", Depth::new(5), 89941194); }

#[test]
fn test_legal_moves_5() {
    test_pos(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ",
        Depth::new(5),
        164075551,
    );
}

fn plegal_with_filter_is_same_as_legal(fen: &str, depth: Depth) {
    let limit = UciLimit { depth, ..Default::default() };
    let ct = CancellationToken::default();
    let debug = DebugMode::default();

    search::perft::perft_inner_collect(
        &mut Position::from_fen(fen).unwrap(),
        limit.depth,
        &limit,
        &ct,
        &debug,
        |_, _, _, _| {},
        move |pos, list| {
            let list_legal = list;
            fold_moves::<opt::AllLegal, _, _, _>(pos, (), |_, m| {
                list_legal.push(m);
                ControlFlow::Continue::<(), ()>(())
            })
            .continue_value()
            .unwrap();

            let mut list_filtered_plegal = MoveList::default();
            _ = fold_moves::<opt::AllPseudoLegal, _, _, _>(pos, (), |_, m| {
                if pos.is_legal(m) {
                    list_filtered_plegal.push(m);
                }
                ControlFlow::Continue::<(), ()>(())
            });

            assert_eq!(
                list_legal.len(),
                list_filtered_plegal.len(),
                "Move count mismatch in position: {} \nExpected: {} \nGot: {} \nDiff: {:?}",
                crate::core::position::FenExport(pos),
                list_filtered_plegal,
                list_legal,
                {
                    let expected = list_filtered_plegal.iter().collect::<HashSet<_>>();
                    let result = list_legal.iter().collect::<HashSet<_>>();
                    expected.symmetric_difference(&result).cloned().collect_vec()
                }
            );
        },
    );
}

#[test]
pub fn plegal_with_filter_is_same_as_legal_test_0() {
    magics::init();
    zobrist::init();

    plegal_with_filter_is_same_as_legal("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", Depth::new(5))
}

#[test]
pub fn plegal_with_filter_is_same_as_legal_test_1() {
    magics::init();
    zobrist::init();

    plegal_with_filter_is_same_as_legal("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", Depth::new(5))
}

#[test]
pub fn plegal_with_filter_is_same_as_legal_test_2() {
    magics::init();
    zobrist::init();

    plegal_with_filter_is_same_as_legal("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", Depth::new(6))
}

#[test]
pub fn plegal_with_filter_is_same_as_legal_test_3() {
    magics::init();
    zobrist::init();

    plegal_with_filter_is_same_as_legal("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", Depth::new(5))
}

#[test]
pub fn plegal_with_filter_is_same_as_legal_test_4() {
    magics::init();
    zobrist::init();

    plegal_with_filter_is_same_as_legal("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ", Depth::new(5))
}

#[test]
pub fn plegal_with_filter_is_same_as_legal_test_5() {
    magics::init();
    zobrist::init();

    plegal_with_filter_is_same_as_legal("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ", Depth::new(5))
}

/// Ground-truth validator that determines if a move satisfies `O`
fn is_valid_move_for_opts<O: Options>(pos: &Position, m: Move) -> bool {
    let is_check = pos.does_check(m) != CheckState::None;
    let flag = m.get_flag();
    let is_promo = flag.is_promo();
    let is_capture = flag.is_capture();
    let is_quiet = !is_promo && !is_capture;

    if is_promo {
        if is_check {
            O::promo_checks()
        }
        else {
            O::promo_nochecks()
        }
    }
    else if is_capture {
        if is_check {
            O::capture_checks()
        }
        else {
            O::capture_nochecks()
        }
    }
    else if is_quiet {
        if is_check {
            O::quiet_checks()
        }
        else {
            O::quiet_nochecks()
        }
    }
    else {
        unreachable!()
    }
}

fn options_match_oracle_perft<O: Options>(fen: &str, depth: Depth) {
    let limit = UciLimit { depth, ..Default::default() };
    let ct = CancellationToken::default();
    let debug = DebugMode::default();

    search::perft::perft_inner_collect(
        &mut Position::from_fen(fen).unwrap(),
        limit.depth,
        &limit,
        &ct,
        &debug,
        |_, _, _, _| {},
        move |pos, _| {
            let mut all_legals = MoveList::default();
            _ = fold_moves::<opt::AllLegal, _, _, _>(pos, (), |_, m| {
                all_legals.push(m);
                ControlFlow::Continue::<(), ()>(())
            });

            let mut expected = MoveList::default();
            for &m in all_legals.iter() {
                if is_valid_move_for_opts::<O>(pos, m) {
                    expected.push(m);
                }
            }

            let mut actual = MoveList::default();
            _ = fold_moves::<O, _, _, _>(pos, (), |_, m| {
                actual.push(m);
                ControlFlow::Continue::<(), ()>(())
            });

            let expected_set: HashSet<_> = expected.iter().copied().collect();
            let actual_set: HashSet<_> = actual.iter().copied().collect();

            assert_eq!(
                actual_set,
                expected_set,
                "\nOptions move mismatch in perft tree!\nPosition FEN: {}\nExpected ({} moves): {:?}\nGot ({} moves): {:?}\nDiff: {:?}",
                FenExport(pos),
                expected.len(),
                expected,
                actual.len(),
                actual,
                expected_set.symmetric_difference(&actual_set).cloned().collect_vec()
            );
        },
    );
}

#[test]
pub fn test_threats_perft_startpos() {
    magics::init();
    zobrist::init();

    options_match_oracle_perft::<Threats>("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", Depth::new(4));
}

#[test]
pub fn test_threats_perft_kiwipete() {
    magics::init();
    zobrist::init();

    options_match_oracle_perft::<Threats>("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", Depth::new(3));
}

#[test]
pub fn test_threats_perft_pos3() {
    magics::init();
    zobrist::init();

    options_match_oracle_perft::<Threats>("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", Depth::new(4));
}

#[test]
pub fn test_threats_perft_pos4_promotions() {
    magics::init();
    zobrist::init();

    options_match_oracle_perft::<Threats>("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", Depth::new(3));
}

#[test]
pub fn test_threats_perft_discovered_checks_fuzzer() {
    magics::init();
    zobrist::init();

    options_match_oracle_perft::<Threats>("8/8/8/8/k2N3R/8/8/3K4 w - - 0 1", Depth::new(4));
    options_match_oracle_perft::<Threats>("8/8/5k2/8/3P4/2B5/8/3K4 w - - 0 1", Depth::new(4));
}

#[test]
pub fn test_checks_perft_discovered_checks() {
    magics::init();
    zobrist::init();

    // knight
    options_match_oracle_perft::<Checks>("8/8/8/8/k2N3R/8/8/3K4 w - - 0 1", Depth::new(4));

    // pawn quiet
    options_match_oracle_perft::<Checks>("8/8/5k2/8/3P4/2B5/8/3K4 w - - 0 1", Depth::new(4));
    // pawn capture
    options_match_oracle_perft::<Checks>("8/8/5k2/2n5/3P4/2B5/8/3K4 w - - 0 1", Depth::new(4));
    // pawn blocked
    options_match_oracle_perft::<Checks>("8/8/5k2/3n4/3P4/2B5/8/3K4 w - - 0 1", Depth::new(4));

    // rook
    options_match_oracle_perft::<Checks>("8/8/5k2/8/3R4/2B5/8/3K4 w - - 0 1", Depth::new(4));

    // bishop
    options_match_oracle_perft::<Checks>("8/8/5k2/8/5B2/5R2/8/3K4 w - - 0 1", Depth::new(4));
}

// todo: test that options work (i.e. if gen_captures is false no captures will
// be generated)
