use uom::si::{information::mebibyte, u64::Information};

use super::*;
use crate::{
    core::{move_iter::sliding_piece::magics, params::C_IdHceParams, search::limit::UciLimit},
    math::DefaultLmrParams,
};

fn run_search(fen: &str, depth: u8) {
    math::init(DefaultLmrParams);
    magics::init();
    zobrist::init();

    let mut pos = Position::from_fen(fen).unwrap();
    let limit = UciLimit {
        depth: Depth::new(depth),
        ..Default::default()
    };
    let debug = DebugMode::default();
    let ct = CancellationToken::new();
    let hash_size = Information::new::<mebibyte>(16);
    let mut tt = TT::new_of_size(hash_size);
    let mut hh = HH::new();
    let mut timeman = TimeMan::new(C_IdHceParams);
    go::<C_IdHceParams>(
        &mut pos,
        limit,
        &mut timeman,
        &debug,
        ct,
        &mut tt,
        &mut hh,
        &mut HceEvaluator,
        C_IdHceParams,
    );
}

#[test]
fn no_segfault_perpetual_check() { run_search("8/6pk/7p/pp6/3p4/3P3P/rP2RKP1/8 w - - 6 43", 12); }

#[test]
fn no_segfault_in_check() {
    // king is in check from the rook on a2; only king moves are legal.
    run_search("8/6pk/7p/pp6/3p4/3P3P/rP2RKP1/8 w - - 6 43", 8);
    // Double check: only king moves resolve it.
    run_search("4k3/8/8/8/8/8/4r3/R3K2b w Q - 0 1", 6);
    // Single check with many non-resolving pseudo-legal moves.
    run_search("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", 8);
}
