use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use engine::core::{color::perspectives::White, move_iter::sliding_piece::magics, position::Position, search::id::HceThreatener, zobrist};

struct ThreatTestCase {
    name: &'static str,
    fen: &'static str,
}

const TEST_CASES: &[ThreatTestCase] = &[
    // =========================================================================
    // 1. DIRECT CHECKS
    // =========================================================================
    ThreatTestCase {
        name: "direct_check_pawn",
        fen: "4k3/8/8/8/8/4p3/8/3K4 w - - 0 1",
    },
    ThreatTestCase {
        name: "direct_check_knight",
        fen: "4k3/8/8/8/6n1/8/8/3K4 w - - 0 1",
    },
    ThreatTestCase {
        name: "direct_check_slider",
        fen: "4k3/8/4r3/8/8/8/8/3K4 w - - 0 1",
    },
    // =========================================================================
    // 2. DISCOVERED CHECKS
    // =========================================================================
    ThreatTestCase {
        name: "discovered_check_piece_move",
        fen: "4k3/8/7r/8/7n/8/8/7K w - - 0 1",
    },
    ThreatTestCase {
        name: "discovered_check_pawn_push",
        fen: "8/3k4/8/8/8/1r1p1K2/8/8 w - - 0 1",
    },
    // =========================================================================
    // 3. NO CHECKS AVAILABLE
    // =========================================================================
    ThreatTestCase {
        name: "no_check_high_see_capture",
        fen: "3r3k/6pp/8/8/7q/8/PP4p1/K1NBRN2 w - - 0 1",
    },
    ThreatTestCase {
        name: "no_check_quiet_startpos",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    },
    ThreatTestCase {
        name: "no_check_quiet_endgame",
        fen: "7k/5ppp/8/8/8/8/PP6/K6N w - - 0 1",
    },
    // =========================================================================
    // 4. REALISTIC POSITIONS
    // =========================================================================
    ThreatTestCase {
        name: "realistic_middlegame_karpov_kasparov",
        fen: "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 1 9",
    },
    ThreatTestCase {
        name: "realistic_tactical_endgame",
        fen: "8/3p4/8/K2p3r/1R3p1k/4P3/6P1/8 w - - 0 1",
    },
];

fn bench_hce_threatener(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let threatener = HceThreatener;

    let mut group = c.benchmark_group("threats/positions");

    for test_case in TEST_CASES {
        let pos = Position::from_fen(test_case.fen).expect("Valid FEN");

        group.bench_with_input(BenchmarkId::new("threat_white", test_case.name), &pos, |b, pos| {
            b.iter(|| black_box(threatener.threat::<White>(black_box(pos))));
        });
    }
    group.finish();

    let positions: Vec<Position> = TEST_CASES.iter().map(|tc| Position::from_fen(tc.fen).unwrap()).collect();

    c.bench_function("threats/batch_mix", |b| {
        b.iter(|| {
            for pos in &positions {
                black_box(threatener.threat::<White>(black_box(pos)));
            }
        });
    });
}

criterion_group!(benches, bench_hce_threatener);
criterion_main!(benches);
