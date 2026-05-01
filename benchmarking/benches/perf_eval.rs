use std::fmt::Display;

use engine::{
    core::{
        r#move::Move,
        move_iter::sliding_piece::magics,
        position::{EpdLineImport, FenExport, Position},
        search::{
            limit::UciLimit,
            mcts::{
                self, HceParts, MctsConfig, mcts,
                node::{self, Branch, node_state},
                strategy::{MctsUci, UciCp, UciScore},
            },
        },
        zobrist,
    },
    math::entropy,
    misc::{CancellationToken, DebugMode},
    uci::tokens::Tokenizer,
};
use itertools::Itertools;
use tabled::{Table, Tabled};

struct Test {
    id: String,
    pos: Position,
    find: Option<Vec<Move>>,
    avoid: Option<Vec<Move>>,
}

impl TryFrom<EpdLineImport<'_, '_>> for Test {
    type Error = Box<dyn std::error::Error>;

    fn try_from(epd: EpdLineImport) -> Result<Self, Self::Error> {
        let (pos, ops) = epd.try_into().expect("Parse EPD");

        let id = ops
            .iter()
            .find(|op| matches!(op.0.as_ref(), "id"))
            .expect("Eret lines should have an id")
            .1
            .clone();

        let bm = ops
            .iter()
            .find(|op| matches!(op.0.as_ref(), "bm"))
            .map(|op| {
                op.1.split_ascii_whitespace()
                    .map(|mov| Move::from_san(mov, &pos).expect("Parse SAN from EPD line"))
                    .collect_vec()
            });

        let am = ops
            .iter()
            .find(|op| matches!(op.0.as_ref(), "am"))
            .map(|op| {
                op.1.split_ascii_whitespace()
                    .map(|mov| Move::from_san(mov, &pos).expect("Parse SAN from EPD line"))
                    .collect_vec()
            });

        Ok(Test { id, pos, find: bm, avoid: am })
    }
}

fn print_dataset(tests: &[Test]) {
    #[derive(Tabled)]
    struct EretEntry {
        #[tabled(rename = "ID")]
        id: String,
        #[tabled(rename = "AM")]
        am: String,
        #[tabled(rename = "BM")]
        bm: String,
        #[tabled(rename = "FEN")]
        fen: String,
    }

    let entries: Vec<EretEntry> = tests
        .iter()
        .map(|test| {
            let am_str = test
                .avoid
                .as_ref()
                .map_or(String::new(), |am| am.iter().join(", "));
            let bm_str = test
                .find
                .as_ref()
                .map_or(String::new(), |bm| bm.iter().join(", "));
            let fen = FenExport(&test.pos).to_string();
            EretEntry {
                id: test.id.clone(),
                am: am_str,
                bm: bm_str,
                fen,
            }
        })
        .collect();

    let table = Table::new(entries);
    println!("{table}");
}

struct Solution<Diagnostic> {
    best_move: Move,
    score: UciScore,
    diagnostic: Diagnostic,
}

struct MctsDiagnostic {
    visits: u32,
    value: node::Value,
    root_entropy: f32,
}

impl Display for MctsDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "visits: {}, value: {}, root_entropy: {:.3}",
            self.visits, self.value, self.root_entropy
        )
    }
}

struct TestResult<Diagnostic> {
    id: String,
    success: bool,
    solution: Solution<Diagnostic>,
}

trait PerfRunner {
    type Diagnostic;

    fn run(&self, pos: Position, limit: UciLimit) -> Solution<Self::Diagnostic>;
}

#[derive(Debug, Clone, Copy)]
struct MctsHceRunner;

impl PerfRunner for MctsHceRunner {
    type Diagnostic = MctsDiagnostic;
    fn run(&self, mut pos: Position, limit: UciLimit) -> Solution<Self::Diagnostic> {
        struct Config;
        impl MctsConfig for Config {
            type Parts = mcts::HceParts;
            type Strat = MctsUci;
        }

        let ct = CancellationToken::new();

        let state = &mut mcts::SearchState::default();
        let parts = HceParts::default();
        let strat = &mut MctsUci::new(limit, DebugMode::off(), ct, None);
        let result = mcts::<1, Config, _>(&mut pos, &parts, state, strat)
            .expect("mcts should've completed in the given time");

        let tree = &state.tree;
        let pv = tree.principal_line();
        let root = tree
            .try_node::<node_state::Evaluated>(tree.root())
            .expect("root should be evaluated after the search");

        let root_policies = tree
            .branches(root.id())
            .iter()
            .map(Branch::policy)
            .collect::<Vec<_>>();

        let root_entropy = entropy(root_policies.iter().cloned());

        Solution {
            best_move: result,
            score: strat
                .determine_score(tree, pv.len())
                .expect("root node should be evaluated after the search"),
            diagnostic: MctsDiagnostic {
                visits: root.visits(),
                value: root.value(),
                root_entropy,
            },
        }
    }
}

fn run_perf_eval<Runner: PerfRunner>(test: Test, runner: Runner) -> TestResult<Runner::Diagnostic> {
    let limit = UciLimit {
        wtime: 0,
        winc: 10000,
        btime: 0,
        binc: 10000,
        lag_buf: 0,
        ..Default::default()
    };

    let Test { pos, id, find, avoid } = test;

    let solution = runner.run(pos, limit);

    let has_found = find.is_none_or(|find| find.contains(&solution.best_move));
    let has_avoided = avoid.is_none_or(|avoid| !avoid.contains(&solution.best_move));

    TestResult {
        id,
        success: has_found && has_avoided,
        solution,
    }
}

fn main() {
    magics::init();
    zobrist::init();

    let epd = include_str!("../resources/eret.epd");

    let tests: Vec<Test> = epd
        .lines()
        .map(|line| {
            let tok = &mut Tokenizer::new(line);
            let epd = EpdLineImport(tok);
            let test = Test::try_from(epd).expect("Parse EPD line into Test");
            test
        })
        .collect_vec();

    #[cfg(debug_assertions)]
    {
        print_dataset(&tests);
    }

    let runner = MctsHceRunner;

    let solutions = tests
        .into_iter()
        .map(|test| run_perf_eval(test, runner))
        .collect::<Vec<_>>();

    print_solution(&solutions);
}

fn print_solution<D: Display>(results: &[TestResult<D>]) {
    #[derive(Tabled)]
    struct SolutionRow {
        #[tabled(rename = "ID")]
        id: String,
        #[tabled(rename = "Success")]
        success: bool,
        #[tabled(rename = "Best Move")]
        best_move: String,
        #[tabled(rename = "Score")]
        score: String,
        #[tabled(rename = "Diagnostic")]
        diagnostic: String,
    }

    let mut rows = Vec::new();
    let mut success_count = 0;

    for result in results.iter() {
        let success = result.success;
        let best_move = result.solution.best_move.to_string();
        let score_str = match result.solution.score {
            UciScore::Centipawns(UciCp(cp)) => format!("{:.2}", cp.v() as f64 / 100.0),
            UciScore::Mate(ply) => format!("Mate in {}", ply.abs()),
            _ => format!("{}", result.solution.score),
        };
        let diagnostic = result.solution.diagnostic.to_string();

        rows.push(SolutionRow {
            id: result.id.clone(),
            success,
            best_move,
            score: score_str,
            diagnostic,
        });

        if success {
            success_count += 1;
        }
    }

    let total_tests = rows.len();
    let success_rate = (success_count as f64 / total_tests as f64) * 100.0;

    println!("\nPerformance Evaluation Results:\n");
    let table = Table::new(rows);
    println!("{table}\n");

    println!("Summary:");
    println!("  Total tests:     {}", total_tests);
    println!("  Success count:   {}", success_count);
    println!("  Success rate:    {:.2}%", success_rate);
}
