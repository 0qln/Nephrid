use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

const ENGINE_BIN: &str = env!("CARGO_BIN_EXE_NEPHRID");

#[test]
fn test_ponder_miss_outputs_ponder_move() {
    let mut child = Command::new(ENGINE_BIN)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn engine binary");

    let mut stdin = child.stdin.take().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();

    // Helper closure to read the next line, handle EOF, and print for debugging
    let mut read_engine_line = || -> String {
        line.clear();
        reader.read_line(&mut line).expect("Failed to read from engine");
        let trimmed = line.trim().to_string();
        if !trimmed.is_empty() {
            println!("Engine: {}", trimmed);
        }
        trimmed
    };

    // 1. Send uci and wait for uciok
    writeln!(stdin, "uci").unwrap();
    stdin.flush().unwrap();
    loop {
        let out = read_engine_line();
        if out == "uciok" {
            break;
        }
    }

    // 2. Send isready and wait for readyok (Ensures engine is fully initialized)
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    loop {
        let out = read_engine_line();
        if out == "readyok" {
            break;
        }
    }

    // 3. Start pondering
    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go ponder wtime 300000 btime 300000").unwrap();
    stdin.flush().unwrap();

    // 4. Wait for the engine to prove it is searching, then send stop.
    loop {
        let out = read_engine_line();
        if out.starts_with("info") {
            // Engine is actively searching! We can now simulate the ponder miss.
            writeln!(stdin, "stop").unwrap();
            stdin.flush().unwrap();
            break;
        }
    }

    // 5. Read lines until we find "bestmove"
    let bestmove_line;
    loop {
        let out = read_engine_line();
        if out.starts_with("bestmove") {
            bestmove_line = out;
            break;
        }
    }

    // 6. Assert the engine's behavior
    assert!(
        !bestmove_line.is_empty(),
        "Engine exited before outputting a bestmove"
    );
    assert!(
        bestmove_line.contains("ponder"),
        "Engine outputted a bestmove, but forgot the ponder move! Output: {}",
        bestmove_line
    );

    // 7. Cleanly shut down the engine
    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();

    // Wait for the process to exit to avoid zombie processes
    let _ = child.wait();
}

/// Helper function to extract the `nodes` value from an `info` string
/// e.g., "info depth 5 nodes 2540 nps 120000" -> Some(2540)
fn extract_nodes(line: &str) -> Option<u64> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if let Some(pos) = parts.iter().position(|&s| s == "nodes") {
        if pos + 1 < parts.len() {
            return parts[pos + 1].parse::<u64>().ok();
        }
    }
    None
}

#[test]
fn test_ponder_miss_retains_cached_tree() {
    let mut child = Command::new(ENGINE_BIN)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn engine binary");

    let mut stdin = child.stdin.take().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();

    let mut read_engine_line = || -> String {
        line.clear();
        reader.read_line(&mut line).expect("Failed to read from engine");
        let trimmed = line.trim().to_string();
        if !trimmed.is_empty() {
            println!("Engine: {}", trimmed); // Prints to test output for debugging
        }
        trimmed
    };

    // 1. Boot up
    writeln!(stdin, "uci").unwrap();
    stdin.flush().unwrap();
    loop { if read_engine_line() == "uciok" { break; } }

    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    loop { if read_engine_line() == "readyok" { break; } }

    // 2. Setup the predicted line (e.g., White played e2e4, we guess Black plays e7e5)
    writeln!(stdin, "position startpos moves e2e4 e7e5").unwrap();
    writeln!(stdin, "go ponder wtime 30000 btime 30000").unwrap();
    stdin.flush().unwrap();

    // 3. Wait until the engine has built a sizable tree (>2000 nodes)
    loop {
        let out = read_engine_line();
        if out.starts_with("info") {
            if let Some(nodes) = extract_nodes(&out) {
                if nodes > 2000 {
                    // Tree is sufficiently populated! Interrupt the ponder.
                    writeln!(stdin, "stop").unwrap();
                    stdin.flush().unwrap();
                    break;
                }
            }
        }
    }

    // 4. Wait for the engine to acknowledge the stop and output bestmove
    loop {
        if read_engine_line().starts_with("bestmove") {
            break;
        }
    }

    // 5. Simulate the Ponder Miss! (Black actually played c7c5)
    // This is exactly a 1-ply divergence, which should trigger the Rollback logic.
    writeln!(stdin, "position startpos moves e2e4 c7c5").unwrap();
    writeln!(stdin, "go wtime 295000 btime 295000").unwrap();
    stdin.flush().unwrap();

    // 6. Capture the first info line of the new search
    let first_search_nodes;
    loop {
        let out = read_engine_line();
        if out.starts_with("info") {
            if let Some(nodes) = extract_nodes(&out) {
                first_search_nodes = nodes;
                break; // We only care about the very first node report
            }
        }
    }

    // 7. Stop the second search and quit cleanly
    writeln!(stdin, "stop").unwrap();
    stdin.flush().unwrap();
    loop {
        if read_engine_line().starts_with("bestmove") { break; }
    }
    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    let _ = child.wait();

    // 8. Assert that the tree was retained!
    // If it dropped the tree, `nodes` would be around 1-50. 
    // If it successfully did a 1-ply rollback, it should retain the branch data on `e2e4 c7c5`.
    assert!(
        first_search_nodes > 1000,
        "Failed caching! Expected cached tree with >1000 nodes, but the first info reported only {} nodes. The tree was dropped!",
        first_search_nodes
    );
}