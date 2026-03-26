use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use ntest::timeout;
use assert_cmd::prelude::*;
use regex::Regex;

#[test]
#[timeout(10000)]
fn test_ponder_miss_outputs_ponder_move() {
    let mut child = Command::cargo_bin("nephrid").unwrap()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn engine binary");

    let mut stdin = child.stdin.take().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");
    let mut reader = BufReader::new(stdout);

    // 1. Send uci and wait for uciok
    write_engine_line(&mut stdin, "uci");
    loop { if read_engine_line(&mut reader) == "uciok" { break; } }

    write_engine_line(&mut stdin, "isready");
    loop { if read_engine_line(&mut reader) == "readyok" { break; } }

    // 3. Start pondering
    write_engine_line(&mut stdin, "position startpos");
    write_engine_line(&mut stdin, "go ponder wtime 300000 btime 300000");

    // 4. Wait for the engine to prove it is searching, then send stop.
    loop {
        let line = read_engine_line(&mut reader);
        if line.starts_with("info") {
            // Only stop once we have an info line with a PV containing at least two moves,  
            // so the engine has a chance to produce a ponder move.  
            if let Some(pv_start) = line.find(" pv ") {  
                let pv = &line[pv_start + 4..];  
                if pv.split_whitespace().count() >= 2 {  
                    write_engine_line(&mut stdin, "stop");  
                    break;  
                }  
            } 
            // Engine is actively searching! We can now simulate the ponder miss.
            write_engine_line(&mut stdin, "stop");
            break;
        }
    }

    // 5. Read lines until we find "bestmove"
    let bestmove_line;
    loop {
        let out = read_engine_line(&mut reader);
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
    write_engine_line(&mut stdin, "quit");

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

fn read_engine_line(reader: &mut BufReader<ChildStdout>) -> String {
    let mut line = String::new();
    line.clear();
    reader.read_line(&mut line).expect("Failed to read from engine");
    let trimmed = line.trim().to_string();
    if !trimmed.is_empty() {
        println!("Engine: {}", trimmed);
    }
    trimmed
}

fn write_engine_line(stdin: &mut ChildStdin, line: &str) {
    writeln!(stdin, "{line}").unwrap();
    println!("Gui: {}", line);
    stdin.flush().unwrap();
}

#[test]
#[timeout(10000)]
fn test_ponder_miss_retains_cached_tree() {
    let mut child = Command::cargo_bin("nephrid").unwrap()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn engine binary");

    let mut stdin = child.stdin.take().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");
    let mut reader = BufReader::new(stdout);

    // 1. Boot up
    write_engine_line(&mut stdin, "uci");
    loop { if read_engine_line(&mut reader) == "uciok" { break; } }

    write_engine_line(&mut stdin, "isready");
    loop { if read_engine_line(&mut reader) == "readyok" { break; } }

    // 2. Wait until the engine has built a sizable tree on the base move
    write_engine_line(&mut stdin, "position startpos moves e2e4");
    write_engine_line(&mut stdin, "go nodes 5000000");
    loop { if read_engine_line(&mut reader).starts_with("bestmove") { break; } }

    let hit_move;
    let miss_move;
    let pattern = r"(?P<prefix>[-*])\s(?P<state>\w*)\s(?P<move>[a-z1-8]{4,5})\s+v\s+(?P<value>[\d.]*)/(?P<visits>\d*)\s+";
    let re = Regex::new(pattern).unwrap();

    write_engine_line(&mut stdin, "mcts d");

    let mut child_nodes: Vec<(String, u64)> = Vec::new();
    loop {
        let out = read_engine_line(&mut reader);

        if out == "---" { break; }
        
        if let Some(captures) = re.captures(&out) {
            let mov = captures.name("move").map(|m| m.as_str()).unwrap_or("");
            let visits_str = captures.name("visits").map(|m| m.as_str()).unwrap_or("0");
            let visits = visits_str.parse::<u64>().unwrap_or(0);
            child_nodes.push((mov.to_string(), visits));
        }
    }

    child_nodes.sort_by(|a, b| b.1.cmp(&a.1));
    if child_nodes.len() >= 2 {
        hit_move = child_nodes[0].0.clone();  // Most visited (Ponder move)
        miss_move = child_nodes[1].0.clone(); // Second most visited (Actual move)
        
        println!("Most visited (hit): {} with {} visits", hit_move, child_nodes[0].1);
        println!("Second visited (miss): {} with {} visits", miss_move, child_nodes[1].1);
    } else {
        panic!("Not enough child nodes dumped by MCTS to pick a hit and a miss!");
    }

    // 3. Setup the predicted line (e.g., White played e2e4, we guess Black plays e7e5)
    write_engine_line(&mut stdin, &format!("position startpos moves e2e4 {miss_move}"));
    write_engine_line(&mut stdin, "go ponder wtime 295000 btime 295000");
    loop {
        let out = read_engine_line(&mut reader);
        if out.starts_with("info") {
            if let Some(nodes) = extract_nodes(&out) {
                if nodes > 2000 {
                    write_engine_line(&mut stdin, "stop");
                    break;
                }
            }
        }
    }

    // 4. Wait for the engine to acknowledge the stop and output bestmove
    loop { if read_engine_line(&mut reader).starts_with("bestmove") { break; } }

    // 5. Simulate the Ponder Miss! (Black actually played c7c5)
    // This is exactly a 1-ply divergence, which should trigger the Rollback logic.
    write_engine_line(&mut stdin, &format!("position startpos moves e2e4 {hit_move}"));
    write_engine_line(&mut stdin, "go wtime 295000 btime 295000");

    // 6. Capture the first info line of the new search
    let first_search_nodes;
    loop {
        let out = read_engine_line(&mut reader);
        if out.starts_with("info") {
            if let Some(nodes) = extract_nodes(&out) {
                first_search_nodes = nodes;
                break; // We only care about the very first node report
            }
        }
    }

    // 7. Stop the second search and quit cleanly
    write_engine_line(&mut stdin, "stop");
    loop { if read_engine_line(&mut reader).starts_with("bestmove") { break; } }

    write_engine_line(&mut stdin, "quit");
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