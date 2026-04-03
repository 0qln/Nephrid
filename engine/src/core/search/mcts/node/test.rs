use super::*;
use crate::core::{
    Move, Position,
    coordinates::squares,
    depth::Depth,
    r#move::move_flags,
    move_iter::sliding_piece::magics,
    search::mcts::node::node_state::{Branching, Leaf, NodeState, Switch},
    zobrist,
};

// --- Utility Functions ---

fn create_position(fen: &str) -> Position {
    zobrist::init();
    magics::init();
    Position::from_fen(fen).unwrap()
}

// --- ID and View Initialization Tests ---

#[test]
fn test_node_id_and_view_initialization() {
    let tree = Tree::new();
    let root_id = tree.root();

    let view = tree.node(root_id);
    assert_eq!(view.state(), NodeState::Leaf);
    assert_eq!(view.visits(), 0);
    assert_eq!(view.value(), Value(0.0));

    let switch = tree.node_switch(root_id);
    match switch {
        Switch::Leaf(leaf_id) => {
            let leaf_view = tree.node(leaf_id);
            assert_eq!(leaf_view.visits(), 0);
        }
        _ => panic!("Expected the switch to yield a Leaf variant"),
    }
}

// --- Tree Tests & Domain Expansions ---

#[test]
fn test_tree_default_initializes_leaf_node() {
    let tree = Tree::default();
    let root = tree.node(tree.root());

    assert_eq!(root.state(), NodeState::Leaf);
    assert_eq!(root.visits(), 0);
    assert_eq!(root.value(), Value(0.0));
}

#[test]
fn test_node_expand_from_standard_position() {
    let pos = create_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut tree = Tree::default();

    let leaf = match tree.node_switch(tree.root()) {
        Switch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    let expanded = tree.expand_node(leaf, &pos, Depth::ROOT);

    assert!(matches!(expanded, ExpandedSwitch::Branching(_)));
    assert_eq!(tree.node(tree.root()).state(), NodeState::Branching);

    // Size should be 21 (1 root + 20 legal starting moves)
    assert_eq!(tree.size(), 21);

    if let Switch::Branching(b) = tree.node_switch(tree.root()) {
        let branches = tree.branches(b);
        assert!(!branches.is_empty());
        for branch in branches {
            let mut pos_copy = pos.clone();
            pos_copy.make_move(branch.mov()); // Asserting this doesn't panic
        }
    }
    else {
        panic!("Position should yield branching state");
    }
}

#[test]
fn test_node_expand_from_checkmate_position_becomes_terminal() {
    // Black is checkmated
    let pos = create_position("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1");
    let mut tree = Tree::default();

    let leaf = match tree.node_switch(tree.root()) {
        Switch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    let expanded = tree.expand_node(leaf, &pos, Depth::ROOT);

    assert!(matches!(expanded, ExpandedSwitch::Terminal(_)));
    assert_eq!(tree.node(tree.root()).state(), NodeState::Terminal);
    assert_eq!(tree.size(), 1); // Size should remain 1 as no branches are created
}

#[test]
fn test_node_expand_from_stalemate_position_becomes_terminal() {
    let pos = create_position("k6R/8/1K6/8/8/8/8/8 b - - 0 1");
    let mut tree = Tree::default();

    let leaf = match tree.node_switch(tree.root()) {
        Switch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    let expanded = tree.expand_node(leaf, &pos, Depth::ROOT);

    assert!(matches!(expanded, ExpandedSwitch::Terminal(_)));
    assert_eq!(tree.node(tree.root()).state(), NodeState::Terminal);
}

// --- Node Branch Sorting and Selection ---

#[test]
fn test_node_sort_by_visits() {
    let pos = create_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut tree = Tree::default();

    let leaf = tree.node_switch(tree.root()).get::<Leaf>().unwrap();
    tree.expand_node(leaf, &pos, Depth::ROOT);
    let branching = tree.node_switch(tree.root()).get::<Branching>().unwrap();

    // Tweak visits manually via private array (allowed in same-module tests)
    tree.arena.nodes[tree.arena.branches[0].node().index()].visits = 3;
    tree.arena.nodes[tree.arena.branches[1].node().index()].visits = 1;
    tree.arena.nodes[tree.arena.branches[2].node().index()].visits = 2;

    // Transition to Evaluated state to satisfy `sort_branches_by` constraints
    let eval_node = tree.skip_policy(branching);

    tree.sort_branches_by(eval_node, |child_a, _, child_b, _| {
        // Sort descending by visits
        child_b.visits().cmp(&child_a.visits())
    });

    let sorted_branches = tree.branches(eval_node);
    assert_eq!(
        tree.arena.nodes[sorted_branches[0].node().index()].visits(),
        3
    );
    assert_eq!(
        tree.arena.nodes[sorted_branches[1].node().index()].visits(),
        2
    );
    assert_eq!(
        tree.arena.nodes[sorted_branches[2].node().index()].visits(),
        1
    );
}

// --- Tree Advancing & Traversal ---

#[test]
fn test_tree_best_move_on_empty_tree() {
    let tree = Tree::default();
    assert!(tree.maybe_best_move(tree.root()).is_none());
}

#[test]
fn test_tree_advance_best_moves_root() {
    let pos = create_position("k7/8/8/8/8/8/2PPPP2/K7 w - - 0 1");
    let mut tree = Tree::default();
    let mut back_buffer = Tree::default();

    let leaf = tree.node_switch(tree.root()).get::<Leaf>().unwrap();
    tree.expand_node(leaf, &pos, Depth::ROOT);
    let branching = tree.node_switch(tree.root()).get::<Branching>().unwrap();

    // Force a high visit count on the first branch to make it "best"
    let target_node_id = tree.branches(branching)[0].node();
    tree.arena.nodes[target_node_id.index()].visits = 100;

    // Advance GC
    tree.advance_to(&mut back_buffer, target_node_id);

    // The new tree root should have the forced 100 visits and be back to a Leaf
    // state
    assert_eq!(tree.node(tree.root()).visits(), 100);
    assert_eq!(tree.node(tree.root()).state(), NodeState::Leaf);
}

#[test]
fn test_tree_advance_to_specific_move() {
    let pos = create_position("k7/8/8/8/8/8/2PPPP2/K7 w - - 0 1");
    let mut tree = Tree::default();
    let mut back_buffer = Tree::default();

    let leaf = tree.node_switch(tree.root()).get::<Leaf>().unwrap();
    tree.expand_node(leaf, &pos, Depth::ROOT);

    let branching = tree.node_switch(tree.root()).get::<Branching>().unwrap();
    let target_node_id = tree.branches(branching)[0].node();

    tree.advance_to(&mut back_buffer, target_node_id);

    assert_eq!(tree.node(tree.root()).state(), NodeState::Leaf);
}

#[test]
fn test_tree_principal_variation_from_leaf() {
    let tree = Tree::default();
    let pv = tree.principal_line();
    assert!(pv.is_empty());
}

#[test]
fn test_tree_principal_variation_simple_path() {
    let mut tree = Tree::default();
    tree.arena.clear();

    // Manually construct an interconnected 3-level tree: root(0) -> mid(1) ->
    // leaf(2) CRITICAL: We MUST use NodeState::Evaluated so `principal_line` is
    // allowed to traverse them!
    tree.arena.nodes.push(NodeData {
        branch_start: 0,
        branch_count: MoveIndex::from(1),
        visits: 10,
        value: Value(0.),
        state: NodeState::Evaluated,
    });
    tree.arena.nodes.push(NodeData {
        branch_start: 1,
        branch_count: MoveIndex::from(1),
        visits: 5,
        value: Value(0.),
        state: NodeState::Evaluated,
    });
    tree.arena.nodes.push(NodeData {
        branch_start: 0,
        branch_count: MoveIndex::from(0),
        visits: 5,
        value: Value(0.),
        state: NodeState::Leaf,
    });

    tree.arena.branches.push(Branch {
        node: RtNodeId::new(1),
        policy: 0.8,
        mov: Move::new(squares::E2, squares::E4, move_flags::QUIET),
    });
    tree.arena.branches.push(Branch {
        node: RtNodeId::new(2),
        policy: 0.7,
        mov: Move::new(squares::E7, squares::E5, move_flags::QUIET),
    });

    let pv = tree.principal_line();

    assert_eq!(pv.len(), 2);
    assert_eq!(
        pv.0[0].mov(),
        Move::new(squares::E2, squares::E4, move_flags::QUIET)
    );
    assert_eq!(
        pv.0[1].mov(),
        Move::new(squares::E7, squares::E5, move_flags::QUIET)
    );
}

// ================================================
// VALUE AND PROVEN STATE COMPARISONS
// ================================================

#[test]
fn test_value_win_beats_unproven() {
    let win = Value::proven_win();
    let unproven = Value(5000.0); // Extreme heuristic advantage, but unproven

    assert!(win > unproven);
}

#[test]
fn test_value_win_beats_loss() {
    let win = Value::proven_win();
    let loss = Value::proven_loss();

    assert!(win > loss);
}

#[test]
fn test_value_unproven_beats_loss() {
    let unproven = Value(-5000.0); // Extreme heuristic disadvantage, but unproven
    let loss = Value::proven_loss(); // Guaranteed loss

    // We MUST prefer an unexplored/unproven move over a guaranteed, proven loss!
    assert!(unproven > loss);
}
