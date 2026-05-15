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
    let tree = DAG::default();
    let root_id = tree.root();

    let view = tree.node(root_id);
    assert_eq!(view.state(), NodeState::Leaf);
    assert_eq!(view.visits(), VisitCount(0));
    assert_eq!(view.value(), Value(0.0));

    let switch = tree.node_switch(root_id);
    match switch {
        Switch::Leaf(leaf_id) => {
            let leaf_view = tree.node(leaf_id);
            assert_eq!(leaf_view.visits(), VisitCount(0));
        }
        _ => panic!("Expected the switch to yield a Leaf variant"),
    }
}

// --- Tree Tests & Domain Expansions ---

#[test]
fn test_tree_default_initializes_leaf_node() {
    let tree = DAG::default();
    let root = tree.node(tree.root());

    assert_eq!(root.state(), NodeState::Leaf);
    assert_eq!(root.visits(), VisitCount(0));
    assert_eq!(root.value(), Value(0.0));
}

#[test]
fn test_node_expand_from_standard_position() {
    let mut pos = create_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut tree = DAG::default();

    let leaf = match tree.node_switch(tree.root()) {
        Switch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    let expanded = tree.expand_node(leaf, &mut pos, Depth::ROOT);

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
    let mut pos = create_position("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1");
    let mut tree = DAG::default();

    let leaf = match tree.node_switch(tree.root()) {
        Switch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    let expanded = tree.expand_node(leaf, &mut pos, Depth::ROOT);

    assert!(matches!(expanded, ExpandedSwitch::Terminal(_)));
    assert_eq!(tree.node(tree.root()).state(), NodeState::Terminal);
    assert_eq!(tree.size(), 1); // Size should remain 1 as no branches are created
}

#[test]
fn test_node_expand_from_stalemate_position_becomes_terminal() {
    let mut pos = create_position("k6R/8/1K6/8/8/8/8/8 b - - 0 1");
    let mut tree = DAG::default();

    let leaf = match tree.node_switch(tree.root()) {
        Switch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    let expanded = tree.expand_node(leaf, &mut pos, Depth::ROOT);

    assert!(matches!(expanded, ExpandedSwitch::Terminal(_)));
    assert_eq!(tree.node(tree.root()).state(), NodeState::Terminal);
}

// --- Tree Advancing & Traversal ---

#[test]
fn test_tree_best_move_on_empty_tree() {
    let tree = DAG::default();
    assert!(tree.maybe_best_move(tree.root()).is_none());
}

#[test]
fn test_tree_advance_best_moves_root() {
    let mut pos = create_position("k7/8/8/8/8/8/2PPPP2/K7 w - - 0 1");
    let mut tree = DAG::default();
    let mut back_buffer = DAG::default();

    let leaf = tree.node_switch(tree.root()).get::<Leaf>().unwrap();
    tree.expand_node(leaf, &mut pos, Depth::ROOT);
    let branching = tree.node_switch(tree.root()).get::<Branching>().unwrap();

    // Force a high visit count on the first branch to make it "best"
    let target_node_id = tree.branches(branching)[0].node();
    tree.node_data_mut(target_node_id).visits = VisitCount(100);

    // Advance GC
    tree.advance_to(&mut back_buffer, target_node_id);

    // The new tree root should have the forced 100 visits and be back to a Leaf
    // state
    assert_eq!(tree.node(tree.root()).visits(), VisitCount(100));
    assert_eq!(tree.node(tree.root()).state(), NodeState::Leaf);
}

#[test]
fn test_tree_advance_to_specific_move() {
    let mut pos = create_position("k7/8/8/8/8/8/2PPPP2/K7 w - - 0 1");
    let mut tree = DAG::default();
    let mut back_buffer = DAG::default();

    let leaf = tree.node_switch(tree.root()).get::<Leaf>().unwrap();
    tree.expand_node(leaf, &mut pos, Depth::ROOT);

    let branching = tree.node_switch(tree.root()).get::<Branching>().unwrap();
    let target_node_id = tree.branches(branching)[0].node();

    tree.advance_to(&mut back_buffer, target_node_id);

    assert_eq!(tree.node(tree.root()).state(), NodeState::Leaf);
}

#[test]
fn test_tree_principal_variation_from_leaf() {
    let tree = DAG::default();
    let pv = tree.principal_line();
    assert!(pv.count() == 0);
}

// #[test]
// fn test_tree_principal_variation_simple_path() {
//     let mut tree = DAG::default();
//     tree.arena.clear();

//     // Manually construct an interconnected 3-level tree: root(0) -> mid(1)
// ->     // leaf(2) CRITICAL: We MUST use NodeState::Evaluated so
// `principal_line` is     // allowed to traverse them!
//     tree.arena.nodes.push(NodeData {
//         branch_start: 0,
//         branch_count: MoveIndex::from(1),
//         visits: VisitCount(10),
//         value: Value(0.),
//         state: NodeState::Evaluated,
//     });
//     tree.arena.nodes.push(NodeData {
//         branch_start: 1,
//         branch_count: MoveIndex::from(1),
//         visits: VisitCount(5),
//         value: Value(0.),
//         state: NodeState::Evaluated,
//     });
//     tree.arena.nodes.push(NodeData {
//         branch_start: 0,
//         branch_count: MoveIndex::from(0),
//         visits: VisitCount(5),
//         value: Value(0.),
//         state: NodeState::Leaf,
//     });

//     tree.arena.branches.push(Branch {
//         node: RtNodeId::new(1),
//         policy: Probability::new(0.8),
//         mov: Move::new(squares::E2, squares::E4, move_flags::QUIET),
//     });
//     tree.arena.branches.push(Branch {
//         node: RtNodeId::new(2),
//         policy: Probability::new(0.7),
//         mov: Move::new(squares::E7, squares::E5, move_flags::QUIET),
//     });

//     let pv = tree.principal_line();

//     assert_eq!(pv.len(), 2);
//     assert_eq!(
//         pv.0[0].mov(),
//         Move::new(squares::E2, squares::E4, move_flags::QUIET)
//     );
//     assert_eq!(
//         pv.0[1].mov(),
//         Move::new(squares::E7, squares::E5, move_flags::QUIET)
//     );
// }

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

// ================================================
// ADVANCE_TO AND GARBAGE COLLECTION TESTS
// ================================================

// Helper to create dummy hashes for testing (using u64)
fn hash(id: u64) -> zobrist::Hash {
    zobrist::Hash::from_v(id)
}

#[test]
fn test_advance_to_preserves_subtree_and_discards_siblings() {
    let mut tree = DAG::new(&Position::default()); // or create empty
    let mut back_buffer = DAG::new(&Position::default());

    // Clear the arena to build our custom DAG
    tree.arena.clear();

    // Define node hashes
    let root_hash = hash(0);
    let sibling_hash = hash(1);
    let target_hash = hash(2);
    let grandchild_hash = hash(3);

    // Node 0: Root (Evaluated)
    tree.arena.nodes.insert(
        root_hash,
        NodeData {
            branch_start: 0,
            branch_count: MoveIndex::from(2),
            visits: VisitCount(10),
            value: Value(0.),
            state: NodeState::Evaluated,
        },
    );
    // Node 1: Sibling (Leaf)
    tree.arena.nodes.insert(
        sibling_hash,
        NodeData {
            branch_start: 0,
            branch_count: MoveIndex::from(0),
            visits: VisitCount(2),
            value: Value(-1.),
            state: NodeState::Leaf,
        },
    );
    // Node 2: Target (Evaluated)
    tree.arena.nodes.insert(
        target_hash,
        NodeData {
            branch_start: 2, // after root's branches
            branch_count: MoveIndex::from(1),
            visits: VisitCount(8),
            value: Value(1.),
            state: NodeState::Evaluated,
        },
    );
    // Node 3: Grandchild (Leaf)
    tree.arena.nodes.insert(
        grandchild_hash,
        NodeData {
            branch_start: 0,
            branch_count: MoveIndex::from(0),
            visits: VisitCount(5),
            value: Value(2.),
            state: NodeState::Leaf,
        },
    );

    // Root branches
    tree.arena.branches.push(Branch {
        is_init: true,
        node: RtNodeId::from(sibling_hash),
        policy: Probability::new(0.1),
        mov: Move::new(squares::A1, squares::A2, move_flags::QUIET),
    });
    tree.arena.branches.push(Branch {
        is_init: true,
        node: RtNodeId::from(target_hash),
        policy: Probability::new(0.9),
        mov: Move::new(squares::A1, squares::B1, move_flags::QUIET),
    });
    // Target branches
    tree.arena.branches.push(Branch {
        is_init: true,
        node: RtNodeId::from(grandchild_hash),
        policy: Probability::new(1.0),
        mov: Move::new(squares::B1, squares::B2, move_flags::QUIET),
    });

    // Manually set tree root (normally set by Tree::new)
    tree.root = RtNodeId::from(root_hash);
    tree.size = tree.arena.nodes.len();

    // Execute GC advance to target node
    tree.advance_to(&mut back_buffer, RtNodeId::from(target_hash));

    // Assert Global Tree State
    assert_eq!(tree.size(), 2); // Target + Grandchild
    assert_eq!(tree.maxheight(), Height(2));

    // Assert New Root (formerly Target)
    let new_root = tree.node(tree.root());
    assert_eq!(new_root.visits(), VisitCount(8));
    assert_eq!(new_root.value(), Value(1.0));
    assert_eq!(new_root.state(), NodeState::Evaluated);

    let branches = tree.branches_rt(tree.root());
    assert_eq!(branches.len(), 1);

    // Assert Retained Child (formerly Grandchild)
    let child = tree.node(branches[0].node().unwrap());
    assert_eq!(child.visits(), VisitCount(5));
    assert_eq!(child.value(), Value(2.0));
    assert_eq!(child.state(), NodeState::Leaf);
}

#[test]
fn test_advance_to_leaf_node_resets_tree_size_and_height() {
    let mut tree = DAG::default();
    let mut back_buffer = DAG::default();
    tree.arena.clear();

    let root_hash = hash(0);
    let leaf_hash = hash(1);

    // Node 0: Root (Evaluated)
    tree.arena.nodes.insert(
        root_hash,
        NodeData {
            branch_start: 0,
            branch_count: MoveIndex::from(1),
            visits: VisitCount(10),
            value: Value(0.),
            state: NodeState::Evaluated,
        },
    );
    // Node 1: Leaf Target
    tree.arena.nodes.insert(
        leaf_hash,
        NodeData {
            branch_start: 0,
            branch_count: MoveIndex::from(0),
            visits: VisitCount(5),
            value: Value(5.5),
            state: NodeState::Leaf,
        },
    );

    tree.arena.branches.push(Branch {
        is_init: true,
        node: RtNodeId::from(leaf_hash),
        policy: Probability::new(1.0),
        mov: Move::new(squares::E2, squares::E4, move_flags::QUIET),
    });

    tree.root = RtNodeId::from(root_hash);
    tree.size = tree.arena.nodes.len();

    // Advance to leaf
    tree.advance_to(&mut back_buffer, RtNodeId::from(leaf_hash));

    assert_eq!(tree.size(), 1);
    assert_eq!(tree.maxheight(), Height::ROOT);

    let new_root = tree.node(tree.root());
    assert_eq!(new_root.visits(), VisitCount(5));
    assert_eq!(new_root.value(), Value(5.5));
    assert_eq!(new_root.state(), NodeState::Leaf);
    assert_eq!(tree.branches_rt(tree.root()).len(), 0);
}

#[test]
fn test_advance_to_deeper_level_updates_pointers_correctly() {
    let mut pos = create_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut tree = DAG::default();
    let mut back_buffer = DAG::default();

    // Expand Root -> Level 1
    let leaf = tree.node_switch(tree.root()).get::<Leaf>().unwrap();
    tree.expand_node(leaf, &mut pos, Depth::ROOT);

    // Grab first child and expand it -> Level 2
    let level_1_branch = tree.branches_rt(tree.root())[0].clone();
    let child_leaf = tree
        .node_switch(level_1_branch.node())
        .get::<Leaf>()
        .unwrap();

    // Fake a position for child expansion to simulate depth
    let mut pos_copy = pos.clone();
    pos_copy.make_move(level_1_branch.mov());
    tree.expand_node(child_leaf, &mut pos_copy, Depth::new(1));

    let initial_size_before_gc = tree.size();

    // Advance to Level 1
    tree.advance_to(&mut back_buffer, level_1_branch.node());

    // We expect the tree size to shrink drastically because we discarded 19 of the
    // 20 initial moves and only kept the 1 move we advanced to + its newly
    // expanded children.
    assert!(tree.size() < initial_size_before_gc);
    assert_eq!(tree.maxheight(), Height(2));

    // Ensure the new root has branches (since we expanded it prior to advance_to)
    assert!(!tree.branches_rt(tree.root()).is_empty());
}
