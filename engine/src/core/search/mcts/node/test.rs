use super::*;
use crate::core::{
    Move, Position,
    coordinates::squares,
    r#move::move_flags,
    move_iter::sliding_piece::magics,
    search::mcts::node::node_state::{Branching, Leaf, NodeState, NodeSwitch},
    zobrist,
};

// --- Utility Functions ---

fn create_position(fen: &str) -> Position {
    zobrist::init();
    magics::init();
    Position::from_fen(fen).unwrap()
}

// --- Type-safe Reference Tests (CtNodeRef / RtNodeRef) ---

#[test]
fn test_ct_node_ref_initialization() {
    let leaf_node = Node::<Leaf>::new_leaf();
    let ct_leaf = CtNodeRef::new(leaf_node);

    assert_eq!(ct_leaf.inner.state.get(), NodeState::Leaf);
    assert_eq!(ct_leaf.borrow().visits(), 0);
    assert_eq!(ct_leaf.borrow().value(), Value(0.0));
}

#[test]
fn test_rt_node_ref_conversions() {
    let leaf_node = Node::<Leaf>::new_leaf();
    let ct_leaf = CtNodeRef::new(leaf_node);

    let rt_ref = RtNodeRef::from_ct(ct_leaf);
    assert_eq!(rt_ref.state(), NodeState::Leaf);

    let switch = rt_ref.into_ct();
    match switch {
        NodeSwitch::Leaf(leaf) => assert_eq!(leaf.borrow().visits(), 0),
        _ => panic!("Expected the switch to yield a Leaf variant"),
    }
}

// --- Tree Tests & Domain Expansions ---

#[test]
fn test_tree_default_initializes_leaf_node() {
    let tree = Tree::default();
    let root = tree.get_root();

    assert_eq!(root.state(), NodeState::Leaf);
    assert_eq!(root.borrow().visits(), 0);
    assert_eq!(root.borrow().value(), Value(0.0));
}

#[test]
fn test_node_expand_from_standard_position() {
    let pos = create_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut tree = Tree::default();

    let leaf = match tree.get_root().into_ct() {
        NodeSwitch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    let expanded = tree.expand_node(leaf, &pos, Depth::ROOT);

    assert!(matches!(expanded, ExpandedRefSwitch::Branching(_)));
    assert_eq!(tree.get_root().state(), NodeState::Branching);
    assert_eq!(tree.size(), 21);

    // Verify branches contain legal moves
    if let NodeSwitch::Branching(b) = tree.get_root().into_ct() {
        assert!(!b.borrow().branches().is_empty());
        for branch in b.borrow().branches() {
            let mut pos_copy = pos.clone();
            pos_copy.make_move(branch.mov()); // Asserting this doesn't panic
        }
    }
}

#[test]
fn test_node_expand_from_checkmate_position_becomes_terminal() {
    // Black is checkmated
    let pos = create_position("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1");
    let mut tree = Tree::default();

    let leaf = match tree.get_root().into_ct() {
        NodeSwitch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    let expanded = tree.expand_node(leaf, &pos, Depth::ROOT);

    assert!(matches!(expanded, ExpandedRefSwitch::Terminal(_)));
    assert_eq!(tree.get_root().state(), NodeState::Terminal);
    assert_eq!(tree.size(), 1); // Size should remain 1 as no branches are created
}

#[test]
fn test_node_expand_from_stalemate_position_becomes_terminal() {
    let pos = create_position("k6R/8/1K6/8/8/8/8/8 b - - 0 1");
    let mut tree = Tree::default();

    let leaf = match tree.get_root().into_ct() {
        NodeSwitch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    let expanded = tree.expand_node(leaf, &pos, Depth::ROOT);

    assert!(matches!(expanded, ExpandedRefSwitch::Terminal(_)));
    assert_eq!(tree.get_root().state(), NodeState::Terminal);
}

// --- Node Branch Sorting and Selection ---

#[test]
fn test_node_sort_by_visits() {
    let mut data = NodeData::default();

    let branch1 = Branch::new(Move::null(), 0.5, RtNodeRef::new(Node::<Leaf>::new_leaf()));
    let branch2 = Branch::new(
        Move::new(squares::E2, squares::E4, move_flags::QUIET),
        0.5,
        RtNodeRef::new(Node::<Leaf>::new_leaf()),
    );
    let branch3 = Branch::new(
        Move::new(squares::D2, squares::D4, move_flags::QUIET),
        0.5,
        RtNodeRef::new(Node::<Leaf>::new_leaf()),
    );

    branch1.node.borrow_mut().data.visits = 3;
    branch2.node.borrow_mut().data.visits = 1;
    branch3.node.borrow_mut().data.visits = 2;

    data.branches.push(branch1);
    data.branches.push(branch2);
    data.branches.push(branch3);

    // Build branching node using unsafe to bypass standard expansion requirements
    let mut node = unsafe { Node::<Branching>::new(data) };

    node.sort_by(|b| b.visits());

    assert_eq!(node.branches()[0].visits(), 1);
    assert_eq!(node.branches()[1].visits(), 2);
    assert_eq!(node.branches()[2].visits(), 3);
}

// --- Tree Advancing & Traversal ---

#[test]
fn test_tree_best_move_on_empty_tree() {
    let tree = Tree::default();
    assert!(tree.best_move().is_none());
}

#[test]
fn test_tree_advance_best_moves_root() {
    let pos = create_position("k7/8/8/8/8/8/2PPPP2/K7 w - - 0 1");
    let mut tree = Tree::default();

    let leaf = match tree.get_root().into_ct() {
        NodeSwitch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    let expanded = tree.expand_node(leaf, &pos, Depth::ROOT);

    if let ExpandedRefSwitch::Branching(b) = expanded {
        // Force a high visit count on the first branch to make it "best"
        let first_branch_node = b.borrow().branches()[0].node();
        first_branch_node.borrow_mut().data.visits = 100;
    }
    else {
        panic!("Position should yield branches");
    }

    // Advance to the highest visited node
    tree.advance_best();

    // The new tree root should have the forced 100 visits and be back to a Leaf
    // state
    assert_eq!(tree.get_root().borrow().visits(), 100);
    assert_eq!(tree.get_root().state(), NodeState::Leaf);
}

#[test]
fn test_tree_advance_to_predicate() {
    let pos = create_position("k7/8/8/8/8/8/2PPPP2/K7 w - - 0 1");
    let mut tree = Tree::default();

    let leaf = match tree.get_root().into_ct() {
        NodeSwitch::Leaf(l) => l,
        _ => panic!("Expected leaf"),
    };

    tree.expand_node(leaf, &pos, Depth::ROOT);

    // Identify target move
    let target_move = if let NodeSwitch::Branching(b) = tree.get_root().into_ct() {
        b.borrow().branches()[0].mov()
    }
    else {
        panic!("No branches generated");
    };

    tree.advance_to(|b| b.mov() == target_move);

    assert_eq!(tree.get_root().state(), NodeState::Leaf);
}

#[test]
fn test_tree_principal_variation_from_leaf() {
    let tree = Tree::default();
    let pv = tree.principal_variation();
    assert!(pv.is_empty());
}

#[test]
fn test_tree_principal_variation_simple_path() {
    // Manually construct an interconnected 3-level tree: root -> mid -> leaf
    let leaf_data = NodeData::default();
    let leaf_node = RtNodeRef::new(unsafe { Node::<Leaf>::new(leaf_data) });

    let mut mid_data = NodeData::default();
    let b2 = Branch::new(
        Move::new(squares::E7, squares::E5, move_flags::QUIET),
        0.7,
        leaf_node.clone(),
    );
    mid_data.branches.push(b2);
    let mid_node = RtNodeRef::new(unsafe { Node::<Branching>::new(mid_data) });

    let mut root_data = NodeData::default();
    let b1 = Branch::new(
        Move::new(squares::E2, squares::E4, move_flags::QUIET),
        0.8,
        mid_node.clone(),
    );
    root_data.branches.push(b1);
    let root_node = RtNodeRef::new(unsafe { Node::<Branching>::new(root_data) });

    let tree = Tree {
        root: root_node,
        size: 3,
        // mindepth: Depth::new(3),
        maxdepth: Depth::new(3),
    };

    // Synthetically inflate visits to force `select_best` behavior for PV
    mid_node.borrow_mut().data.visits = 5;
    leaf_node.borrow_mut().data.visits = 5;

    let pv = tree.principal_variation();

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
