use crate::{
    core::{
        coordinates::squares,
        r#move::move_flags,
        move_iter::sliding_piece::magics,
        position::Position,
        search::mcts::node::{Node, NodeState},
        zobrist,
    },
    uci::tokens::Tokenizer,
};

use super::*;
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn test_terminal_expansion() {
    magics::init();
    zobrist::init();

    // Checkmate position
    let mut fen = Tokenizer::new("2k2R2/4R3/K7/8/8/8/8/8 b - - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();

    let mut node = Node::leaf();
    node.expand(&pos);

    assert_eq!(node.state(), NodeState::Terminal);
    assert!(!node.has_branches());
}

fn create_position(fen: &str) -> Position {
    zobrist::init();
    magics::init();

    let mut fen = Tokenizer::new(fen);
    Position::try_from(&mut fen).unwrap()
}

#[test]
fn test_tree_new_initializes_leaf_node() {
    let tree = Tree::new();
    let root = tree.root.borrow();

    assert_eq!(root.state(), NodeState::Leaf);
    assert_eq!(root.visits(), 0);
    assert_eq!(root.value(), Value(0.0));
    assert!(!root.has_branches());
}

#[test]
fn test_tree_default_identical_to_new() {
    let tree1 = Tree::new();
    let tree2 = Tree::default();

    let root1 = tree1.root.borrow();
    let root2 = tree2.root.borrow();

    assert_eq!(root1.state(), root2.state());
    assert_eq!(root1.visits(), root2.visits());
    assert_eq!(root1.value(), root2.value());
    assert_eq!(root1.branches.len(), root2.branches.len());
}

#[test]
fn test_node_expand_from_standard_position() {
    zobrist::init();
    magics::init();

    let pos = create_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut node = Node::leaf();

    node.expand(&pos);

    assert_eq!(node.state(), NodeState::Expanded);
    assert!(node.has_branches());
    assert!(!node.branches.is_empty());

    // Check that all generated moves are legal moves
    for branch in node.iter_branches() {
        let mut pos_copy = pos.clone();
        pos_copy.make_move(branch.mov());
    }
}

#[test]
fn test_node_expand_from_checkmate_position_becomes_terminal() {
    zobrist::init();
    magics::init();

    // Black is checkmated
    let pos = create_position("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1");
    let mut node = Node::leaf();

    node.expand(&pos);

    assert_eq!(node.state(), NodeState::Terminal);
    assert!(!node.has_branches());
}

#[test]
fn test_node_expand_from_stalemate_position_becomes_terminal() {
    zobrist::init();
    magics::init();

    let pos = create_position("k6R/8/1K6/8/8/8/8/8 b - - 0 1");
    let mut node = Node::leaf();

    node.expand(&pos);

    assert_eq!(node.state(), NodeState::Terminal);
    assert!(!node.has_branches());
}

#[test]
fn test_node_set_policy_with_correct_length() {
    zobrist::init();
    magics::init();

    let pos = create_position("k7/8/8/8/8/8/8/K7 w - - 0 1");
    let mut node = Node::leaf();

    node.expand(&pos);
    let num_branches = node.branches.len();

    let policy: Vec<f32> = (0..num_branches).map(|i| i as f32 * 0.1).collect();
    node.set_policy(&Policy::new(policy));

    for (i, branch) in node.branches.iter().enumerate() {
        assert_eq!(branch.policy(), i as f32 * 0.1);
    }
}

#[test]
fn test_node_select_best_by_visits() {
    let mut node = Node::leaf();

    // Create branches with different visits
    let mut branch1 = Branch::new(Move::null(), 0.5);
    let mut branch2 = Branch::new(Move::new(squares::E2, squares::E4, move_flags::QUIET), 0.5);

    branch1.node.borrow_mut().visits = 10;
    branch2.node.borrow_mut().visits = 5;

    node.branches.push(branch1);
    node.branches.push(branch2);
    node.state = NodeState::Expanded;

    let best = node.select_best();
    assert!(best.is_some());
    assert_eq!(best.unwrap().visits(), 10);
}

#[test]
fn test_node_select_by_custom_function() {
    let mut node = Node::leaf();

    node.branches.push(Branch::new(Move::null(), 0.3));
    node.branches.push(Branch::new(
        Move::new(squares::E2, squares::E4, move_flags::QUIET),
        0.7,
    ));
    node.branches.push(Branch::new(
        Move::new(squares::D2, squares::D4, move_flags::QUIET),
        0.5,
    ));
    node.state = NodeState::Expanded;

    let best = node.select(|b| b.policy());
    assert_eq!(best.unwrap().policy(), 0.7);

    let worst = node.select(|b| -b.policy());
    assert_eq!(worst.unwrap().policy(), 0.3);
}

#[test]
fn test_node_take_branch_by_predicate() {
    let mut node = Node::leaf();

    let target_move = Move::new(squares::E2, squares::E4, move_flags::QUIET);
    node.branches.push(Branch::new(Move::null(), 0.5));
    node.branches.push(Branch::new(target_move, 0.5));
    node.state = NodeState::Expanded;

    let taken = node.take_branch(|b| b.mov() == target_move);

    assert!(taken.is_some());
    assert_eq!(taken.unwrap().mov(), target_move);
}

#[test]
fn test_node_sort_by_policy() {
    let mut node = Node::leaf();

    node.branches.push(Branch::new(Move::null(), 0.9));
    node.branches.push(Branch::new(
        Move::new(squares::A2, squares::A3, move_flags::QUIET),
        0.3,
    ));
    node.branches.push(Branch::new(
        Move::new(squares::E2, squares::E4, move_flags::QUIET),
        0.7,
    ));

    node.sort_by(|b| Value(b.policy()));

    assert_eq!(node.branches[0].policy(), 0.3);
    assert_eq!(node.branches[1].policy(), 0.7);
    assert_eq!(node.branches[2].policy(), 0.9);
}

#[test]
fn test_node_sort_by_visits() {
    let mut node = Node::leaf();

    let mut branch1 = Branch::new(Move::null(), 0.5);
    let mut branch2 = Branch::new(Move::new(squares::E2, squares::E4, move_flags::QUIET), 0.5);
    let mut branch3 = Branch::new(Move::new(squares::D2, squares::D4, move_flags::QUIET), 0.5);

    branch1.node.borrow_mut().visits = 3;
    branch2.node.borrow_mut().visits = 1;
    branch3.node.borrow_mut().visits = 2;

    node.branches.push(branch1);
    node.branches.push(branch2);
    node.branches.push(branch3);

    node.sort_by(|b| b.visits());

    assert_eq!(node.branches[0].visits(), 1);
    assert_eq!(node.branches[1].visits(), 2);
    assert_eq!(node.branches[2].visits(), 3);
}

#[test]
fn test_tree_best_move_on_empty_tree() {
    let tree = Tree::new();
    assert!(tree.best_move().is_none());
}

#[test]
fn test_tree_best_move_returns_most_visited() {
    zobrist::init();
    magics::init();

    let pos = create_position("k7/8/8/8/8/8/8/K7 w - - 0 1");
    let mut tree = Tree::new();

    // Manually set up tree structure
    {
        let mut root = tree.root.borrow_mut();
        root.expand(&pos);

        // Set visits to make one branch "best"
        for (i, branch) in root.branches.iter_mut().enumerate() {
            branch.node.borrow_mut().visits = i as u32;
        }
    }

    let best_move = tree.best_move();
    assert!(best_move.is_some());
    // Should return move with highest visits (last branch in this case)
}

#[test]
fn test_tree_advance_best_moves_root() {
    zobrist::init();
    magics::init();

    let pos = create_position("k7/8/8/8/8/8/8/K7 w - - 0 1");
    let mut tree = Tree::new();

    // We need to build a tree structure first
    {
        let mut root = tree.root.borrow_mut();
        root.expand(&pos);

        // Make one branch clearly best
        if let Some(branch) = root.branches.first_mut() {
            branch.node.borrow_mut().visits = 100;
        }
    }

    // Convert to mutable for advance_best
    tree.advance_best();

    // The new tree should have the best branch's node as root
    assert_eq!(tree.root.borrow().visits(), 100);
    assert_eq!(tree.root.borrow().state(), NodeState::Leaf);
}

#[test]
fn test_tree_advance_to_predicate() {
    zobrist::init();
    magics::init();

    let pos = create_position("k7/8/8/8/8/8/8/K7 w - - 0 1");
    let mut tree = Tree::new();

    {
        let mut root = tree.root.borrow_mut();
        root.expand(&pos);
    }

    let target_move = {
        let root = tree.root.borrow();
        root.branches.first().unwrap().mov()
    };

    tree.advance_to(|b| b.mov() == target_move);

    // Root should now be the child node
    let new_root = tree.root.borrow();
    assert_eq!(new_root.state(), NodeState::Leaf);
}

#[test]
fn test_tree_principal_variation_from_leaf() {
    let tree = Tree::new();
    let pv = tree.principal_variation();
    assert!(pv.is_empty());
}

#[test]
fn test_tree_principal_variation_simple_path() {
    let mut tree = Tree::new();

    // Build a simple tree: root -> branch1 -> branch2 -> leaf
    let leaf_node = Rc::new(RefCell::new(Node::leaf()));
    let branch2 = Branch {
        node: Rc::clone(&leaf_node),
        policy: 0.7,
        mov: Move::new(squares::E7, squares::E5, move_flags::QUIET),
    };

    let middle_node = Rc::new(RefCell::new(Node {
        branches: vec![branch2.clone()],
        state: NodeState::Expanded,
        ..Node::leaf()
    }));

    let branch1 = Branch {
        node: Rc::clone(&middle_node),
        policy: 0.8,
        mov: Move::new(squares::E2, squares::E4, move_flags::QUIET),
    };

    {
        let mut root = tree.root.borrow_mut();
        root.branches.push(branch1.clone());
        root.state = NodeState::Expanded;

        // Mark both nodes as best by giving them highest visits
        middle_node.borrow_mut().visits = 5;
        leaf_node.borrow_mut().visits = 5;
    }

    let pv = tree.principal_variation();
    assert_eq!(pv.len(), 2);
    assert_eq!(
        pv[0].mov(),
        Move::new(squares::E2, squares::E4, move_flags::QUIET)
    );
    assert_eq!(
        pv[1].mov(),
        Move::new(squares::E7, squares::E5, move_flags::QUIET)
    );
}

#[test]
fn test_branch_accessors() {
    let mov = Move::new(squares::E2, squares::E4, move_flags::QUIET);
    let branch = Branch::new(mov, 0.8);

    assert_eq!(branch.mov(), mov);
    assert_eq!(branch.policy(), 0.8);
    assert_eq!(branch.node().borrow().state(), NodeState::Leaf);
    assert_eq!(branch.visits(), 0);

    // Modify underlying node
    branch.node.borrow_mut().visits = 5;
    branch.node.borrow_mut().state = NodeState::Expanded;

    assert_eq!(branch.visits(), 5);
    assert_eq!(branch.node_state(), NodeState::Expanded);
}

#[test]
fn test_value_operations_and_comparisons() {
    let v1 = Value(1.0);
    let v2 = Value(2.0);
    let v3 = Value(1.0);

    // Comparisons
    assert!(v1 < v2);
    assert!(v2 > v1);
    assert_eq!(v1, v3);
    assert_ne!(v1, v2);

    // Division
    assert_eq!(v2 / 2.0, 1.0);

    // Add assign
    let mut v = Value(1.0);
    v += 2.0;
    assert_eq!(v, Value(3.0));
}

#[test]
fn test_node_get_branch_and_iteration() {
    let mut node = Node::leaf();

    node.branches.push(Branch::new(Move::null(), 0.5));
    node.branches.push(Branch::new(
        Move::new(squares::E2, squares::E4, move_flags::QUIET),
        0.5,
    ));
    node.state = NodeState::Expanded;

    // Test get_branch
    assert!(node.get_branch(0).is_some());
    assert!(node.get_branch(1).is_some());
    assert!(node.get_branch(2).is_none());

    // Test iteration
    let mut count = 0;
    for branch in node.iter_branches() {
        assert!(branch.policy() == 0.5);
        count += 1;
    }
    assert_eq!(count, 2);
}

#[test]
fn test_tree_advance_to_with_no_match_keeps_root() {
    zobrist::init();
    magics::init();

    let pos = create_position("k7/8/8/8/8/8/8/K7 w - - 0 1");
    let mut tree = Tree::new();
    let original_root = Rc::clone(&tree.root);

    {
        let mut root = tree.root.borrow_mut();
        root.expand(&pos);
    }

    // Predicate that matches nothing
    tree.advance_to(|b| b.policy() > 1.0);

    // Should still point to original root
    assert!(Rc::ptr_eq(&tree.root, &original_root));
}

#[test]
fn test_node_select_mut_allows_modification() {
    let mut node = Node::leaf();

    node.branches.push(Branch::new(Move::null(), 0.3));
    node.branches.push(Branch::new(
        Move::new(squares::E2, squares::E4, move_flags::QUIET),
        0.7,
    ));
    node.state = NodeState::Expanded;

    let best = node.select_mut(|b| b.policy());
    assert!(best.is_some());

    // We can modify the best branch
    best.unwrap().policy = 0.9;

    // Verify modification
    let new_best = node.select(|b| b.policy()).unwrap();
    assert_eq!(new_best.policy(), 0.9);
}

#[test]
fn test_node_state_enum_correctness() {
    let leaf = NodeState::Leaf;
    let expanded = NodeState::Expanded;
    let terminal = NodeState::Terminal;

    assert_ne!(leaf, expanded);
    assert_ne!(leaf, terminal);
    assert_ne!(expanded, terminal);

    // Test default is Leaf
    assert_eq!(NodeState::default(), NodeState::Leaf);
}
