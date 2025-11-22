use crate::{
    core::{
        color::colors,
        coordinates::squares,
        r#move::move_flags,
        move_iter::sliding_piece::magics,
        piece::{Piece, piece_type},
        position::Position,
        search::mcts::{NodeState, Tree},
    },
    misc::ConstFrom,
    uci::tokens::Tokenizer,
};

#[test]
fn initialization() {
    magics::init();

    let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();
    let tree = Tree::new(&pos);

    assert_eq!(tree.root.state, NodeState::Branch);
    assert!(!tree.root.children.is_empty());
    assert!(
        tree.root
            .children
            .iter()
            .all(|c| c.state == NodeState::Leaf)
    );
}

#[test]
fn growth() {
    let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut pos = Position::try_from(&mut fen).unwrap();
    let mut tree = Tree::new(&pos);

    // Initial state checks
    assert_eq!(tree.root.score.playouts, 0);

    // Perform one growth iteration
    tree.grow(&mut pos);

    // Verify backpropagation
    assert_eq!(tree.root.score.playouts, 1);
    assert!(tree.root.children.iter().any(|c| c.score.playouts == 1));
}

#[test]
fn selects_unexpanded_leaf() {
    magics::init();

    // Setup position with one legal move
    let mut fen = Tokenizer::new("K7/8/1k6/8/8/8/8/8 w - - 0 1");
    let mut pos = Position::try_from(&mut fen).unwrap();
    let mut tree = Tree::new(&pos);

    tree.select_leaf_mut(&mut pos);
    let stack = tree.selection_buffer;

    // Should select the only child
    assert_eq!(stack.len(), 1);
    unsafe {
        let node = stack[0].as_ref();
        assert_eq!(node.state, NodeState::Leaf);
        assert_eq!(node.score.playouts, 0);
    }
    // Verify move was made
    assert!(pos.get_piece(squares::B8) == Piece::from_c((colors::WHITE, piece_type::KING)));
}

#[test]
fn expands_leaf_and_selects_child() {
    magics::init();

    // Position with known follow-up moves
    let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut pos = Position::try_from(&mut fen).unwrap();
    let mut tree = Tree::new(&pos);

    // Force initial selection to mark node as visited
    for _ in 0..20 {
        tree.grow(&mut pos.clone());
    }

    tree.select_leaf_mut(&mut pos);
    let stack = tree.selection_buffer;

    // Should have root -> branch -> leaf
    assert_eq!(stack.len(), 2);
    unsafe {
        let branch_node = stack[0].as_ref();
        assert_eq!(branch_node.state, NodeState::Branch);

        let leaf_node = stack[1].as_ref();
        assert_eq!(leaf_node.state, NodeState::Leaf);
    }
}

#[test]
fn handles_terminal_nodes_through_expansion() {
    magics::init();

    let fen = "7k/P7/6K1/5nn1/6n1/5nn1/8/8 w - - 0 1";
    let mut fen = Tokenizer::new(fen);
    let pos = Position::try_from(&mut fen).unwrap();
    let mut tree = Tree::new(&pos);

    // Grow tree to create branch nodes
    for _ in 0..100 {
        tree.grow(&mut pos.clone());
    }

    // Verify terminal handling
    let node = tree
        .root
        .children
        .iter()
        .find(|n| n.mov.get_flag() == move_flags::PROMOTION_ROOK)
        .unwrap();
    assert_eq!(node.state, NodeState::Terminal);
    assert!(node.children.is_empty());
}

#[test]
fn traverses_multiple_branch_nodes() {
    magics::init();

    // Setup deep tree with known structure
    let mut fen =
        Tokenizer::new("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    let mut pos = Position::try_from(&mut fen).unwrap();
    let mut tree = Tree::new(&pos);

    // Grow tree to create branch nodes
    for _ in 0..100 {
        tree.grow(&mut pos.clone());
    }

    tree.select_leaf_mut(&mut pos);
    let stack = tree.selection_buffer;

    // Verify path depth and node types
    assert!(stack.len() >= 2);
    unsafe {
        assert!(stack.iter().any(|n| n.as_ref().state == NodeState::Branch));
        assert!(stack.last().unwrap().as_ref().state == NodeState::Leaf);
    }
}
