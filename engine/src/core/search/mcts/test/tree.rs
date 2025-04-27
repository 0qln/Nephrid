use burn_cuda::{Cuda, CudaDevice};

use crate::{
    core::{
        color::Color, coordinates::Square, r#move::MoveFlag, move_iter::sliding_piece::magics, piece::{Piece, PieceType}, position::Position, search::mcts::{eval::{self, model::{Model, ModelConfig}}, NodeState, Tree}, zobrist
    },
    misc::ConstFrom, uci::tokens::Tokenizer,
};

#[test]
fn initialization() {
    magics::init();
    zobrist::init();
    
    let device = CudaDevice::new(0);
    let model = ModelConfig::new().init::<eval::Backend>(&device);

    let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();

    let tree = Tree::new(&pos, &model);

    assert_eq!(tree.root.state, NodeState::Expanded);
    assert!(!tree.root.branches.is_empty());
    assert!(tree
        .root
        .branches
        .iter()
        .all(|c| c.node_state() == NodeState::Leaf));
}

// #[test]
// fn growth() {
//     magics::init();
//     zobrist::init();
//
//     let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//     let mut pos = Position::try_from(&mut fen).unwrap();
//     
//     let device = CudaDevice::new(0);
//     let model = ModelConfig::new().init::<eval::Backend>(&device);
//
//     let mut tree = Tree::new(&pos, &model);
//
//     // Initial state checks. The root node should have been evaluated.
//     assert_eq!(tree.root.score.visits, 1);
//
//     // Perform one growth iteration
//     tree.grow(&mut pos, &model);
//
//     // Verify backpropagation
//     assert_eq!(tree.root.score.visits, 2);
//     assert!(tree.root.children.iter().any(|c| c.score.playouts == 1));
// }

// #[test]
// fn selects_unexpanded_leaf() {
//     magics::init();
//     zobrist::init();
//
//     let device = CudaDevice::new(0);
//     let model = ModelConfig::new().init::<eval::Backend>(&device);
//
//     // Setup position with one legal move
//     let mut fen = Tokenizer::new("K7/8/1k6/8/8/8/8/8 w - - 0 1");
//     let mut pos = Position::try_from(&mut fen).unwrap();
//
//     let mut tree = Tree::new(&pos, &model);
//
//     tree.select_leaf_mut(&mut pos, &model);
//     let stack = tree.selection_buffer;
//
//     // Should select the only child
//     assert_eq!(stack.len(), 1);
//     unsafe {
//         let node = stack[0].as_ref();
//         assert_eq!(node.state, NodeState::Leaf);
//         assert_eq!(node.score.visits, 0);
//     }
//     // Verify move was made
//     assert!(pos.get_piece(Square::B8) == Piece::from_c((Color::WHITE, PieceType::KING)));
// }

// #[test]
// fn expands_leaf_and_selects_child() {
//     magics::init();
//     zobrist::init();
//
//     let device = CudaDevice::new(0);
//     let model = ModelConfig::new().init::<eval::Backend>(&device);
//
//     // Position with known follow-up moves
//     let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//     let mut pos = Position::try_from(&mut fen).unwrap();
//
//     let mut tree = Tree::new(&pos, &model);
//
//     // Force initial selection to mark node as visited
//     for _ in 0..20 {
//         tree.grow(&mut pos.clone(), &model);
//     }
//     
//     println!("{:#?}", tree.root);
//
//     tree.select_leaf_mut(&mut pos, &model);
//     let stack = tree.selection_buffer;
//
//     // Should have root -> branch -> leaf
//     assert_eq!(stack.len(), 2);
//     unsafe {
//         println!("{:#?}", stack);
//
//         let branch_node = stack[0].as_ref();
//         assert_eq!(branch_node.state, NodeState::Expanded);
//
//         let leaf_node = stack[1].as_ref();
//         assert_eq!(leaf_node.state, NodeState::Leaf);
//     }
// }

// #[test]
// fn handles_terminal_nodes_through_expansion() {
//     magics::init();
//
//     let fen = "7k/P7/6K1/5nn1/6n1/5nn1/8/8 w - - 0 1";
//     let mut fen = Tokenizer::new(fen);
//     let pos = Position::try_from(&mut fen).unwrap();
//     let mut tree = Tree::new(&pos);
//
//     // Grow tree to create branch nodes
//     for _ in 0..100 {
//         tree.grow(&mut pos.clone());
//     }
//
//     // Verify terminal handling
//     let node = tree
//         .root
//         .children
//         .iter()
//         .find(|n| n.mov.get_flag() == MoveFlag::PROMOTION_ROOK)
//         .unwrap();
//     assert_eq!(node.state, NodeState::Terminal);
//     assert!(node.children.is_empty());
// }

// #[test]
// fn traverses_multiple_branch_nodes() {
//     magics::init();
//
//     // Setup deep tree with known structure
//     let mut fen = Tokenizer::new("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
//     let mut pos = Position::try_from(&mut fen).unwrap();
//     let mut tree = Tree::new(&pos);
//
//     // Grow tree to create branch nodes
//     for _ in 0..100 {
//         tree.grow(&mut pos.clone());
//     }
//
//     tree.select_leaf_mut(&mut pos);
//     let stack = tree.selection_buffer;
//
//     // Verify path depth and node types
//     assert!(stack.len() >= 2);
//     unsafe {
//         assert!(stack.iter().any(|n| n.as_ref().state == NodeState::Branch));
//         assert!(stack.last().unwrap().as_ref().state == NodeState::Leaf);
//     }
// }
//