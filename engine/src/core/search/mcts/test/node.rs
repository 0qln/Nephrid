use burn_cuda::CudaDevice;

use crate::{core::{
    r#move::Move, move_iter::sliding_piece::magics, position::Position, search::mcts::{eval::{self, model::ModelConfig}, Node, NodeState}, zobrist
}, uci::tokens::Tokenizer};

// #[test]
// fn puct_selection() {
//     magics::init();
//     zobrist::init();
//     
//     let device = CudaDevice::new(0);
//     let model = ModelConfig::new().init::<eval::Backend>(&device);
//
//     let mut fen = Tokenizer::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//     let pos = Position::try_from(&mut fen).unwrap();
//
//     let mut root = Node::root(&pos, &model);
//     _ = root.children.split_off(2);
//
//     root.score.visits = 1;
//     root.score.value = 0.0;
//
//     let child1 = &mut root.children[0];
//     child1.score.playouts = 1;
//     child1.score.quality = 1.0;
//
//     // Should select child2 due to infinite PUCT
//     let selected = root.select_mut();
//     assert_eq!(selected.score.playouts, 0);
// }

// #[test]
// fn test_terminal_expansion() {
//     magics::init();
//
//     // Checkmate position
//     let mut fen = Tokenizer::new("2k2R2/4R3/K7/8/8/8/8/8 b - - 0 1");
//     let pos = Position::try_from(&mut fen).unwrap();
//
//     let mut node = Node::leaf(Move::null());
//     node.expand(&pos);
//
//     assert_eq!(node.state, NodeState::Terminal);
//     assert!(node.children.is_empty());
// }
