use crate::engine::{
    fen::Fen,
    move_iter::sliding_piece::magics,
    position::Position,
    r#move::Move,
    search::mcts::{Node, NodeState},
};

#[test]
fn ucb_selection() {
    magics::init(0xdead_beef);

    let mut fen = Fen::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();

    let mut root = Node::root(&pos);
    root.score.playouts = 1;
    root.score.wins = 0.0;

    let mut child1 = Node::leaf(Move::null());
    child1.score.playouts = 1;
    child1.score.wins = 1.0;

    let child2 = Node::leaf(Move::null()); // 0 playouts

    root.children = vec![child1, child2];
    root.state = NodeState::Branch;

    // Should select child2 due to infinite UCB
    let selected = root.select_mut();
    assert_eq!(selected.score.playouts, 0);
}

#[test]
fn test_terminal_expansion() {
    magics::init(0xdead_beef);

    // Checkmate position
    let mut fen = Fen::new("2k2R2/4R3/K7/8/8/8/8/8 b - - 0 1");
    let pos = Position::try_from(&mut fen).unwrap();

    let mut node = Node::leaf(Move::null());
    node.expand(&pos);

    assert_eq!(node.state, NodeState::Terminal);
    assert!(node.children.is_empty());
}
