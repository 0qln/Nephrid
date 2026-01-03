use crate::{
    core::{
        color::colors,
        coordinates::squares,
        r#move::{Move, move_flags},
        move_iter::sliding_piece::magics,
        piece::{Piece, piece_type},
        position::Position,
        zobrist,
    },
    misc::ConstFrom,
    uci::tokens::Tokenizer,
};

#[test]
fn same_position_same_key() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    magics::init();
    zobrist::init();

    let mut pos = Position::from_fen(fen).unwrap();

    use move_flags::*;
    use squares::*;

    let key_begin = pos.get_key();

    // make some moves
    pos.make_move(Move::new(G1, F3, QUIET));
    pos.make_move(Move::new(G8, F6, QUIET));

    // get back into the position
    pos.make_move(Move::new(F3, G1, QUIET));
    pos.make_move(Move::new(F6, G8, QUIET));

    let key_end = pos.get_key();

    assert_eq!(key_begin, key_end);
}

#[test]
fn move_piece_back_and_forth() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    magics::init();
    zobrist::init();

    let mut pos = Position::from_fen(fen).unwrap();

    use move_flags::*;
    use squares::*;

    let key_begin = pos.get_key();

    // make some moves
    pos.make_move(Move::new(G1, F3, QUIET));

    // get back into the position
    pos.make_move(Move::new(F3, G1, QUIET));

    let key_end = pos.get_key();

    assert_eq!(key_begin, key_end);
}

#[test]
fn move_piece_back_and_forth_hash_only() {
    magics::init();
    zobrist::init();

    use squares::*;

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let pos = Position::from_fen(fen).unwrap();
    let key_begin = pos.get_key();

    let piece = Piece::from_c((colors::WHITE, piece_type::KNIGHT));
    let mut key = key_begin;
    key.move_piece_sq(G1, F3, piece);
    key.move_piece_sq(F3, G1, piece);

    let key_end = key;

    assert_eq!(key_begin, key_end);
}

#[test]
fn place_piece_and_remove_hash_only() {
    magics::init();
    zobrist::init();

    use squares::*;

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let pos = Position::from_fen(fen).unwrap();
    let key_begin = pos.get_key();

    let piece = Piece::from_c((colors::WHITE, piece_type::KNIGHT));
    let mut key = key_begin;
    key.toggle_piece_sq(G1, piece);
    key.toggle_piece_sq(G1, piece);

    let key_end = key;

    assert_eq!(key_begin, key_end);
}

#[test]
fn place_piece_and_remove_nested_hash_only() {
    magics::init();
    zobrist::init();

    use squares::*;

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let pos = Position::from_fen(fen).unwrap();
    let key_begin = pos.get_key();

    let piece = Piece::from_c((colors::WHITE, piece_type::KNIGHT));
    let mut key = key_begin;
    key.toggle_piece_sq(G1, piece);
    assert_ne!(key_begin, key);
    key.toggle_piece_sq(F3, piece);
    assert_ne!(key_begin, key);
    key.toggle_piece_sq(F3, piece);
    assert_ne!(key_begin, key);
    key.toggle_piece_sq(G1, piece);

    let key_end = key;

    assert_eq!(key_begin, key_end);
}

#[test]
fn place_piece_and_remove_out_of_order_hash_only() {
    magics::init();
    zobrist::init();

    use squares::*;

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let pos = Position::from_fen(fen).unwrap();
    let key_begin = pos.get_key();

    let piece = Piece::from_c((colors::WHITE, piece_type::KNIGHT));
    let mut key = key_begin;
    key.toggle_piece_sq(G1, piece);
    assert_ne!(key_begin, key);
    key.toggle_piece_sq(F3, piece);
    assert_ne!(key_begin, key);
    key.toggle_piece_sq(G1, piece);
    assert_ne!(key_begin, key);
    key.toggle_piece_sq(F3, piece);

    let key_end = key;
    assert_eq!(key_begin, key_end);
}
