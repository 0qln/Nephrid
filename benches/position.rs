use core::fmt;
use std::{fmt::Display, ops::ControlFlow};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use nephrid::{engine::{color::Color, coordinates::Square, fen::Fen, move_iter::{fold_legal_moves, sliding_piece::magics}, piece::{Piece, PieceType}, position::Position}, misc::ConstFrom};

pub fn get_bitboard(c: &mut Criterion) {
    let inputs = [
        Pair(PieceType::PAWN, Color::WHITE),
        Pair(PieceType::QUEEN, Color::WHITE),
        Pair(PieceType::PAWN, Color::BLACK),
        Pair(PieceType::QUEEN, Color::BLACK),
    ];
    
    magics::init();
    let pos = Position::start_position();

    for pair in inputs {
        c.bench_with_input(
            BenchmarkId::new("position::get_bitboard", pair),
            &pair,
            |b, &pair| b.iter(|| pos.get_bitboard(pair.0, pair.1)),
        );
    }
}

pub fn put_piece(c: &mut Criterion) {
    let inputs = [
        Pair(Square::A3, Piece::from_c((Color::WHITE, PieceType::PAWN))),
        Pair(Square::B4, Piece::from_c((Color::WHITE, PieceType::QUEEN))),
        Pair(Square::E4, Piece::from_c((Color::BLACK, PieceType::PAWN))),
        Pair(Square::H5, Piece::from_c((Color::BLACK, PieceType::QUEEN))),
    ];

    magics::init();
    let mut pos = Position::default();

    for pair in inputs {
        c.bench_with_input(
            BenchmarkId::new("position::put_piece", pair),
            &pair,
            |b, &pair| b.iter(|| unsafe { pos.put_piece_unsafe(pair.0, pair.1) }),
        );
    }
}

pub fn remove_piece(c: &mut Criterion) {
    let inputs = [
        Square::A1,
        Square::B4,
        Square::E2,
        Square::H8,
    ];

    magics::init();
    let mut pos = Position::start_position();

    for sq in inputs {
        c.bench_with_input(
            BenchmarkId::new("position::remove_piece", sq),
            &sq,
            |b, &sq| b.iter(|| unsafe { pos.remove_piece_unsafe(sq) }),
        );
    }
}

pub fn move_piece(c: &mut Criterion) {
    let inputs = [
        Pair(Square::A1, Square::A2),
        Pair(Square::E2, Square::E4),
        Pair(Square::H8, Square::H7),
    ];

    magics::init();
    let mut pos = Position::start_position();

    for pair in inputs {
        c.bench_with_input(
            BenchmarkId::new("position::move_piece", pair),
            &pair,
            |b, &pair| b.iter(|| unsafe { pos.move_piece_unsafe(pair.0, pair.1) }),
        );
    }
}

pub fn make_move(c: &mut Criterion) {
    magics::init();

    let mut fen = Fen::new("2n1k3/1P6/8/4pP2/8/6B1/P2P4/R3K2R w KQ e6 0 1");
    let pos = Position::try_from(&mut fen).unwrap();
    let moves = fold_legal_moves(&pos, Vec::new(), |mut acc, m| {
        acc.push(m);
        ControlFlow::Continue::<(), _>(acc)
    }).continue_value().unwrap();
    
    for m in moves.into_iter().unique_by(|m| m.get_flag()) {
        c.bench_with_input(
            BenchmarkId::new("position::make_move", m),
            &m,
            |b, &m| b.iter(|| pos.clone().make_move(m)),
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct Pair<T1, T2>(T1, T2);

impl<T1: Display, T2: Display> fmt::Display for Pair<T1, T2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

criterion_group!(benches, 
    get_bitboard,
    put_piece,
    remove_piece,
    move_piece,
    make_move,
);

criterion_main!(benches);
