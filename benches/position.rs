use core::fmt;
use std::fmt::Display;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nephrid::{engine::{color::Color, coordinates::Square, move_iter::sliding_piece::magics, piece::{Piece, PieceType}, position::Position}, misc::ConstFrom};

pub fn get_bitboard(c: &mut Criterion) {
    let inputs = [
        Pair(PieceType::PAWN, Color::WHITE),
        Pair(PieceType::QUEEN, Color::WHITE),
        Pair(PieceType::PAWN, Color::BLACK),
        Pair(PieceType::QUEEN, Color::BLACK),
    ];
    
    magics::init(0xdead_beef);
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

    magics::init(0xdead_beef);
    let mut pos = Position::default();

    for pair in inputs {
        c.bench_with_input(
            BenchmarkId::new("position::put_piece", pair),
            &pair,
            |b, &pair| b.iter(|| unsafe { pos.put_piece_unsafe(pair.0, pair.1) }),
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
);
criterion_main!(benches);
