use core::fmt;
use std::{fmt::Display, ops::ControlFlow};

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use engine::{
    core::{
        color::Color,
        coordinates::Square,
        move_iter::{fold_legal_moves, sliding_piece::magics},
        piece::{Piece, PieceType},
        position::Position,
        zobrist,
    },
    misc::ConstFrom,
    uci::tokens::Tokenizer,
};
use itertools::Itertools;

pub fn get_bitboard(c: &mut Criterion) {
    let inputs = [
        Pair(piece_type::PAWN, colors::WHITE),
        Pair(piece_type::QUEEN, colors::WHITE),
        Pair(piece_type::PAWN, colors::BLACK),
        Pair(piece_type::QUEEN, colors::BLACK),
    ];

    magics::init();
    zobrist::init();
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
        Pair(Square::A3, Piece::from_c((colors::WHITE, piece_type::PAWN))),
        Pair(
            Square::B4,
            Piece::from_c((colors::WHITE, piece_type::QUEEN)),
        ),
        Pair(Square::E4, Piece::from_c((colors::BLACK, piece_type::PAWN))),
        Pair(
            Square::H5,
            Piece::from_c((colors::BLACK, piece_type::QUEEN)),
        ),
    ];

    magics::init();
    zobrist::init();
    let pos = Position::default();

    for pair in inputs {
        c.bench_with_input(
            BenchmarkId::new("position::put_piece", pair),
            &pair,
            |b, &pair| {
                b.iter_batched(
                    || pos.clone(),
                    |mut pos| unsafe { pos.put_piece_unsafe(pair.0, pair.1) },
                    BatchSize::PerIteration,
                )
            },
        );
    }
}

pub fn remove_piece(c: &mut Criterion) {
    let inputs = [Square::A1, Square::E2, Square::H8];

    magics::init();
    zobrist::init();
    let pos = Position::start_position();

    for sq in inputs {
        c.bench_with_input(
            BenchmarkId::new("position::remove_piece", sq),
            &sq,
            |b, &sq| {
                b.iter_batched(
                    || pos.clone(),
                    |mut pos| unsafe { pos.remove_piece_unsafe(sq) },
                    BatchSize::PerIteration,
                )
            },
        );
    }
}

pub fn move_piece(c: &mut Criterion) {
    let inputs = [
        Pair(Square::A1, Square::A3),
        Pair(Square::E2, Square::E4),
        Pair(Square::H8, Square::H5),
    ];

    magics::init();
    zobrist::init();
    let pos = Position::start_position();

    for pair in inputs {
        c.bench_with_input(
            BenchmarkId::new("position::move_piece", pair),
            &pair,
            |b, &pair| {
                b.iter_batched(
                    || pos.clone(),
                    |mut pos| unsafe { pos.move_piece_unsafe(pair.0, pair.1) },
                    BatchSize::PerIteration,
                )
            },
        );
    }
}

pub fn make_move(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let mut fen = Tokenizer::new("2n1k3/1P6/8/4pP2/8/6B1/P2P4/R3K2R w KQ e6 0 1");
    let pos = Position::try_from(&mut fen).unwrap();
    let moves = fold_legal_moves(&pos, Vec::new(), |mut acc, m| {
        acc.push(m);
        ControlFlow::Continue::<(), _>(acc)
    })
    .continue_value()
    .unwrap();

    for m in moves.into_iter().unique_by(|m| m.get_flag()) {
        c.bench_with_input(BenchmarkId::new("position::make_move", m), &m, |b, &m| {
            b.iter_batched(
                || pos.clone(),
                |mut pos| pos.make_move(m),
                BatchSize::PerIteration,
            )
        });
    }
}

#[derive(Debug, Clone, Copy)]
struct Pair<T1, T2>(T1, T2);

impl<T1: Display, T2: Display> fmt::Display for Pair<T1, T2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

criterion_group!(
    benches,
    get_bitboard,
    put_piece,
    remove_piece,
    move_piece,
    make_move,
);

criterion_main!(benches);
