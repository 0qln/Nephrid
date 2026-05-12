use crate::core::{color::colors, coordinates::pawn_utils};

use super::*;

#[test]
fn msb() {
    assert_eq!(Bitboard::empty().msb(), None);
    for lo in Bitboard::full() {
        for hi in Bitboard::split_north(lo) {
            let bb = Bitboard::from(lo) | Bitboard::from(hi);
            assert_eq!(bb.msb(), Some(hi));
        }
    }
}

#[test]
fn lsb() {
    assert_eq!(Bitboard::empty().lsb(), None);
    for lo in Bitboard::full() {
        for hi in Bitboard::split_north(lo) {
            let bb = Bitboard::from(lo) | Bitboard::from(hi);
            assert_eq!(bb.lsb(), Some(lo));
        }
    }
}

#[test]
fn split_north() {
    assert_eq!(
        Bitboard::split_north(A1),
        Bitboard { v: 0xfffffffffffffffeu64 }
    );
    assert_eq!(
        Bitboard::split_north(B1),
        Bitboard { v: 0xfffffffffffffffcu64 }
    );
    assert_eq!(
        Bitboard::split_north(A5),
        Bitboard { v: 0xfffffffe00000000u64 }
    );
    assert_eq!(
        Bitboard::split_north(G8),
        Bitboard { v: 0x8000000000000000u64 }
    );
    assert_eq!(Bitboard::split_north(H8), Bitboard { v: 0 });
}

#[test]
fn split_south() {
    assert_eq!(Bitboard::split_south(A1), Bitboard { v: 0 });
    assert_eq!(Bitboard::split_south(B1), Bitboard { v: 1 });
    assert_eq!(Bitboard::split_south(A5), Bitboard { v: 0xffffffffu64 });
    assert_eq!(
        Bitboard::split_south(G8),
        Bitboard { v: 0x3fffffffffffffffu64 }
    );
    assert_eq!(
        Bitboard::split_south(H8),
        Bitboard { v: 0x7fffffffffffffffu64 }
    );
}

#[test]
fn fill_north_south() {
    /*
    white frontfill     black rearfill
    1 1 1 . . 1 1 1     1 1 1 1 . 1 1 1
    1 1 1 . . 1 1 1     1 1 1 1 . 1 1 1
    1 1 1 . . 1 1 1     1 . 1 1 . . . 1
    1 1 1 . . 1 1 1     . . . 1 . . . .
    1 1 1 . . 1 1 1  ^  . . . . . . . .
    . 1 1 . . . 1 1  |   . . . . . . . .
    . 1 1 . . . 1 1  |   . . . . . . . .
    . . . . . . . .  |   . . . . . . . .
                   north
    white pawns         black pawns
    . . . . . . . .     . . . . . . . .
    . . . . . . . .     . 1 . . . 1 1 .
    . . . . . . . .     1 . 1 . . . . 1
    . . . . . . . .     . . . 1 . . . .
    1 . . . . 1 . .     . . . . . . . .
    . . 1 . . . . .     . . . . . . . .
    . 1 1 . . . 1 1     . . . . . . . .
    . . . . . . . .     . . . . . . . .
                   south
    white rearfill      black frontfill
    . . . . . . . .  |  . . . . . . . .
    . . . . . . . .  |  . 1 . . . 1 1 .
    . . . . . . . .  |  1 1 1 . . 1 1 1
    . . . . . . . .  v  1 1 1 1 . 1 1 1
    1 . . . . 1 . .     1 1 1 1 . 1 1 1
    1 . 1 . . 1 . .     1 1 1 1 . 1 1 1
    1 1 1 . . 1 1 1     1 1 1 1 . 1 1 1
    1 1 1 . . 1 1 1     1 1 1 1 . 1 1 1
    */

    let white_pawns = Bitboard::from(A4)
        | Bitboard::from(B2)
        | Bitboard::from(C2)
        | Bitboard::from(C3)
        | Bitboard::from(F4)
        | Bitboard::from(G2)
        | Bitboard::from(H2);

    let black_pawns = Bitboard::from(A6)
        | Bitboard::from(B7)
        | Bitboard::from(C6)
        | Bitboard::from(D5)
        | Bitboard::from(F7)
        | Bitboard::from(G7)
        | Bitboard::from(H6);

    let white_frontfill = white_pawns.fill::<{ pawn_utils::single_step(colors::WHITE).v() }>();
    let white_rearfill = white_pawns.fill::<{ pawn_utils::back_step(colors::WHITE).v() }>();
    let black_frontfill = black_pawns.fill::<{ pawn_utils::single_step(colors::BLACK).v() }>();
    let black_rearfill = black_pawns.fill::<{ pawn_utils::back_step(colors::BLACK).v() }>();

    for pawn in white_pawns {
        let file = File::from(pawn);
        let frontfill = white_frontfill & Bitboard::from(file);
        let rearfill = white_rearfill & Bitboard::from(file);

        for rank in Rank::from(pawn)..=ranks::_8 {
            assert!(
                (Bitboard::from(rank) & frontfill).pop_cnt() == 1,
                "square {} should be in white frontfill",
                Square::from((file, rank))
            );
        }

        for rank in ranks::_1..=Rank::from(pawn) {
            assert!(
                (Bitboard::from(rank) & rearfill).pop_cnt() == 1,
                "square {} should be in white rearfill",
                Square::from((file, rank))
            );
        }
    }

    for pawn in black_pawns {
        let file = File::from(pawn);
        let frontfill = black_frontfill & Bitboard::from(file);
        let rearfill = black_rearfill & Bitboard::from(file);

        for rank in Rank::from(pawn)..=ranks::_8 {
            assert!(
                (Bitboard::from(rank) & rearfill).pop_cnt() == 1,
                "square {} should be in black rearfill",
                Square::from((file, rank))
            );
        }

        for rank in ranks::_1..=Rank::from(pawn) {
            assert!(
                (Bitboard::from(rank) & frontfill).pop_cnt() == 1,
                "square {} should be in black frontfill",
                Square::from((file, rank))
            );
        }
    }
}

#[test]
fn span_north_south() {
    /*
    white frontspans    black rearspans
    1 1 1 . . 1 1 1     1 1 1 1 . 1 1 1
    1 1 1 . . 1 1 1     1 . 1 1 . . . 1
    1 1 1 . . 1 1 1     . . . 1 . . . .
    1 1 1 . . 1 1 1     . . . . . . . .
    . 1 1 . . . 1 1  ^  . . . . . . . .
    . 1 1 . . . 1 1  |  . . . . . . . .
    . . . . . . . .     . . . . . . . .
    . . . . . . . .     . . . . . . . .
                   north
    white pawns         black pawns
    . . . . . . . .     . . . . . . . .
    . . . . . . . .     . 1 . . . 1 1 .
    . . . . . . . .     1 . 1 . . . . 1
    . . . . . . . .     . . . 1 . . . .
    1 . . . . 1 . .     . . . . . . . .
    . . 1 . . . . .     . . . . . . . .
    . 1 1 . . . 1 1     . . . . . . . .
    . . . . . . . .     . . . . . . . .
                   south
    white rearspans     black frontspans
    . . . . . . . .  |  . . . . . . . .
    . . . . . . . .  v  . . . . . . . .
    . . . . . . . .     . 1 . . . 1 1 .
    . . . . . . . .     1 1 1 . . 1 1 1
    . . . . . . . .     1 1 1 1 . 1 1 1
    1 . . . . 1 . .     1 1 1 1 . 1 1 1
    1 . 1 . . 1 . .     1 1 1 1 . 1 1 1
    1 1 1 . . 1 1 1     1 1 1 1 . 1 1 1
    */

    let white_pawns = Bitboard::from(A4)
        | Bitboard::from(B2)
        | Bitboard::from(C2)
        | Bitboard::from(C3)
        | Bitboard::from(F4)
        | Bitboard::from(G2)
        | Bitboard::from(H2);

    let black_pawns = Bitboard::from(A6)
        | Bitboard::from(B7)
        | Bitboard::from(C6)
        | Bitboard::from(D5)
        | Bitboard::from(F7)
        | Bitboard::from(G7)
        | Bitboard::from(H6);

    let white_frontspan = white_pawns.span::<{ pawn_utils::single_step(colors::WHITE).v() }>();
    let white_rearspan = white_pawns.span::<{ pawn_utils::back_step(colors::WHITE).v() }>();
    let black_frontspan = black_pawns.span::<{ pawn_utils::single_step(colors::BLACK).v() }>();
    let black_rearspan = black_pawns.span::<{ pawn_utils::back_step(colors::BLACK).v() }>();

    for pawn in white_pawns {
        let file = File::from(pawn);
        let frontspan = white_frontspan & Bitboard::from(file);
        let rearspan = white_rearspan & Bitboard::from(file);

        for rank in (Rank::from(pawn) + 1)..=ranks::_8 {
            assert!(
                (Bitboard::from(rank) & frontspan).pop_cnt() == 1,
                "square {} should be in white frontspan",
                Square::from((file, rank))
            );
        }

        for rank in ranks::_1..Rank::from(pawn) {
            assert!(
                (Bitboard::from(rank) & rearspan).pop_cnt() == 1,
                "square {} should be in white rearspan",
                Square::from((file, rank))
            );
        }
    }

    for pawn in black_pawns {
        let file = File::from(pawn);
        let frontspan = black_frontspan & Bitboard::from(file);
        let rearspan = black_rearspan & Bitboard::from(file);

        for rank in (Rank::from(pawn) + 1)..=ranks::_8 {
            assert!(
                (Bitboard::from(rank) & rearspan).pop_cnt() == 1,
                "square {} should be in black rearspan",
                Square::from((file, rank))
            );
        }

        for rank in ranks::_1..Rank::from(pawn) {
            assert!(
                (Bitboard::from(rank) & frontspan).pop_cnt() == 1,
                "square {} should be in black frontspan",
                Square::from((file, rank))
            );
        }
    }
}
