use super::*;

#[test]
fn test_to_mate_score_we_force_mate() {
    // Odd plies: We are forcing the mate.
    // 1 ply = we move and it's mate (Mate in 1)
    assert_eq!(Ply { v: 1 }.to_mate_score(), 1);

    // 3 plies = we move, they move, we mate (Mate in 2)
    assert_eq!(Ply { v: 3 }.to_mate_score(), 2);

    // 5 plies = Mate in 3
    assert_eq!(Ply { v: 5 }.to_mate_score(), 3);

    // 7 plies = Mate in 4
    assert_eq!(Ply { v: 7 }.to_mate_score(), 4);
}

#[test]
fn test_to_mate_score_opponent_forces_mate() {
    // Even plies: Opponent is forcing the mate.
    // Note: The method calculates the absolute number of moves.
    // The negative sign for UCI output (e.g., Mate -1) is handled by the caller.

    // 2 plies = they move, we move into mate (Absolute moves = 1)
    assert_eq!(Ply { v: 2 }.to_mate_score(), 1);

    // 4 plies = Absolute moves = 2
    assert_eq!(Ply { v: 4 }.to_mate_score(), 2);

    // 6 plies = Absolute moves = 3
    assert_eq!(Ply { v: 6 }.to_mate_score(), 3);
}

#[test]
fn test_to_mate_score_zero() {
    // Edge case: 0 plies (already checkmated)
    assert_eq!(Ply { v: 0 }.to_mate_score(), 0);
}
