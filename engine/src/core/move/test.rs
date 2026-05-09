use super::*;
use crate::core::{coordinates::squares, search::mcts::nn::PolicyHeadIndex};

#[test]
fn non_promo_moves_unique_indices() {
    let move1 = Move::new(squares::A1, squares::A2, move_flags::QUIET);
    let move2 = Move::new(squares::A1, squares::A3, move_flags::QUIET);
    assert_ne!(
        PolicyHeadIndex::from(move1),
        PolicyHeadIndex::from(move2),
        "Non-promo moves with different to squares must have different indices"
    );
}

#[test]
fn promo_moves_same_file_different_types_unique() {
    let from = squares::B7;
    let to = squares::B8;
    let moves = [
        Move::new(from, to, move_flags::PROMOTION_KNIGHT),
        Move::new(from, to, move_flags::PROMOTION_BISHOP),
        Move::new(from, to, move_flags::PROMOTION_ROOK),
        Move::new(from, to, move_flags::PROMOTION_QUEEN),
    ];
    let indices: Vec<PolicyHeadIndex> = moves.iter().map(|m| PolicyHeadIndex::from(*m)).collect();
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            assert_ne!(
                indices[i], indices[j],
                "Same from-file promotions must have unique indices"
            );
        }
    }
}

#[test]
fn capture_vs_non_capture_promo_unique_indices() {
    let from = squares::C7;
    let non_cap = Move::new(from, squares::C8, move_flags::PROMOTION_QUEEN);
    let cap = Move::new(from, squares::D8, move_flags::CAPTURE_PROMOTION_QUEEN);
    assert_ne!(
        PolicyHeadIndex::from(non_cap),
        PolicyHeadIndex::from(cap),
        "Capture/non-capture promotions must have different indices"
    );
}

#[test]
fn different_from_file_promos_unique() {
    let promo1 = Move::new(squares::A7, squares::A8, move_flags::PROMOTION_QUEEN);
    let promo2 = Move::new(squares::B7, squares::B8, move_flags::PROMOTION_QUEEN);
    assert_ne!(
        PolicyHeadIndex::from(promo1),
        PolicyHeadIndex::from(promo2),
        "Promotions from different files must have unique indices"
    );
}

#[test]
fn promo_and_non_promo_indices_dont_overlap() {
    let non_promo = Move::new(squares::A1, squares::A2, move_flags::QUIET);
    let promo = Move::new(squares::A7, squares::A8, move_flags::PROMOTION_QUEEN);
    assert!(
        PolicyHeadIndex::from(non_promo).v() < Move::MASK_SQ,
        "Non-promo index should be below 4096"
    );
    assert!(
        PolicyHeadIndex::from(promo).v() > Move::MASK_SQ,
        "Promo index should be above 4096"
    );
}

#[test]
fn all_promo_indices_within_expected_range() {
    let from = squares::H7; // Max file (7)
    let max_promo = Move::new(from, squares::H8, move_flags::CAPTURE_PROMOTION_QUEEN);
    let idx: PolicyHeadIndex = max_promo.into();
    assert!(
        idx.v() <= 4096 + 7 + 8,
        "Promo indices must not exceed 4111"
    );
}

#[test]
fn promo_indices_from_same_file_are_consecutive_without_gaps() {
    let from = squares::A7;
    let to_non_cap = squares::A8;
    let to_cap = squares::B8;

    // Generate all 8 possible promotion moves from this file
    let moves = [
        // Non-capture promotions
        Move::new(from, to_non_cap, move_flags::PROMOTION_KNIGHT),
        Move::new(from, to_non_cap, move_flags::PROMOTION_BISHOP),
        Move::new(from, to_non_cap, move_flags::PROMOTION_ROOK),
        Move::new(from, to_non_cap, move_flags::PROMOTION_QUEEN),
        // Capture promotions
        Move::new(from, to_cap, move_flags::CAPTURE_PROMOTION_KNIGHT),
        Move::new(from, to_cap, move_flags::CAPTURE_PROMOTION_BISHOP),
        Move::new(from, to_cap, move_flags::CAPTURE_PROMOTION_ROOK),
        Move::new(from, to_cap, move_flags::CAPTURE_PROMOTION_QUEEN),
    ];

    // Convert to indices and sort
    let mut indices: Vec<PolicyHeadIndex> = moves.iter().map(|m| PolicyHeadIndex::from(*m)).collect();
    indices.sort();

    // Verify there are exactly 8 unique consecutive indices
    assert_eq!(indices.len(), 8, "Should have 8 promotion indices");
    for i in 1..indices.len() {
        assert_eq!(
            indices[i].v(),
            indices[i - 1].v() + 1,
            "Gap detected between promotion indices {:?} and {:?}",
            indices[i - 1],
            indices[i]
        );
    }

    // Verify they occupy the exact expected range for this file
    assert_eq!(indices[0].v(), 4096, "First promo index should be 4096");
    assert_eq!(indices[7].v(), 4103, "Last promo index should be 4103");
}

#[test]
fn non_promo_indices_use_contiguous_low_range() {
    // Test first and last possible non-promo indices
    let min_move = Move::new(squares::A1, squares::A1, move_flags::QUIET);
    let max_move = Move::new(squares::H8, squares::H8, move_flags::CAPTURE);

    assert_eq!(
        PolicyHeadIndex::from(min_move),
        PolicyHeadIndex::new(0),
        "Lowest non-promo index should be 0"
    );
    assert_eq!(
        PolicyHeadIndex::from(max_move),
        PolicyHeadIndex::new(0x0FFF), // 4095
        "Highest non-promo index should be 4095"
    );
}
