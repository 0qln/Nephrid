use super::*;
use crate::core::coordinates::Square;

#[test]
fn non_promo_moves_unique_indices() {
    let move1 = Move::new(Square::A1, Square::A2, MoveFlag::QUIET);
    let move2 = Move::new(Square::A1, Square::A3, MoveFlag::QUIET);
    assert_ne!(
        usize::from(move1),
        usize::from(move2),
        "Non-promo moves with different to squares must have different indices"
    );
}

#[test]
fn promo_moves_same_file_different_types_unique() {
    let from = Square::B7;
    let to = Square::B8;
    let moves = [
        Move::new(from, to, MoveFlag::PROMOTION_KNIGHT),
        Move::new(from, to, MoveFlag::PROMOTION_BISHOP),
        Move::new(from, to, MoveFlag::PROMOTION_ROOK),
        Move::new(from, to, MoveFlag::PROMOTION_QUEEN),
    ];
    let indices: Vec<usize> = moves.iter().map(|m| usize::from(*m)).collect();
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
    let from = Square::C7;
    let non_cap = Move::new(from, Square::C8, MoveFlag::PROMOTION_QUEEN);
    let cap = Move::new(from, Square::D8, MoveFlag::CAPTURE_PROMOTION_QUEEN);
    assert_ne!(
        usize::from(non_cap),
        usize::from(cap),
        "Capture/non-capture promotions must have different indices"
    );
}

#[test]
fn different_from_file_promos_unique() {
    let promo1 = Move::new(Square::A7, Square::A8, MoveFlag::PROMOTION_QUEEN);
    let promo2 = Move::new(Square::B7, Square::B8, MoveFlag::PROMOTION_QUEEN);
    assert_ne!(
        usize::from(promo1),
        usize::from(promo2),
        "Promotions from different files must have unique indices"
    );
}

#[test]
fn promo_and_non_promo_indices_dont_overlap() {
    let non_promo = Move::new(Square::A1, Square::A2, MoveFlag::QUIET);
    let promo = Move::new(Square::A7, Square::A8, MoveFlag::PROMOTION_QUEEN);
    assert!(
        usize::from(non_promo) < Move::MASK_SQ as usize,
        "Non-promo index should be below 4096"
    );
    assert!(
        usize::from(promo) > Move::MASK_SQ as usize,
        "Promo index should be above 4096"
    );
}

#[test]
fn all_promo_indices_within_expected_range() {
    let from = Square::H7; // Max file (7)
    let max_promo = Move::new(from, Square::H8, MoveFlag::CAPTURE_PROMOTION_QUEEN);
    let idx: usize = max_promo.into();
    assert!(idx <= 4096 + 7 + 8, "Promo indices must not exceed 4111");
}

#[test]
fn promo_indices_from_same_file_are_consecutive_without_gaps() {
    let from = Square::A7;
    let to_non_cap = Square::A8;
    let to_cap = Square::B8;
    
    // Generate all 8 possible promotion moves from this file
    let moves = [
        // Non-capture promotions
        Move::new(from, to_non_cap, MoveFlag::PROMOTION_KNIGHT),
        Move::new(from, to_non_cap, MoveFlag::PROMOTION_BISHOP),
        Move::new(from, to_non_cap, MoveFlag::PROMOTION_ROOK),
        Move::new(from, to_non_cap, MoveFlag::PROMOTION_QUEEN),
        // Capture promotions
        Move::new(from, to_cap, MoveFlag::CAPTURE_PROMOTION_KNIGHT),
        Move::new(from, to_cap, MoveFlag::CAPTURE_PROMOTION_BISHOP),
        Move::new(from, to_cap, MoveFlag::CAPTURE_PROMOTION_ROOK),
        Move::new(from, to_cap, MoveFlag::CAPTURE_PROMOTION_QUEEN),
    ];

    // Convert to indices and sort
    let mut indices: Vec<usize> = moves.iter().map(|m| usize::from(*m)).collect();
    indices.sort();

    // Verify there are exactly 8 unique consecutive indices
    assert_eq!(indices.len(), 8, "Should have 8 promotion indices");
    for i in 1..indices.len() {
        assert_eq!(
            indices[i],
            indices[i-1] + 1,
            "Gap detected between promotion indices {} and {}",
            indices[i-1],
            indices[i]
        );
    }

    // Verify they occupy the exact expected range for this file
    assert_eq!(indices[0], 4096, "First promo index should be 4096");
    assert_eq!(indices[7], 4103, "Last promo index should be 4103");
}

#[test]
fn non_promo_indices_use_contiguous_low_range() {
    // Test first and last possible non-promo indices
    let min_move = Move::new(Square::A1, Square::A1, MoveFlag::QUIET);
    let max_move = Move::new(Square::H8, Square::H8, MoveFlag::CAPTURE);
    
    assert_eq!(
        usize::from(min_move), 0,
        "Lowest non-promo index should be 0"
    );
    assert_eq!(
        usize::from(max_move), 0x0FFF, // 4095
        "Highest non-promo index should be 4095"
    );
}