use crate::{
    core::{
        color::colors,
        piece::{Piece, piece_type},
    },
    misc::ConstFrom,
};

#[test]
fn parse_piece() {
    assert_eq!(
        Ok(Piece::from_c((colors::WHITE, piece_type::PAWN))),
        Piece::try_from('P')
    );
    assert_eq!(
        Ok(Piece::from_c((colors::WHITE, piece_type::KING))),
        Piece::try_from('K')
    );
    assert_eq!(
        Ok(Piece::from_c((colors::BLACK, piece_type::PAWN))),
        Piece::try_from('p')
    );
    assert_eq!(
        Ok(Piece::from_c((colors::BLACK, piece_type::KING))),
        Piece::try_from('k')
    );
}
