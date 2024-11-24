use crate::{
    engine::{
        bitboard::Bitboard, castling::{CastlingRights, CastlingSide}, color::Color, coordinates::{File, Rank, Square}, r#move::{Move, MoveFlag}, position::Position
    },
    misc::{ConstFrom, PostIncrement},
};

pub fn gen_pseudo_legal_castling(pos: &Position, color: Color) -> impl Iterator<Item = Move> {
    let rank = match color {
        Color::WHITE => Rank::_1,
        Color::BLACK => Rank::_8,
        _ => unreachable!(),
    };
    let from = Square::from_c((File::E, rank));
    let castling = pos.get_castling().clone();
    return Gen {
        state: 0,
        castling,
        from,
        rank,
        color,
    };

    // probably has a simpler solution, but this is just temporary.
    struct Gen {
        state: u8,
        castling: CastlingRights,
        from: Square,
        rank: Rank,
        color: Color,
    }
    impl Iterator for Gen {
        type Item = Move;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            match self.state.post_incr(1) {
                0 => {
                    match self.castling.is_true(CastlingSide::KING_SIDE, self.color) {
                        true => {
                            let to = Square::from_c((File::G, self.rank));
                            Some(Move::new(self.from, to, MoveFlag::KING_CASTLE))
                        }
                        false => self.next(),
                    }
                }
                1 => {
                    match self.castling.is_true(CastlingSide::QUEEN_SIDE, self.color) {
                        true => {
                            let to = Square::from_c((File::C, self.rank));
                            Some(Move::new(self.from, to, MoveFlag::QUEEN_CASTLE))
                        }
                        false => self.next(),
                    }
                }
                _ => None,
            }
        }
    }

    // todo: uncomment when gen_blocks feature works again.
    //
    // gen {
    //     if castling.is_true(CastlingSide::KING_SIDE, color) {
    //         let to = Square::from_c((File::G, rank));
    //         yield Move::new(from, to, MoveFlag::KING_CASTLE);
    //     }
    //     if castling.is_true(CastlingSide::QUEEN_SIDE, color) {
    //         let to = Square::from_c((File::C, rank));
    //         yield Move::new(from, to, MoveFlag::QUEEN_CASTLE);
    //     }
    // }.into_iter()
}

// todo: can be computed at compile time
//
pub fn compute_attacks(sq: Square) -> Bitboard {
    let file = File::from_c(sq);
    let rank = Rank::from_c(sq);

    let mut result = Bitboard::empty();

    if file > File::A {
        // Safety: file is in range 1.., so file - 1 is still a valid file.
        let west = unsafe { File::from_v(file.v() - 1) };
        result |= Bitboard::from_c(west);
    }

    if file < File::H {
        // Safety: file is in range 0..7, so file + 1 is still a valid file.
        let east = unsafe { File::from_v(file.v() + 1) };
        result |= Bitboard::from_c(east);
    }

    if rank > Rank::_1 {
        // Safety: rank is in range 1.., so rank - 1 is still a valid rank.
        let south = unsafe { Rank::from_v(rank.v() - 1) };
        result |= Bitboard::from_c(south);
    }

    if rank < Rank::_8 {
        // Safety: rank is in range 0..7, so rank + 1 is still a valid rank.
        let north = unsafe { Rank::from_v(rank.v() + 1) };
        result |= Bitboard::from_c(north);
    }

    result
}
