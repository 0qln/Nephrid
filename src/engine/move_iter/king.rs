use crate::{
    engine::{
        bitboard::Bitboard,
        castling::{CastlingRights, CastlingSide},
        color::Color,
        coordinates::{File, Rank, Square},
        piece::PieceType,
        position::Position,
        r#move::{Move, MoveFlag},
    },
    misc::{ConstFrom, PostIncrement},
};

use const_for::const_for;

use super::{bishop::{self, Bishop}, queen::{self, Queen}, rook::{self, Rook}, sliding_piece::Attacks};

pub fn gen_legals_check_some(pos: &Position) -> impl Iterator<Item = Move> {
    let color = pos.get_turn();
    let moves = gen_legals_check_none(pos);
    let nstm_attacks = pos.get_nstm_attacks();
    let king_bb = pos.get_bitboard(PieceType::KING, color);
    let occupancy = pos.get_occupancy();
    let checkers = pos.get_checkers();

    // Make sure that we move the king out of check
    let checker_rooks = checkers & pos.get_bitboard(PieceType::ROOK, !color);
    let checker_bishops = checkers & pos.get_bitboard(PieceType::BISHOP, !color);
    // todo: queen can be merged into rook and bishop?
    let checker_queens = checkers & pos.get_bitboard(PieceType::QUEEN, !color);
    moves.filter(move |m| {
        let to = Bitboard::from_c(m.get_to());

        // When the king has moved and a sliding piece was a checker, the attacks of
        // that sliding piece will have changed.
        // Note that, in no case does a king move cause an enemy attack to get covered
        // without the king being in check after he moved, which is why we can just
        // append the new attacks of the sliding piece to the existing attacks.
        // The new 'nstm_attacks' are not really nstm_attacks, but only reflect nstm_attacks
        // which are relevant to checking whether the our king is in check.
        let occupancy_after_king_move = (occupancy ^ king_bb) | to;
        let nstm_attacks = checker_rooks.fold(nstm_attacks, |acc, checker| {
            acc | Rook::compute_attacks(checker, occupancy_after_king_move)
        });
        let nstm_attacks = checker_bishops.fold(nstm_attacks, |acc, checker| {
            acc | Bishop::compute_attacks(checker, occupancy_after_king_move)
        });
        let nstm_attacks = checker_queens.fold(nstm_attacks, |acc, checker| {
            acc | Queen::compute_attacks(checker, occupancy_after_king_move)
        });

        (to & nstm_attacks).is_empty()
    })
}

pub fn gen_legals_check_none(pos: &Position) -> impl Iterator<Item = Move> {
    let color = pos.get_turn();
    let nstm_attacks = pos.get_nstm_attacks();
    let occupancy = pos.get_occupancy();
    let enemies = pos.get_color_bb(!color);
    let king_bb = pos.get_bitboard(PieceType::KING, color);
    let king = king_bb.lsb().unwrap();
    let attacks = lookup_attacks(king);
    let targets = attacks & !nstm_attacks;

    let quiets = {
        let targets = targets & !occupancy;
        targets.map(move |target| Move::new(king, target, MoveFlag::QUIET))
    };

    let captures = {
        let targets = targets & enemies;
        targets.map(move |target| Move::new(king, target, MoveFlag::CAPTURE))
    };

    captures.chain(quiets)
}

pub fn gen_legal_castling(pos: &Position, color: Color) -> impl Iterator<Item = Move> + '_ {
    let rank = match color {
        Color::WHITE => Rank::_1,
        Color::BLACK => Rank::_8,
        _ => unreachable!(),
    };
    let from = Square::from_c((File::E, rank));
    return Gen {
        state: 0,
        pos,
        from,
        rank,
        color,
    };

    // probably has a simpler solution, but this is just temporary.
    struct Gen<'a> {
        state: u8,
        pos: &'a Position,
        from: Square,
        rank: Rank,
        color: Color,
    }
    impl Iterator for Gen<'_> {
        type Item = Move;

        fn next(&mut self) -> Option<Self::Item> {
            let castling = self.pos.get_castling();
            match self.state.post_incr(1) {
                0 => match castling.is_true(CastlingSide::KING_SIDE, self.color) {
                    false => self.next(),
                    true => {
                        const TABU_MASK: [Bitboard; 2] = [
                            Bitboard { v: 0x60_u64 },
                            Bitboard {
                                v: 0x6000000000000000_u64,
                            },
                        ];
                        let nstm_attacks = self.pos.get_nstm_attacks();
                        let tabus = nstm_attacks | self.pos.get_occupancy();
                        if !(tabus & TABU_MASK[self.color.v() as usize]).is_empty() {
                            return self.next();
                        }
                        let to = Square::from_c((File::G, self.rank));
                        Some(Move::new(self.from, to, MoveFlag::KING_CASTLE))
                    }
                },
                1 => match castling.is_true(CastlingSide::QUEEN_SIDE, self.color) {
                    false => self.next(),
                    true => {
                        const BLOCK_MASK: [Bitboard; 2] = [
                            Bitboard { v: 0xE_u64 },
                            Bitboard {
                                v: 0xE00000000000000_u64,
                            },
                        ];
                        const CHECK_MASK: [Bitboard; 2] = [
                            Bitboard { v: 0xC_u64 },
                            Bitboard {
                                v: 0xC00000000000000_u64,
                            },
                        ];
                        let to = Square::from_c((File::C, self.rank));
                        let nstm_attacks = self.pos.get_nstm_attacks();
                        let blockers = self.pos.get_occupancy();
                        let blocked = BLOCK_MASK[self.color.v() as usize] & blockers;
                        let checked = CHECK_MASK[self.color.v() as usize] & nstm_attacks;
                        if !(blocked | checked).is_empty() {
                            return self.next();
                        }
                        Some(Move::new(self.from, to, MoveFlag::QUEEN_CASTLE))
                    }
                },
                _ => None,
            }
        }
    }
}

pub fn gen_pseudo_legal_castling(pos: &Position, color: Color) -> impl Iterator<Item = Move> {
    let rank = match color {
        Color::WHITE => Rank::_1,
        Color::BLACK => Rank::_8,
        _ => unreachable!(),
    };
    let from = Square::from_c((File::E, rank));
    let castling = pos.get_castling();
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
                0 => match self.castling.is_true(CastlingSide::KING_SIDE, self.color) {
                    true => {
                        let to = Square::from_c((File::G, self.rank));
                        Some(Move::new(self.from, to, MoveFlag::KING_CASTLE))
                    }
                    false => self.next(),
                },
                1 => match self.castling.is_true(CastlingSide::QUEEN_SIDE, self.color) {
                    true => {
                        let to = Square::from_c((File::C, self.rank));
                        Some(Move::new(self.from, to, MoveFlag::QUEEN_CASTLE))
                    }
                    false => self.next(),
                },
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

pub fn lookup_attacks(sq: Square) -> Bitboard {
    static ATTACKS: [Bitboard; 64] = {
        let mut attacks = [Bitboard::empty(); 64];
        const_for!(sq in Square::A1_C..(Square::H8_C+1) => {
            // Safety: we are only iterating over valid squares.
            let sq = unsafe { Square::from_v(sq) };
            attacks[sq.v() as usize] = compute_attacks(sq);
        });
        attacks
    };
    // Safety: sq is in range 0..64
    unsafe {
        *ATTACKS.get_unchecked(sq.v() as usize)
    }
}

pub const fn compute_attacks(sq: Square) -> Bitboard {
    let file = File::from_c(sq);
    let rank = Rank::from_c(sq);
    let king = Bitboard::from_c(sq);

    let mut files = Bitboard::from_c(file);
    if file.v() > File::A_C {
        // Safety: file is in range 1.., so file - 1 is still a valid file.
        let west = unsafe { File::from_v(file.v() - 1) };
        files.v |= Bitboard::from_c(west).v;
    }

    if file.v() < File::H_C {
        // Safety: file is in range 0..7, so file + 1 is still a valid file.
        let east = unsafe { File::from_v(file.v() + 1) };
        files.v |= Bitboard::from_c(east).v;
    }

    let mut ranks = Bitboard::from_c(rank);
    if rank.v() > Rank::_1_C {
        // Safety: rank is in range 1.., so rank - 1 is still a valid rank.
        let south = unsafe { Rank::from_v(rank.v() - 1) };
        ranks.v |= Bitboard::from_c(south).v;
    }

    if rank.v() < Rank::_8_C {
        // Safety: rank is in range 0..7, so rank + 1 is still a valid rank.
        let north = unsafe { Rank::from_v(rank.v() + 1) };
        ranks.v |= Bitboard::from_c(north).v;
    }

    files.and_c(ranks).and_not_c(king)
}
