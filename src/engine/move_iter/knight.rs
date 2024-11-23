use crate::{
    engine::{
        bitboard::Bitboard,
        color::{Color, TColor},
        coordinates::{CompassRose, Square, TCompassRose},
        piece::PieceType,
        position::Position,
        r#move::{Move, MoveFlag},
    },
    misc::ConstFrom,
};

#[derive(Clone, Copy)]
struct PseudoLegalKnightMovesInfo {
    enemies: Bitboard,
    allies: Bitboard,
}

impl PseudoLegalKnightMovesInfo {
    #[inline]
    pub fn new(pos: &Position, color: Color) -> Self {
        Self {
            enemies: pos.get_color_bb(!color),
            allies: pos.get_color_bb(color),
        }
    }
}

// todo: flag as const generic?

struct PseudoLegalKnightMoves {
    from: Square,
    to: Bitboard,
    flag: MoveFlag,
}

impl PseudoLegalKnightMoves {
    #[inline]
    fn new_quiets(info: PseudoLegalKnightMovesInfo, knight: Square, attacks: Bitboard) -> Self {
        Self {
            from: knight,
            to: attacks & !info.allies & !info.enemies,
            flag: MoveFlag::QUIET,
        }
    }

    #[inline]
    fn new_captures(info: PseudoLegalKnightMovesInfo, knight: Square, attacks: Bitboard) -> Self {
        Self {
            from: knight,
            to: attacks & info.enemies,
            flag: MoveFlag::CAPTURE,
        }
    }
}

impl Iterator for PseudoLegalKnightMoves {
    type Item = Move;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let to = self.to.pop_lsb();
        to.map(|sq| Move::new(self.from, sq, self.flag))
    }
}

pub fn gen_pseudo_legals<const C: TColor>(position: &Position) -> impl Iterator<Item = Move> {
    Color::assert_variant(C); // Safety
    let color = unsafe { Color::from_v(C) };
    let info = PseudoLegalKnightMovesInfo::new(position, color);
    position
        .get_bitboard(PieceType::KNIGHT, color)
        .map(move |knight| {
            let attacks = compute_attacks(knight);
            let moves: [fn(PseudoLegalKnightMovesInfo, Square, Bitboard) -> PseudoLegalKnightMoves; 2] = [
                PseudoLegalKnightMoves::new_quiets,
                PseudoLegalKnightMoves::new_captures,
            ];
            moves
                .into_iter()
                .map(move |f| f(info, knight, attacks))
                .flatten()
        })
        .flatten()
}

// todo: the attacks can be precomputed.

#[inline]
const fn compute_attacks(sq: Square) -> Bitboard {
    let knight = Bitboard::from_c(sq);
    compute_attacks_multiple(knight)
}

const fn compute_attacks_multiple(knights: Bitboard) -> Bitboard {
    let mut result = Bitboard::empty();
    compute_atttack::<{ CompassRose::NONOWE_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::NONOEA_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::NOWEWE_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::NOEAEA_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::SOSOWE_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::SOSOEA_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::SOWEWE_C }>(knights, &mut result);
    compute_atttack::<{ CompassRose::SOEAEA_C }>(knights, &mut result);
    return result;
}

#[inline]
const fn compute_atttack<const DIR: TCompassRose>(knight: Bitboard, attacks: &mut Bitboard) {
    let attack_sqrs = Bitboard::full().shift(CompassRose::new(-DIR));
    attacks.v |= knight.and_c(attack_sqrs).shift(CompassRose::new(DIR)).v;
}
