use std::{iter::once, mem, sync::Once};

use itertools::Itertools;
use rand::{rngs::SmallRng, RngCore, SeedableRng};

use crate::engine::{
    bitboard::Bitboard,
    coordinates::Square,
    move_iter::{bishop::Bishop, map_bits, rook::Rook},
};

use super::Attacks;

#[cfg(test)]
mod test;

#[const_trait]
pub trait MagicGen {
    fn relevant_occupancy(sq: Square) -> Bitboard;
    fn relevant_occupancy_num_combinations() -> usize;
}

#[derive(Debug, Clone, Copy)]
pub struct Magic<'a> {
    ptr: &'a [Bitboard],
    mask: Bitboard,
    magic: u64,
    shift: u8,
}

impl Magic<'_> {
    pub fn new(ptr: &[Bitboard], mask: Bitboard, magic: u64, shift: u8) -> Magic<'_> {
        Magic {
            ptr,
            mask,
            magic,
            shift,
        }
    }

    #[inline(always)]
    pub fn key(&self, occupancy: Bitboard) -> usize {
        let relevant_occ = occupancy & self.mask;
        let key = relevant_occ.v.wrapping_mul(self.magic) >> self.shift;
        key as usize
    }

    #[inline(always)]
    pub fn get(&self, occupancy: Bitboard) -> Bitboard {
        unsafe { *self.ptr.get_unchecked(self.key(occupancy)) }
    }
}

#[derive(Debug, Default, Clone)]
pub struct MagicInfo {
    ptr_off: usize,
    ptr_size: usize,
    mask: Bitboard,
    magic: u64,
    shift: u8,
}

impl MagicInfo {
    pub fn init<'a>(self, table: &'a AttackTable) -> Magic<'a> {
        let ptr = &table.0[self.ptr_off..self.ptr_off + self.ptr_size];
        Magic::new(ptr, self.mask, self.magic, self.shift)
    }

    pub fn key(&self, occupancy: Bitboard) -> usize {
        let relevant_occ = occupancy & self.mask;
        let key = relevant_occ.v.wrapping_mul(self.magic) >> self.shift;
        key as usize
    }
}

pub struct MagicTable<'a>([Magic<'a>; 64]);

impl<'a> MagicTable<'a> {
    #[inline(always)]
    pub fn get<'b>(&'b self, sq: Square) -> &'b Magic<'a> {
        let index = sq.v() as usize;
        // Safety: A square is in range 0..64
        unsafe { self.0.get_unchecked(index) }
    }

    /// Safety:
    /// The caller has to assert, that the table is initialized
    /// in a single threaded context. Idealy, at the start of the program.
    pub unsafe fn init<I: IntoIterator<Item = MagicInfo>>(
        &mut self,
        table: &'a AttackTable,
        magics: I,
    ) {
        for (sq, m) in magics.into_iter().enumerate() {
            self.0[sq] = m.init(table);
        }
    }
}

pub struct AttackTable([Bitboard; 0x1A480]);

pub static mut ATTACK_TABLE: AttackTable = unsafe { mem::zeroed() };

pub static mut ROOK_MAGICS: MagicTable = MagicTable(
    [Magic {
        ptr: &[],
        mask: Bitboard::empty(),
        magic: 0,
        shift: 0,
    }; 64],
);

pub static mut BISHOP_MAGICS: MagicTable = MagicTable(
    [Magic {
        ptr: &[],
        mask: Bitboard::empty(),
        magic: 0,
        shift: 0,
    }; 64],
);

static INIT: Once = Once::new();

#[allow(static_mut_refs)]
pub fn init(seed: u64) {
    INIT.call_once(|| unsafe {
        let mut rng = SmallRng::seed_from_u64(seed);
        let table = &mut ATTACK_TABLE;
        let rook_magics = find_magics::<Rook>(table, &mut rng, None);
        let bishop_magics = find_magics::<Bishop>(table, &mut rng, Some(&rook_magics[63]));
        ROOK_MAGICS.init(table, rook_magics);
        BISHOP_MAGICS.init(table, bishop_magics);
    });
}

fn find_magics<'a, T: MagicGen + Attacks>(
    table: &'a mut AttackTable,
    rng: &mut SmallRng,
    prev: Option<&MagicInfo>,
) -> [MagicInfo; 64] {
    let mut result: [MagicInfo; 64] = unsafe { mem::zeroed() };
    for sq in Square::A1..=Square::H8 {
        let idx = sq.v() as usize;
        result[idx] = find_magic::<T>(
            table,
            sq,
            rng,
            idx.checked_sub(1).map_or(prev, |x| Some(&result[x])),
        );
    }
    result
}

fn find_magic<T: MagicGen + Attacks>(
    table: &mut AttackTable,
    sq: Square,
    rng: &mut SmallRng,
    prev: Option<&MagicInfo>,
) -> MagicInfo {
    let max_size = T::relevant_occupancy_num_combinations();
    let full_blockers = T::relevant_occupancy(sq);

    let blockers = (0..max_size)
        .map(|x| map_bits(x, full_blockers))
        .take_while(|&x| x != full_blockers)
        .chain(once(full_blockers))
        .collect_vec();

    let attacks = blockers
        .iter()
        .map(|&x| T::compute_attacks(sq, x))
        .collect_vec();

    let mask = T::relevant_occupancy(sq);
    let shift = 64 - mask.pop_cnt() as u8;
    let idx = prev.map_or(0, |x: &MagicInfo| x.ptr_off + x.ptr_size);
    let mut m = MagicInfo {
        mask,
        magic: 0,
        shift,
        ptr_off: idx,
        ptr_size: attacks.len(),
    };

    let mut verify_magic = |magic: u64, ptr: &mut [Bitboard]| {
        m.magic = magic;

        // check each blocker composition
        for (&occ, &attack) in blockers.iter().zip(attacks.iter()) {
            let key = m.key(occ);

            // that won't work :/
            if key >= ptr.len() {
                return false;
            }

            // free spot?
            if ptr[key] == Bitboard::empty() {
                // use the spot.
                ptr[key] = attack;
                continue;
            }

            // contructive collision?
            if ptr[key] == attack {
                // thats fine, continue.
                continue;
            }

            // bad magic, verification fails.
            return false;
        }

        return true;
    };

    let ptr = &mut table.0[idx..idx + attacks.len()];
    while !verify_magic(rn(rng), ptr) {
        ptr.fill(Bitboard::empty());
    }

    m
}

fn rn(rng: &mut SmallRng) -> u64 {
    rng.next_u64() & rng.next_u64() & rng.next_u64()
}
