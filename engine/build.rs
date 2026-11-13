//! Build-time generation of the magic-bitboard tables.
//!
//! This is a self-contained copy of the attack-generation + magic-search logic
//! that used to run at first use in `sliding_piece::magics::init()`. A build
//! script cannot depend on the crate it is building, so the minimal bitboard
//! arithmetic is reproduced here (as plain `u64`s). The generated `static`s are
//! `include!`d by `sliding_piece::magics`, which keeps the hot-path `Magic::get`
//! unchanged and removes the runtime `Once`/`static mut` initialization.
//!
//! The correctness oracle is `sliding_piece::magics::test`, which compares
//! `lookup_attacks` (these generated tables) against the engine's independent
//! `compute_attacks` for every square and blocker subset.

use std::{env, fmt::Write, fs, path::PathBuf};

use rand::{RngCore, SeedableRng, rngs::SmallRng};

type BB = u64;

const fn bit(sq: usize) -> BB {
    1u64 << sq
}

const EDGES: BB = 0xFF00_0000_0000_00FF | 0x0101_0101_0101_0101 | 0x8080_8080_8080_8080;

// ---- attack generation ----

fn in_board(f: i32, r: i32) -> bool {
    (0..8).contains(&f) && (0..8).contains(&r)
}

/// Squares along one ray from `sq` (exclusive) in direction `(df, dr)`, up to
/// and including the first blocker, stopping at the board edge.
fn ray(sq: usize, occ: BB, df: i32, dr: i32) -> BB {
    let (mut f, mut r) = ((sq % 8) as i32 + df, (sq / 8) as i32 + dr);
    let mut out = 0u64;
    while in_board(f, r) {
        let t = (r * 8 + f) as usize;
        out |= bit(t);
        if occ & bit(t) != 0 {
            break;
        }
        f += df;
        r += dr;
    }
    out
}

fn rook_attacks(sq: usize, occ: BB) -> BB {
    ray(sq, occ, 1, 0) | ray(sq, occ, -1, 0) | ray(sq, occ, 0, 1) | ray(sq, occ, 0, -1)
}

fn bishop_attacks(sq: usize, occ: BB) -> BB {
    ray(sq, occ, 1, 1) | ray(sq, occ, 1, -1) | ray(sq, occ, -1, 1) | ray(sq, occ, -1, -1)
}

// ---- relevant-occupancy masks (mirror `Rook`/`Bishop::relevant_occupancy`) ----

fn relevant_rook(sq: usize) -> BB {
    let file_bb = 0x0101_0101_0101_0101u64 << (sq % 8);
    let rank_bb = 0xFFu64 << (8 * (sq / 8));
    let rel_file = file_bb & !0xFF & !(0xFF << 56);
    let rel_rank = rank_bb & !0x0101_0101_0101_0101 & !0x8080_8080_8080_8080;
    !bit(sq) & (rel_file | rel_rank)
}

fn relevant_bishop(sq: usize) -> BB {
    bishop_attacks(sq, 0) & !EDGES
}

// ---- magic search (mirror `find_magics` / `find_magic`) ----

struct MagicOut {
    mask: BB,
    magic: u64,
    shift: u8,
    ptr_off: usize,
    ptr_size: usize,
}

fn rn(rng: &mut SmallRng) -> u64 {
    rng.next_u64() & rng.next_u64() & rng.next_u64()
}

/// Map the low bits of `bits` onto the set bits of `mask` (subset enumeration).
fn map_bits(mut bits: usize, mask: BB) -> BB {
    let mut acc = 0u64;
    let mut m = mask;
    while m != 0 {
        let pos = m.trailing_zeros();
        acc |= ((bits & 1) as u64) << pos;
        bits >>= 1;
        m &= m - 1;
    }
    acc
}

fn find_magic(
    table: &mut [BB],
    sq: usize,
    attacks_fn: fn(usize, BB) -> BB,
    mask: BB,
    rng: &mut SmallRng,
    ptr_off: usize,
) -> MagicOut {
    let n = 1usize << mask.count_ones();

    // Every subset of the relevant occupancy, with the full set appended last.
    let mut blockers = Vec::with_capacity(n);
    for x in 0..n {
        let b = map_bits(x, mask);
        if b != mask {
            blockers.push(b);
        }
    }
    blockers.push(mask);

    let attacks: Vec<BB> = blockers.iter().map(|&b| attacks_fn(sq, b)).collect();
    let shift = (64 - mask.count_ones()) as u8;

    loop {
        let candidate = rn(rng);

        let mut ok = true;
        for (&occ, &att) in blockers.iter().zip(attacks.iter()) {
            let key = ((occ & mask).wrapping_mul(candidate) >> shift) as usize;
            if key >= n {
                ok = false;
                break;
            }
            let slot = &mut table[ptr_off + key];
            if *slot == 0 {
                *slot = att;
            } else if *slot != att {
                ok = false;
                break;
            }
        }
        if ok {
            return MagicOut { mask, magic: candidate, shift, ptr_off, ptr_size: n };
        }

        // Reset the slice this candidate wrote into before the next attempt.
        table[ptr_off..ptr_off + n].fill(0);
    }
}

fn find_magics(
    table: &mut [BB],
    attacks_fn: fn(usize, BB) -> BB,
    relevant_fn: fn(usize) -> BB,
    rng: &mut SmallRng,
    start_off: usize,
) -> Vec<MagicOut> {
    let mut out = Vec::with_capacity(64);
    let mut off = start_off;
    for sq in 0..64 {
        let m = find_magic(table, sq, attacks_fn, relevant_fn(sq), rng, off);
        off += m.ptr_size;
        out.push(m);
    }
    out
}

fn main() {
    // The total table size is a function of the relevant masks only, so it must
    // stay fixed at the constant `AttackTable` is sized for.
    let total = (0..64).map(|sq| 1usize << relevant_rook(sq).count_ones()).sum::<usize>()
        + (0..64).map(|sq| 1usize << relevant_bishop(sq).count_ones()).sum::<usize>();
    assert_eq!(total, 0x1A480, "attack table size drifted from 0x1A480");

    let mut table = vec![0u64; 0x1A480];

    let mut rook_rng = SmallRng::seed_from_u64(2947477585843578870);
    let rook = find_magics(&mut table, rook_attacks, relevant_rook, &mut rook_rng, 0);

    let mut bishop_rng = SmallRng::seed_from_u64(16342894268649274550);
    let bishop_start = rook[63].ptr_off + rook[63].ptr_size;
    let bishop = find_magics(&mut table, bishop_attacks, relevant_bishop, &mut bishop_rng, bishop_start);

    let mut src = String::with_capacity(5_000_000);
    writeln!(src, "// @generated by build.rs — do not edit").unwrap();

    writeln!(src, "static ATTACK_TABLE: AttackTable = AttackTable([").unwrap();
    for &v in &table {
        writeln!(src, "    Bitboard {{ v: 0x{v:016x} }},").unwrap();
    }
    writeln!(src, "]);").unwrap();

    for (name, list) in [("ROOK_MAGICS", &rook), ("BISHOP_MAGICS", &bishop)] {
        writeln!(src, "static {name}: MagicTable<'static> = MagicTable([").unwrap();
        for m in list {
            writeln!(
                src,
                "    Magic {{ ptr: &ATTACK_TABLE.0[{}..{}], mask: Bitboard {{ v: 0x{:016x} }}, magic: 0x{:016x}, shift: {} }},",
                m.ptr_off,
                m.ptr_off + m.ptr_size,
                m.mask,
                m.magic,
                m.shift,
            )
            .unwrap();
        }
        writeln!(src, "]);").unwrap();
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::write(out_dir.join("magics.generated.rs"), src).unwrap();
}
