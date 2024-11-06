use std::{cmp::max, result};

use crate::engine::{bitboard::Bitboard, coordinates::{CompassRose, File, Rank, Square}};

use super::{get_key, MagicBits, MagicData};

const MAGICS: [MagicData; 64] = [
    144134980375683200i64, 198193571466100736i64, 5800671507789123584i64, -4251366159715139576i64, 8791061949192342016i64, 8214574671053455872i64, 36101365071609984i64, 3602879977680307201i64, 140877074808834i64, 2343208813714481153i64, 281750125150464i64, 6918092115335448896i64, 4785108968427776i64, 308637861786157568i64, 180425468661530628i64, 2305983748849561856i64, 158879434407968i64, 2319354083510599697i64, 153122939234943008i64, 38291592236368064i64, 141287311296512i64, 2306406508990038656i64, 5550831676370724963i64, -8881096265985653759i64, 2603645682091170i64, 40542568202641408i64, 18578525286056064i64, 4504703436067080i64, 2533846021570832i64, 1729947408034693248i64, 235462631291421264i64, 4512679188496513i64, 324329544607596672i64, 5665785582206985i64, -5683507543196823552i64, 4540158690986496i64, 2251840624264192i64, 145289468650193408i64, 1697852447261090i64, 4684887658709647428i64, 600175053996457984i64, 40532534089498625i64, 54151506018041872i64, 1152939930022969368i64, 577023874593587204i64, 563104580665354i64, 20275020458885121i64, 578889028049436676i64, 36029621990328384i64, 1154364338757206272i64, 2305994744339109376i64, 4611844349041344640i64, 2314863540214700544i64, 5263019149165330944i64, -9222809052474441216i64, -8934855700681520640i64, 36310480301599335i64, 2432295668556005409i64, 2312609404312252546i64, 9306267004708357i64, 1171217412453436465i64, 2416744219160117570i64, 6990149589216067786i64, 4399262925026i64,
];

const BITS: [MagicBits; 64] = [
  12, 11, 11, 11, 11, 11, 11, 12,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  12, 11, 11, 11, 11, 11, 11, 12  
];

// todo: should this be heap allocated?
static ATTACKS: [Box<[Bitboard]>; 64];

fn init_rook_attacks(sq: Square) -> Box<[Bitboard]> {
    let mut key_max = 0;
    let max_blockers = relevant_occupancy(sq);
    let num_blocker_compositions = 1 << max_blockers.v.count_ones();
    let result = Vec::<Bitboard>::new();
    
    for i in 0..num_blocker_compositions {
        let occupied = map_bits(i, max_blockers);
        let computed = compute_attacks(sq, occupied);
        let sq_usize: usize = sq.into();
        let key = get_key(computed, MAGICS[sq_usize], BITS[sq_usize]);
        result[key] = computed;
        key_max = max(key, key_max);
    }
    
    Box::new(result.as_slice())
}

/// Maps the bits of the index into allowed bits (defined by mask).
/// If the mask does not specify atleast the number of bits in 
/// needed for a complete mapping, the remaining bits are cut off.
fn map_bits(mut bits: usize, mask: Bitboard) -> Bitboard {
    mask.fold(Bitboard::empty(), |acc, pos| {
        let val = bits & 1;
        bits >>= 1;
        acc | (val << pos)
    })
}

pub fn lookup_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    let relevant_occupancy = relevant_occupancy(sq);
    let sq: usize = sq.into();
    let magic = MAGICS[sq];
    let bits = BITS[sq];
    let key = get_key(relevant_occupancy, magic, bits);
    (&ATTACKS[sq])[key as usize]
}

fn relevant_occupancy(sq: Square) -> Bitboard {
    let mut result = relevant_squares_from_file(File::from(sq)) | relevant_squares_from_rank(Rank::from(sq));
    result &= !Bitboard::from(sq);
    result
}

// todo: make const
fn relevant_squares_from_file(file: File) -> Bitboard {
    let relevant_ranks = Bitboard::full() ^ Bitboard::from(Rank::_1) ^ Bitboard::from(Rank::_8);
    let file = Bitboard::from(file);
    relevant_ranks & file
}

fn relevant_squares_from_rank(rank: Rank) -> Bitboard {
    let relevant_files = Bitboard::full() ^ Bitboard::from(File::A) ^ Bitboard::from(File::H);
    let rank = Bitboard::from(rank);
    relevant_files & rank
}

pub fn compute_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    let file = File::from(sq);
    let rank = Rank::from(sq);
    let file_bb = Bitboard::from(file);
    let rank_bb = Bitboard::from(rank);
    let nort_bb = Bitboard::split_north(sq);
    let sout_bb = Bitboard::split_south(sq);
    let mut result = Bitboard::default();
    
    // south
    let ray = file_bb & sout_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.msb();
    let moves = (Bitboard::split_north(nearest) << CompassRose::SOUT) & ray;
    result |= moves;
    
    // north
    let ray = file_bb & nort_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.lsb();
    let moves = (Bitboard::split_south(nearest) << CompassRose::NORT) & ray;
    result |= moves;
    
    // west
    let ray = rank_bb & sout_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.msb();
    let moves = (Bitboard::split_north(nearest) << CompassRose::WEST) & ray;
    result |= moves;
    
    // east
    let ray = rank_bb & nort_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.lsb();
    let moves = (Bitboard::split_south(nearest) << CompassRose::EAST) & ray;
    result |= moves;
    
    result
}