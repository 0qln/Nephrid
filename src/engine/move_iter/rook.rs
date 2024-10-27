use crate::engine::{bitboard::Bitboard, coordinates::{CompassRose, File, Rank, Square}};

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
    let moves = (Bitboard::split_north(nearest) << CompassRose::Sout) & ray;
    result |= moves;
    
    // north
    let ray = file_bb & nort_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.lsb();
    let moves = (Bitboard::split_south(nearest) << CompassRose::Nort) & ray;
    result |= moves;
    
    // west
    let ray = rank_bb & sout_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.msb();
    let moves = (Bitboard::split_north(nearest) << CompassRose::West) & ray;
    result |= moves;
    
    // east
    let ray = rank_bb & nort_bb;
    let occupands = occupancy & ray;
    let nearest = occupands.lsb();
    let moves = (Bitboard::split_south(nearest) << CompassRose::East) & ray;
    result |= moves;
    
    result
}