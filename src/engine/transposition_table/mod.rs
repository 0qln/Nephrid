use super::zobrist;
use super::ConfigOptionType;

pub struct Hash {
    key: zobrist::Hash,
} 

// static mut ENTRIES_COUNT: ConfigOptionType = super::CONFIG.iter().find(|cfg| cfg.cfg_name == "Hash").unwrap().cfg_type;
// static mut ENTRIES: Box<[Hash]> = Box::();

pub fn set(key: zobrist::Hash, entry: Hash) {
   todo!("Comp") 
}

pub fn get(key: zobrist::Hash) -> Option<Hash> {
    todo!()
}

pub fn set_order(order: u8) {
    todo!("Set entries array to size 2 ^ <order>")
}
