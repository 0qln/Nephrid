use super::*;

#[test]
fn push_increments_occurrences() {
    let mut table = RepetitionTable::<3>::default();
    let h = zobrist::Hash::from_v(5);

    assert_eq!(table.get(h), None);

    table.push(h);
    assert_eq!(table.get(h), Some(1));

    table.push(h);
    assert_eq!(table.get(h), Some(2));
}

#[test]
fn pop_decrements_or_removes_occurrences() {
    let mut table = RepetitionTable::<3>::default();
    let h = zobrist::Hash::from_v(5);

    table.push(h);
    table.push(h);

    table.pop(h);
    assert_eq!(table.get(h), Some(1));

    table.pop(h);
    assert_eq!(table.get(h), None);
}

#[test]
fn pop_non_existent_hash_is_noop() {
    let mut table = RepetitionTable::<3>::default();
    let h = zobrist::Hash::from_v(5);

    table.pop(h); // Should not panic or affect the table
    assert_eq!(table.get(h), None);
}

#[test]
fn multiple_entries_in_same_bucket() {
    const N: usize = 3;
    let mut table = RepetitionTable::<N>::default();
    let h1 = zobrist::Hash::from_v(0); // bucket 0
    let h2 = zobrist::Hash::from_v(3); // bucket 0

    table.push(h1);
    table.push(h2);
    assert_eq!(table.get(h1), Some(1));
    assert_eq!(table.get(h2), Some(1));

    table.push(h1);
    table.pop(h2);

    assert_eq!(table.get(h1), Some(2));
    assert_eq!(table.get(h2), None);
}

#[test]
fn entries_in_different_buckets() {
    const N: usize = 3;
    let mut table = RepetitionTable::<N>::default();
    let h1 = zobrist::Hash::from_v(0); // bucket 0
    let h2 = zobrist::Hash::from_v(1); // bucket 1

    table.push(h1);
    table.push(h2);

    table.pop(h1);
    assert_eq!(table.get(h1), None);
    assert_eq!(table.get(h2), Some(1));
}

#[test]
fn push_after_pop_to_zero() {
    let mut table = RepetitionTable::<3>::default();
    let h = zobrist::Hash::from_v(5);

    table.push(h);
    table.pop(h);

    table.push(h);
    assert_eq!(table.get(h), Some(1));
}

#[test]
fn swap_remove_preserves_other_entries() {
    const N: usize = 3;
    let mut table = RepetitionTable::<N>::default();
    let h1 = zobrist::Hash::from_v(0);
    let h2 = zobrist::Hash::from_v(3);
    let h3 = zobrist::Hash::from_v(6);

    table.push(h1);
    table.push(h2);
    table.push(h3);

    table.pop(h2); // Removes h2 via swap_remove

    assert_eq!(table.get(h1), Some(1));
    assert_eq!(table.get(h3), Some(1));
}

#[test]
fn multiple_operations() {
    let mut table = RepetitionTable::<3>::default();
    let h = zobrist::Hash::from_v(5);

    for _ in 0..5 {
        table.push(h);
    }
    assert_eq!(table.get(h), Some(5));

    for _ in 0..4 {
        table.pop(h);
    }
    assert_eq!(table.get(h), Some(1));

    table.pop(h);
    assert_eq!(table.get(h), None);
}

#[test]
fn cloning_doesnt_allocate_on_stack() {
    let original = RepetitionTable::<{ 2 << 20 }>::default();
    // this should crash if it allocates on the stack.
    let cloned = original.clone();
    assert_eq!(original, cloned);
}