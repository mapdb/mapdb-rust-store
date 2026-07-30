//! Streaming descending iterator (spec 03 §7 second cut): parity with an
//! oracle across the bounds matrix on deeply split trees, weak-consistency
//! behavior under mid-iteration mutation, empty-range/inverted-range edges,
//! and the O(log n)-per-attempt `poll_last_entry` / `floor` paths it now backs.

use std::sync::Arc;

use mapdb_rust_store::btree::BTreeMap;
use mapdb_rust_store::ser::long::LongFormat;
use mapdb_rust_store::store::StoreOnHeap;

fn new_map(
    max: usize,
) -> (
    Arc<StoreOnHeap>,
    BTreeMap<StoreOnHeap, LongFormat, LongFormat>,
) {
    let store = Arc::new(StoreOnHeap::new(true));
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, max).unwrap();
    (store, map)
}

fn desc_keys(
    map: &BTreeMap<StoreOnHeap, LongFormat, LongFormat>,
    lo: Option<i64>,
    lo_inc: bool,
    hi: Option<i64>,
    hi_inc: bool,
) -> Vec<i64> {
    map.descending_entry_iter(lo, lo_inc, hi, hi_inc)
        .unwrap()
        .map(|r| r.unwrap().0)
        .collect()
}

/// Oracle: the same range from a std BTreeMap, reversed.
fn oracle(keys: &[i64], lo: Option<i64>, lo_inc: bool, hi: Option<i64>, hi_inc: bool) -> Vec<i64> {
    let mut v: Vec<i64> = keys
        .iter()
        .copied()
        .filter(|k| match lo {
            None => true,
            Some(l) => *k > l || (lo_inc && *k == l),
        })
        .filter(|k| match hi {
            None => true,
            Some(h) => *k < h || (hi_inc && *k == h),
        })
        .collect();
    v.sort_unstable();
    v.reverse();
    v
}

#[test]
fn descending_matches_oracle_across_bounds_matrix_on_a_deep_tree() {
    // max_node_size=4 → several levels at 1000 keys; only even keys exist so
    // every probe key parity (present/absent) is exercised.
    let (_s, map) = new_map(4);
    let keys: Vec<i64> = (0..1000).map(|i| i * 2).collect();
    for &k in &keys {
        map.put(k, k * 10).unwrap();
    }
    let probes = [
        None,
        Some(-5),
        Some(0),
        Some(3),
        Some(400),
        Some(999),
        Some(1998),
        Some(2500),
    ];
    for &lo in &probes {
        for &hi in &probes {
            for lo_inc in [true, false] {
                for hi_inc in [true, false] {
                    assert_eq!(
                        desc_keys(&map, lo, lo_inc, hi, hi_inc),
                        oracle(&keys, lo, lo_inc, hi, hi_inc),
                        "lo={lo:?}/{lo_inc} hi={hi:?}/{hi_inc}"
                    );
                }
            }
        }
    }
}

#[test]
fn descending_values_come_with_their_keys() {
    let (_s, map) = new_map(4);
    for i in 0..300i64 {
        map.put(i, i * 7).unwrap();
    }
    let entries: Vec<(i64, i64)> = map
        .descending_entry_iter(Some(10), true, Some(200), false)
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(entries.first(), Some(&(199, 199 * 7)));
    assert_eq!(entries.last(), Some(&(10, 70)));
    assert!(
        entries.windows(2).all(|w| w[0].0 > w[1].0),
        "strictly descending"
    );
    assert!(entries.iter().all(|(k, v)| *v == k * 7));
}

#[test]
fn inverted_and_empty_ranges_yield_nothing() {
    let (_s, map) = new_map(4);
    for i in 0..100i64 {
        map.put(i, i).unwrap();
    }
    assert!(
        desc_keys(&map, Some(60), true, Some(40), true).is_empty(),
        "inverted"
    );
    assert!(
        desc_keys(&map, Some(40), false, Some(41), false).is_empty(),
        "open gap"
    );
    let (_s2, empty) = new_map(4);
    assert!(
        desc_keys(&empty, None, true, None, true).is_empty(),
        "empty map"
    );
}

#[test]
fn descending_streams_after_removes_leave_sparse_leaves() {
    // Remove most keys so the walk crosses leaves that went empty/sparse.
    let (_s, map) = new_map(4);
    for i in 0..500i64 {
        map.put(i, i).unwrap();
    }
    for i in 0..500i64 {
        if i % 25 != 0 {
            map.remove(&i).unwrap();
        }
    }
    let keys: Vec<i64> = (0..500).filter(|i| i % 25 == 0).collect();
    assert_eq!(
        desc_keys(&map, None, true, None, true),
        oracle(&keys, None, true, None, true)
    );
}

#[test]
fn descending_is_weakly_consistent_never_duplicating_under_mutation() {
    // Streaming property: keys at/above the consumed position never reappear,
    // even when re-inserted mid-iteration; keys inserted below MAY appear
    // (weak consistency permits both) but order stays strictly descending.
    let (_s, map) = new_map(4);
    for i in 0..200i64 {
        map.put(i, i).unwrap();
    }
    let mut it = map.descending_entry_iter(None, true, None, true).unwrap();
    let mut got: Vec<i64> = Vec::new();
    for _ in 0..50 {
        got.push(it.next().unwrap().unwrap().0);
    }
    assert_eq!(*got.last().unwrap(), 150);
    // Mutate around the frontier: delete just below it, re-insert far above it,
    // and add brand-new keys on both sides of it.
    map.remove(&149).unwrap();
    map.put(199, 9999).unwrap();
    map.put(1000, 1).unwrap();
    map.put(-7, 1).unwrap();
    for r in it {
        got.push(r.unwrap().0);
    }
    assert!(
        got.windows(2).all(|w| w[0] > w[1]),
        "strictly descending, no duplicates"
    );
    assert!(!got[50..].contains(&149), "removed key resurfaced");
    assert_eq!(
        got[50..].iter().filter(|k| **k >= 150).count(),
        0,
        "consumed frontier re-emitted"
    );
    assert!(
        got.contains(&-7),
        "new key below the frontier belongs to the tail"
    );
}

#[test]
fn early_drop_is_cheap_and_poll_last_is_a_single_descent_shape() {
    let (_s, map) = new_map(4);
    for i in 0..1000i64 {
        map.put(i, i).unwrap();
    }
    // Early stop: taking 3 must not require visiting the whole range.
    let top3: Vec<i64> = map
        .descending_entry_iter(None, true, None, true)
        .unwrap()
        .take(3)
        .map(|r| r.unwrap().0)
        .collect();
    assert_eq!(top3, vec![999, 998, 997]);
    // poll_last drains correctly from the top under bounds.
    assert_eq!(
        map.poll_last_entry(None, true, Some(500), true).unwrap(),
        Some((500, 500))
    );
    assert_eq!(
        map.poll_last_entry(None, true, Some(500), true).unwrap(),
        Some((499, 499))
    );
    assert_eq!(
        map.poll_last_entry(Some(998), true, None, true).unwrap(),
        Some((999, 999))
    );
    assert_eq!(
        map.poll_last_entry(Some(998), false, None, true).unwrap(),
        None
    );
}

#[test]
fn floor_and_lower_ride_the_descending_path() {
    let (_s, map) = new_map(4);
    for i in (0..1000i64).step_by(10) {
        map.put(i, i).unwrap();
    }
    assert_eq!(map.floor_entry(&555).unwrap(), Some((550, 550)));
    assert_eq!(map.floor_entry(&550).unwrap(), Some((550, 550)));
    assert_eq!(map.lower_entry(&550).unwrap(), Some((540, 540)));
    assert_eq!(map.floor_entry(&-1).unwrap(), None);
    assert_eq!(map.last_entry().unwrap(), Some((990, 990)));
    // Range-view composition: floor within a sub_map window.
    let view = map.sub_map(100, true, 500, false);
    assert_eq!(view.floor_entry(&555).unwrap(), Some((490, 490)));
    assert_eq!(view.last_entry().unwrap(), Some((490, 490)));
}
