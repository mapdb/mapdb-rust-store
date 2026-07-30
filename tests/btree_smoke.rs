//! BTreeMap smoke tests: basic ops, splits, iteration, pump, CAS, reopen.

use std::sync::Arc;

use mapdb_rust_store::btree::BTreeMap;
use mapdb_rust_store::ser::long::LongFormat;
use mapdb_rust_store::store::{Store, StoreOnHeap};

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

#[test]
fn put_get_remove_basic() {
    let (_s, map) = new_map(8);
    assert_eq!(map.get(&1).unwrap(), None);
    assert_eq!(map.put(1, 100).unwrap(), None);
    assert_eq!(map.put(2, 200).unwrap(), None);
    assert_eq!(map.get(&1).unwrap(), Some(100));
    assert_eq!(map.get(&2).unwrap(), Some(200));
    assert_eq!(map.put(1, 111).unwrap(), Some(100)); // overwrite returns old
    assert_eq!(map.get(&1).unwrap(), Some(111));
    assert!(map.contains_key(&1).unwrap());
    assert!(!map.contains_key(&99).unwrap());
    assert_eq!(map.remove(&1).unwrap(), Some(111));
    assert_eq!(map.get(&1).unwrap(), None);
    assert_eq!(map.remove(&1).unwrap(), None);
    assert_eq!(map.size_long().unwrap(), 1);
}

#[test]
fn splits_many_ascending() {
    let (_s, map) = new_map(4); // small nodes → many splits
    for i in 0..1000i64 {
        assert_eq!(map.put(i, i * 10).unwrap(), None);
    }
    for i in 0..1000i64 {
        assert_eq!(map.get(&i).unwrap(), Some(i * 10), "key {i}");
    }
    assert_eq!(map.size_long().unwrap(), 1000);
    // ascending iteration order
    let entries = map.entries().unwrap();
    assert_eq!(entries.len(), 1000);
    for (idx, (k, v)) in entries.iter().enumerate() {
        assert_eq!(*k, idx as i64);
        assert_eq!(*v, idx as i64 * 10);
    }
}

#[test]
fn splits_many_descending_insert() {
    let (_s, map) = new_map(4);
    for i in (0..500i64).rev() {
        map.put(i, i).unwrap();
    }
    for i in 0..500i64 {
        assert_eq!(map.get(&i).unwrap(), Some(i));
    }
    let ks: Vec<i64> = map.entries().unwrap().into_iter().map(|(k, _)| k).collect();
    assert!(ks.windows(2).all(|w| w[0] < w[1]), "sorted");
}

#[test]
fn random_ops_vs_std() {
    use std::collections::BTreeMap as Std;
    let (_s, map) = new_map(6);
    let mut model = Std::new();
    // deterministic pseudo-random
    let mut x: u64 = 0x1234_5678;
    for _ in 0..5000 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let k = (x >> 33) as i64 % 300;
        let op = (x >> 20) & 3;
        match op {
            0 | 1 => {
                let v = k * 7;
                assert_eq!(map.put(k, v).unwrap(), model.insert(k, v));
            }
            2 => {
                assert_eq!(map.remove(&k).unwrap(), model.remove(&k));
            }
            _ => {
                assert_eq!(map.get(&k).unwrap(), model.get(&k).copied());
            }
        }
    }
    let got = map.entries().unwrap();
    let want: Vec<(i64, i64)> = model.into_iter().collect();
    assert_eq!(got, want);
}

#[test]
fn cas_ops() {
    let (_s, map) = new_map(8);
    map.put(1, 10).unwrap();
    assert_eq!(map.put_if_absent(1, 999).unwrap(), Some(10)); // present, no change
    assert_eq!(map.get(&1).unwrap(), Some(10));
    assert_eq!(map.put_if_absent(2, 20).unwrap(), None); // inserted
    assert_eq!(map.get(&2).unwrap(), Some(20));
    assert_eq!(map.replace(&1, 11).unwrap(), Some(10));
    assert_eq!(map.replace(&99, 1).unwrap(), None); // absent
    assert!(map.replace_if(&1, &11, 12).unwrap());
    assert!(!map.replace_if(&1, &11, 13).unwrap()); // old mismatch
    assert_eq!(map.get(&1).unwrap(), Some(12));
    assert!(!map.remove_if(&1, &99).unwrap()); // value mismatch
    assert!(map.remove_if(&1, &12).unwrap());
    assert_eq!(map.get(&1).unwrap(), None);
}

#[test]
fn bounded_iteration() {
    let (_s, map) = new_map(4);
    for i in 0..100i64 {
        map.put(i, i).unwrap();
    }
    let collect = |lo, lo_i, hi, hi_i| -> Vec<i64> {
        map.entry_iter(lo, lo_i, hi, hi_i)
            .unwrap()
            .map(|e| e.unwrap().0)
            .collect()
    };
    assert_eq!(
        collect(Some(10), true, Some(15), true),
        vec![10, 11, 12, 13, 14, 15]
    );
    assert_eq!(
        collect(Some(10), false, Some(15), false),
        vec![11, 12, 13, 14]
    );
    assert_eq!(collect(Some(95), true, None, true).len(), 5);
    assert_eq!(collect(None, true, Some(4), true), vec![0, 1, 2, 3, 4]);
    // descending
    let desc: Vec<i64> = map
        .descending_entry_iter(Some(10), true, Some(13), true)
        .unwrap()
        .map(|e| e.unwrap().0)
        .collect();
    assert_eq!(desc, vec![13, 12, 11, 10]);
}

#[test]
fn poll_entries() {
    let (_s, map) = new_map(4);
    for i in 0..20i64 {
        map.put(i, i).unwrap();
    }
    assert_eq!(
        map.poll_first_entry(None, true, None, true).unwrap(),
        Some((0, 0))
    );
    assert_eq!(
        map.poll_last_entry(None, true, None, true).unwrap(),
        Some((19, 19))
    );
    assert_eq!(map.size_long().unwrap(), 18);
}

#[test]
fn pump_bulk_build() {
    let store = Arc::new(StoreOnHeap::new(true));
    let entries: Vec<(i64, i64)> = (0..2000).map(|i| (i, i * 2)).collect();
    let map =
        BTreeMap::create_from_sorted(store.clone(), LongFormat, LongFormat, 16, entries.clone())
            .unwrap();
    assert_eq!(map.size_long().unwrap(), 2000);
    for &(k, v) in &entries {
        assert_eq!(map.get(&k).unwrap(), Some(v), "key {k}");
    }
    // still fully functional: insert + read back
    map.put(10000, 42).unwrap();
    assert_eq!(map.get(&10000).unwrap(), Some(42));
    assert_eq!(map.entries().unwrap().len(), 2001);
}

#[test]
fn pump_rejects_unsorted() {
    let store = Arc::new(StoreOnHeap::new(true));
    let bad = vec![(1i64, 1i64), (3, 3), (2, 2)];
    let r = BTreeMap::create_from_sorted(store, LongFormat, LongFormat, 8, bad);
    assert!(matches!(r, Err(mapdb_rust_store::DbError::NotSorted)));
}

#[test]
fn pump_empty_source() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map =
        BTreeMap::create_from_sorted(store, LongFormat, LongFormat, 8, Vec::<(i64, i64)>::new())
            .unwrap();
    assert_eq!(map.size_long().unwrap(), 0);
    map.put(5, 5).unwrap();
    assert_eq!(map.get(&5).unwrap(), Some(5));
}

#[test]
fn reopen_persists_root() {
    let store = Arc::new(StoreOnHeap::new(true));
    let rrr;
    {
        let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 4).unwrap();
        for i in 0..200i64 {
            map.put(i, i * 3).unwrap();
        }
        rrr = map.root_recid_recid();
    } // drop map → release lease
    let map2 = BTreeMap::open(store.clone(), rrr, LongFormat, LongFormat, 4).unwrap();
    for i in 0..200i64 {
        assert_eq!(map2.get(&i).unwrap(), Some(i * 3));
    }
}

#[test]
fn duplicate_open_rejected() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 8).unwrap();
    let rrr = map.root_recid_recid();
    let dup = BTreeMap::open(store.clone(), rrr, LongFormat, LongFormat, 8);
    assert!(matches!(
        dup,
        Err(mapdb_rust_store::DbError::AlreadyOpen { .. })
    ));
    drop(map);
    // after drop, reopen ok
    let _m2 = BTreeMap::open(store, rrr, LongFormat, LongFormat, 8).unwrap();
}

#[test]
fn verify_after_ops() {
    let (store, map) = new_map(4);
    for i in 0..300i64 {
        map.put(i, i).unwrap();
        if i % 3 == 0 {
            map.remove(&(i / 2)).unwrap();
        }
    }
    store.verify().unwrap();
}
