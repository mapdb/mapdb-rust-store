//! Concurrency + byte-store tests. The multi-thread stress exercises the
//! Lehman-Yao writer protocol (node-lock table, split publish ordering,
//! move-right) under real contention; the byte-store tests exercise the
//! `on_bytes` binary-search read path (StoreOnHeap only uses `on_object`).

use std::collections::BTreeSet;
use std::sync::Arc;
use std::thread;

use mapdb_rust_store::btree::BTreeMap;
use mapdb_rust_store::ser::long::LongFormat;
use mapdb_rust_store::store::{Store, StoreDirect, StoreOnHeap};

#[test]
fn concurrent_disjoint_writers() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 8).unwrap();
    let threads = 8;
    let per = 4000i64;
    thread::scope(|s| {
        for t in 0..threads {
            let map = map.clone();
            s.spawn(move || {
                // disjoint key ranges by thread → every insert is a distinct key
                for i in 0..per {
                    let k = t as i64 * per + i;
                    map.put(k, k * 2).unwrap();
                }
            });
        }
    });
    // every key present with the right value; ascending + complete
    let total = threads as i64 * per;
    assert_eq!(map.size_long().unwrap(), total as u64);
    let entries = map.entries().unwrap();
    assert_eq!(entries.len(), total as usize);
    for (idx, (k, v)) in entries.iter().enumerate() {
        assert_eq!(*k, idx as i64);
        assert_eq!(*v, idx as i64 * 2);
    }
    store.verify().unwrap();
}

#[test]
fn concurrent_mixed_ops_invariants() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 6).unwrap();
    let threads = 6;
    thread::scope(|s| {
        for t in 0..threads {
            let map = map.clone();
            s.spawn(move || {
                let mut x = 0x9E3779B9u64 ^ ((t as u64) << 32);
                for _ in 0..8000 {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let k = (x >> 33) as i64 % 2000;
                    match (x >> 20) & 3 {
                        0 | 1 => {
                            map.put(k, k).unwrap();
                        }
                        2 => {
                            map.remove(&k).unwrap();
                        }
                        _ => {
                            if let Some(v) = map.get(&k).unwrap() {
                                assert_eq!(v, k); // value invariant: v == k always
                            }
                        }
                    }
                }
            });
        }
    });
    store.verify().unwrap();
    // invariant sweep: strictly ascending, no duplicates, values == keys
    let entries = map.entries().unwrap();
    let mut seen = BTreeSet::new();
    let mut prev: Option<i64> = None;
    for (k, v) in &entries {
        assert_eq!(*v, *k, "value invariant");
        assert!(seen.insert(*k), "duplicate key {k}");
        if let Some(p) = prev {
            assert!(p < *k, "not ascending: {p} >= {k}");
        }
        prev = Some(*k);
    }
    // link-chain reachability: iteration count matches size_long
    assert_eq!(entries.len() as u64, map.size_long().unwrap());
}

#[test]
fn concurrent_root_grow_contention() {
    // Many threads insert the SAME overlapping key range into a fresh small-node
    // tree, so the root splits/grows repeatedly under heavy contention — the
    // schedule that can double-grow a root if root identity is sampled wrong.
    // Repeated to widen the race window; the tree must stay correct.
    for _round in 0..12 {
        let store = Arc::new(StoreOnHeap::new(true));
        let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 4).unwrap();
        let n = 600i64;
        thread::scope(|s| {
            for _ in 0..12 {
                let map = map.clone();
                s.spawn(move || {
                    for k in 0..n {
                        map.put(k, k).unwrap(); // same value → LWW deterministic
                    }
                });
            }
        });
        store.verify().unwrap();
        let entries = map.entries().unwrap();
        assert_eq!(entries.len() as i64, n, "round {_round}: wrong entry count");
        for (i, (k, v)) in entries.iter().enumerate() {
            assert_eq!(*k, i as i64, "round {_round}: key gap/dup at {i}");
            assert_eq!(*v, i as i64, "round {_round}: wrong value at {i}");
        }
    }
}

#[test]
fn concurrent_readers_during_writes() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 8).unwrap();
    for i in 0..1000i64 {
        map.put(i, i).unwrap();
    }
    thread::scope(|s| {
        // one writer growing the tree
        {
            let map = map.clone();
            s.spawn(move || {
                for i in 1000..10000i64 {
                    map.put(i, i).unwrap();
                }
            });
        }
        // readers must never see a torn/absent value for a key known to exist
        for _ in 0..4 {
            let map = map.clone();
            s.spawn(move || {
                for _ in 0..20000 {
                    for k in [0i64, 500, 999] {
                        assert_eq!(map.get(&k).unwrap(), Some(k));
                    }
                }
            });
        }
    });
    store.verify().unwrap();
}

// ---------------- byte-store read path (on_bytes / binary search) ----------------

#[test]
fn bytestore_put_get_splits() {
    let store = Arc::new(StoreDirect::new_heap().unwrap());
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 4).unwrap();
    for i in 0..2000i64 {
        map.put(i * 3, i).unwrap();
    }
    // exercise the binary-search on_bytes path for hits and misses
    for i in 0..2000i64 {
        assert_eq!(map.get(&(i * 3)).unwrap(), Some(i), "hit {i}");
        assert_eq!(map.get(&(i * 3 + 1)).unwrap(), None, "miss {i}");
    }
    assert_eq!(map.size_long().unwrap(), 2000);
    let entries = map.entries().unwrap();
    assert_eq!(entries.len(), 2000);
    assert!(entries.windows(2).all(|w| w[0].0 < w[1].0));
    store.verify().unwrap();
}

#[test]
fn bytestore_reopen_and_pump() {
    use mapdb_rust_store::store::StoreDirect;
    let store = Arc::new(StoreDirect::new_heap().unwrap());
    let entries: Vec<(i64, i64)> = (0..3000).map(|i| (i, i + 1)).collect();
    let map =
        BTreeMap::create_from_sorted(store.clone(), LongFormat, LongFormat, 16, entries).unwrap();
    let rrr = map.root_recid_recid();
    for i in 0..3000i64 {
        assert_eq!(map.get(&i).unwrap(), Some(i + 1));
    }
    drop(map);
    // reopen and keep working (binary-search path)
    let m2 = BTreeMap::open(store.clone(), rrr, LongFormat, LongFormat, 16).unwrap();
    assert_eq!(m2.get(&1500).unwrap(), Some(1501));
    m2.put(99999, 7).unwrap();
    assert_eq!(m2.get(&99999).unwrap(), Some(7));
    store.verify().unwrap();
}

#[test]
fn bytestore_concurrent() {
    let store = Arc::new(StoreDirect::new_heap_ts(true).unwrap());
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 8).unwrap();
    thread::scope(|s| {
        for t in 0..4 {
            let map = map.clone();
            s.spawn(move || {
                for i in 0..3000i64 {
                    let k = t as i64 * 3000 + i;
                    map.put(k, k).unwrap();
                }
            });
        }
    });
    assert_eq!(map.size_long().unwrap(), 12000);
    for k in [0i64, 5000, 11999] {
        assert_eq!(map.get(&k).unwrap(), Some(k));
    }
    store.verify().unwrap();
}
