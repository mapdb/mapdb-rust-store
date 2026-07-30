//! External-value `BTreeMap` (`valueInline=false`) ported from
//! Java `BTreeMapExternalValuesTest`: values live in separate store records; the
//! value read barrier (74b9963) prevents a lock-free reader observing a recid a
//! concurrent remove deleted and the store reused; removes must not leak records.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use mapdb_rust_store::btree::BTreeMap;
use mapdb_rust_store::ser::long::LongFormat;
use mapdb_rust_store::ser::string_group::StringGroupFormat;
use mapdb_rust_store::store::{Store, StoreByteArray, StoreDirect, StoreOnHeap};
use mapdb_rust_store::DbError;

fn recid_count<S: Store>(store: &S) -> usize {
    store.get_all_recids().unwrap().len()
}

// One body per backend (the collection bound `S: Store + StoreLease` uses a
// crate-private trait, so it cannot be named in a generic test helper — generate
// concrete bodies with a macro instead).
macro_rules! ops_views_reopen_test {
    ($name:ident, $store_expr:expr) => {
        #[test]
        fn $name() {
            let store = Arc::new($store_expr);
            let map = BTreeMap::create_external_values(
                store.clone(),
                LongFormat,
                StringGroupFormat,
                4,
                true,
            )
            .unwrap();
            assert!(!map.value_inline());
            for i in 0..40i64 {
                assert_eq!(map.put(i, format!("v{i}")).unwrap(), None);
            }
            assert_eq!(map.put(5, "updated".into()).unwrap(), Some("v5".into()));
            assert_eq!(map.get(&5).unwrap(), Some("updated".into()));
            assert_eq!(map.replace(&6, "six".into()).unwrap(), Some("v6".into()));
            assert!(map.replace_if(&7, &"v7".into(), "seven".into()).unwrap());
            assert!(!map.replace_if(&7, &"v7".into(), "wrong".into()).unwrap());
            assert!(map.remove_if(&8, &"v8".into()).unwrap());
            assert_eq!(
                map.sub_map(9, true, 12, false).remove(&9).unwrap(),
                Some("v9".into())
            );
            let first = map
                .poll_first_entry(None, true, None, true)
                .unwrap()
                .unwrap();
            assert_eq!(first.0, 0);
            assert_eq!(first.1, "v0");
            assert_eq!(map.size_long().unwrap(), 37);

            let rrr = map.root_recid_recid();
            let cr = map.counter_recid();
            drop(map); // release the RW lease (D12) before reopening

            let reopened = BTreeMap::open_external_values(
                store.clone(),
                rrr,
                LongFormat,
                StringGroupFormat,
                4,
                cr,
            )
            .unwrap();
            assert_eq!(reopened.get(&5).unwrap(), Some("updated".into()));
            assert_eq!(reopened.get(&7).unwrap(), Some("seven".into()));
            assert_eq!(reopened.entries().unwrap().len(), 37);
            reopened.clear().unwrap();
            assert!(reopened.is_empty().unwrap());
            assert_eq!(reopened.size_long().unwrap(), 0);
            store.verify().unwrap();
            store.close().unwrap();
        }
    };
}

ops_views_reopen_test!(operations_views_and_reopen_on_heap, StoreOnHeap::new(true));
ops_views_reopen_test!(
    operations_views_and_reopen_byte_array,
    StoreByteArray::new(true)
);
ops_views_reopen_test!(
    operations_views_and_reopen_direct,
    StoreDirect::new_heap_ts(true).unwrap()
);

#[test]
fn iterator_resumes_lazily_across_removes_updates_inserts_and_split() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map =
        BTreeMap::create_external_values(store, LongFormat, StringGroupFormat, 4, false).unwrap();
    for key in [10i64, 20, 30, 40] {
        map.put(key, format!("v{key}")).unwrap();
    }

    let mut iter = map.iter().unwrap();
    assert_eq!(iter.next().unwrap().unwrap(), (10, "v10".into()));

    // Mutate between pull steps. Re-inserting the resume key and inserting a
    // smaller key must not make either appear again. Keys inserted ahead may be
    // observed by this weakly-consistent iterator. The fifth live key forces a
    // split with max_node_size=4, exercising the fresh lower-bound re-filter.
    assert_eq!(map.remove(&20).unwrap(), Some("v20".into()));
    assert_eq!(map.remove(&10).unwrap(), Some("v10".into()));
    map.put(10, "v10-new".into()).unwrap();
    map.put(5, "v5".into()).unwrap();
    map.put(15, "v15".into()).unwrap();
    map.put(25, "v25".into()).unwrap();
    map.put(30, "v30-new".into()).unwrap();

    let rest: Vec<(i64, String)> = iter.map(|r| r.unwrap()).collect();
    assert_eq!(
        rest,
        vec![
            (15, "v15".into()),
            (25, "v25".into()),
            (30, "v30-new".into()),
            (40, "v40".into()),
        ]
    );
}

#[test]
fn iterator_does_not_snapshot_before_first_pull() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map =
        BTreeMap::create_external_values(store, LongFormat, StringGroupFormat, 4, false).unwrap();
    map.put(1, "one".into()).unwrap();
    map.put(2, "two".into()).unwrap();

    let mut iter = map.iter().unwrap();
    map.clear().unwrap();
    assert!(iter.next().is_none());
}

#[test]
fn iterator_honors_bounds_and_fuses_after_error() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map =
        BTreeMap::create_external_values(store.clone(), LongFormat, StringGroupFormat, 4, false)
            .unwrap();
    for key in 0..10i64 {
        map.put(key, format!("v{key}")).unwrap();
    }

    let inclusive: Vec<i64> = map
        .entry_iter(Some(3), true, Some(6), true)
        .unwrap()
        .map(|r| r.unwrap().0)
        .collect();
    assert_eq!(inclusive, vec![3, 4, 5, 6]);
    let exclusive: Vec<i64> = map
        .entry_iter(Some(3), false, Some(6), false)
        .unwrap()
        .map(|r| r.unwrap().0)
        .collect();
    assert_eq!(exclusive, vec![4, 5]);

    let mut iter = map.iter().unwrap();
    assert!(iter.next().unwrap().is_ok());
    store.close().unwrap();
    assert!(matches!(iter.next(), Some(Err(DbError::StoreClosed))));
    assert!(
        iter.next().is_none(),
        "iterator did not fuse after an error"
    );
}

#[test]
fn descending_and_floor_compose_with_external_streaming_iterator() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map =
        BTreeMap::create_external_values(store, LongFormat, StringGroupFormat, 4, false).unwrap();
    for key in [1i64, 3, 5, 7, 9] {
        map.put(key, format!("v{key}")).unwrap();
    }

    let descending: Vec<i64> = map
        .descending()
        .entries()
        .unwrap()
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    assert_eq!(descending, vec![9, 7, 5, 3, 1]);
    assert_eq!(map.floor_entry(&6).unwrap(), Some((5, "v5".into())));
}

#[test]
fn paused_iterator_holds_no_external_value_barrier() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map =
        BTreeMap::create_external_values(store, LongFormat, StringGroupFormat, 4, false).unwrap();
    map.put(1, "one".into()).unwrap();
    map.put(2, "two".into()).unwrap();

    let mut iter = map.iter().unwrap();
    assert_eq!(iter.next().unwrap().unwrap(), (1, "one".into()));

    thread::scope(|scope| {
        let worker_map = map.clone();
        let (started_tx, started_rx) = mpsc::channel();
        let (completed_tx, completed_rx) = mpsc::channel();
        let worker = scope.spawn(move || {
            started_tx.send(()).unwrap();
            let removed = worker_map.remove(&2).unwrap();
            completed_tx.send(()).unwrap();
            removed
        });

        // Start timing only after the worker is scheduled and about to remove,
        // avoiding a false failure caused merely by slow thread startup. Always
        // drop the iterator before joining so a retained-guard regression cannot
        // hang the test process.
        let worker_started = started_rx.recv_timeout(Duration::from_secs(30)).is_ok();
        let completed_while_paused =
            worker_started && completed_rx.recv_timeout(Duration::from_secs(30)).is_ok();
        drop(iter);
        assert_eq!(worker.join().unwrap(), Some("two".into()));
        assert!(
            worker_started,
            "remove worker did not start within 30 seconds"
        );
        assert!(
            completed_while_paused,
            "remove was blocked while the iterator was idle"
        );
    });
}

// Readers hold the external read barrier across store.get, which keeps a concurrent
// remove from deleting + reusing the recid mid-read.
#[test]
fn reader_never_observes_reused_external_value_under_concurrent_remove() {
    let store = Arc::new(StoreDirect::new_heap_ts(true).unwrap());
    let map =
        BTreeMap::create_external_values(store.clone(), LongFormat, StringGroupFormat, 8, false)
            .unwrap();
    map.put(1, "one".into()).unwrap();
    let baseline = recid_count(&*store);
    let done = Arc::new(AtomicBool::new(false));

    thread::scope(|s| {
        let reader = {
            let map = map.clone();
            let done = done.clone();
            s.spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    if let Some(v) = map.get(&1).unwrap() {
                        assert_eq!(v, "one", "reader observed a reused/garbage value");
                    }
                }
            })
        };
        for _ in 0..4000 {
            map.remove(&1).unwrap();
            map.put(2, "unrelated".into()).unwrap();
            map.remove(&2).unwrap();
            map.put(1, "one".into()).unwrap();
        }
        done.store(true, Ordering::Relaxed);
        reader.join().unwrap();
    });

    // No external value record may leak (bounded structural growth only).
    assert!(
        recid_count(&*store) <= baseline + 2,
        "external value records leaked: {} > {}",
        recid_count(&*store),
        baseline + 2
    );
    store.close().unwrap();
}

#[test]
fn iterator_never_observes_reused_external_value_under_concurrent_remove() {
    let store = Arc::new(StoreDirect::new_heap_ts(true).unwrap());
    let map =
        BTreeMap::create_external_values(store.clone(), LongFormat, StringGroupFormat, 8, false)
            .unwrap();
    for key in 0..16i64 {
        map.put(key, format!("v{key}")).unwrap();
    }
    let done = Arc::new(AtomicBool::new(false));

    thread::scope(|scope| {
        let reader = {
            let map = map.clone();
            let done = done.clone();
            scope.spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    let mut previous = None;
                    for entry in map.iter().unwrap() {
                        let (key, value) = entry.unwrap();
                        assert_eq!(value, format!("v{key}"), "iterator observed a reused value");
                        if let Some(p) = previous {
                            assert!(p < key, "iterator emitted duplicate/out-of-order keys");
                        }
                        previous = Some(key);
                    }
                }
            })
        };

        for i in 0..2000i64 {
            let key = i & 15;
            map.remove(&key).unwrap();
            let transient = 1000 + key;
            map.put(transient, format!("v{transient}")).unwrap();
            map.remove(&transient).unwrap();
            map.put(key, format!("v{key}")).unwrap();
        }
        done.store(true, Ordering::Relaxed);
        reader.join().unwrap();
    });

    store.verify().unwrap();
    store.close().unwrap();
}

/// Multi-threaded churn on one external-value map, then a full clear: no external
/// value recids may leak (bounded getAllRecids count after the map empties).
#[test]
fn concurrent_churn_leaks_no_external_value_recids() {
    let store = Arc::new(StoreDirect::new_heap_ts(true).unwrap());
    let map =
        BTreeMap::create_external_values(store.clone(), LongFormat, StringGroupFormat, 8, false)
            .unwrap();
    let baseline = recid_count(&*store); // structure only, empty map
    let key_count = 256i64;
    let threads = 6usize;
    let iterations = 4000usize;

    thread::scope(|s| {
        for t in 0..threads {
            let map = map.clone();
            s.spawn(move || {
                let mut state = t as u64 + 1;
                let mut next = || {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 33) as u32
                };
                for i in 0..iterations {
                    let key = (next() as i64) % key_count;
                    match i & 3 {
                        0 => {
                            map.put(key, format!("t{t}v{i}")).unwrap();
                        }
                        1 => {
                            map.get(&key).unwrap();
                        }
                        2 => {
                            map.remove(&key).unwrap();
                        }
                        _ => {
                            let mut seen = 0;
                            for e in map.iter().unwrap() {
                                e.unwrap();
                                seen += 1;
                                if seen >= 32 {
                                    break;
                                }
                            }
                        }
                    }
                }
            });
        }
    });

    for key in 0..key_count {
        map.remove(&key).unwrap();
    }
    store.verify().unwrap();
    // The emptied map keeps its split leaf/dir nodes (mapdb3 semantics: no merge on
    // remove), bounded by the key count, but every external VALUE recid must be gone.
    let bound = baseline + key_count as usize;
    assert!(
        recid_count(&*store) <= bound,
        "external value records leaked under churn: {} > {}",
        recid_count(&*store),
        bound
    );
    store.close().unwrap();
}
