//! Feature A (O(1) size counter) and Feature B (modification listeners) for
//! `BTreeMap`, ported from Java `BTreeMapCounterListenerTest`. Runs on
//! `StoreOnHeap` (thread-safe) so the concurrent cases exercise the real
//! node-lock + CAS-counter protocol.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use mapdb_rust_store::btree::BTreeMap;
use mapdb_rust_store::listener::FnListener;
use mapdb_rust_store::ser::long::LongFormat;
use mapdb_rust_store::store::StoreOnHeap;

type Map = BTreeMap<StoreOnHeap, LongFormat, LongFormat>;

fn counter_map(max: usize) -> (Arc<StoreOnHeap>, Map) {
    let store = Arc::new(StoreOnHeap::new(true));
    let map =
        BTreeMap::create_with_counter(store.clone(), LongFormat, LongFormat, max, true).unwrap();
    (store, map)
}

fn plain_map(max: usize) -> (Arc<StoreOnHeap>, Map) {
    let store = Arc::new(StoreOnHeap::new(true));
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, max).unwrap();
    (store, map)
}

fn traversal_count(m: &Map) -> u64 {
    m.entries().unwrap().len() as u64
}

// ---------------- counter: sequential ----------------

#[test]
fn counter_disabled_by_default() {
    let (_s, m) = plain_map(8);
    assert_eq!(m.counter_recid(), 0);
    m.put(1, 1).unwrap();
    assert_eq!(m.size_long().unwrap(), 1); // traversal fallback
}

#[test]
fn counter_enabled_exposes_recid() {
    let (_s, m) = counter_map(8);
    assert!(m.counter_recid() > 0);
    assert_eq!(m.size_long().unwrap(), 0);
}

#[test]
fn counter_insert_update_remove_clear() {
    let (_s, m) = counter_map(6);

    for i in 0..100i64 {
        assert_eq!(m.put(i, i).unwrap(), None);
        assert_eq!(m.size_long().unwrap(), (i + 1) as u64);
    }
    assert_eq!(m.size_long().unwrap(), 100);
    assert_eq!(traversal_count(&m), m.size_long().unwrap());

    // updates do NOT change the counter
    for i in 0..100i64 {
        assert_eq!(m.put(i, i * 10).unwrap(), Some(i));
    }
    assert_eq!(m.size_long().unwrap(), 100);

    // putIfAbsent on present key: no change
    assert_eq!(m.put_if_absent(0, 999).unwrap(), Some(0));
    assert_eq!(m.size_long().unwrap(), 100);
    // putIfAbsent on absent key: +1
    assert_eq!(m.put_if_absent(1000, 1).unwrap(), None);
    assert_eq!(m.size_long().unwrap(), 101);
    m.remove(&1000).unwrap();
    assert_eq!(m.size_long().unwrap(), 100);

    // replace of present key: no change
    assert_eq!(m.replace(&0, 7).unwrap(), Some(0));
    assert_eq!(m.size_long().unwrap(), 100);
    // replace of absent key: no change
    assert_eq!(m.replace(&5000, 1).unwrap(), None);
    assert_eq!(m.size_long().unwrap(), 100);

    // removes
    for i in 0..50i64 {
        let expect = if i == 0 { 7 } else { i * 10 };
        assert_eq!(m.remove(&i).unwrap(), Some(expect));
        assert_eq!(m.size_long().unwrap(), (100 - (i + 1)) as u64);
    }
    assert_eq!(m.size_long().unwrap(), 50);
    assert_eq!(traversal_count(&m), m.size_long().unwrap());

    // remove of absent key: no change
    assert_eq!(m.remove(&0).unwrap(), None);
    assert_eq!(m.size_long().unwrap(), 50);

    // clear resets to 0
    m.clear().unwrap();
    assert_eq!(m.size_long().unwrap(), 0);
    assert_eq!(traversal_count(&m), 0);
}

#[test]
fn counter_matches_traversal_after_mixed_ops() {
    let (_s, m) = counter_map(4); // small nodes -> many splits
                                  // deterministic pseudo-random sequence (portable, no rand dep)
    let mut state: u64 = 42;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut refm: std::collections::BTreeMap<i64, i64> = std::collections::BTreeMap::new();
    for _ in 0..5000 {
        let k = (next() % 500) as i64;
        if next() % 2 == 0 {
            let prev = m.put(k, k).unwrap();
            let was_present = refm.insert(k, k).is_some();
            assert_eq!(was_present, prev.is_some());
        } else {
            let prev = m.remove(&k).unwrap();
            let was_present = refm.remove(&k).is_some();
            assert_eq!(was_present, prev.is_some());
        }
        assert_eq!(refm.len() as u64, m.size_long().unwrap());
    }
    assert_eq!(traversal_count(&m), m.size_long().unwrap());
}

// ---------------- counter: concurrent ----------------

fn run_disjoint(m: &Map, threads: usize, per_thread: i64, insert: bool) {
    thread::scope(|s| {
        for t in 0..threads {
            let base = t as i64 * per_thread;
            let m = m.clone();
            s.spawn(move || {
                for i in 0..per_thread {
                    if insert {
                        m.put(base + i, base + i).unwrap();
                    } else {
                        m.remove(&(base + i)).unwrap();
                    }
                }
            });
        }
    });
}

#[test]
fn counter_concurrent_disjoint_keys() {
    let (_s, m) = counter_map(8);
    let threads = 8usize;
    let per = 5000i64;
    run_disjoint(&m, threads, per, true);
    let expected = threads as u64 * per as u64;
    assert_eq!(m.size_long().unwrap(), expected);
    assert_eq!(traversal_count(&m), m.size_long().unwrap());
}

#[test]
fn counter_concurrent_put_remove_same_keyspace() {
    let (_s, m) = counter_map(6);
    let keyspace = 2000i64;
    let mut k = 0;
    while k < keyspace {
        m.put(k, k).unwrap();
        k += 2;
    }
    let threads = 8usize;
    let ops = 20000usize;
    thread::scope(|s| {
        for t in 0..threads {
            let m = m.clone();
            s.spawn(move || {
                let mut state = t as u64 + 1;
                let mut next = || {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 33) as u32
                };
                for _ in 0..ops {
                    let key = (next() as i64) % keyspace;
                    if next() % 2 == 0 {
                        m.put(key, key).unwrap();
                    } else {
                        m.remove(&key).unwrap();
                    }
                }
            });
        }
    });
    // after all threads join, the counter must equal the actual live entry count.
    assert_eq!(traversal_count(&m), m.size_long().unwrap());
}

#[test]
fn counter_concurrent_insert_then_remove_to_empty() {
    let (_s, m) = counter_map(6);
    let threads = 6usize;
    let per = 4000i64;
    run_disjoint(&m, threads, per, true);
    assert_eq!(m.size_long().unwrap(), threads as u64 * per as u64);
    assert_eq!(traversal_count(&m), m.size_long().unwrap());
    run_disjoint(&m, threads, per, false);
    assert_eq!(m.size_long().unwrap(), 0);
    assert_eq!(traversal_count(&m), 0);
}

// ---------------- listeners ----------------

type Event = (i64, Option<i64>, Option<i64>);

#[allow(clippy::type_complexity)]
fn record_listener(
    sink: Arc<Mutex<Vec<Event>>>,
) -> Arc<
    FnListener<
        impl Fn(&i64, Option<&i64>, Option<&i64>, bool) -> mapdb_rust_store::Result<()> + Send + Sync,
    >,
> {
    Arc::new(FnListener(
        move |k: &i64, o: Option<&i64>, n: Option<&i64>, _t: bool| {
            sink.lock().unwrap().push((*k, o.copied(), n.copied()));
            Ok(())
        },
    ))
}

#[test]
fn listener_insert_update_remove() {
    let (_s, m) = plain_map(8);
    let events = Arc::new(Mutex::new(Vec::new()));
    m.modification_listener_add(record_listener(events.clone()));

    m.put(5, 50).unwrap(); // insert: old=None new=50
    m.put(5, 60).unwrap(); // update: old=50 new=60
    assert_eq!(m.remove(&5).unwrap(), Some(60)); // remove: old=60 new=None

    let ev = events.lock().unwrap();
    assert_eq!(
        *ev,
        vec![
            (5, None, Some(50)),
            (5, Some(50), Some(60)),
            (5, Some(60), None),
        ]
    );
}

#[test]
fn listener_replace_and_conditional_ops() {
    let (_s, m) = plain_map(8);
    let events = Arc::new(Mutex::new(Vec::new()));
    m.modification_listener_add(record_listener(events.clone()));

    m.put(1, 10).unwrap(); // insert
    m.replace(&1, 20).unwrap(); // update via replace(K,V): old=10 new=20
    m.replace_if(&1, &20, 30).unwrap(); // update via replace(K,V,V): old=20 new=30
                                        // failed conditional ops fire NOTHING:
    assert!(!m.replace_if(&1, &999, 40).unwrap()); // value mismatch
    assert_eq!(m.replace(&2, 5).unwrap(), None); // absent
    assert_eq!(m.put_if_absent(1, 77).unwrap(), Some(30)); // present
    assert!(!m.remove_if(&1, &999).unwrap()); // value mismatch
    assert_eq!(m.put_if_absent(2, 22).unwrap(), None); // insert
    assert!(m.remove_if(&2, &22).unwrap()); // conditional remove success

    let ev = events.lock().unwrap();
    assert_eq!(
        *ev,
        vec![
            (1, None, Some(10)),
            (1, Some(10), Some(20)),
            (1, Some(20), Some(30)),
            (2, None, Some(22)),
            (2, Some(22), None),
        ]
    );
}

#[test]
fn listener_clear_fires_removal_for_every_entry() {
    let (_s, m) = counter_map(4);
    let events = Arc::new(Mutex::new(Vec::new()));
    m.modification_listener_add(record_listener(events.clone()));

    for i in 0..25i64 {
        m.put(i, i * 10).unwrap();
    }
    events.lock().unwrap().clear();
    m.clear().unwrap();

    let ev = events.lock().unwrap();
    assert_eq!(ev.len(), 25);
    for i in 0..25i64 {
        assert_eq!(ev[i as usize], (i, Some(i * 10), None));
    }
    assert_eq!(m.size_long().unwrap(), 0);
    assert_eq!(traversal_count(&m), 0);
}

#[test]
fn multiple_listeners_all_fire() {
    let (_s, m) = plain_map(8);
    let a = Arc::new(AtomicU64::new(0));
    let b = Arc::new(AtomicU64::new(0));
    let a2 = a.clone();
    let b2 = b.clone();
    m.modification_listener_add(Arc::new(FnListener(
        move |_k: &i64, _o: Option<&i64>, _n: Option<&i64>, _t| {
            a2.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
    )));
    m.modification_listener_add(Arc::new(FnListener(
        move |_k: &i64, _o: Option<&i64>, _n: Option<&i64>, _t| {
            b2.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
    )));
    m.put(1, 1).unwrap();
    m.put(1, 2).unwrap();
    m.remove(&1).unwrap();
    assert_eq!(a.load(Ordering::SeqCst), 3);
    assert_eq!(b.load(Ordering::SeqCst), 3);
}

#[test]
fn listener_concurrent_event_count_matches_ops() {
    let (_s, m) = counter_map(6);
    let count = Arc::new(AtomicU64::new(0));
    let null_old = Arc::new(AtomicU64::new(0));
    let c2 = count.clone();
    let n2 = null_old.clone();
    m.modification_listener_add(Arc::new(FnListener(
        move |_k: &i64, o: Option<&i64>, _n: Option<&i64>, _t| {
            c2.fetch_add(1, Ordering::SeqCst);
            if o.is_none() {
                n2.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        },
    )));
    let threads = 6usize;
    let per = 3000i64;
    run_disjoint(&m, threads, per, true); // all inserts
    let expected = threads as u64 * per as u64;
    assert_eq!(count.load(Ordering::SeqCst), expected);
    assert_eq!(null_old.load(Ordering::SeqCst), expected); // every event is an insert (old==None)
    assert_eq!(m.size_long().unwrap(), expected);
}

// ---------------- bulk build ----------------

#[test]
fn bulk_build_counter() {
    let store = Arc::new(StoreOnHeap::new(true));
    let n = 3000i64;
    let entries: Vec<(i64, i64)> = (0..n).map(|i| (i, i * 2)).collect();
    let m = BTreeMap::create_from_sorted_counter(store, LongFormat, LongFormat, 16, entries, true)
        .unwrap();

    assert!(m.counter_recid() > 0);
    assert_eq!(m.size_long().unwrap(), n as u64);
    assert_eq!(traversal_count(&m), m.size_long().unwrap());

    // counter keeps tracking after the bulk build
    assert_eq!(m.put(n, 1).unwrap(), None);
    assert_eq!(m.size_long().unwrap(), (n + 1) as u64);
    m.remove(&0).unwrap();
    assert_eq!(m.size_long().unwrap(), n as u64);
}

#[test]
fn bulk_build_no_counter() {
    let store = Arc::new(StoreOnHeap::new(true));
    let entries: Vec<(i64, i64)> = (0..100).map(|i| (i, i)).collect();
    let m = BTreeMap::create_from_sorted_counter(store, LongFormat, LongFormat, 16, entries, false)
        .unwrap();
    assert_eq!(m.counter_recid(), 0);
    assert_eq!(m.size_long().unwrap(), 100); // traversal fallback
}

#[test]
fn bulk_build_empty_counter_starts_at_zero_and_tracks() {
    let store = Arc::new(StoreOnHeap::new(true));
    let entries: Vec<(i64, i64)> = Vec::new();
    let m = BTreeMap::create_from_sorted_counter(store, LongFormat, LongFormat, 16, entries, true)
        .unwrap();
    assert!(m.counter_recid() > 0);
    assert_eq!(m.size_long().unwrap(), 0);
    assert_eq!(m.put(1, 10).unwrap(), None);
    assert_eq!(m.size_long().unwrap(), 1);
}

// ---------------- reopen ----------------

#[test]
fn reopen_with_counter_recid() {
    let (store, m) = counter_map(8);
    for i in 0..200i64 {
        m.put(i, i).unwrap();
    }
    let rrr = m.root_recid_recid();
    let cr = m.counter_recid();
    assert_eq!(m.size_long().unwrap(), 200);
    // drop the first handle so the reopen can acquire the RW lease (D12).
    drop(m);

    let re = BTreeMap::open_with_counter(store, rrr, LongFormat, LongFormat, 8, cr).unwrap();
    assert_eq!(re.size_long().unwrap(), 200);
    re.put(200, 200).unwrap();
    assert_eq!(re.size_long().unwrap(), 201);
}
