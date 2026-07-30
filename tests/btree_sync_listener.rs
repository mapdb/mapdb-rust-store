//! Synchronous modification listeners, ported from Java
//! `BTreeMapSyncListenerTest`. A FAILING synchronous listener (the Rust idiom for
//! Java's throwing listener — it returns `Err`) runs while the map holds the
//! covering leaf's node lock, after the mutation and counter update commit. It
//! must never leak the node lock (a leaked spin lock hangs every later op on that
//! leaf forever) nor desync the size counter — this drills the put-update,
//! non-split insert, SPLITTING insert, remove and replace fire points, on inline
//! and external-value maps. Follow-up ops run in a bounded-join thread so a
//! lock-leak regression fails fast instead of hanging the build.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use mapdb_rust_store::btree::BTreeMap;
use mapdb_rust_store::ser::long::LongFormat;
use mapdb_rust_store::ser::string_group::StringGroupFormat;
use mapdb_rust_store::store::{Store, StoreOnHeap};
use mapdb_rust_store::{DbError, MapModificationListener, Result, SynchronousMapModificationListener};

type Map = BTreeMap<StoreOnHeap, LongFormat, StringGroupFormat>;

/// Sync listener that fails once when armed, then disarms.
struct ArmedThrow {
    armed: AtomicBool,
}

impl MapModificationListener<i64, String> for ArmedThrow {
    fn modify(
        &self,
        _key: &i64,
        _old: Option<&String>,
        _new: Option<&String>,
        _triggered: bool,
    ) -> Result<()> {
        if self.armed.swap(false, Ordering::SeqCst) {
            return Err(DbError::corrupt_msg("listener boom"));
        }
        Ok(())
    }
}
impl SynchronousMapModificationListener<i64, String> for ArmedThrow {}

fn is_boom(e: &DbError) -> bool {
    format!("{e}").contains("listener boom")
}

fn expect_boom<F: FnOnce() -> Result<T>, T>(listener: &Arc<ArmedThrow>, op: F) {
    listener.armed.store(true, Ordering::SeqCst);
    match op() {
        Err(e) if is_boom(&e) => {}
        Err(e) => panic!("wrong error: {e}"),
        Ok(_) => panic!("armed listener did not fail"),
    }
}

/// `size_long` (the O(1) counter) must equal the iterated entry count and the expectation.
fn assert_consistent(map: &Map, expected: u64) {
    assert_eq!(map.size_long().unwrap(), expected, "counter desynced");
    assert_eq!(
        map.entries().unwrap().len() as u64,
        expected,
        "counter vs iteration"
    );
}

/// Run `op` in a fresh thread with a BOUNDED join: a leaked node lock spins
/// forever, and this must fail the test rather than hang the build.
fn assert_completes<F>(map: &Map, op: F)
where
    F: FnOnce(Map) -> Result<()> + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    let map = map.clone();
    thread::spawn(move || {
        let r = op(map);
        let _ = tx.send(r);
    });
    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("operation failed: {e}"),
        Err(_) => panic!("operation did not complete within 30s (leaked node lock?)"),
    }
}

fn drill(store: &Arc<StoreOnHeap>, map: &Map) {
    let listener = Arc::new(ArmedThrow {
        armed: AtomicBool::new(false),
    });
    map.modification_listener_add_sync(listener.clone());
    map.put(0, "v0".into()).unwrap();
    map.put(1, "v1".into()).unwrap();

    // non-split insert (leaf below maxNodeSize=4)
    expect_boom(&listener, || map.put(2, "v2".into()));
    assert_eq!(map.get(&2).unwrap(), Some("v2".into()));
    assert_consistent(map, 3);
    assert_completes(map, |m| m.put(2, "v2b".into()).map(|_| ())); // SAME leaf: lock must not leak
    assert_eq!(map.get(&2).unwrap(), Some("v2b".into()));

    // put over an existing key (update branch)
    expect_boom(&listener, || map.put(0, "v0b".into()));
    assert_eq!(map.get(&0).unwrap(), Some("v0b".into()));
    assert_consistent(map, 3);
    assert_completes(map, |m| m.put(0, "v0c".into()).map(|_| ()));

    // SPLITTING insert: the 5th key overflows maxNodeSize=4 and splits the ROOT
    // leaf. The listener failure is surfaced only AFTER separator/root propagation
    // completes, so the tree stays fully operable — including the LATER split of B,
    // which under a skipped root propagation would spin forever on a level-1 left
    // edge that was never created.
    map.put(3, "v3".into()).unwrap();
    expect_boom(&listener, || map.put(4, "v4".into()));
    assert_eq!(map.get(&4).unwrap(), Some("v4".into()));
    assert_consistent(map, 5);
    assert_completes(map, |m| m.put(5, "v5".into()).map(|_| ())); // fills B (split left B={2,3,4})
    assert_completes(map, |m| m.put(6, "v6".into()).map(|_| ())); // overflows B: forces B's own split
    assert_completes(map, |m| m.put(7, "v7".into()).map(|_| ()));
    assert_eq!(map.get(&5).unwrap(), Some("v5".into()));
    assert_eq!(map.get(&6).unwrap(), Some("v6".into()));
    assert_eq!(map.get(&7).unwrap(), Some("v7".into()));
    assert_consistent(map, 8);
    map.remove(&6).unwrap();
    map.remove(&7).unwrap();
    assert_consistent(map, 6);

    // remove
    expect_boom(&listener, || map.remove(&5));
    assert_eq!(map.get(&5).unwrap(), None);
    assert_consistent(map, 5);
    assert_completes(map, |m| m.put(5, "back".into()).map(|_| ()));
    assert_consistent(map, 6);

    // replace
    expect_boom(&listener, || map.replace(&1, "v1b".into()));
    assert_eq!(map.get(&1).unwrap(), Some("v1b".into()));
    assert_consistent(map, 6);
    assert_completes(map, |m| m.remove(&1).map(|_| ()));
    assert_consistent(map, 5);

    store.verify().unwrap();
}

#[test]
fn throwing_listener_leaves_inline_map_usable() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map = BTreeMap::create_with_counter(store.clone(), LongFormat, StringGroupFormat, 4, true)
        .unwrap();
    drill(&store, &map);
}

#[test]
fn throwing_listener_leaves_external_value_map_usable() {
    let store = Arc::new(StoreOnHeap::new(true));
    let map =
        BTreeMap::create_external_values(store.clone(), LongFormat, StringGroupFormat, 4, true)
            .unwrap();
    drill(&store, &map);
}

/// A PANICKING sync listener (Java `RuntimeException|Error`) on the root-splitting
/// insert must be caught, converted to a listener error, and must NOT skip
/// separator/root propagation — so a later split of the right half B cannot hang
/// waiting for a level-1 left edge that was never created.
#[test]
fn panicking_sync_listener_on_root_split_does_not_hang_later_b_split() {
    struct PanicOnce {
        armed: AtomicBool,
    }
    impl MapModificationListener<i64, String> for PanicOnce {
        fn modify(
            &self,
            _k: &i64,
            _o: Option<&String>,
            _n: Option<&String>,
            _t: bool,
        ) -> Result<()> {
            if self.armed.swap(false, Ordering::SeqCst) {
                panic!("listener kaboom");
            }
            Ok(())
        }
    }
    impl SynchronousMapModificationListener<i64, String> for PanicOnce {}

    let store = Arc::new(StoreOnHeap::new(true));
    let map: Map =
        BTreeMap::create_with_counter(store.clone(), LongFormat, StringGroupFormat, 4, true)
            .unwrap();
    let listener = Arc::new(PanicOnce {
        armed: AtomicBool::new(false),
    });
    map.modification_listener_add_sync(listener.clone());

    for i in 0..4i64 {
        map.put(i, format!("v{i}")).unwrap();
    } // root leaf full (maxNodeSize=4)

    // Arm the panic; the 5th key splits the ROOT leaf and the listener panics
    // under the leaf lock. The panic is caught, propagation completes, and the
    // (converted) listener error is surfaced.
    listener.armed.store(true, Ordering::SeqCst);
    let err = map.put(4, "v4".into()).unwrap_err();
    assert!(format!("{err}").contains("panicked"), "got: {err}");
    assert_eq!(map.get(&4).unwrap(), Some("v4".into())); // insert committed
    assert_eq!(map.size_long().unwrap(), 5); // counter bumped, not poisoned

    // Now force B (the new right half) to split. Under a skipped root propagation
    // this would spin forever on the missing level-1 left edge; the bounded join
    // turns a regression into a fast failure instead of a hang.
    for i in 5..12i64 {
        assert_completes(&map, move |m| m.put(i, format!("v{i}")).map(|_| ()));
    }
    assert_consistent(&map, 12);
    store.verify().unwrap();
}

/// A failing sync listener must not hide the event from LATER sync listeners.
#[test]
fn throwing_sync_listener_still_delivers_to_later_sync_listener() {
    struct Boom;
    impl MapModificationListener<i64, String> for Boom {
        fn modify(
            &self,
            _k: &i64,
            _o: Option<&String>,
            _n: Option<&String>,
            _t: bool,
        ) -> Result<()> {
            Err(DbError::corrupt_msg("listener boom"))
        }
    }
    impl SynchronousMapModificationListener<i64, String> for Boom {}

    struct Recorder(Arc<Mutex<Vec<String>>>);
    impl MapModificationListener<i64, String> for Recorder {
        fn modify(&self, k: &i64, o: Option<&String>, n: Option<&String>, _t: bool) -> Result<()> {
            let os = o.map(|s| s.as_str()).unwrap_or("null");
            let ns = n.map(|s| s.as_str()).unwrap_or("null");
            self.0.lock().unwrap().push(format!("{k}:{os}:{ns}"));
            Ok(())
        }
    }
    impl SynchronousMapModificationListener<i64, String> for Recorder {}

    let store = Arc::new(StoreOnHeap::new(true));
    let map = BTreeMap::create_with_counter(store.clone(), LongFormat, StringGroupFormat, 4, true)
        .unwrap();
    let later = Arc::new(Mutex::new(Vec::new()));
    map.modification_listener_add_sync(Arc::new(Boom));
    map.modification_listener_add_sync(Arc::new(Recorder(later.clone())));

    let err = map.put(1, "v1".into()).unwrap_err();
    assert!(is_boom(&err));
    assert_eq!(*later.lock().unwrap(), vec!["1:null:v1".to_string()]);
    assert_eq!(map.get(&1).unwrap(), Some("v1".into()));
}

/// Same continuation for ordinary (deferred) listeners.
#[test]
fn throwing_deferred_listener_still_delivers_to_later_listener() {
    use mapdb_rust_store::FnListener;
    let store = Arc::new(StoreOnHeap::new(true));
    let map = BTreeMap::create_with_counter(store.clone(), LongFormat, StringGroupFormat, 4, true)
        .unwrap();
    let later = Arc::new(Mutex::new(Vec::new()));
    map.modification_listener_add(Arc::new(FnListener(
        |_k: &i64, _o: Option<&String>, _n: Option<&String>, _t| {
            Err(DbError::corrupt_msg("listener boom"))
        },
    )));
    let l2 = later.clone();
    map.modification_listener_add(Arc::new(FnListener(
        move |k: &i64, o: Option<&String>, n: Option<&String>, _t| {
            let os = o.map(|s| s.as_str()).unwrap_or("null");
            let ns = n.map(|s| s.as_str()).unwrap_or("null");
            l2.lock().unwrap().push(format!("{k}:{os}:{ns}"));
            Ok(())
        },
    )));

    let err = map.put(1, "v1".into()).unwrap_err();
    assert!(is_boom(&err));
    assert_eq!(*later.lock().unwrap(), vec!["1:null:v1".to_string()]);
    assert_eq!(map.get(&1).unwrap(), Some("v1".into()));
}
