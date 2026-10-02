//! `get_current_size()` after `close()` reports 0 (owner ruling 2026-10-02,
//! plan T3) on StoreDirect (heap and file) and StoreWAL: after a single close,
//! after repeated closes, and while another thread closes the store. Also the
//! `Serializer` re-entry contract: a callback may query its store's metrics
//! (plan T3b) without deadlocking, under `get` and `compare_and_swap`.

use mapdb_rust_store::error::Result;
use mapdb_rust_store::io::{DataInput2, DataOutput2};
use mapdb_rust_store::ser::serializers::LongSer;
use mapdb_rust_store::ser::Serializer;
use mapdb_rust_store::store::{Store, StoreDirect, StoreWAL};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

fn tmp(kind: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "mapdb5_closed_size_{}_{}_{}",
        kind,
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("store")
}

fn fill(s: &impl Store) {
    for i in 0..200i64 {
        s.put(&i, &LongSer).unwrap();
    }
}

fn closed_reports_zero(s: impl Store) {
    fill(&s);
    assert!(s.get_current_size() > 0, "open store reports its footprint");
    s.close().unwrap();
    assert_eq!(s.get_current_size(), 0);
    s.close().unwrap();
    assert_eq!(s.get_current_size(), 0);
}

#[test]
fn direct_heap_closed_size_is_zero() {
    closed_reports_zero(StoreDirect::new_heap().unwrap());
}

#[test]
fn direct_file_closed_size_is_zero() {
    closed_reports_zero(StoreDirect::open_file(&tmp("direct")).unwrap());
}

#[test]
fn wal_closed_size_is_zero() {
    closed_reports_zero(StoreWAL::open(&tmp("wal")).unwrap());
}

/// Readers loop on the size while another thread closes: every value is either
/// the open footprint or 0, never a panic, and 0 once the close has returned.
fn stress<S: Store + Sync>(make: impl Fn() -> S) {
    for _ in 0..200 {
        let s = make();
        fill(&s);
        let open = s.get_current_size();
        assert!(open > 0);
        let closed = AtomicBool::new(false);
        std::thread::scope(|t| {
            for _ in 0..4 {
                t.spawn(|| loop {
                    let done = closed.load(Ordering::Acquire);
                    let v = s.get_current_size();
                    assert!(v == open || v == 0, "size {v} is neither {open} nor 0");
                    if done {
                        assert_eq!(v, 0);
                        break;
                    }
                });
            }
            t.spawn(|| {
                std::thread::yield_now();
                s.close().unwrap();
                closed.store(true, Ordering::Release);
            });
        });
    }
}

#[test]
fn direct_heap_concurrent_close() {
    stress(|| StoreDirect::new_heap().unwrap());
}

#[test]
fn direct_file_concurrent_close() {
    stress(|| StoreDirect::open_file(&tmp("direct_c")).unwrap());
}

#[test]
fn wal_concurrent_close() {
    stress(|| StoreWAL::open(&tmp("wal_c")).unwrap());
}

/// Every metric the `Serializer` contract lets a callback query on its store.
fn query_metrics(s: &impl Store) {
    let _ = (
        s.get_current_size(),
        s.is_closed(),
        s.is_tx(),
        s.is_read_only(),
        s.is_thread_safe(),
        s.structural_generation(),
    );
}

/// Serializer whose `deserialize` reports when it is running, waits for the
/// go signal, then queries the store's size from inside the callback.
struct SizeProbe<S> {
    store: Arc<S>,
    inside: Mutex<mpsc::Sender<()>>,
    go: Mutex<mpsc::Receiver<()>>,
    seen: AtomicU64,
}

impl<S: Store> Serializer<i64> for SizeProbe<S> {
    fn serialize(&self, out: &mut DataOutput2, value: &i64) {
        LongSer.serialize(out, value)
    }
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<i64> {
        self.inside.lock().unwrap().send(()).unwrap();
        self.go.lock().unwrap().recv().unwrap();
        query_metrics(&*self.store);
        self.seen
            .store(self.store.get_current_size(), Ordering::Release);
        LongSer.deserialize(input, size)
    }
    fn compare(&self, a: &i64, b: &i64) -> std::cmp::Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &i64, b: &i64) -> bool {
        a == b
    }
}

/// A serializer callback (run under the store's shared commit barrier) queries
/// the size while close() already waits for that barrier exclusively. The size
/// query must not queue behind the waiting writer: that would deadlock the
/// reader against the close it blocks.
fn callback_size_query_with_waiting_close<S: Store + Send + Sync + 'static>(store: S) {
    let store = Arc::new(store);
    let recid = store.put(&7i64, &LongSer).unwrap();
    let open = store.get_current_size();
    let (inside_tx, inside_rx) = mpsc::channel();
    let (go_tx, go_rx) = mpsc::channel();
    let probe = Arc::new(SizeProbe {
        store: store.clone(),
        inside: Mutex::new(inside_tx),
        go: Mutex::new(go_rx),
        seen: AtomicU64::new(u64::MAX),
    });
    let (done_tx, done_rx) = mpsc::channel();
    {
        let (store, probe) = (store.clone(), probe.clone());
        std::thread::spawn(move || {
            let v = store.get(recid, &*probe).unwrap();
            done_tx.send(v).unwrap();
        });
    }
    inside_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let (closed_tx, closed_rx) = mpsc::channel();
    {
        let store = store.clone();
        std::thread::spawn(move || {
            store.close().unwrap();
            closed_tx.send(()).unwrap();
        });
    }
    // Let close() reach and queue on the exclusive barrier the reader holds.
    std::thread::sleep(Duration::from_millis(300));
    assert!(closed_rx.try_recv().is_err(), "close waits for the reader");
    go_tx.send(()).unwrap();
    let v = done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("size query inside a serializer deadlocked behind a waiting close()");
    assert_eq!(v, Some(7));
    assert_eq!(probe.seen.load(Ordering::Acquire), open);
    closed_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(store.get_current_size(), 0);
}

#[test]
fn direct_heap_callback_size_query_with_waiting_close() {
    callback_size_query_with_waiting_close(StoreDirect::new_heap().unwrap());
}

#[test]
fn direct_file_callback_size_query_with_waiting_close() {
    callback_size_query_with_waiting_close(StoreDirect::open_file(&tmp("direct_cb")).unwrap());
}

#[test]
fn wal_callback_size_query_with_waiting_close() {
    callback_size_query_with_waiting_close(StoreWAL::open(&tmp("wal_cb")).unwrap());
}

/// Serializer whose every callback queries the store's metrics.
struct MetricsProbe<S> {
    store: Arc<S>,
    calls: AtomicU64,
}

impl<S: Store> Serializer<i64> for MetricsProbe<S> {
    fn serialize(&self, out: &mut DataOutput2, value: &i64) {
        query_metrics(&*self.store);
        self.calls.fetch_add(1, Ordering::Relaxed);
        LongSer.serialize(out, value)
    }
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<i64> {
        query_metrics(&*self.store);
        self.calls.fetch_add(1, Ordering::Relaxed);
        LongSer.deserialize(input, size)
    }
    fn compare(&self, a: &i64, b: &i64) -> std::cmp::Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &i64, b: &i64) -> bool {
        query_metrics(&*self.store);
        self.calls.fetch_add(1, Ordering::Relaxed);
        a == b
    }
}

/// compare_and_swap runs the serializer under the store's exclusive lock
/// (StoreWAL's state write guard); a metric query from that callback must not
/// try to take the same lock. Covers staged and committed records.
fn callback_metrics_under_cas<S: Store + Send + Sync + 'static>(store: S) {
    let store = Arc::new(store);
    let staged = store.put(&1i64, &LongSer).unwrap();
    let committed = store.put(&2i64, &LongSer).unwrap();
    store.commit().unwrap();
    let staged2 = store.put(&3i64, &LongSer).unwrap();
    let (done_tx, done_rx) = mpsc::channel();
    {
        let store = store.clone();
        std::thread::spawn(move || {
            let probe = MetricsProbe {
                store: store.clone(),
                calls: AtomicU64::new(0),
            };
            let r = (
                store
                    .compare_and_swap(committed, Some(&2), Some(&20), &probe)
                    .unwrap(),
                store
                    .compare_and_swap(staged2, Some(&3), Some(&30), &probe)
                    .unwrap(),
                store
                    .compare_and_swap(staged, Some(&9), Some(&10), &probe)
                    .unwrap(),
                store.get(committed, &probe).unwrap(),
            );
            done_tx
                .send((r, probe.calls.load(Ordering::Relaxed)))
                .unwrap();
        });
    }
    let (r, calls) = done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("metric query inside a compare_and_swap callback deadlocked");
    assert_eq!(r, (true, true, false, Some(20)));
    assert!(calls > 0, "the probe ran");
}

#[test]
fn wal_callback_metrics_under_cas() {
    callback_metrics_under_cas(StoreWAL::open(&tmp("wal_cas")).unwrap());
}

#[test]
fn direct_heap_callback_metrics_under_cas() {
    callback_metrics_under_cas(StoreDirect::new_heap().unwrap());
}

#[test]
fn direct_file_callback_metrics_under_cas() {
    callback_metrics_under_cas(StoreDirect::open_file(&tmp("direct_cas")).unwrap());
}
