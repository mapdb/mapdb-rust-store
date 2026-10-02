//! `get_current_size()` after `close()` reports 0 (owner ruling 2026-10-02,
//! plan T3) on StoreDirect (heap and file) and StoreWAL: after a single close,
//! after repeated closes, and while another thread closes the store.

use mapdb_rust_store::ser::serializers::LongSer;
use mapdb_rust_store::store::{Store, StoreDirect, StoreWAL};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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
