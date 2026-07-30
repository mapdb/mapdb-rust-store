//! BTreeMap over a transactional store (StoreWAL): the structural `left_edges`
//! cache must stay consistent with the tx-visible tree across a `rollback()`
//! that shrinks the tree height while the map object stays open (found in review). Without the tx-refresh, a post-rollback root grow would append onto
//! a stale, too-long `left_edges` vector (panic in debug / stale-recid append in
//! release).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use mapdb_rust_store::btree::BTreeMap;
use mapdb_rust_store::ser::long::LongFormat;
use mapdb_rust_store::store::{Store, StoreTx, StoreWAL};

fn tmp() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!("mapdb5_btree_wal_{}_{}.wal", std::process::id(), n));
    let _ = std::fs::remove_file(&p);
    let mut c = p.clone().into_os_string();
    c.push(".ckpt");
    let _ = std::fs::remove_file(PathBuf::from(c));
    p
}

#[test]
fn rollback_then_regrow_keeps_left_edges_consistent() {
    let p = tmp();
    let store = Arc::new(StoreWAL::open(&p).unwrap());
    // small nodes so a modest key count forces several root grows
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 4).unwrap();
    store.commit().unwrap(); // committed baseline: empty map, height 1

    let n = 200i64;
    // Uncommitted inserts grow the tree to height >= 2 (left_edges lengthens).
    for k in 0..n {
        map.put(k, k).unwrap();
    }
    assert_eq!(map.size_long().unwrap(), n as u64);

    // Revert to the committed empty tree; the map object (and its now-stale,
    // longer left_edges) stays open.
    store.rollback().unwrap();
    assert_eq!(
        map.size_long().unwrap(),
        0,
        "rollback should empty the tree"
    );

    // Re-grow on the SAME open map: the first put must resync left_edges with
    // the reverted height-1 tree so root grow's level accounting is honest.
    for k in 0..n {
        map.put(k, k * 10).unwrap();
    }
    assert_eq!(map.size_long().unwrap(), n as u64);
    let entries = map.entries().unwrap();
    assert_eq!(entries.len() as i64, n);
    for (i, (k, v)) in entries.iter().enumerate() {
        assert_eq!(*k, i as i64, "key gap/dup at {i}");
        assert_eq!(*v, i as i64 * 10, "wrong value at {i}");
    }

    // Commit the regrown tree and reopen to confirm it is well-formed on disk.
    store.commit().unwrap();
    let rrr = map.root_recid_recid();
    drop(map);
    store.close().unwrap();
    let store2 = Arc::new(StoreWAL::open(&p).unwrap());
    let m2 = BTreeMap::open(store2.clone(), rrr, LongFormat, LongFormat, 4).unwrap();
    assert_eq!(m2.size_long().unwrap(), n as u64);
    for k in 0..n {
        assert_eq!(m2.get(&k).unwrap(), Some(k * 10));
    }
    store2.close().unwrap();
    let _ = std::fs::remove_file(&p);
}

#[test]
fn repeated_rollback_cycles_advance_generation() {
    // Several uncommitted-grow → rollback cycles on the same open map: each
    // rollback bumps the store's structural_generation, so the next put resyncs
    // left_edges exactly once. The tree must stay correct across every cycle and
    // the final committed state must survive reopen.
    let p = tmp();
    let store = Arc::new(StoreWAL::open(&p).unwrap());
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 4).unwrap();
    store.commit().unwrap(); // baseline: empty

    for cycle in 0..4 {
        let n = 80i64 + cycle as i64 * 40; // varying height each cycle
        for k in 0..n {
            map.put(k, k).unwrap();
        }
        assert_eq!(
            map.size_long().unwrap(),
            n as u64,
            "cycle {cycle} pre-rollback"
        );
        store.rollback().unwrap();
        assert_eq!(map.size_long().unwrap(), 0, "cycle {cycle} post-rollback");
    }

    // Final committed write after all the reverts.
    let n = 300i64;
    for k in 0..n {
        map.put(k, k + 7).unwrap();
    }
    store.commit().unwrap();
    assert_eq!(map.size_long().unwrap(), n as u64);
    let rrr = map.root_recid_recid();
    drop(map);
    store.close().unwrap();

    let store2 = Arc::new(StoreWAL::open(&p).unwrap());
    let m2 = BTreeMap::open(store2.clone(), rrr, LongFormat, LongFormat, 4).unwrap();
    assert_eq!(m2.size_long().unwrap(), n as u64);
    for k in 0..n {
        assert_eq!(m2.get(&k).unwrap(), Some(k + 7));
    }
    store2.close().unwrap();
    let _ = std::fs::remove_file(&p);
}
