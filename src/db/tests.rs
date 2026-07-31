//! Ported DB-facade tests (adapted from the Java `db` test suite to the typed
//! Rust API). Untyped `db.get(name)` scenarios are re-expressed as typed reopen;
//! HTreeMap/hashMap/indexTree scenarios are out of scope (see PORTING-GAPS).

#![allow(clippy::bool_assert_comparison)]

use crate::db::atomic::STRING_NULLABLE;
use crate::db::bind;
use crate::db::catalog::{NameCatalog, CATALOG_SER, RECID_CATALOG};
use crate::db::{DBMaker, DB};
use crate::error::DbError;
use crate::listener::{FnListener, MapModificationListener};
use crate::ser::bytearray::ByteArrayFormat as ByteArrayFmt;
use crate::ser::columnar::{ColumnType, ColumnarValueFormat};
use crate::ser::families::CompressionSerializer;
use crate::ser::long::LongFormat;
use crate::ser::object_array::ObjectArrayFormat;
use crate::ser::serializers::{StringSer, LONG};
use crate::ser::string_group::StringGroupFormat;
use crate::ser::tuple::{TupleComponent, TupleFormat};
use crate::ser::value::Value;
use crate::ser::Serializer;
use crate::store::{Store, StoreByteArray};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// A fresh, not-yet-existing temp file path in the scratchpad.
fn fresh_file() -> std::path::PathBuf {
    let dir = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    let n = TMP_SEQ.fetch_add(1, Ordering::SeqCst);
    let mut p = std::path::PathBuf::from(dir);
    p.push(format!(
        "mapdb5-w4-{}-{}-{}.db",
        std::process::id(),
        n,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_file(&p);
    p
}

fn cleanup(p: &std::path::Path) {
    let _ = std::fs::remove_file(p);
    let mut ckpt = p.as_os_str().to_os_string();
    ckpt.push(".ckpt");
    let _ = std::fs::remove_file(std::path::PathBuf::from(ckpt));
}

fn sg() -> StringGroupFormat {
    StringGroupFormat
}

// ============================ DBSmokeTest ============================

#[test]
fn catalog_at_recid_1() {
    assert_eq!(RECID_CATALOG, 1);
    let db = DBMaker::memory_db().make().unwrap();
    assert!(db.get_name_catalog().unwrap().is_empty());
    db.close().unwrap();
}

#[test]
fn tree_map_round_trip() {
    let db = DBMaker::memory_db().make().unwrap();
    let m = db.tree_map("t", LongFormat, sg()).create().unwrap();
    for i in 0..100 {
        m.put(i, format!("v{i}")).unwrap();
    }
    assert_eq!(m.get(&50).unwrap(), Some("v50".to_string()));
    assert_eq!(m.first_entry().unwrap().map(|(k, _)| k), Some(0));
    assert_eq!(m.last_entry().unwrap().map(|(k, _)| k), Some(99));
    assert!(db.exists("t").unwrap());
    assert_eq!(db.get_type("t").unwrap().as_deref(), Some("TreeMap"));
    db.close().unwrap();
}

#[test]
fn tree_set_add_through_view() {
    let db = DBMaker::memory_db().make().unwrap();
    let ts = db.tree_set("ts", LongFormat).create().unwrap();
    assert!(ts.add(3).unwrap());
    assert!(ts.add(1).unwrap());
    assert!(ts.add(2).unwrap());
    assert!(!ts.add(2).unwrap());
    assert_eq!(ts.first().unwrap(), Some(1));
    assert_eq!(ts.last().unwrap(), Some(3));
    assert_eq!(ts.size_long().unwrap(), 3);
    assert!(ts.contains(&2).unwrap());
    assert_eq!(db.get_type("ts").unwrap().as_deref(), Some("TreeSet"));
    db.close().unwrap();
}

#[test]
fn tree_set_navigation_surface() {
    let db = DBMaker::memory_db().make().unwrap();
    let ts = db.tree_set("ts", LongFormat).create().unwrap();
    for k in [10, 20, 30, 40, 50] {
        assert!(ts.add(k).unwrap());
    }
    // Point navigation (Java lower/floor/ceiling/higher).
    assert_eq!(ts.lower(&30).unwrap(), Some(20));
    assert_eq!(ts.floor(&30).unwrap(), Some(30));
    assert_eq!(ts.floor(&35).unwrap(), Some(30));
    assert_eq!(ts.ceiling(&30).unwrap(), Some(30));
    assert_eq!(ts.ceiling(&35).unwrap(), Some(40));
    assert_eq!(ts.higher(&30).unwrap(), Some(40));
    assert_eq!(ts.lower(&10).unwrap(), None);
    assert_eq!(ts.higher(&50).unwrap(), None);
    // Bounded live views.
    assert_eq!(
        ts.sub_set(20, true, 40, false).to_vec().unwrap(),
        vec![20, 30]
    );
    assert_eq!(ts.head_set(30, true).to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(ts.tail_set(30, false).to_vec().unwrap(), vec![40, 50]);
    assert_eq!(ts.descending_to_vec().unwrap(), vec![50, 40, 30, 20, 10]);
    // A view is live: a backing add appears, and a view remove/poll writes through.
    let tail = ts.tail_set(30, true);
    assert_eq!(tail.to_vec().unwrap(), vec![30, 40, 50]);
    assert!(ts.add(45).unwrap()); // backing change visible in the existing view
    assert_eq!(tail.to_vec().unwrap(), vec![30, 40, 45, 50]);
    assert_eq!(tail.first().unwrap(), Some(30));
    assert_eq!(tail.ceiling(&41).unwrap(), Some(45));
    assert!(tail.remove(&45).unwrap()); // write-through remove
    assert!(!ts.contains(&45).unwrap());
    // A descending view reverses orientation and its poll_first takes the greatest.
    let desc = ts.descending_set();
    assert_eq!(desc.first().unwrap(), Some(50));
    assert_eq!(desc.poll_first().unwrap(), Some(50)); // removes 50 from the backing set
    assert!(!ts.contains(&50).unwrap());
    // Out-of-range add is not offered on a view; bounds are enforced on contains.
    assert!(!ts.head_set(25, false).contains(&30).unwrap());
    // Polling removes from the ends of the whole set.
    assert_eq!(ts.poll_first().unwrap(), Some(10));
    assert_eq!(ts.poll_last().unwrap(), Some(40));
    assert_eq!(ts.to_vec().unwrap(), vec![20, 30]);
    assert_eq!(ts.size_long().unwrap(), 2);
    db.close().unwrap();
}

#[test]
fn tree_set_nested_and_descending_views() {
    let db = DBMaker::memory_db().make().unwrap();
    let ts = db.tree_set("ts", LongFormat).create().unwrap();
    for k in [10, 20, 30, 40, 50, 60, 70] {
        assert!(ts.add(k).unwrap());
    }

    // ---- ascending nested views with inclusive/exclusive equal endpoints ----
    let sub = ts.sub_set(20, true, 60, true); // [20,30,40,50,60]
    assert_eq!(sub.to_vec().unwrap(), vec![20, 30, 40, 50, 60]);
    // Nested sub-view intersects the parent bounds; args in ascending orientation.
    assert_eq!(
        sub.sub_set(30, true, 50, false).to_vec().unwrap(),
        vec![30, 40]
    );
    assert_eq!(
        sub.sub_set(20, false, 60, false).to_vec().unwrap(),
        vec![30, 40, 50]
    );
    assert_eq!(sub.head_set(40, true).to_vec().unwrap(), vec![20, 30, 40]);
    assert_eq!(sub.head_set(40, false).to_vec().unwrap(), vec![20, 30]);
    assert_eq!(sub.tail_set(40, true).to_vec().unwrap(), vec![40, 50, 60]);
    assert_eq!(sub.tail_set(40, false).to_vec().unwrap(), vec![50, 60]);

    // ---- descending view: orientation-mapped navigation ----
    let desc = ts.descending_set();
    assert_eq!(desc.to_vec().unwrap(), vec![70, 60, 50, 40, 30, 20, 10]);
    assert_eq!(desc.first().unwrap(), Some(70));
    assert_eq!(desc.last().unwrap(), Some(10));
    // In descending order: lower=backing-higher, floor=backing-ceiling,
    // ceiling=backing-floor, higher=backing-lower.
    assert_eq!(desc.lower(&40).unwrap(), Some(50));
    assert_eq!(desc.floor(&40).unwrap(), Some(40));
    assert_eq!(desc.ceiling(&40).unwrap(), Some(40));
    assert_eq!(desc.higher(&40).unwrap(), Some(30));
    assert_eq!(desc.lower(&70).unwrap(), None); // nothing "before" the greatest
    assert_eq!(desc.higher(&10).unwrap(), None); // nothing "after" the least

    // ---- descending nested/bounded view with ordered traversal ----
    let dsub = desc.sub_set(60, true, 20, true); // args in descending order
    assert_eq!(dsub.to_vec().unwrap(), vec![60, 50, 40, 30, 20]);
    assert_eq!(dsub.first().unwrap(), Some(60));
    assert_eq!(dsub.head_set(40, true).to_vec().unwrap(), vec![60, 50, 40]);
    assert_eq!(dsub.tail_set(40, false).to_vec().unwrap(), vec![30, 20]);
    // Navigation on the bounded descending view honors the nested bounds.
    assert_eq!(dsub.lower(&40).unwrap(), Some(50));
    assert_eq!(dsub.floor(&40).unwrap(), Some(40));
    assert_eq!(dsub.ceiling(&40).unwrap(), Some(40));
    assert_eq!(dsub.higher(&40).unwrap(), Some(30));
    assert_eq!(dsub.lower(&60).unwrap(), None); // 70 is above the upper bound
    assert_eq!(dsub.higher(&20).unwrap(), None); // 10 is below the lower bound

    // ---- bounded polls honor the nested bounds (write-through) ----
    assert_eq!(dsub.poll_first().unwrap(), Some(60)); // greatest IN RANGE
    assert_eq!(dsub.poll_last().unwrap(), Some(20)); // least IN RANGE
    assert!(ts.contains(&70).unwrap()); // above dsub's range — untouched
    assert!(ts.contains(&10).unwrap()); // below dsub's range — untouched
    assert_eq!(ts.to_vec().unwrap(), vec![10, 30, 40, 50, 70]);
    assert_eq!(desc.first().unwrap(), Some(70)); // unbounded view still sees 70

    // ---- bounded clear write-through: only in-range elements disappear ----
    ts.sub_set(30, true, 50, true).clear().unwrap();
    assert!(!ts.contains(&30).unwrap());
    assert!(!ts.contains(&40).unwrap());
    assert!(!ts.contains(&50).unwrap());
    assert!(ts.contains(&10).unwrap()); // below the cleared range — retained
    assert!(ts.contains(&70).unwrap()); // above the cleared range — retained
    assert_eq!(ts.to_vec().unwrap(), vec![10, 70]);

    // ---- whole-set descending polls take the extremes ----
    assert_eq!(desc.poll_first().unwrap(), Some(70)); // greatest
    assert_eq!(desc.poll_last().unwrap(), Some(10)); // least
    assert!(desc.first().unwrap().is_none());
    assert!(ts.is_empty().unwrap());

    db.close().unwrap();
}

#[test]
fn atomics_all_kinds() {
    let db = DBMaker::memory_db().make().unwrap();
    let al = db.atomic_long_init("al", 10).create().unwrap();
    assert_eq!(al.get().unwrap(), 10);
    assert_eq!(al.increment_and_get().unwrap(), 11);
    assert!(al.compare_and_set(11, 20).unwrap());
    assert!(!al.compare_and_set(11, 30).unwrap());

    let ai = db.atomic_integer("ai").create().unwrap();
    assert_eq!(ai.get().unwrap(), 0);
    assert_eq!(ai.add_and_get(5).unwrap(), 5);

    let ab = db.atomic_boolean_init("ab", true).create().unwrap();
    assert!(ab.get().unwrap());
    ab.set(false).unwrap();
    assert!(!ab.get().unwrap());

    let as_ = db.atomic_string("as").create().unwrap();
    assert_eq!(as_.get().unwrap(), None);
    as_.set_str("hi").unwrap();
    assert_eq!(as_.get().unwrap(), Some("hi".to_string()));

    let av = db
        .atomic_var("av", StringSer, Some("init".to_string()))
        .create()
        .unwrap();
    assert_eq!(av.get().unwrap(), Some("init".to_string()));
    av.set_value(&"next".to_string()).unwrap();
    assert_eq!(av.get().unwrap(), Some("next".to_string()));
    db.close().unwrap();
}

#[test]
fn numeric_atomics_int_long_value() {
    let db = DBMaker::memory_db().make().unwrap();
    let v = db.atomic_long_init("long", 42).create().unwrap();
    let i = db.atomic_integer_init("int", 7).create().unwrap();
    assert_eq!(v.int_value().unwrap(), 42);
    assert_eq!(i.long_value().unwrap(), 7);
    db.compact().unwrap();
    db.close().unwrap();
}

#[test]
fn persist_across_reopen_wal() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let t = db.tree_map("t", LongFormat, sg()).create().unwrap();
        t.put(7, "seven".to_string()).unwrap();
        db.atomic_long_init("counter", 99).create().unwrap();
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        assert!(db.exists("t").unwrap());
        let t = db.tree_map("t", LongFormat, sg()).open().unwrap();
        assert_eq!(t.get(&7).unwrap(), Some("seven".to_string()));
        assert_eq!(db.atomic_long("counter").open().unwrap().get().unwrap(), 99);
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn create_or_open_returns_same_instance() {
    let db = DBMaker::memory_db().make().unwrap();
    let a = db.tree_map("t", LongFormat, sg()).create_or_open().unwrap();
    let b = db.tree_map("t", LongFormat, sg()).create_or_open().unwrap();
    assert!(a.shares_state_with(&b));
    db.close().unwrap();
}

#[test]
fn create_twice_fails() {
    let db = DBMaker::memory_db().make().unwrap();
    db.atomic_long("x").create().unwrap();
    assert!(matches!(
        db.atomic_long("x").create(),
        Err(DbError::WrongConfiguration(_))
    ));
    db.close().unwrap();
}

#[test]
fn type_mismatch_fails() {
    let db = DBMaker::memory_db().make().unwrap();
    db.atomic_long("x").create().unwrap();
    assert!(matches!(
        db.tree_map("x", LongFormat, sg()).open(),
        Err(DbError::WrongConfiguration(_))
    ));
    db.close().unwrap();
}

#[test]
fn rename_and_delete() {
    let db = DBMaker::memory_db().make().unwrap();
    let m = db.tree_map("old", LongFormat, sg()).create().unwrap();
    m.put(1, "a".to_string()).unwrap();
    db.rename("old", "new").unwrap();
    assert!(!db.exists("old").unwrap());
    assert!(db.exists("new").unwrap());
    let m2 = db.tree_map("new", LongFormat, sg()).open().unwrap();
    assert_eq!(m2.get(&1).unwrap(), Some("a".to_string()));

    assert!(db.delete("new").unwrap());
    assert!(!db.exists("new").unwrap());
    assert!(!db.delete("new").unwrap());
    db.close().unwrap();
}

#[test]
fn rollback_clears_cache_wal() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        db.atomic_long_init("committed", 1).create().unwrap();
        db.commit().unwrap();
        db.atomic_long_init("uncommitted", 2).create().unwrap();
        assert!(db.exists("uncommitted").unwrap());
        db.rollback().unwrap();
        assert!(!db.exists("uncommitted").unwrap());
        assert!(db.exists("committed").unwrap());
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn rollback_unsupported_on_non_tx() {
    let db = DBMaker::memory_db().make().unwrap();
    assert!(matches!(db.rollback(), Err(DbError::Unsupported(_))));
    db.close().unwrap();
}

#[test]
fn bad_name_rejected() {
    let db = DBMaker::memory_db().make().unwrap();
    assert!(matches!(
        db.atomic_long("bad#name").create(),
        Err(DbError::WrongConfiguration(_))
    ));
    db.close().unwrap();
}

#[test]
fn transaction_enable_rejected_for_memory() {
    assert!(matches!(
        DBMaker::memory_db().transaction_enable().make(),
        Err(DbError::WrongConfiguration(_))
    ));
}

#[test]
fn get_all_names() {
    let db = DBMaker::memory_db().make().unwrap();
    db.atomic_long("a").create().unwrap();
    db.atomic_long("b").create().unwrap();
    assert_eq!(db.get_all_names().unwrap().len(), 2);
    db.close().unwrap();
}

#[test]
fn close_is_idempotent() {
    let db = DBMaker::memory_db().make().unwrap();
    db.close().unwrap();
    db.close().unwrap();
    assert!(db.is_closed());
    assert!(matches!(
        db.atomic_long("x").create(),
        Err(DbError::StoreClosed)
    ));
}

// ============================ DBEdgeCaseTest ============================

#[test]
fn fresh_stores_initialize_with_empty_catalog() {
    let a = DBMaker::memory_db().make().unwrap();
    assert!(a.get_name_catalog().unwrap().is_empty());
    assert!(a.get_all_names().unwrap().is_empty());
    a.close().unwrap();
    let b = DBMaker::heap_db().make().unwrap();
    assert!(b.get_name_catalog().unwrap().is_empty());
    b.close().unwrap();
    let c = DBMaker::memory_byte_array_db().make().unwrap();
    assert!(c.get_name_catalog().unwrap().is_empty());
    c.close().unwrap();
    let d = DBMaker::memory_direct_db().make().unwrap();
    assert!(d.get_name_catalog().unwrap().is_empty());
    d.close().unwrap();
}

#[test]
fn reopen_store_with_valid_catalog_works() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        db.atomic_long_init("counter", 7).create().unwrap();
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        assert!(db.exists("counter").unwrap());
        assert_eq!(db.atomic_long("counter").open().unwrap().get().unwrap(), 7);
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn db_over_polluted_store_recid1_fails_cleanly() {
    // A foreign writer occupies recid 1 with a non-catalog record.
    let raw = Arc::new(StoreByteArray::new(true));
    let recid = raw.put(&123456789i64, &LONG).unwrap();
    assert_eq!(recid.get(), RECID_CATALOG);
    let res = DB::new(raw);
    assert!(matches!(
        res.err(),
        Some(DbError::WrongConfiguration(_)) | Some(DbError::DataCorruption(_))
    ));
}

#[test]
fn create_open_semantics_across_reopen() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        assert!(matches!(
            db.atomic_long("x").open(),
            Err(DbError::WrongConfiguration(_))
        ));
        db.atomic_long_init("x", 1).create_or_open().unwrap(); // creates
        assert!(matches!(
            db.atomic_long("x").create(),
            Err(DbError::WrongConfiguration(_))
        ));
        assert_eq!(
            db.atomic_long("x").create_or_open().unwrap().get().unwrap(),
            1
        );
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        assert!(matches!(
            db.atomic_long("x").create(),
            Err(DbError::WrongConfiguration(_))
        ));
        assert_eq!(db.atomic_long("x").open().unwrap().get().unwrap(), 1);
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn delete_frees_atomic_and_allows_recreate() {
    let db = DBMaker::memory_db().make().unwrap();
    db.atomic_long_init("c", 5).create().unwrap();
    assert!(db.exists("c").unwrap());
    assert!(db.delete("c").unwrap());
    assert!(!db.exists("c").unwrap());
    assert_eq!(db.get_type("c").unwrap(), None);
    assert!(!db.delete("c").unwrap());
    let recreated = db.atomic_long_init("c", 42).create().unwrap();
    assert_eq!(recreated.get().unwrap(), 42);
    db.close().unwrap();
}

#[test]
fn rename_onto_existing_and_nonexistent_throw() {
    let db = DBMaker::memory_db().make().unwrap();
    db.atomic_long("a").create().unwrap();
    db.atomic_long("b").create().unwrap();
    assert!(matches!(
        db.rename("a", "b"),
        Err(DbError::WrongConfiguration(_))
    ));
    assert!(db.exists("a").unwrap() && db.exists("b").unwrap());
    assert!(matches!(
        db.rename("nope", "x"),
        Err(DbError::WrongConfiguration(_))
    ));
    db.close().unwrap();
}

#[test]
fn many_named_atomics_survive_reopen() {
    let f = fresh_file();
    let n = 200;
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        for i in 0..n {
            db.atomic_long_init(&format!("al{i}"), i as i64)
                .create()
                .unwrap();
        }
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        assert_eq!(db.get_all_names().unwrap().len(), n);
        for i in 0..n {
            assert_eq!(
                db.atomic_long(&format!("al{i}"))
                    .open()
                    .unwrap()
                    .get()
                    .unwrap(),
                i as i64
            );
        }
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn tree_map_byte_array_keys_and_values_round_trip() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let m = db
            .tree_map("b", ByteArrayFmt, ByteArrayFmt)
            .create()
            .unwrap();
        for i in 0..200u32 {
            m.put(vec![(i >> 8) as u8, i as u8], vec![i as u8, 99])
                .unwrap();
        }
        assert_eq!(m.get(&vec![0, 50]).unwrap(), Some(vec![50, 99]));
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let m = db.tree_map("b", ByteArrayFmt, ByteArrayFmt).open().unwrap();
        for i in 0..200u32 {
            assert_eq!(
                m.get(&vec![(i >> 8) as u8, i as u8]).unwrap(),
                Some(vec![i as u8, 99])
            );
        }
        assert_eq!(m.first_entry().unwrap().map(|(k, _)| k), Some(vec![0, 0]));
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn tree_set_deep_splits_survive_reopen() {
    let f = fresh_file();
    let n: i64 = 600;
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let ts = db
            .tree_set("ts", LongFormat)
            .max_node_size(8)
            .create()
            .unwrap();
        for i in 0..n {
            ts.add((i * 37) % n).unwrap();
        }
        assert_eq!(ts.size_long().unwrap(), n as u64);
        assert!(!ts.add(0).unwrap());
        assert!(ts.add(n).unwrap());
        ts.remove(&n).unwrap();
        assert_eq!(ts.first().unwrap(), Some(0));
        assert_eq!(ts.last().unwrap(), Some(n - 1));
        assert!(ts.remove(&123).unwrap());
        assert!(!ts.contains(&123).unwrap());
        assert_eq!(ts.size_long().unwrap(), (n - 1) as u64);
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let ts = db.tree_set("ts", LongFormat).open().unwrap();
        assert_eq!(ts.size_long().unwrap(), (n - 1) as u64);
        let v = ts.to_vec().unwrap();
        let mut prev = i64::MIN;
        for x in v {
            assert!(x > prev);
            prev = x;
        }
        db.close().unwrap();
    }
    cleanup(&f);
}

// ============================ DBReadOnlyMakerTest ============================

#[test]
fn read_only_reopen_reads_but_rejects_writes() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).make().unwrap();
        let m = db.tree_map("m", LongFormat, sg()).create().unwrap();
        m.put(1, "one".to_string()).unwrap();
        m.put(2, "two".to_string()).unwrap();
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let ro = DBMaker::file_db(&f).read_only().make().unwrap();
        let rm = ro.tree_map("m", LongFormat, sg()).open().unwrap();
        assert_eq!(rm.get(&1).unwrap(), Some("one".to_string()));
        assert_eq!(rm.get(&2).unwrap(), Some("two".to_string()));
        assert!(matches!(
            rm.put(3, "three".to_string()),
            Err(DbError::ReadOnly)
        ));
        ro.commit().unwrap(); // harmless no-op
        ro.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn read_only_plus_transaction_enable_rejected() {
    let f = fresh_file();
    assert!(matches!(
        DBMaker::file_db(&f).read_only().transaction_enable().make(),
        Err(DbError::WrongConfiguration(_))
    ));
    assert!(matches!(
        DBMaker::file_db(&f).transaction_enable().read_only().make(),
        Err(DbError::WrongConfiguration(_))
    ));
    cleanup(&f);
}

#[test]
fn read_only_on_empty_store_fails_cleanly() {
    let f = fresh_file();
    let err = DBMaker::file_db(&f).read_only().make().err().unwrap();
    assert!(matches!(err, DbError::WrongConfiguration(m) if m.to_lowercase().contains("read")));
    cleanup(&f);
}

#[test]
fn read_only_on_empty_memory_store_fails_cleanly() {
    let err = DBMaker::memory_db().read_only().make().err().unwrap();
    assert!(matches!(err, DbError::WrongConfiguration(m) if m.to_lowercase().contains("read")));
}

#[test]
fn file_mmap_enable_is_harmless_noop() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f)
            .file_mmap_enable()
            .file_mmap_enable_if_supported()
            .make()
            .unwrap();
        db.tree_map("m", LongFormat, sg())
            .create()
            .unwrap()
            .put(1, "x".to_string())
            .unwrap();
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).file_mmap_enable().make().unwrap();
        assert_eq!(
            db.tree_map("m", LongFormat, sg())
                .open()
                .unwrap()
                .get(&1)
                .unwrap(),
            Some("x".to_string())
        );
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn file_delete_after_open_removes_file_but_db_still_works() {
    let f = fresh_file();
    let db = DBMaker::file_db(&f)
        .file_delete_after_open()
        .make()
        .unwrap();
    assert!(!f.exists(), "backing file should be gone right after open");
    let m = db.tree_map("m", LongFormat, sg()).create().unwrap();
    m.put(1, "alive".to_string()).unwrap();
    db.commit().unwrap();
    assert_eq!(m.get(&1).unwrap(), Some("alive".to_string()));
    db.close().unwrap();
    assert!(!f.exists());
    cleanup(&f);
}

#[test]
fn file_delete_after_open_requires_file_db() {
    assert!(matches!(
        DBMaker::memory_db().file_delete_after_open().make(),
        Err(DbError::WrongConfiguration(_))
    ));
}

#[test]
fn file_delete_after_open_rejects_wal() {
    let f = fresh_file();
    assert!(matches!(
        DBMaker::file_db(&f)
            .transaction_enable()
            .file_delete_after_open()
            .make(),
        Err(DbError::WrongConfiguration(_))
    ));
    cleanup(&f);
}

#[test]
fn legacy_maker_aliases_compose() {
    let db = DBMaker::memory_db()
        .executor_enable()
        .cleaner_hack_enable()
        .file_channel_enable()
        .checksum_store_enable()
        .close_on_jvm_shutdown()
        .close_on_jvm_shutdown_weak_reference()
        .make()
        .unwrap();
    db.close().unwrap();
}

// ============================ DBQueueTest ============================

#[test]
fn queue_families_persist_and_dispatch() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        db.queue("fifo", StringSer)
            .create()
            .unwrap()
            .add("first".to_string())
            .unwrap();
        db.stack("stack", StringSer)
            .create()
            .unwrap()
            .add("top".to_string())
            .unwrap();
        let circular = db
            .circular_queue("circular", StringSer, 2)
            .create()
            .unwrap();
        circular.add("a".to_string()).unwrap();
        circular.add("b".to_string()).unwrap();
        circular.add("c".to_string()).unwrap();
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let fifo = db.queue("fifo", StringSer).open().unwrap();
        assert_eq!(fifo.poll().unwrap(), Some("first".to_string()));
        let stack = db.stack("stack", StringSer).open().unwrap();
        assert_eq!(stack.poll().unwrap(), Some("top".to_string()));
        let circular = db.circular_queue("circular", StringSer, 2).open().unwrap();
        assert_eq!(circular.poll().unwrap(), Some("b".to_string()));
        assert_eq!(circular.poll().unwrap(), Some("c".to_string()));
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn db_close_wakes_blocked_consumer() {
    let db = DBMaker::memory_db().make().unwrap();
    let queue = db.queue("queue", StringSer).create().unwrap();
    let q2 = Arc::clone(&queue);
    let handle = std::thread::spawn(move || q2.take());
    std::thread::sleep(std::time::Duration::from_millis(100));
    db.close().unwrap();
    let res = handle.join().unwrap();
    assert!(matches!(res, Err(DbError::StoreClosed)));
}

// ============================ DBTreeCounterListenerTest ============================

#[test]
fn counter_tracks_insert_remove_replace_and_clear() {
    let db = DBMaker::memory_db().make().unwrap();
    let m = db
        .tree_map("c", LongFormat, sg())
        .counter_enable()
        .create()
        .unwrap();
    assert!(m.counter_recid() > 0);
    for i in 0..100 {
        assert_eq!(m.put(i, format!("v{i}")).unwrap(), None);
        assert_eq!(m.size_long().unwrap(), (i + 1) as u64);
    }
    assert_eq!(
        m.put(0, "changed".to_string()).unwrap(),
        Some("v0".to_string())
    );
    assert_eq!(m.size_long().unwrap(), 100);
    assert_eq!(
        m.replace(&0, "again".to_string()).unwrap(),
        Some("changed".to_string())
    );
    assert_eq!(m.size_long().unwrap(), 100);
    assert_eq!(m.replace(&1000, "x".to_string()).unwrap(), None);
    assert_eq!(m.size_long().unwrap(), 100);
    for i in 0..40 {
        assert!(m.remove(&i).unwrap().is_some());
        assert_eq!(m.size_long().unwrap(), (100 - (i + 1)) as u64);
    }
    assert_eq!(m.remove(&0).unwrap(), None);
    assert_eq!(m.size_long().unwrap(), 60);
    m.clear().unwrap();
    assert_eq!(m.size_long().unwrap(), 0);
    db.close().unwrap();
}

#[test]
fn counter_persists_across_reopen() {
    let f = fresh_file();
    let stored;
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let m = db
            .tree_map("t", LongFormat, sg())
            .counter_enable()
            .create()
            .unwrap();
        for i in 0..500 {
            m.put(i, format!("v{i}")).unwrap();
        }
        assert_eq!(m.size_long().unwrap(), 500);
        stored = db
            .get_name_catalog()
            .unwrap()
            .get("t#counterRecid")
            .unwrap()
            .clone();
        assert!(stored.parse::<u64>().unwrap() > 0);
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        assert_eq!(
            db.get_name_catalog().unwrap().get("t#counterRecid"),
            Some(&stored)
        );
        let re = db.tree_map("t", LongFormat, sg()).open().unwrap();
        assert_eq!(re.counter_recid().to_string(), stored);
        assert_eq!(re.size_long().unwrap(), 500);
        for i in 500..600 {
            re.put(i, format!("v{i}")).unwrap();
        }
        assert_eq!(re.size_long().unwrap(), 600);
        db.commit().unwrap();
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn plain_tree_map_reopens_without_counter() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let m = db.tree_map("p", LongFormat, sg()).create().unwrap();
        for i in 0..300 {
            m.put(i, format!("v{i}")).unwrap();
        }
        assert_eq!(m.counter_recid(), 0);
        assert_eq!(
            db.get_name_catalog()
                .unwrap()
                .get("p#counterRecid")
                .map(String::as_str),
            Some("0")
        );
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let m = db.tree_map("p", LongFormat, sg()).open().unwrap();
        assert_eq!(m.counter_recid(), 0);
        assert_eq!(m.size_long().unwrap(), 300);
        assert_eq!(m.get(&42).unwrap(), Some("v42".to_string()));
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn modification_listener_captures_insert_update_remove() {
    let db = DBMaker::memory_db().make().unwrap();
    #[allow(clippy::type_complexity)]
    let events: Arc<Mutex<Vec<(i64, Option<String>, Option<String>)>>> =
        Arc::new(Mutex::new(Vec::new()));
    let ev = events.clone();
    let listener: Arc<dyn MapModificationListener<i64, String>> = Arc::new(FnListener(
        move |k: &i64, o: Option<&String>, n: Option<&String>, _t| {
            ev.lock().unwrap().push((*k, o.cloned(), n.cloned()));
            Ok(())
        },
    ));
    let m = db
        .tree_map("m", LongFormat, sg())
        .modification_listener(listener)
        .create()
        .unwrap();
    m.put(5, "fifty".to_string()).unwrap();
    m.put(5, "sixty".to_string()).unwrap();
    assert_eq!(m.remove(&5).unwrap(), Some("sixty".to_string()));
    let e = events.lock().unwrap();
    assert_eq!(e.len(), 3);
    assert_eq!(e[0], (5, None, Some("fifty".to_string())));
    assert_eq!(
        e[1],
        (5, Some("fifty".to_string()), Some("sixty".to_string()))
    );
    assert_eq!(e[2], (5, Some("sixty".to_string()), None));
    db.close().unwrap();
}

#[test]
fn listener_is_applied_once_to_cached_handle() {
    let db = DBMaker::memory_db().make().unwrap();
    let created = db.tree_map("t", LongFormat, sg()).create().unwrap();
    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let ev = events.clone();
    let listener: Arc<dyn MapModificationListener<i64, String>> = Arc::new(FnListener(
        move |k: &i64, o: Option<&String>, n: Option<&String>, _t| {
            ev.lock().unwrap().push(format!("{k}:{o:?}:{n:?}"));
            Ok(())
        },
    ));
    let a = db
        .tree_map("t", LongFormat, sg())
        .modification_listener(listener.clone())
        .open()
        .unwrap();
    let b = db
        .tree_map("t", LongFormat, sg())
        .modification_listener(listener.clone())
        .open()
        .unwrap();
    assert!(created.shares_state_with(&a));
    assert!(created.shares_state_with(&b));
    created.put(1, "one".to_string()).unwrap();
    // registered once despite two opens (Arc-identity dedup)
    assert_eq!(events.lock().unwrap().len(), 1);
    db.close().unwrap();
}

#[test]
fn tree_map_create_from_iterator_with_counter() {
    let db = DBMaker::memory_db().make().unwrap();
    let n = 2000i64;
    let entries: Vec<(i64, String)> = (0..n).map(|i| (i, format!("v{i}"))).collect();
    let m = db
        .tree_map("bulk", LongFormat, sg())
        .counter_enable()
        .create_from(entries)
        .unwrap();
    assert!(m.counter_recid() > 0);
    assert_eq!(m.size_long().unwrap(), n as u64);
    assert_eq!(m.get(&0).unwrap(), Some("v0".to_string()));
    assert_eq!(m.get(&1999).unwrap(), Some("v1999".to_string()));
    assert_eq!(db.get_type("bulk").unwrap().as_deref(), Some("TreeMap"));
    assert_eq!(m.put(n, "new".to_string()).unwrap(), None);
    assert_eq!(m.size_long().unwrap(), (n + 1) as u64);
    db.close().unwrap();
}

#[test]
fn create_from_existing_name_and_unsorted_throw() {
    let db = DBMaker::memory_db().make().unwrap();
    db.tree_map("dup", LongFormat, sg()).create().unwrap();
    assert!(matches!(
        db.tree_map("dup", LongFormat, sg())
            .create_from(vec![(1i64, "a".to_string())]),
        Err(DbError::WrongConfiguration(_))
    ));
    let bad = vec![
        (3i64, "c".to_string()),
        (1, "a".to_string()),
        (2, "b".to_string()),
    ];
    assert!(matches!(
        db.tree_map("us", LongFormat, sg()).create_from(bad),
        Err(DbError::NotSorted)
    ));
    assert!(!db.exists("us").unwrap());
    db.close().unwrap();
}

#[test]
fn tree_set_counter_and_create_from() {
    let db = DBMaker::memory_db().make().unwrap();
    let src: Vec<i64> = (0..1000).map(|i| i * 2).collect();
    let s = db
        .tree_set("bulkset", LongFormat)
        .counter_enable()
        .create_from(src)
        .unwrap();
    assert_eq!(s.size_long().unwrap(), 1000);
    assert_eq!(s.first().unwrap(), Some(0));
    assert_eq!(s.last().unwrap(), Some(1998));
    assert!(s.contains(&500).unwrap());
    assert!(!s.contains(&501).unwrap());
    assert_eq!(db.get_type("bulkset").unwrap().as_deref(), Some("TreeSet"));

    let bad = vec![5i64, 2];
    assert!(matches!(
        db.tree_set("badset", LongFormat).create_from(bad),
        Err(DbError::NotSorted)
    ));
    assert!(!db.exists("badset").unwrap());
    db.close().unwrap();
}

// ============================ DBTreeExternalValuesTest ============================

#[test]
fn external_values_catalog_persistence() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let m = db
            .tree_map("tree", LongFormat, sg())
            .values_outside_nodes_enable()
            .counter_enable()
            .create()
            .unwrap();
        m.put(1, "one".to_string()).unwrap();
        m.put(2, "two".to_string()).unwrap();
        assert!(!m.value_inline());
        assert_eq!(
            db.get_name_catalog()
                .unwrap()
                .get("tree#valueInline")
                .map(String::as_str),
            Some("false")
        );
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let m = db.tree_map("tree", LongFormat, sg()).open().unwrap();
        assert!(!m.value_inline());
        assert_eq!(m.get(&1).unwrap(), Some("one".to_string()));
        assert_eq!(m.size_long().unwrap(), 2);
        db.close().unwrap();
    }
    cleanup(&f);
}

// ============================ DBParameterizedCatalogTest ============================

#[test]
fn parameterized_codecs_reopen_without_resupply() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let tuple = TupleFormat::of(&[
            TupleComponent::Str,
            TupleComponent::Long,
            TupleComponent::Int,
        ]);
        let tuples = db.tree_map("tuples", tuple, sg()).create().unwrap();
        tuples
            .put(
                vec![Value::Str("tenant".into()), Value::Long(7), Value::Int(1)],
                "one".to_string(),
            )
            .unwrap();
        tuples
            .put(
                vec![Value::Str("tenant".into()), Value::Long(7), Value::Int(2)],
                "two".to_string(),
            )
            .unwrap();

        let compressed = ObjectArrayFormat::new(CompressionSerializer::with_level(StringSer, 6));
        let cv = db
            .tree_map("compressed", LongFormat, compressed)
            .create()
            .unwrap();
        cv.put(1, "abcdefghij".repeat(100)).unwrap();

        let columnar = ColumnarValueFormat::of(&[ColumnType::Long, ColumnType::Int]);
        let cm = db
            .tree_map("columnar", LongFormat, columnar)
            .create()
            .unwrap();
        cm.put(1, vec![Value::Long(9), Value::Int(4)]).unwrap();
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let tuple = TupleFormat::of(&[
            TupleComponent::Str,
            TupleComponent::Long,
            TupleComponent::Int,
        ]);
        let tuples = db.tree_map("tuples", tuple, sg()).open().unwrap();
        assert_eq!(
            tuples
                .get(&vec![
                    Value::Str("tenant".into()),
                    Value::Long(7),
                    Value::Int(2)
                ])
                .unwrap(),
            Some("two".to_string())
        );
        let compressed = ObjectArrayFormat::new(CompressionSerializer::with_level(StringSer, 6));
        let cv = db
            .tree_map("compressed", LongFormat, compressed)
            .open()
            .unwrap();
        assert_eq!(cv.get(&1).unwrap(), Some("abcdefghij".repeat(100)));
        let columnar = ColumnarValueFormat::of(&[ColumnType::Long, ColumnType::Int]);
        let cm = db
            .tree_map("columnar", LongFormat, columnar)
            .open()
            .unwrap();
        assert_eq!(
            cm.get(&1).unwrap(),
            Some(vec![Value::Long(9), Value::Int(4)])
        );
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn wrong_descriptor_on_reopen_is_rejected() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        db.tree_map("t", LongFormat, sg()).create().unwrap();
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        // wrong value format (Long instead of String) must be rejected
        assert!(matches!(
            db.tree_map("t", LongFormat, LongFormat).open(),
            Err(DbError::WrongConfiguration(_))
        ));
        db.close().unwrap();
    }
    cleanup(&f);
}

// ============================ BindTest ============================

#[test]
fn bind_secondary_indexes_histogram_and_delete_sink() {
    let db = DBMaker::memory_db().make().unwrap();
    let primary = db.tree_map("p", LongFormat, sg()).create().unwrap();

    let lengths: bind::SecondaryMap<i64, usize> = bind::SecondaryMap::new();
    bind::secondary_value(&primary, lengths.clone(), |_k, v: &String| v.len()).unwrap();
    primary.put(1, "red,round".to_string()).unwrap();
    assert_eq!(lengths.get(&1), Some(9));

    let inverse: bind::SecondaryMap<String, i64> = bind::SecondaryMap::new();
    bind::map_inverse(&primary, inverse.clone()).unwrap();
    let keys: bind::SecondarySet<(String, i64)> = bind::SecondarySet::new();
    bind::secondary_keys(&primary, keys.clone(), |_k, v: &String| {
        v.split(',').map(|s| s.to_string()).collect()
    })
    .unwrap();
    let histogram: bind::SecondaryMap<char, i64> = bind::SecondaryMap::new();
    bind::histogram(&primary, histogram.clone(), |_k, v: &String| {
        v.chars().next().unwrap()
    })
    .unwrap();
    let deleted: bind::SecondaryMap<i64, String> = bind::SecondaryMap::new();
    bind::map_put_after_delete(&primary, deleted.clone());

    primary.put(2, "blue,square".to_string()).unwrap();
    primary.put(1, "red,square".to_string()).unwrap();
    assert_eq!(inverse.get(&"red,square".to_string()), Some(1));
    assert!(!inverse.contains_key(&"red,round".to_string()));
    assert!(keys.contains(&("square".to_string(), 1)));
    assert!(!keys.contains(&("round".to_string(), 1)));
    assert_eq!(histogram.get(&'r'), Some(1));
    assert_eq!(histogram.get(&'b'), Some(1));

    primary.remove(&2).unwrap();
    assert_eq!(deleted.get(&2), Some("blue,square".to_string()));
    assert!(!histogram.contains_key(&'b'));
    db.close().unwrap();
}

#[test]
fn bind_size_and_self_bind_rejection() {
    let db = DBMaker::memory_db().make().unwrap();
    let primary = db.tree_map("p", LongFormat, sg()).create().unwrap();
    primary.put(1, "one".to_string()).unwrap();
    let counter_recid = db.store().put(&0i64, &LONG).unwrap();
    let counter = crate::db::AtomicLong::new(Arc::clone(db.store()), counter_recid);
    bind::size(&primary, &counter).unwrap();
    assert_eq!(counter.get().unwrap(), 1);
    primary.put(2, "two".to_string()).unwrap();
    primary.put(2, "two updated".to_string()).unwrap();
    primary.remove(&1).unwrap();
    assert_eq!(counter.get().unwrap(), 1);

    // Direct self-binding is rejected.
    let clone = primary.clone();
    assert!(bind::reject_self_bind(&primary, &clone).is_err());
    db.close().unwrap();
}

#[test]
fn bind_unique_secondary_rejects_duplicate_derived_keys() {
    let db = DBMaker::memory_db().make().unwrap();
    let primary = db.tree_map("p", LongFormat, sg()).create().unwrap();
    let unique: bind::SecondaryMap<usize, i64> = bind::SecondaryMap::new();
    bind::secondary_key(&primary, unique, |_k, v: &String| v.len()).unwrap();
    primary.put(1, "same".to_string()).unwrap();
    // "size" also has length 4 → duplicate derived key → listener error propagates.
    assert!(primary.put(2, "size".to_string()).is_err());
    db.close().unwrap();
}

#[test]
fn bind_concurrent_same_key_writers_keep_secondary_consistent() {
    let db = DBMaker::memory_db().make().unwrap();
    let primary = db.tree_map("p", LongFormat, sg()).create().unwrap();
    let secondary: bind::SecondaryMap<i64, String> = bind::SecondaryMap::new();
    bind::secondary_value(&primary, secondary.clone(), |_k, v: &String| {
        format!("d:{v}")
    })
    .unwrap();

    let key_count = 4i64;
    let threads = 8;
    let iters = 2000;
    let barrier = Arc::new(std::sync::Barrier::new(threads));
    let mut handles = Vec::new();
    for t in 0..threads {
        let p = primary.clone();
        let b = barrier.clone();
        handles.push(std::thread::spawn(move || {
            b.wait();
            for i in 0..iters {
                let key = (i as i64) % key_count;
                p.put(key, format!("t{t}-i{i}")).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    for key in 0..key_count {
        let pv = primary.get(&key).unwrap();
        assert_eq!(secondary.get(&key), pv.map(|v| format!("d:{v}")));
    }
    db.close().unwrap();
}

// ============================ golden catalog bytes (§7) ============================

fn recid1_bytes(store: &StoreByteArray) -> Vec<u8> {
    let cat = store
        .get(NonZeroU64::new(RECID_CATALOG).unwrap(), &CATALOG_SER)
        .unwrap()
        .unwrap();
    let mut out = crate::io::DataOutput2::new();
    CATALOG_SER.serialize(&mut out, &cat);
    out.into_vec()
}

fn expect_catalog(pairs: &[(&str, &str)]) -> NameCatalog {
    let mut c = NameCatalog::new();
    for (k, v) in pairs {
        c.insert(k.to_string(), v.to_string());
    }
    c
}

/// Assert the DB catalog contains exactly the expected rows and that the on-store
/// recid-1 bytes decode back to them (MDBC byte-parity per §7 row).
fn assert_catalog_row(db: &DB<StoreByteArray>, expected: &[(&str, &str)]) {
    let cat = db.get_name_catalog().unwrap();
    let exp = expect_catalog(expected);
    assert_eq!(cat, exp, "catalog content mismatch");
    let bytes = recid1_bytes(db.store());
    let mut input = crate::io::SliceInput::new(&bytes);
    let decoded = CATALOG_SER
        .deserialize(&mut input, Some(bytes.len()))
        .unwrap();
    assert_eq!(decoded, exp, "recid-1 bytes decode mismatch");
}

#[test]
fn golden_atomic_long_row() {
    let db = DB::make_byte_array().unwrap();
    let a = db.atomic_long_init("al", 5).create().unwrap();
    let recid = a.recid().get();
    assert_catalog_row(
        &db,
        &[("al#type", "AtomicLong"), ("al#recid", &recid.to_string())],
    );
    db.close().unwrap();
}

#[test]
fn golden_atomic_integer_boolean_string_rows() {
    let db = DB::make_byte_array().unwrap();
    let ai = db.atomic_integer_init("ai", 1).create().unwrap();
    let ab = db.atomic_boolean_init("ab", true).create().unwrap();
    let as_ = db.atomic_string("as").create().unwrap();
    assert_catalog_row(
        &db,
        &[
            ("ai#type", "AtomicInteger"),
            ("ai#recid", &ai.recid().get().to_string()),
            ("ab#type", "AtomicBoolean"),
            ("ab#recid", &ab.recid().get().to_string()),
            ("as#type", "AtomicString"),
            ("as#recid", &as_.recid().get().to_string()),
        ],
    );
    db.close().unwrap();
}

#[test]
fn golden_atomic_var_row() {
    let db = DB::make_byte_array().unwrap();
    let av = db
        .atomic_var("av", StringSer, Some("x".to_string()))
        .create()
        .unwrap();
    assert_catalog_row(
        &db,
        &[
            ("av#type", "AtomicVar"),
            ("av#recid", &av.recid().get().to_string()),
            ("av#serializer", "STRING"),
        ],
    );
    db.close().unwrap();
}

#[test]
fn golden_tree_map_row() {
    let db = DB::make_byte_array().unwrap();
    let m = db
        .tree_map("t", LongFormat, sg())
        .max_node_size(32)
        .create()
        .unwrap();
    assert_catalog_row(
        &db,
        &[
            ("t#type", "TreeMap"),
            ("t#keySerializer", "LONG"),
            ("t#valueSerializer", "STRING"),
            ("t#rootRecidRecid", &m.root_recid_recid().to_string()),
            ("t#maxNodeSize", "32"),
            ("t#counterRecid", "0"),
            ("t#valueInline", "true"),
        ],
    );
    db.close().unwrap();
}

#[test]
fn golden_tree_map_counter_and_external_rows() {
    let db = DB::make_byte_array().unwrap();
    let m = db
        .tree_map("t", LongFormat, sg())
        .max_node_size(16)
        .values_outside_nodes_enable()
        .counter_enable()
        .create()
        .unwrap();
    assert_catalog_row(
        &db,
        &[
            ("t#type", "TreeMap"),
            ("t#keySerializer", "LONG"),
            ("t#valueSerializer", "STRING"),
            ("t#rootRecidRecid", &m.root_recid_recid().to_string()),
            ("t#maxNodeSize", "16"),
            ("t#counterRecid", &m.counter_recid().to_string()),
            ("t#valueInline", "false"),
        ],
    );
    db.close().unwrap();
}

#[test]
fn golden_tree_set_row() {
    let db = DB::make_byte_array().unwrap();
    let s = db
        .tree_set("s", LongFormat)
        .max_node_size(32)
        .create()
        .unwrap();
    assert_catalog_row(
        &db,
        &[
            ("s#type", "TreeSet"),
            ("s#serializer", "LONG"),
            ("s#rootRecidRecid", &s.root_recid_recid().to_string()),
            ("s#maxNodeSize", "32"),
            ("s#counterRecid", "0"),
        ],
    );
    db.close().unwrap();
}

#[test]
fn golden_queue_stack_circular_rows() {
    let db = DB::make_byte_array().unwrap();
    let q = db.queue("fifo", StringSer).create().unwrap();
    let s = db.stack("lifo", StringSer).create().unwrap();
    let c = db.circular_queue("circ", StringSer, 4).create().unwrap();
    assert_catalog_row(
        &db,
        &[
            ("fifo#type", "Queue"),
            ("fifo#headerRecid", &q.header_recid().get().to_string()),
            ("fifo#serializer", "STRING"),
            ("lifo#type", "Stack"),
            ("lifo#headerRecid", &s.header_recid().get().to_string()),
            ("lifo#serializer", "STRING"),
            ("circ#type", "CircularQueue"),
            ("circ#headerRecid", &c.header_recid().get().to_string()),
            ("circ#serializer", "STRING"),
        ],
    );
    db.close().unwrap();
}

// ============================ legacy defaults (§7) ============================

#[test]
fn legacy_absent_counter_and_value_inline_defaults() {
    use crate::btree::BTreeMap;
    let store = Arc::new(StoreByteArray::new(true));
    // 1. Reserve recid 1 for the catalog (empty for now).
    let recid = store.put(&NameCatalog::new(), &CATALOG_SER).unwrap();
    assert_eq!(recid.get(), RECID_CATALOG);
    // 2. Build a real inline TreeMap (records allocated after recid 1).
    let root_recid_recid = {
        let m = BTreeMap::create(Arc::clone(&store), LongFormat, sg(), 32).unwrap();
        m.put(1, "one".to_string()).unwrap();
        m.put(2, "two".to_string()).unwrap();
        let r = m.root_recid_recid();
        // 3. Drop `m` here (end of scope) to release its D12 RW lease.
        r
    };
    // 4. Write a LEGACY catalog with no counterRecid / valueInline keys.
    let mut cat = NameCatalog::new();
    cat.insert("t#type".into(), "TreeMap".into());
    cat.insert("t#keySerializer".into(), "LONG".into());
    cat.insert("t#valueSerializer".into(), "STRING".into());
    cat.insert("t#rootRecidRecid".into(), root_recid_recid.to_string());
    cat.insert("t#maxNodeSize".into(), "32".into());
    store
        .update(
            NonZeroU64::new(RECID_CATALOG).unwrap(),
            Some(&cat),
            &CATALOG_SER,
        )
        .unwrap();
    // 5. Open via the facade — must default counter→0 and valueInline→true.
    let db = DB::new(Arc::clone(&store)).unwrap();
    let m = db.tree_map("t", LongFormat, sg()).open().unwrap();
    assert_eq!(m.counter_recid(), 0);
    assert!(m.value_inline());
    assert_eq!(m.get(&1).unwrap(), Some("one".to_string()));
    assert_eq!(m.get(&2).unwrap(), Some("two".to_string()));
    db.close().unwrap();
}

#[test]
fn atomic_string_nullable_record_bytes() {
    let mut out = crate::io::DataOutput2::new();
    STRING_NULLABLE.serialize(&mut out, &None);
    assert_eq!(out.into_vec(), vec![0x00]);
    let mut out2 = crate::io::DataOutput2::new();
    STRING_NULLABLE.serialize(&mut out2, &Some("hi".to_string()));
    assert_eq!(out2.into_vec(), vec![0x01, 0x82, b'h', b'i']);
}

// ============================ hand-derived golden §7 rows ============================

/// Serialize `cat`, assert byte-for-byte equality with `expected`, and decode
/// `expected` back to `cat` (the decoder is checked INDEPENDENTLY of the encoder).
fn assert_golden_bytes(cat: &NameCatalog, expected: &[u8]) {
    let mut out = crate::io::DataOutput2::new();
    CATALOG_SER.serialize(&mut out, cat);
    assert_eq!(out.into_vec(), expected, "serialize != hand-derived bytes");
    let mut input = crate::io::SliceInput::new(expected);
    let decoded = CATALOG_SER
        .deserialize(&mut input, Some(expected.len()))
        .unwrap();
    assert_eq!(&decoded, cat, "decode(hand bytes) != catalog");
}

#[test]
fn golden_hand_derived_atomic_long_row_bytes() {
    // Hand-computed MDBC bytes for {al#recid -> "7", al#type -> "AtomicLong"}.
    //   header  4D 44 42 43 | 00 00 00 01 | 00
    //   count 2 = packInt(2) = 0x82
    //   "al#recid" (8) = 0x88 + bytes ; "7" (1) = 0x81 '7'
    //   "al#type"  (7) = 0x87 + bytes ; "AtomicLong" (10) = 0x8A + bytes
    let mut cat = NameCatalog::new();
    cat.insert("al#recid".into(), "7".into());
    cat.insert("al#type".into(), "AtomicLong".into());
    let expected: Vec<u8> = vec![
        0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00, // header + repr
        0x82, // count = 2
        0x88, b'a', b'l', b'#', b'r', b'e', b'c', b'i', b'd', // key "al#recid"
        0x81, b'7', // value "7"
        0x87, b'a', b'l', b'#', b't', b'y', b'p', b'e', // key "al#type"
        0x8A, b'A', b't', b'o', b'm', b'i', b'c', b'L', b'o', b'n', b'g', // value
    ];
    assert_golden_bytes(&cat, &expected);
}

#[test]
fn golden_hand_derived_atomic_var_row_bytes() {
    // Hand-computed MDBC bytes for
    //   {v#recid -> "9", v#serializer -> "STRING", v#type -> "AtomicVar"}.
    //   count 3 = 0x83
    //   "v#recid"(7)=0x87 ; "9"(1)=0x81
    //   "v#serializer"(12)=0x8C ; "STRING"(6)=0x86
    //   "v#type"(6)=0x86 ; "AtomicVar"(9)=0x89
    let mut cat = NameCatalog::new();
    cat.insert("v#recid".into(), "9".into());
    cat.insert("v#serializer".into(), "STRING".into());
    cat.insert("v#type".into(), "AtomicVar".into());
    let expected: Vec<u8> = vec![
        0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00, // header + repr
        0x83, // count = 3
        0x87, b'v', b'#', b'r', b'e', b'c', b'i', b'd', // "v#recid"
        0x81, b'9', // "9"
        0x8C, b'v', b'#', b's', b'e', b'r', b'i', b'a', b'l', b'i', b'z', b'e',
        b'r', // "v#serializer"
        0x86, b'S', b'T', b'R', b'I', b'N', b'G', // "STRING"
        0x86, b'v', b'#', b't', b'y', b'p', b'e', // "v#type"
        0x89, b'A', b't', b'o', b'm', b'i', b'c', b'V', b'a', b'r', // "AtomicVar"
    ];
    assert_golden_bytes(&cat, &expected);
}

// ============================ hostile catalog rejection (§6) ============================

/// A well-formed TreeMap catalog row, mutated by `mutate` into (usually) a
/// hostile variant. The `rootRecidRecid` is a phantom recid; `DB::new` only
/// VALIDATES the catalog, it does not open the collection.
fn treemap_catalog(mutate: impl FnOnce(&mut NameCatalog)) -> NameCatalog {
    let mut c = NameCatalog::new();
    c.insert("t#type".into(), "TreeMap".into());
    c.insert("t#keySerializer".into(), "LONG".into());
    c.insert("t#valueSerializer".into(), "STRING".into());
    c.insert("t#rootRecidRecid".into(), "2".into());
    c.insert("t#maxNodeSize".into(), "32".into());
    mutate(&mut c);
    c
}

/// Install `cat` at recid 1 of a fresh non-empty store and assert `DB::new`
/// rejects it as corruption.
fn assert_db_new_rejects(cat: &NameCatalog) {
    let store = Arc::new(StoreByteArray::new(true));
    let recid = store.put(&NameCatalog::new(), &CATALOG_SER).unwrap();
    assert_eq!(recid.get(), RECID_CATALOG);
    store
        .update(
            NonZeroU64::new(RECID_CATALOG).unwrap(),
            Some(cat),
            &CATALOG_SER,
        )
        .unwrap();
    let res = DB::new(store);
    assert!(
        matches!(res, Err(DbError::DataCorruption(_))),
        "expected DataCorruption, got {:?}",
        res.err()
    );
}

#[test]
fn hostile_catalog_rejected_at_open() {
    // Control: a well-formed catalog opens (phantom root is fine — no open).
    {
        let store = Arc::new(StoreByteArray::new(true));
        let recid = store.put(&NameCatalog::new(), &CATALOG_SER).unwrap();
        assert_eq!(recid.get(), RECID_CATALOG);
        let good = treemap_catalog(|_| {});
        store
            .update(
                NonZeroU64::new(RECID_CATALOG).unwrap(),
                Some(&good),
                &CATALOG_SER,
            )
            .unwrap();
        assert!(DB::new(store).is_ok());
    }
    // maxNodeSize below the split bound.
    assert_db_new_rejects(&treemap_catalog(|c| {
        c.insert("t#maxNodeSize".into(), "2".into());
    }));
    // Unknown field for the type.
    assert_db_new_rejects(&treemap_catalog(|c| {
        c.insert("t#bogus".into(), "x".into());
    }));
    // Unknown #type discriminator.
    assert_db_new_rejects(&treemap_catalog(|c| {
        c.insert("t#type".into(), "Frobnicator".into());
    }));
    // rootRecidRecid must be >= 1.
    assert_db_new_rejects(&treemap_catalog(|c| {
        c.insert("t#rootRecidRecid".into(), "0".into());
    }));
    // Non-decimal recid.
    assert_db_new_rejects(&treemap_catalog(|c| {
        c.insert("t#rootRecidRecid".into(), "abc".into());
    }));
    // Non-boolean valueInline.
    assert_db_new_rejects(&treemap_catalog(|c| {
        c.insert("t#valueInline".into(), "maybe".into());
    }));
}

// ============================ failure injection ============================

#[test]
fn catalog_save_failure_leaves_in_memory_catalog_intact() {
    // A read-only reopen makes any catalog save (here, rename) fail; the in-memory
    // catalog must survive that failure untouched (publish_catalog stages first).
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).make().unwrap();
        db.atomic_long_init("x", 1).create().unwrap();
        db.commit().unwrap();
        db.close().unwrap();
    }
    {
        let db = DBMaker::file_db(&f).read_only().make().unwrap();
        assert_eq!(db.get_type("x").unwrap().as_deref(), Some("AtomicLong"));
        // The save inside rename fails on the read-only store.
        assert!(db.rename("x", "y").is_err());
        // Catalog is unchanged: "x" still present, "y" never created.
        assert_eq!(db.get_type("x").unwrap().as_deref(), Some("AtomicLong"));
        assert!(db.get_type("y").unwrap().is_none());
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn delete_after_open_preserves_file_on_failed_make() {
    let f = fresh_file();
    // Build a real StoreDirect file whose recid-1 catalog is hostile.
    {
        let store = Arc::new(crate::store::StoreDirect::open_file(&f).unwrap());
        let recid = store.put(&NameCatalog::new(), &CATALOG_SER).unwrap();
        assert_eq!(recid.get(), RECID_CATALOG);
        let bad = treemap_catalog(|c| {
            c.insert("t#maxNodeSize".into(), "2".into()); // < 4 -> validation fails
        });
        store
            .update(
                NonZeroU64::new(RECID_CATALOG).unwrap(),
                Some(&bad),
                &CATALOG_SER,
            )
            .unwrap();
        store.commit().unwrap();
        store.close().unwrap();
    }
    assert!(f.exists());
    // Opening with delete-after-open must FAIL validation and NOT unlink the file.
    let res = DBMaker::file_db(&f).file_delete_after_open().make();
    assert!(res.is_err(), "hostile catalog must fail make()");
    assert!(
        f.exists(),
        "a failing make() must NOT unlink a pre-existing file (M10)"
    );
    cleanup(&f);
}

// ============================ more Bind coverage ============================

#[test]
fn bind_installs_over_prepopulated_primary() {
    let db = DBMaker::memory_db().make().unwrap();
    let primary = db.tree_map("p", LongFormat, sg()).create().unwrap();
    // Populate BEFORE binding — the initial scan must fill the empty secondary.
    primary.put(1, "alpha".to_string()).unwrap();
    primary.put(2, "beta".to_string()).unwrap();
    let lengths: bind::SecondaryMap<i64, usize> = bind::SecondaryMap::new();
    bind::secondary_value(&primary, lengths.clone(), |_k, v: &String| v.len()).unwrap();
    assert_eq!(lengths.get(&1), Some(5));
    assert_eq!(lengths.get(&2), Some(4));
    // Live updates still apply.
    primary.put(3, "xy".to_string()).unwrap();
    assert_eq!(lengths.get(&3), Some(2));
    db.close().unwrap();
}

#[test]
fn bind_map_inverse_rejects_duplicate_value() {
    let db = DBMaker::memory_db().make().unwrap();
    let primary = db.tree_map("p", LongFormat, sg()).create().unwrap();
    let inverse: bind::SecondaryMap<String, i64> = bind::SecondaryMap::new();
    bind::map_inverse(&primary, inverse.clone()).unwrap();
    primary.put(1, "dup".to_string()).unwrap();
    // A second key mapping to the SAME value breaks the inverse's uniqueness.
    assert!(matches!(
        primary.put(2, "dup".to_string()),
        Err(DbError::WrongConfiguration(_))
    ));
    db.close().unwrap();
}

#[test]
fn bind_self_bind_via_persistent_secondary_rejected() {
    let db = DBMaker::memory_db().make().unwrap();
    let primary = db.tree_map("p", LongFormat, sg()).create().unwrap();
    // Binding the map into a clone of ITSELF (a persistent-map secondary) is
    // rejected via the SecMap identity check.
    let res = bind::secondary_value(&primary, primary.clone(), |_k, v: &String| v.clone());
    assert!(matches!(res, Err(DbError::WrongConfiguration(_))));
    db.close().unwrap();
}

#[test]
fn bind_secondary_value_survives_node_splits() {
    let db = DBMaker::memory_db().make().unwrap();
    let primary = db
        .tree_map("p", LongFormat, sg())
        .max_node_size(4)
        .create()
        .unwrap();
    let mirror: bind::SecondaryMap<i64, String> = bind::SecondaryMap::new();
    bind::secondary_value(&primary, mirror.clone(), |_k, v: &String| format!("m:{v}")).unwrap();
    // More than max_node_size entries forces leaf splits under the sync listener.
    for i in 0..20i64 {
        primary.put(i, format!("v{i}")).unwrap();
    }
    for i in 0..20i64 {
        assert_eq!(mirror.get(&i), Some(format!("m:v{i}")));
    }
    primary.remove(&7).unwrap();
    assert_eq!(mirror.get(&7), None);
    assert_eq!(mirror.len(), 19);
    db.close().unwrap();
}

// ============================ rollback + cache-type coverage ============================

#[test]
fn rollback_then_reopen_queue() {
    let f = fresh_file();
    {
        let db = DBMaker::file_db(&f).transaction_enable().make().unwrap();
        let q = db.queue("q", StringSer).create().unwrap();
        q.add("committed".to_string()).unwrap();
        db.commit().unwrap();
        // Uncommitted enqueue, then roll back.
        q.add("rolled-back".to_string()).unwrap();
        // Drop our handle so the rollback's cache-clear releases the only lease.
        drop(q);
        db.rollback().unwrap();
        // Rollback cleared the cache; reopen a fresh handle from the reverted state.
        let q2 = db.queue("q", StringSer).open().unwrap();
        assert_eq!(q2.poll().unwrap(), Some("committed".to_string()));
        assert_eq!(q2.poll().unwrap(), None);
        db.close().unwrap();
    }
    cleanup(&f);
}

#[test]
fn cached_type_mismatch_same_descriptor_different_concrete_type() {
    use crate::io::{DataInput2, DataOutput2};
    // Two DISTINCT Rust serializer types that both persist the CUSTOM marker
    // (ser_descriptor -> None) and delegate their wire format to STRING.
    #[derive(Clone)]
    struct CustomA;
    #[derive(Clone)]
    struct CustomB;
    macro_rules! custom_string_ser {
        ($t:ty) => {
            impl Serializer<String> for $t {
                fn serialize(&self, out: &mut DataOutput2, v: &String) {
                    StringSer.serialize(out, v)
                }
                fn deserialize(
                    &self,
                    i: &mut dyn DataInput2,
                    s: Option<usize>,
                ) -> crate::error::Result<String> {
                    StringSer.deserialize(i, s)
                }
                fn compare(&self, a: &String, b: &String) -> std::cmp::Ordering {
                    StringSer.compare(a, b)
                }
                fn equals(&self, a: &String, b: &String) -> bool {
                    StringSer.equals(a, b)
                }
            }
            impl crate::db::descriptor::SerDescriptor for $t {
                fn ser_descriptor(&self) -> Option<String> {
                    None
                }
            }
        };
    }
    custom_string_ser!(CustomA);
    custom_string_ser!(CustomB);

    let db = DBMaker::memory_db().make().unwrap();
    // Create "v" with CustomA — catalog serializer persisted as "CUSTOM".
    let a = db
        .atomic_var("v", CustomA, Some("x".to_string()))
        .create()
        .unwrap();
    // Reopen with the SAME concrete type: descriptor matches AND the cached
    // concrete type matches -> cache HIT (shared-state clone).
    let a2 = db.atomic_var("v", CustomA, None).open().unwrap();
    assert_eq!(a2.get().unwrap(), a.get().unwrap());
    // Reopen with a DIFFERENT concrete type but the SAME ("CUSTOM") descriptor:
    // verify passes, but the cached handle's concrete type differs -> mismatch.
    let res = db.atomic_var("v", CustomB, None).open();
    assert!(
        matches!(res, Err(DbError::CachedTypeMismatch(_))),
        "got {:?}",
        res.err()
    );
    db.close().unwrap();
}

// ============================ review fix round (C1, R3–R6) ============================

#[test]
fn delete_map_clears_entries_but_keeps_structural_records() {
    // C1: `delete()` must clear a map's ENTRIES (Java `obj.clear()`) but must NOT
    // free its structural root / counter records — a still-live clone keeps using
    // them, and freeing would let the store reuse the recid under the clone.
    let db = DBMaker::memory_db().make().unwrap();
    let map = db
        .tree_map("a", LongFormat, sg())
        .counter_enable()
        .create()
        .unwrap();
    map.put(1, "x".to_string()).unwrap();
    map.put(2, "y".to_string()).unwrap();
    let clone = map.clone();
    assert!(db.delete("a").unwrap());
    // Entries were cleared.
    assert_eq!(clone.size_long().unwrap(), 0);
    assert_eq!(clone.get(&1).unwrap(), None);
    // Root + counter records survived: the retained clone is still fully usable.
    clone.put(3, "z".to_string()).unwrap();
    assert_eq!(clone.get(&3).unwrap(), Some("z".to_string()));
    assert_eq!(clone.size_long().unwrap(), 1);
    db.close().unwrap();
}

#[test]
fn delete_queue_frees_records_and_closes_clones() {
    // C1: deleting a queue frees its node + header records and globally closes the
    // shared handle, so a retained clone observes StoreClosed (safe — no clone can
    // write the freed recids).
    let db = DBMaker::memory_db().make().unwrap();
    let q = db.queue("q", StringSer).create().unwrap();
    q.add("a".to_string()).unwrap();
    let q2 = Arc::clone(&q);
    assert!(db.delete("q").unwrap());
    assert!(matches!(q2.poll(), Err(DbError::StoreClosed)));
    db.close().unwrap();
}

#[test]
fn max_node_size_out_of_range_rejected_at_create() {
    // R3: the create path enforces the SAME 4..=1<<20 bound the reopen validator
    // uses, so a create can never persist a value that bricks the next open.
    let db = DBMaker::memory_db().make().unwrap();
    let too_big = (1usize << 20) + 1;
    assert!(matches!(
        db.tree_map("m", LongFormat, sg())
            .max_node_size(too_big)
            .create(),
        Err(DbError::WrongConfiguration(_))
    ));
    assert!(matches!(
        db.tree_set("s", LongFormat).max_node_size(too_big).create(),
        Err(DbError::WrongConfiguration(_))
    ));
    assert!(matches!(
        db.tree_map("lo", LongFormat, sg())
            .max_node_size(3)
            .create(),
        Err(DbError::WrongConfiguration(_))
    ));
    // Nothing was persisted for the rejected names.
    assert!(!db.exists("m").unwrap());
    assert!(!db.exists("s").unwrap());
    assert!(!db.exists("lo").unwrap());
    // The exact maximum is accepted.
    db.tree_map("ok", LongFormat, sg())
        .max_node_size(1 << 20)
        .create()
        .unwrap();
    db.close().unwrap();
}

#[test]
fn bind_unique_index_atomic_under_concurrent_distinct_leaves() {
    // R4: two writers on DIFFERENT primary leaves whose derived unique keys
    // collide must not both insert — putIfAbsent makes exactly one win.
    for _ in 0..30 {
        let db = DBMaker::memory_db().make().unwrap();
        let primary = db
            .tree_map("p", LongFormat, sg())
            .max_node_size(4)
            .create()
            .unwrap();
        // Populate 0..40 with DISTINCT-length values (derived key = length), which
        // forces many leaves so keys 0 and 39 live on different leaves.
        for i in 0..40i64 {
            primary.put(i, "a".repeat((i + 1) as usize)).unwrap();
        }
        let unique: bind::SecondaryMap<usize, i64> = bind::SecondaryMap::new();
        bind::secondary_key(&primary, unique.clone(), |_k, v: &String| v.len()).unwrap();
        // Concurrently update two far-apart keys to the SAME new length (100),
        // colliding on derived key 100.
        let p1 = primary.clone();
        let p2 = primary.clone();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let b1 = barrier.clone();
        let b2 = barrier.clone();
        let h1 = std::thread::spawn(move || {
            b1.wait();
            p1.put(0, "a".repeat(100))
        });
        let h2 = std::thread::spawn(move || {
            b2.wait();
            p2.put(39, "a".repeat(100))
        });
        let r1 = h1.join().unwrap();
        let r2 = h2.join().unwrap();
        let errs = [r1.is_err(), r2.is_err()].iter().filter(|e| **e).count();
        assert_eq!(
            errs, 1,
            "exactly one writer must be rejected on the collision"
        );
        // Derived key 100 maps to exactly one primary key.
        assert!(unique.get(&100).is_some());
        db.close().unwrap();
    }
}

#[test]
fn histogram_reentrant_category_does_not_deadlock() {
    // R5: the category closure is evaluated BEFORE the histogram lock is taken, so
    // a closure that reads the same histogram cannot self-deadlock the mutex.
    let db = DBMaker::memory_db().make().unwrap();
    let primary = db.tree_map("p", LongFormat, sg()).create().unwrap();
    let hist: bind::SecondaryMap<char, i64> = bind::SecondaryMap::new();
    let hist_reader = hist.clone();
    bind::histogram(&primary, hist.clone(), move |_k, v: &String| {
        let _ = hist_reader.len(); // reentrant read of the same histogram
        v.chars().next().unwrap()
    })
    .unwrap();
    // Run the mutation on a worker and time it out, so a regression (deadlock)
    // fails the test instead of hanging the suite.
    let (tx, rx) = std::sync::mpsc::channel();
    let p = primary.clone();
    std::thread::spawn(move || {
        let _ = tx.send(p.put(1, "apple".to_string()));
    });
    let res = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("histogram category reentrancy deadlocked");
    res.unwrap();
    assert_eq!(hist.get(&'a'), Some(1));
    db.close().unwrap();
}

#[test]
fn queue_open_rejects_header_mode_mismatch() {
    // R6: a catalog #type that disagrees with the header's stored mode is a
    // corrupt catalog<->header pairing (Java `QueueMaker.open2` compares mode).
    use crate::queue::blocking::{Mode, PersistentBlockingQueue};
    let store = Arc::new(StoreByteArray::new(true));
    let recid = store.put(&NameCatalog::new(), &CATALOG_SER).unwrap();
    assert_eq!(recid.get(), RECID_CATALOG);
    // Build a LIFO (mode=1) header directly.
    let q =
        PersistentBlockingQueue::create(Arc::clone(&store), StringSer, Mode::Lifo, i64::MAX as u64)
            .unwrap();
    let header = q.header_recid();
    drop(q);
    // Write a catalog claiming this is a FIFO Queue, pointing at the LIFO header.
    let mut cat = NameCatalog::new();
    cat.insert("s#type".into(), "Queue".into());
    cat.insert("s#headerRecid".into(), header.get().to_string());
    cat.insert("s#serializer".into(), "STRING".into());
    store
        .update(
            NonZeroU64::new(RECID_CATALOG).unwrap(),
            Some(&cat),
            &CATALOG_SER,
        )
        .unwrap();
    let db = DB::new(store).unwrap();
    let res = db.queue("s", StringSer).open();
    assert!(
        matches!(res, Err(DbError::DataCorruption(_))),
        "got {:?}",
        res.err()
    );
    db.close().unwrap();
}

#[test]
fn file_delete_after_close_rejected_on_memory_backend() {
    // R10: fileDeleteAfterClose on a non-file backend is rejected, not ignored.
    assert!(matches!(
        DBMaker::memory_db().file_delete_after_close().make(),
        Err(DbError::WrongConfiguration(_))
    ));
}
