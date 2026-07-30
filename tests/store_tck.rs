//! Store TCK (spec 05 §2): exercises every record state transition and calls
//! `verify()` after each mutation. Run generically against all stores. Exception
//! contract → `DbError` variant assertions.

use mapdb_rust_store::error::{DbError, Result};
use mapdb_rust_store::io::{DataInput2, DataOutput2, SliceInput};
use mapdb_rust_store::ser::serializers::LongSer;
use mapdb_rust_store::ser::Serializer;
use mapdb_rust_store::store::{
    AppendResult, Record, RecordRead, Store, StoreByteArray, StoreDelta, StoreDirect, StoreOnHeap,
    StoreWAL,
};
use std::cmp::Ordering;

const L: LongSer = LongSer;

/// Size-driven raw-bytes serializer (Java `Fixtures.RAW`): content == value, so
/// appended delta bytes are observable (a length-prefixed serializer would hide
/// the delta region). Used by the delta TCK.
struct RawSer;
impl Serializer<Vec<u8>> for RawSer {
    fn serialize(&self, out: &mut DataOutput2, v: &Vec<u8>) {
        out.write_all(v);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<Vec<u8>> {
        let n = size.expect("raw serializer needs a framed size");
        let mut b = vec![0u8; n];
        input.read_fully(&mut b)?;
        Ok(b)
    }
    fn compare(&self, a: &Vec<u8>, b: &Vec<u8>) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &Vec<u8>, b: &Vec<u8>) -> bool {
        a == b
    }
}
const R: RawSer = RawSer;

fn bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..len)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u8
        })
        .collect()
}

fn concat(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

/// A push-down read action that decodes an i64 (for byte stores) or downcasts
/// (for the heap store), recording which branch ran.
#[derive(Default)]
struct ReadProbe {
    saw_null: bool,
    value: Option<i64>,
}
impl RecordRead for ReadProbe {
    fn on_bytes(&mut self, input: &mut SliceInput<'_>, _size: usize) -> mapdb_rust_store::error::Result<i64> {
        let v = input.read_i64()?;
        self.value = Some(v);
        Ok(v)
    }
    fn on_object(&mut self, obj: &dyn std::any::Any) -> mapdb_rust_store::error::Result<i64> {
        let v = *obj.downcast_ref::<i64>().unwrap();
        self.value = Some(v);
        Ok(v)
    }
    fn on_null(&mut self) -> mapdb_rust_store::error::Result<i64> {
        self.saw_null = true;
        Ok(0)
    }
}

fn is_get_void<T: std::fmt::Debug>(r: mapdb_rust_store::error::Result<T>) -> bool {
    matches!(r, Err(DbError::GetVoid(_)))
}

/// Core state machine, generic over any Store.
fn tck_states<S: Store>(s: &S) {
    s.verify().unwrap();

    // Void: a never-allocated recid → GetVoid on read/get/delete.
    let void = std::num::NonZeroU64::new(999_999).unwrap();
    assert!(is_get_void(s.get(void, &L)));
    assert!(is_get_void(s.delete(void)));

    // put → Live
    let r1 = s.put(&42i64, &L).unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(r1, &L).unwrap(), Some(42));
    let mut p = ReadProbe::default();
    assert_eq!(s.read(r1, &mut p).unwrap(), 42);
    assert_eq!(p.value, Some(42));

    // preallocate → Preallocated: get → None, excluded from get_all_recids
    let rp = s.preallocate().unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(rp, &L).unwrap(), None);
    let mut p = ReadProbe::default();
    s.read(rp, &mut p).unwrap();
    assert!(p.saw_null);
    assert!(!s.get_all_recids().unwrap().contains(&rp));
    assert!(s.get_all_recids().unwrap().contains(&r1));

    // update preallocated → Live
    s.update(rp, Some(&7i64), &L).unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(rp, &L).unwrap(), Some(7));
    assert!(s.get_all_recids().unwrap().contains(&rp));

    // update Live → Null content: get → None but record still exists
    s.update(r1, None::<&i64>, &L).unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(r1, &L).unwrap(), None);
    let mut p = ReadProbe::default();
    s.read(r1, &mut p).unwrap();
    assert!(p.saw_null);

    // delete → Deleted: get/read → GetVoid
    s.delete(r1).unwrap();
    s.verify().unwrap();
    assert!(is_get_void(s.get(r1, &L)));
    let mut p = ReadProbe::default();
    assert!(is_get_void(s.read(r1, &mut p)));

    // update of a deleted recid → GetVoid
    assert!(is_get_void(s.update(r1, Some(&1i64), &L)));
}

/// CAS logical-equality semantics.
fn tck_cas<S: Store>(s: &S) {
    let r = s.put(&100i64, &L).unwrap();
    // wrong expected → false, no change
    assert!(!s.compare_and_swap(r, Some(&5i64), Some(&6i64), &L).unwrap());
    assert_eq!(s.get(r, &L).unwrap(), Some(100));
    // right expected → swap
    assert!(s
        .compare_and_swap(r, Some(&100i64), Some(&200i64), &L)
        .unwrap());
    assert_eq!(s.get(r, &L).unwrap(), Some(200));
    s.verify().unwrap();
    // swap Live → Null
    assert!(s
        .compare_and_swap(r, Some(&200i64), None::<&i64>, &L)
        .unwrap());
    assert_eq!(s.get(r, &L).unwrap(), None);
    // Null matches None expected → swap back to Live
    assert!(s
        .compare_and_swap(r, None::<&i64>, Some(&9i64), &L)
        .unwrap());
    assert_eq!(s.get(r, &L).unwrap(), Some(9));
    // expecting non-null on a live value that differs → false
    assert!(!s
        .compare_and_swap(r, None::<&i64>, Some(&1i64), &L)
        .unwrap());
    s.verify().unwrap();
}

fn tck_recid_reuse_and_close<S: Store>(s: &S) {
    let a = s.put(&1i64, &L).unwrap();
    s.delete(a).unwrap();
    // a is now free; a new put may reuse it, but must be Void until then
    assert!(is_get_void(s.get(a, &L)));
    s.verify().unwrap();
    s.close().unwrap();
    assert!(s.is_closed());
    assert!(matches!(s.get(a, &L), Err(DbError::StoreClosed)));
}

/// Delta-capability TCK (spec 02 §1, faithful port of Java `DeltaTCK`): the
/// record content model content == base ++ deltas, capacity refusal at the
/// rounded boundary, and merged-logical-value CAS. Uses the size-driven `RawSer`
/// so appended bytes are observable. Generic over any `StoreDelta`.
fn tck_delta<S: StoreDelta>(s: &S) {
    // --- append grows content byte-exactly within provisioned capacity ---
    let base = bytes(1, 12);
    let r = s.put(&base, &R).unwrap();
    s.update_with_headroom(r, &base, &R, 64).unwrap();
    s.verify().unwrap();
    let (d1, d2, d3) = (vec![10u8, 11, 12], vec![20u8, 21], vec![30u8]);
    assert_eq!(
        s.append(r, &d1).unwrap(),
        AppendResult::NewSize(base.len() + 3)
    );
    assert_eq!(
        s.append(r, &d2).unwrap(),
        AppendResult::NewSize(base.len() + 5)
    );
    assert_eq!(
        s.append(r, &d3).unwrap(),
        AppendResult::NewSize(base.len() + 6)
    );
    s.verify().unwrap();
    let merged = concat(&[&base, &d1, &d2, &d3]);
    assert_eq!(s.get(r, &R).unwrap(), Some(merged));

    // --- refused exactly at the capacity boundary; REFUSED leaves content intact ---
    let base = bytes(2, 8);
    let rb = s.put(&base, &R).unwrap();
    s.update_with_headroom(rb, &base, &R, 40).unwrap();
    s.commit().unwrap();
    let cap_rem = s.capacity_remaining(rb).unwrap();
    assert!(cap_rem >= 40, "headroom must be honoured");
    let fill: Vec<u8> = (0..cap_rem).map(|i| (i + 1) as u8).collect();
    assert_eq!(
        s.append(rb, &fill).unwrap(),
        AppendResult::NewSize(base.len() + cap_rem)
    );
    assert_eq!(s.capacity_remaining(rb).unwrap(), 0, "at boundary");
    let bmerged = concat(&[&base, &fill]);
    assert_eq!(s.append(rb, &[99]).unwrap(), AppendResult::Refused);
    assert_eq!(s.capacity_remaining(rb).unwrap(), 0);
    assert_eq!(s.get(rb, &R).unwrap(), Some(bmerged));
    s.verify().unwrap();

    // --- append on a preallocated record establishes it with delta-only content ---
    let rp = s.preallocate().unwrap();
    assert_eq!(s.get(rp, &R).unwrap(), None, "P record reads null");
    assert!(!s.get_all_recids().unwrap().contains(&rp));
    let d = bytes(3, 14);
    assert_eq!(s.append(rp, &d).unwrap(), AppendResult::NewSize(d.len()));
    s.verify().unwrap();
    assert_eq!(s.get(rp, &R).unwrap(), Some(d.clone()));
    assert!(s.get_all_recids().unwrap().contains(&rp));
    s.commit().unwrap();
    assert_eq!(s.get(rp, &R).unwrap(), Some(d));

    // --- update resets the appended region ---
    let base = bytes(4, 8);
    let ru = s.put(&base, &R).unwrap();
    s.update_with_headroom(ru, &base, &R, 32).unwrap();
    s.append(ru, &[7, 7, 7, 7]).unwrap();
    assert_eq!(
        s.get(ru, &R).unwrap(),
        Some(concat(&[&base, &[7, 7, 7, 7]]))
    );
    let base2 = bytes(40, 10);
    s.update(ru, Some(&base2), &R).unwrap();
    assert_eq!(
        s.get(ru, &R).unwrap(),
        Some(base2),
        "appended region resets"
    );
    s.verify().unwrap();

    // --- update_with_headroom guarantees the headroom is immediately appendable ---
    const H: usize = 48;
    let base = bytes(5, 6);
    let rh = s.put(&base, &R).unwrap();
    s.update_with_headroom(rh, &base, &R, H).unwrap();
    assert!(s.capacity_remaining(rh).unwrap() >= H);
    let block: Vec<u8> = (0..H).map(|i| i as u8).collect();
    assert_eq!(
        s.append(rh, &block).unwrap(),
        AppendResult::NewSize(base.len() + H),
        "headroom bytes must be immediately appendable"
    );
    assert_eq!(s.get(rh, &R).unwrap(), Some(concat(&[&base, &block])));
    s.verify().unwrap();

    // --- delete after appends → GetVoid on every delta op ---
    let base = bytes(6, 8);
    let rd = s.put(&base, &R).unwrap();
    s.update_with_headroom(rd, &base, &R, 32).unwrap();
    s.append(rd, &[1, 2, 3]).unwrap();
    s.delete(rd).unwrap();
    s.verify().unwrap();
    assert!(is_get_void(s.get(rd, &R)));
    assert!(is_get_void(s.append(rd, &[1])));
    assert!(is_get_void(s.capacity_remaining(rd)));
    assert!(is_get_void(s.update(rd, Some(&base), &R)));
    assert!(is_get_void(s.update_with_headroom(rd, &base, &R, 8)));
    assert!(is_get_void(s.compare_and_swap(
        rd,
        Some(&base),
        Some(&base),
        &R
    )));
    s.verify().unwrap();

    // --- zero-length append is a no-op returning the current size ---
    let base = bytes(7, 13);
    let rz = s.put(&base, &R).unwrap();
    assert_eq!(
        s.append(rz, &[]).unwrap(),
        AppendResult::NewSize(base.len())
    );
    assert_eq!(s.get(rz, &R).unwrap(), Some(base.clone()));
    assert_eq!(
        s.append(rz, &[]).unwrap(),
        AppendResult::NewSize(base.len())
    );
    assert_eq!(s.get(rz, &R).unwrap(), Some(base));
    s.verify().unwrap();

    // --- CAS after appends compares the merged logical value ---
    let base = bytes(8, 8);
    let rc = s.put(&base, &R).unwrap();
    s.update_with_headroom(rc, &base, &R, 64).unwrap();
    let (d1, d2) = (vec![40u8, 41, 42], vec![50u8, 51]);
    s.append(rc, &d1).unwrap();
    s.append(rc, &d2).unwrap();
    let merged = concat(&[&base, &d1, &d2]);
    assert_eq!(s.get(rc, &R).unwrap(), Some(merged.clone()));
    // CAS against the base-only (pre-append) image must fail.
    assert!(!s
        .compare_and_swap(rc, Some(&base), Some(&bytes(80, 5)), &R)
        .unwrap());
    // CAS against the merged image must succeed.
    let replacement = bytes(82, 7);
    assert!(s
        .compare_and_swap(rc, Some(&merged), Some(&replacement), &R)
        .unwrap());
    assert_eq!(s.get(rc, &R).unwrap(), Some(replacement));
    s.verify().unwrap();
}

fn run_all<S: Store>(make: impl Fn() -> S) {
    tck_states(&make());
    tck_cas(&make());
    tck_recid_reuse_and_close(&make());
}

#[test]
fn tck_heap() {
    run_all(|| StoreOnHeap::new(true));
    run_all(|| StoreOnHeap::new(false));
}

#[test]
fn tck_bytearray() {
    run_all(|| StoreByteArray::new(true));
    run_all(|| StoreByteArray::new(false));
    tck_delta(&StoreByteArray::new(true));
    tck_delta(&StoreByteArray::new(false));
}

#[test]
fn tck_direct() {
    // heap-backed StoreDirect; verify() (the tiling oracle) runs after each mutation.
    run_all(|| StoreDirect::new_heap().unwrap());
    run_all(|| StoreDirect::new_heap_ts(false).unwrap());
    tck_delta(&StoreDirect::new_heap().unwrap());
    tck_delta(&StoreDirect::new_heap_ts(false).unwrap());
}

/// Unique temp WAL path per store instance (cleaned first so a stale file never
/// leaks state into a fresh store).
fn wal_tmp() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering as O};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, O::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!("mapdb5_wal_tck_{}_{}.wal", std::process::id(), n));
    let _ = std::fs::remove_file(&p);
    let mut c = p.clone().into_os_string();
    c.push(".ckpt");
    let _ = std::fs::remove_file(std::path::PathBuf::from(c));
    p
}

/// StoreWAL is transactional (staged-until-commit): the generic state/CAS/delta
/// TCK must pass identically because reads merge staged mutations over the inner
/// committed image.
#[test]
fn tck_wal() {
    run_all(|| StoreWAL::open(&wal_tmp()).unwrap());
    run_all(|| StoreWAL::open_ts(&wal_tmp(), false).unwrap());
    tck_delta(&StoreWAL::open(&wal_tmp()).unwrap());
    tck_delta(&StoreWAL::open_ts(&wal_tmp(), false).unwrap());
}

/// A non-i64 record type through the heap store's object path + logical CAS.
#[test]
fn heap_object_and_logical_cas() {
    use mapdb_rust_store::ser::serializers::StringSer;
    // A serializer whose logical equality is case-insensitive, to prove CAS uses
    // ser.equals not byte/value equality.
    struct CaseInsensitive;
    impl mapdb_rust_store::ser::Serializer<String> for CaseInsensitive {
        fn serialize(&self, out: &mut mapdb_rust_store::io::DataOutput2, v: &String) {
            StringSer.serialize(out, v)
        }
        fn deserialize(
            &self,
            input: &mut dyn DataInput2,
            size: Option<usize>,
        ) -> mapdb_rust_store::error::Result<String> {
            StringSer.deserialize(input, size)
        }
        fn compare(&self, a: &String, b: &String) -> std::cmp::Ordering {
            a.to_lowercase().cmp(&b.to_lowercase())
        }
        fn equals(&self, a: &String, b: &String) -> bool {
            a.eq_ignore_ascii_case(b)
        }
    }
    let s = StoreOnHeap::new(true);
    let ci = CaseInsensitive;
    let r = s.put(&"Hello".to_string(), &ci).unwrap();
    assert_eq!(s.get(r, &ci).unwrap().as_deref(), Some("Hello"));
    // logical (case-insensitive) equality lets "HELLO" match "Hello"
    assert!(s
        .compare_and_swap(
            r,
            Some(&"HELLO".to_string()),
            Some(&"world".to_string()),
            &ci
        )
        .unwrap());
    assert_eq!(s.get(r, &ci).unwrap().as_deref(), Some("world"));
    s.verify().unwrap();
}

fn _assert_record_bound<T: Record>() {}
#[test]
fn record_bound_holds() {
    _assert_record_bound::<i64>();
    _assert_record_bound::<String>();
    _assert_record_bound::<Vec<u8>>();
}
