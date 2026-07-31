//! StoreWAL-specific tests (spec 02 §7, spec 05): commit durability across
//! reopen (log replay), multi-section replay, delete/append replay, rollback,
//! torn-tail truncation vs mid-log corruption (D4), checkpoint compaction,
//! crash-during-checkpoint temp recovery, auto-checkpoint, and streaming-replay
//! refill edges. The inner StoreDirect is heap-backed, so *all* durability is
//! carried by the log file — reopen rebuilds inner purely from replay.

use mapdb_rust_store::error::{DbError, Result};
use mapdb_rust_store::io::{DataInput2, DataOutput2};
use mapdb_rust_store::ser::serializers::LongSer;
use mapdb_rust_store::ser::Serializer;
use mapdb_rust_store::store::{Store, StoreDelta, StoreTx, StoreWAL};
use std::cmp::Ordering;
use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomOrd};

const L: LongSer = LongSer;

/// Raw-bytes serializer: content == value, so large/linked records round-trip
/// byte-exactly through the log.
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

/// Unique temp WAL path; removes any stale log + ckpt temp so each test starts clean.
fn tmp() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, AtomOrd::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!("mapdb5_wal_it_{}_{}.wal", std::process::id(), n));
    let _ = std::fs::remove_file(&p);
    let mut c = p.clone().into_os_string();
    c.push(".ckpt");
    let _ = std::fs::remove_file(PathBuf::from(c));
    p
}

fn ckpt_of(p: &Path) -> PathBuf {
    let mut c = p.to_path_buf().into_os_string();
    c.push(".ckpt");
    PathBuf::from(c)
}

fn is_corrupt(r: Result<StoreWAL>) -> bool {
    matches!(r, Err(DbError::DataCorruption(_)))
}

#[test]
fn old_framed_magic_is_rejected_without_rewrite() {
    let p = tmp();
    {
        let s = StoreWAL::open(&p).unwrap();
        s.close().unwrap();
    }
    let f = OpenOptions::new().write(true).open(&p).unwrap();
    f.write_all_at(b"MDB5.WAL", 0).unwrap();
    drop(f);
    let before = std::fs::read(&p).unwrap();

    assert!(is_corrupt(StoreWAL::open(&p)));
    assert_eq!(std::fs::read(&p).unwrap(), before);
}

#[test]
fn valid_legacy_headerless_wal_is_migrated() {
    let p = tmp();
    // PREALLOC recid 1, followed by the legacy COMMIT seal and the CRC32 of
    // the operation bytes. A real legacy stream therefore begins with opcode
    // 1, not an ASCII magic byte.
    let ops = [1u8, 0x81];
    let mut legacy = ops.to_vec();
    legacy.push(8);
    legacy.extend_from_slice(&(crc32fast::hash(&ops) as i32).to_be_bytes());
    std::fs::write(&p, legacy).unwrap();

    let s = StoreWAL::open(&p).unwrap();
    s.verify().unwrap();
    s.close().unwrap();
    assert_eq!(&std::fs::read(&p).unwrap()[..8], b"MDBS.WAL");
}

#[test]
fn one_and_two_byte_legacy_tails_are_safe() {
    for tail in [&[1u8][..], &[1u8, 0x81][..]] {
        let p = tmp();
        std::fs::write(&p, tail).unwrap();

        let s = StoreWAL::open(&p).unwrap();
        s.verify().unwrap();
        s.close().unwrap();
        assert_eq!(&std::fs::read(&p).unwrap()[..8], b"MDBS.WAL");
    }
}

// ---------------------------------------------------------------------------
// Durability: only committed state survives a reopen.
// ---------------------------------------------------------------------------

#[test]
fn committed_state_survives_reopen() {
    let p = tmp();
    let (r1, r2);
    {
        let s = StoreWAL::open(&p).unwrap();
        r1 = s.put(&11i64, &L).unwrap();
        r2 = s.put(&22i64, &L).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    let s = StoreWAL::open(&p).unwrap();
    assert_eq!(s.get(r1, &L).unwrap(), Some(11));
    assert_eq!(s.get(r2, &L).unwrap(), Some(22));
    s.verify().unwrap();
}

#[test]
fn uncommitted_state_is_lost_on_reopen() {
    let p = tmp();
    let r;
    {
        let s = StoreWAL::open(&p).unwrap();
        r = s.put(&99i64, &L).unwrap();
        // NO commit.
        s.close().unwrap();
    }
    let s = StoreWAL::open(&p).unwrap();
    // recid was never durably allocated: reads Void (never-allocated).
    assert!(matches!(s.get(r, &L), Err(DbError::GetVoid(_))));
}

#[test]
fn multi_section_replay_and_last_write_wins() {
    let p = tmp();
    let r;
    {
        let s = StoreWAL::open(&p).unwrap();
        r = s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        s.update(r, Some(&2i64), &L).unwrap();
        s.commit().unwrap();
        s.update(r, Some(&3i64), &L).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    let s = StoreWAL::open(&p).unwrap();
    assert_eq!(
        s.get(r, &L).unwrap(),
        Some(3),
        "last committed section wins"
    );
}

#[test]
fn delete_and_append_replay() {
    let p = tmp();
    let (rkeep, rdel, rapp);
    {
        let s = StoreWAL::open(&p).unwrap();
        rkeep = s.put(&7i64, &L).unwrap();
        rdel = s.put(&8i64, &L).unwrap();
        rapp = s.put(&bytes(1, 10), &R).unwrap();
        s.update_with_headroom(rapp, &bytes(1, 10), &R, 64).unwrap();
        s.commit().unwrap();
        // second tx: delete one, append to another.
        s.delete(rdel).unwrap();
        s.append(rapp, &[100, 101, 102]).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    let s = StoreWAL::open(&p).unwrap();
    assert_eq!(s.get(rkeep, &L).unwrap(), Some(7));
    assert!(matches!(s.get(rdel, &L), Err(DbError::GetVoid(_))));
    let mut want = bytes(1, 10);
    want.extend_from_slice(&[100, 101, 102]);
    assert_eq!(s.get(rapp, &R).unwrap(), Some(want));
    s.verify().unwrap();
}

#[test]
fn linked_oversize_record_replays() {
    let p = tmp();
    let big = bytes(42, 200_000); // forces an oversize/linked record
    let r;
    {
        let s = StoreWAL::open(&p).unwrap();
        r = s.put(&big, &R).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    let s = StoreWAL::open(&p).unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(big));
    s.verify().unwrap();
}

// ---------------------------------------------------------------------------
// Rollback discards staged, keeps committed.
// ---------------------------------------------------------------------------

#[test]
fn rollback_discards_staged_keeps_committed() {
    let p = tmp();
    let s = StoreWAL::open(&p).unwrap();
    let r = s.put(&5i64, &L).unwrap();
    s.commit().unwrap();
    // stage an update + a fresh put, then roll back.
    s.update(r, Some(&500i64), &L).unwrap();
    let rtmp = s.put(&123i64, &L).unwrap();
    s.rollback().unwrap();
    assert_eq!(s.get(r, &L).unwrap(), Some(5), "committed value restored");
    assert!(
        matches!(s.get(rtmp, &L), Err(DbError::GetVoid(_))),
        "rolled-back prealloc is freed"
    );
    s.verify().unwrap();
}

// ---------------------------------------------------------------------------
// Torn tail (availability): a truncated final section is dropped; earlier
// committed sections survive. D4.
// ---------------------------------------------------------------------------

fn file_len(p: &PathBuf) -> u64 {
    std::fs::metadata(p).unwrap().len()
}

#[test]
fn torn_tail_body_is_truncated_not_fatal() {
    let p = tmp();
    let (r1, len_after_first);
    {
        let s = StoreWAL::open(&p).unwrap();
        r1 = s.put(&1000i64, &L).unwrap();
        s.commit().unwrap();
        len_after_first = file_len(&p);
        // a second commit whose section we will tear.
        s.put(&2000i64, &L).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    // Chop the file one byte into the second section's body → torn tail.
    let f = OpenOptions::new().write(true).open(&p).unwrap();
    f.set_len(len_after_first + 1).unwrap();
    drop(f);

    let s = StoreWAL::open(&p).unwrap();
    assert_eq!(s.get(r1, &L).unwrap(), Some(1000), "first commit survives");
    s.verify().unwrap();
    // A subsequent commit reuses the truncated tail region and reopens cleanly.
    let r3 = s.put(&3000i64, &L).unwrap();
    s.commit().unwrap();
    s.close().unwrap();
    let s = StoreWAL::open(&p).unwrap();
    assert_eq!(s.get(r1, &L).unwrap(), Some(1000));
    assert_eq!(s.get(r3, &L).unwrap(), Some(3000));
}

#[test]
fn torn_tail_within_section_header_is_truncated() {
    let p = tmp();
    let (r1, len_after_first);
    {
        let s = StoreWAL::open(&p).unwrap();
        r1 = s.put(&11i64, &L).unwrap();
        s.commit().unwrap();
        len_after_first = file_len(&p);
        s.put(&22i64, &L).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    // Chop inside the second section's 25-byte header (only 5 header bytes present).
    let f = OpenOptions::new().write(true).open(&p).unwrap();
    f.set_len(len_after_first + 5).unwrap();
    drop(f);
    let s = StoreWAL::open(&p).unwrap();
    assert_eq!(s.get(r1, &L).unwrap(), Some(11));
    s.verify().unwrap();
}

// ---------------------------------------------------------------------------
// Mid-log corruption (integrity): a damaged section FOLLOWED by a valid one is
// not a torn tail — reopen must refuse. D4.
// ---------------------------------------------------------------------------

#[test]
fn mid_log_body_corruption_is_fatal() {
    let p = tmp();
    let first_body_off;
    {
        let s = StoreWAL::open(&p).unwrap();
        s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        // body of the first section starts right after file header + section header.
        first_body_off = 16 + 25;
        s.put(&2i64, &L).unwrap();
        s.commit().unwrap(); // a valid section AFTER the one we corrupt
        s.close().unwrap();
    }
    // Flip a byte in the FIRST section's body; the valid second section remains.
    let f = OpenOptions::new().read(true).write(true).open(&p).unwrap();
    let mut b = [0u8; 1];
    f.read_exact_at(&mut b, first_body_off).unwrap();
    b[0] ^= 0xFF;
    f.write_all_at(&b, first_body_off).unwrap();
    f.sync_all().unwrap();
    drop(f);

    assert!(
        is_corrupt(StoreWAL::open(&p)),
        "corrupt section followed by a valid one must refuse (not torn tail)"
    );
}

#[test]
fn mid_log_header_corruption_is_fatal() {
    let p = tmp();
    {
        let s = StoreWAL::open(&p).unwrap();
        s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        s.put(&2i64, &L).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    // Corrupt the FIRST section's tag byte (offset 16); its declared bodyLen still
    // points at the valid, correctly-LSN'd second section → mid-log corruption.
    let f = OpenOptions::new().read(true).write(true).open(&p).unwrap();
    let mut b = [0u8; 1];
    f.read_exact_at(&mut b, 16).unwrap();
    b[0] ^= 0x55;
    f.write_all_at(&b, 16).unwrap();
    f.sync_all().unwrap();
    drop(f);
    assert!(is_corrupt(StoreWAL::open(&p)));
}

#[test]
fn unsupported_version_is_rejected() {
    let p = tmp();
    {
        let s = StoreWAL::open(&p).unwrap();
        s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    // Bump the version word (offset 8..12) to an unknown value.
    let f = OpenOptions::new().write(true).open(&p).unwrap();
    f.write_all_at(&99i32.to_be_bytes(), 8).unwrap();
    f.sync_all().unwrap();
    drop(f);
    assert!(is_corrupt(StoreWAL::open(&p)));
}

#[test]
fn nonzero_v1_header_flags_are_rejected_without_rewrite() {
    let p = tmp();
    {
        let s = StoreWAL::open(&p).unwrap();
        s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    // Set the flags word (offset 12..16) nonzero; magic and version stay v1.
    // The open must fail with an EXPLICIT DataCorruption (not fall through to
    // the framed-MDB guard or legacy replay) and leave the file byte-unchanged.
    let f = OpenOptions::new().write(true).open(&p).unwrap();
    f.write_all_at(&1i32.to_be_bytes(), 12).unwrap();
    f.sync_all().unwrap();
    drop(f);
    let before = std::fs::read(&p).unwrap();
    assert!(
        is_corrupt(StoreWAL::open(&p)),
        "nonzero v1 header flags must be rejected as DataCorruption"
    );
    assert_eq!(
        std::fs::read(&p).unwrap(),
        before,
        "failed open must leave the file byte-unchanged"
    );
}

// ---------------------------------------------------------------------------
// Checkpoint: compacts the log to one snapshot section, preserving state.
// ---------------------------------------------------------------------------

#[test]
fn checkpoint_compacts_log_and_preserves_state() {
    let p = tmp();
    let mut recids = Vec::new();
    let s = StoreWAL::open(&p).unwrap();
    for i in 0..50i64 {
        let r = s.put(&i, &L).unwrap();
        s.commit().unwrap();
        recids.push(r);
    }
    let before = file_len(&p);
    s.checkpoint().unwrap();
    let after = file_len(&p);
    assert!(after < before, "checkpoint compacts {before} -> {after}");
    assert!(!ckpt_of(&p).exists(), "temp promoted away by rename");
    // state preserved live, and across reopen from the snapshot section.
    for (i, r) in recids.iter().enumerate() {
        assert_eq!(s.get(*r, &L).unwrap(), Some(i as i64));
    }
    s.verify().unwrap();
    s.close().unwrap();
    let s = StoreWAL::open(&p).unwrap();
    for (i, r) in recids.iter().enumerate() {
        assert_eq!(s.get(*r, &L).unwrap(), Some(i as i64));
    }
    s.verify().unwrap();
    // still writable after a checkpoint.
    let r = s.put(&777i64, &L).unwrap();
    s.commit().unwrap();
    assert_eq!(s.get(r, &L).unwrap(), Some(777));
}

#[test]
fn crash_during_checkpoint_recovers_from_temp() {
    // A checkpoint's temp file, fully fsynced but not yet renamed, must win on
    // reopen when the log is absent (rename is the commit point).
    let p = tmp();
    let r;
    {
        let s = StoreWAL::open(&p).unwrap();
        r = s.put(&314i64, &L).unwrap();
        s.commit().unwrap();
        s.checkpoint().unwrap(); // log is now itself a valid ckpt-format snapshot
        s.close().unwrap();
    }
    // Simulate the crash: copy the (snapshot) log to <file>.ckpt, delete the log.
    std::fs::copy(&p, ckpt_of(&p)).unwrap();
    std::fs::remove_file(&p).unwrap();
    assert!(!p.exists() && ckpt_of(&p).exists());

    let s = StoreWAL::open(&p).unwrap();
    assert_eq!(s.get(r, &L).unwrap(), Some(314), "recovered from ckpt temp");
    assert!(p.exists(), "temp promoted to log");
    assert!(!ckpt_of(&p).exists(), "temp consumed");
    s.verify().unwrap();
}

#[test]
fn auto_checkpoint_bounds_log_growth() {
    let p = tmp();
    let s = StoreWAL::open(&p).unwrap();
    s.set_auto_checkpoint_bytes(4096).unwrap(); // tiny threshold → frequent compaction
    let mut recids = Vec::new();
    for i in 0..200i64 {
        let r = s.put(&i, &L).unwrap();
        s.commit().unwrap();
        recids.push(r);
    }
    // With auto-checkpoint active the log stays far below the naive linear size.
    let sz = file_len(&p);
    assert!(
        sz < 200 * 64,
        "auto-checkpoint should bound log size, got {sz}"
    );
    for (i, r) in recids.iter().enumerate() {
        assert_eq!(s.get(*r, &L).unwrap(), Some(i as i64));
    }
    s.verify().unwrap();
    s.close().unwrap();
    let s = StoreWAL::open(&p).unwrap();
    for (i, r) in recids.iter().enumerate() {
        assert_eq!(s.get(*r, &L).unwrap(), Some(i as i64));
    }
}

// ---------------------------------------------------------------------------
// Streaming replay: a tiny replay window forces refill edges mid-record.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// After close(), every write op (durable OR staged-only) returns StoreClosed —
// close publishes `closed` under the write lock and write ops recheck it there.
// ---------------------------------------------------------------------------

#[test]
fn write_ops_after_close_return_store_closed() {
    let p = tmp();
    let s = StoreWAL::open(&p).unwrap();
    let r = s.put(&1i64, &L).unwrap();
    s.commit().unwrap();
    s.close().unwrap();
    let closed = |res: Result<()>| matches!(res, Err(DbError::StoreClosed));
    assert!(closed(s.update(r, Some(&2i64), &L)));
    assert!(closed(s.delete(r)));
    assert!(closed(s.commit()));
    assert!(closed(s.rollback()));
    assert!(matches!(s.preallocate(), Err(DbError::StoreClosed)));
    assert!(matches!(s.put(&9i64, &L), Err(DbError::StoreClosed)));
    assert!(matches!(s.append(r, &[1]), Err(DbError::StoreClosed)));
    assert!(matches!(s.checkpoint(), Err(DbError::StoreClosed)));
}

// ---------------------------------------------------------------------------
// Headroom that would push a plain record past MAX_CAPACITY must be rejected:
// otherwise commit would emit an invalid cap=0 T_RECORD for
// non-oversize content — a WAL neither Rust nor Java can reopen.
// ---------------------------------------------------------------------------

#[test]
fn headroom_past_max_capacity_is_rejected_and_log_stays_reopenable() {
    use mapdb_rust_store::store::index_val::MAX_CAPACITY;
    let p = tmp();
    let s = StoreWAL::open(&p).unwrap();
    let small = bytes(1, 10); // fits a plain record
    let r = s.put(&small, &R).unwrap();
    s.commit().unwrap();
    // content fits, but content+headroom rounds past MAX_CAPACITY → RecordTooLarge.
    let huge = MAX_CAPACITY; // headroom alone already exceeds the ceiling
    assert!(matches!(
        s.update_with_headroom(r, &small, &R, huge),
        Err(DbError::RecordTooLarge)
    ));
    // usize::MAX headroom must also be rejected (checked arithmetic, no wrap).
    assert!(matches!(
        s.update_with_headroom(r, &small, &R, usize::MAX),
        Err(DbError::RecordTooLarge)
    ));
    // the rejected update staged nothing invalid: a normal commit + reopen works.
    s.update(r, Some(&bytes(2, 20)), &R).unwrap();
    s.commit().unwrap();
    s.close().unwrap();
    let s = StoreWAL::open(&p).unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(bytes(2, 20)));
    s.verify().unwrap();
}

#[test]
fn streaming_replay_with_tiny_window() {
    let p = tmp();
    let mut recids = Vec::new();
    {
        let s = StoreWAL::open_with(&p, true, 8).unwrap();
        for i in 0..30u64 {
            let v = bytes(i, 40 + (i as usize % 17));
            let r = s.put(&v, &R).unwrap();
            s.commit().unwrap();
            recids.push((r, v));
        }
        s.close().unwrap();
    }
    // reopen with an 8-byte replay window: records span many refills.
    let s = StoreWAL::open_with(&p, true, 8).unwrap();
    for (r, v) in &recids {
        assert_eq!(s.get(*r, &R).unwrap(), Some(v.clone()));
    }
    s.verify().unwrap();
}

/// Regression: close() is idempotent — a clean
/// second close returns Ok. The companion poisoned-double-close path (second
/// close must RETRY the directory fsync instead of reporting Ok while the
/// checkpoint rename is still unconfirmed) needs directory-fsync fault
/// injection, which this environment cannot do; it is guarded by the
/// poison-aware early return in `StoreWAL::close`.
#[test]
fn double_close_is_ok() {
    let p = tmp();
    let s = StoreWAL::open(&p).unwrap();
    let _r = s.put(&7i64, &L).unwrap();
    s.commit().unwrap();
    s.close().unwrap();
    s.close().unwrap();
    assert!(s.is_closed());
}
