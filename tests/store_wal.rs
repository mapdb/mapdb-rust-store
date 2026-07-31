//! `StoreWAL` (format v3) black-box tests: commit durability across reopen,
//! multi-section and multi-segment replay, delete/append replay, rollback,
//! torn-tail truncation vs mid-log corruption, segment rollover, the D1 legacy
//! boundary, the store lock, delete-on-close, and streaming-replay refill edges.
//!
//! The inner `StoreDirect` is heap-backed, so *all* durability is carried by the
//! log — a reopen rebuilds the record map purely from replay.
//!
//! Byte-level format tests (the H/S/K/R decision tables, hand-built images,
//! doctored headers) live with the modules that own them, in `wal_segments.rs`
//! and `wal_recover.rs`; the writer's obligations and its fault injection live
//! in `wal.rs`. What is here is what a user can reach through the public API.

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

/// A fresh scratch DIRECTORY plus the base path inside it. v3 stores own a
/// namespace, so every test gets a directory of its own rather than a filename.
fn tmp() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, AtomOrd::Relaxed);
    let dir = std::env::temp_dir().join(format!("mapdb5_wal_it_{}_{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir.join("store.db")
}

/// The segment files of `base`, ascending by sequence number.
fn segments(base: &Path) -> Vec<PathBuf> {
    let prefix = format!("{}.wal.", base.file_name().unwrap().to_str().unwrap());
    let mut v: Vec<PathBuf> = std::fs::read_dir(base.parent().unwrap())
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            let name = p.file_name().unwrap().to_str().unwrap().to_string();
            (name.starts_with(&prefix) && name.len() == prefix.len() + 16).then_some(p)
        })
        .collect();
    v.sort();
    v
}

fn log_len(base: &Path) -> u64 {
    segments(base)
        .iter()
        .map(|p| std::fs::metadata(p).unwrap().len())
        .sum()
}

/// The one segment a single-segment store has.
fn only_segment(base: &Path) -> PathBuf {
    let s = segments(base);
    assert_eq!(s.len(), 1, "expected exactly one segment, got {s:?}");
    s.into_iter().next().unwrap()
}

fn is_corrupt(r: Result<StoreWAL>) -> bool {
    matches!(r, Err(DbError::DataCorruption(_)))
}

/// Everything the store owns, as (name, bytes) — for "a refused open changed
/// nothing" assertions.
fn dir_image(base: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(base.parent().unwrap())
        .unwrap()
        .map(|e| {
            let p = e.unwrap().path();
            (
                p.file_name().unwrap().to_str().unwrap().to_string(),
                std::fs::read(&p).unwrap_or_default(),
            )
        })
        // The lock file is not part of the store's data and a refused open
        // legitimately creates it: §3.1's lock is taken BEFORE anything is
        // inspected, which is what stops two openers inspecting at once.
        .filter(|(name, _)| !name.ends_with(".lock"))
        .collect();
    v.sort();
    v
}

// ---------------------------------------------------------------------------
// D1: the legacy boundary. Three pre-existing artifacts refuse the open, none
// of them is deleted. The v1 opener took the WAL FILE path, so the same call
// site now passes what v3 reads as a BASE — a fresh empty store opened beside
// the user's only durable copy is the outcome these rows exist to prevent.
// ---------------------------------------------------------------------------

#[test]
fn a_v1_single_file_log_refuses_the_open_and_is_not_touched() {
    for (suffix, what) in [(".wal", "v1 log"), ("", "bare base"), (".ckpt", "v1 temp")] {
        let base = tmp();
        let mut p = base.clone().into_os_string();
        p.push(suffix);
        let victim = PathBuf::from(p);
        // A real v1 header: magic, version 1, flags 0.
        let mut v1 = b"MDBS.WAL".to_vec();
        v1.extend_from_slice(&1i32.to_be_bytes());
        v1.extend_from_slice(&0i32.to_be_bytes());
        std::fs::write(&victim, &v1).unwrap();

        let before = dir_image(&base);
        assert!(
            is_corrupt(StoreWAL::open(&base)),
            "{what}: a v1 artifact must refuse the open"
        );
        assert_eq!(dir_image(&base), before, "{what}: nothing may be deleted");
        assert!(
            segments(&base).is_empty(),
            "{what}: no v3 segment may be created beside it"
        );
    }
}

#[test]
fn a_directory_at_a_legacy_name_is_not_a_legacy_artifact() {
    // Regular files only for the two rows that ARE regular-file rows: a
    // directory at the base or at `<base>.wal` is somebody else's, and refusing
    // there would make ordinary layouts unopenable.
    for suffix in ["", ".wal"] {
        let base = tmp();
        let mut p = base.clone().into_os_string();
        p.push(suffix);
        std::fs::create_dir(PathBuf::from(p)).unwrap();
        let s = StoreWAL::open(&base).unwrap();
        s.close().unwrap();
    }
}

#[test]
fn anything_at_all_at_the_ckpt_name_refuses() {
    // D1 makes `.ckpt` an EXISTENCE sentinel, not a regular-file one, and the
    // distinction is deliberate: that file may be the only recoverable copy
    // after a v1 crash, so "there is something at that name and I cannot tell
    // what it is" is not a licence to create a fresh store beside it. This test
    // used to assert the opposite.
    for make in ["file", "dir", "symlink"] {
        let base = tmp();
        let mut p = base.clone().into_os_string();
        p.push(".ckpt");
        let p = PathBuf::from(p);
        match make {
            "file" => std::fs::write(&p, b"old checkpoint").unwrap(),
            "dir" => std::fs::create_dir(&p).unwrap(),
            _ => std::os::unix::fs::symlink("/nonexistent", &p).unwrap(),
        }
        match StoreWAL::open(&base) {
            Err(DbError::DataCorruption(m)) => {
                assert!(m.to_string().contains("v1 checkpoint temp"), "{make}: {m}")
            }
            other => panic!("{make}: expected a refusal, got {:?}", other.map(|_| "Ok")),
        }
    }
}

// ---------------------------------------------------------------------------
// Durability: only committed state survives a reopen.
// ---------------------------------------------------------------------------

#[test]
fn a_fresh_store_creates_one_segment_at_sequence_one() {
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.segment_seqs(), vec![1]);
    assert_eq!(s.next_lsn(), 1);
    // 36-byte header, no sections.
    assert_eq!(log_len(&base), 36);
    assert!(
        s.open_segment_files() <= 1,
        "at most the active segment holds a descriptor"
    );
    s.close().unwrap();
    assert!(only_segment(&base).exists());
}

#[test]
fn committed_state_survives_reopen() {
    let base = tmp();
    let (a, b, c);
    {
        let s = StoreWAL::open(&base).unwrap();
        a = s.put(&42i64, &L).unwrap();
        b = s.put(&bytes(1, 300), &R).unwrap();
        c = s.preallocate().unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(a, &L).unwrap(), Some(42));
    assert_eq!(s.get(b, &R).unwrap(), Some(bytes(1, 300)));
    assert_eq!(s.get(c, &L).unwrap(), None, "preallocated, still null");
    s.verify().unwrap();
}

#[test]
fn uncommitted_state_is_lost_on_reopen() {
    let base = tmp();
    let a;
    {
        let s = StoreWAL::open(&base).unwrap();
        a = s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        s.update(a, Some(&2i64), &L).unwrap(); // never committed
        s.close().unwrap();
    }
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(a, &L).unwrap(), Some(1));
}

#[test]
fn multi_section_replay_and_last_write_wins() {
    let base = tmp();
    let a;
    {
        let s = StoreWAL::open(&base).unwrap();
        a = s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        for v in 2..=20i64 {
            s.update(a, Some(&v), &L).unwrap();
            s.commit().unwrap();
        }
        assert_eq!(s.next_lsn(), 21, "one LSN per committed transaction");
        s.close().unwrap();
    }
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(a, &L).unwrap(), Some(20));
    assert_eq!(s.next_lsn(), 21, "recovery resumes at the same LSN");
    s.verify().unwrap();
}

#[test]
fn delete_and_append_replay() {
    let base = tmp();
    let (a, b);
    {
        let s = StoreWAL::open(&base).unwrap();
        a = s.put(&bytes(3, 40), &R).unwrap();
        b = s.put(&7i64, &L).unwrap();
        s.commit().unwrap();
        // an append against a committed base: logged as a delta stamped with
        // the LSN of the section that established the content.
        s.update_with_headroom(a, &bytes(3, 40), &R, 64).unwrap();
        s.commit().unwrap();
        s.append(a, &[9u8; 8]).unwrap();
        s.delete(b).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    let s = StoreWAL::open(&base).unwrap();
    let mut want = bytes(3, 40);
    want.extend_from_slice(&[9u8; 8]);
    assert_eq!(s.get(a, &R).unwrap(), Some(want));
    assert!(matches!(s.get(b, &L), Err(DbError::GetVoid(_))));
    s.verify().unwrap();
}

#[test]
fn an_append_replays_across_a_segment_boundary() {
    // The delta cites its base image by LSN, and the base is now in a segment
    // the delta is not in. Recovery must still find it.
    let base = tmp();
    let a;
    {
        let s = StoreWAL::open_segment_bytes(&base, 128).unwrap();
        a = s.put(&bytes(5, 60), &R).unwrap();
        s.commit().unwrap();
        s.update_with_headroom(a, &bytes(5, 60), &R, 128).unwrap();
        s.commit().unwrap();
        for i in 0..8 {
            s.append(a, &[i as u8; 4]).unwrap();
            s.commit().unwrap();
        }
        assert!(
            s.segment_seqs().len() > 1,
            "the workload must have rolled over"
        );
        s.close().unwrap();
    }
    let s = StoreWAL::open(&base).unwrap();
    let mut want = bytes(5, 60);
    for i in 0..8 {
        want.extend_from_slice(&[i as u8; 4]);
    }
    assert_eq!(s.get(a, &R).unwrap(), Some(want));
    s.verify().unwrap();
}

#[test]
fn linked_oversize_record_replays() {
    use mapdb_rust_store::store::index_val::MAX_CAPACITY;
    let base = tmp();
    let big = bytes(9, MAX_CAPACITY + 5000); // past the plain-record ceiling
    let a;
    {
        let s = StoreWAL::open(&base).unwrap();
        a = s.put(&big, &R).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(a, &R).unwrap(), Some(big));
    s.verify().unwrap();
}

#[test]
fn rollback_discards_staged_keeps_committed() {
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();
    let a = s.put(&1i64, &L).unwrap();
    s.commit().unwrap();
    let b = s.put(&2i64, &L).unwrap();
    s.update(a, Some(&99i64), &L).unwrap();
    s.rollback().unwrap();
    assert_eq!(s.get(a, &L).unwrap(), Some(1));
    assert!(matches!(s.get(b, &L), Err(DbError::GetVoid(_))));
    // A rollback writes nothing, so no LSN was consumed.
    assert_eq!(s.next_lsn(), 2);
    s.verify().unwrap();
}

// ---------------------------------------------------------------------------
// Rollover: the threshold is checked at a section boundary and only when the
// active segment is nonempty, so one section may exceed it and an oversize
// section gets a segment to itself.
// ---------------------------------------------------------------------------

#[test]
fn the_log_rolls_over_at_the_segment_threshold() {
    let base = tmp();
    let mut recids = Vec::new();
    let s = StoreWAL::open_segment_bytes(&base, 200).unwrap();
    for i in 0..30i64 {
        recids.push(s.put(&i, &L).unwrap());
        s.commit().unwrap();
    }
    let seqs = s.segment_seqs();
    assert!(seqs.len() > 3, "expected several segments, got {seqs:?}");
    assert_eq!(
        seqs,
        (1..=seqs.len() as i64).collect::<Vec<_>>(),
        "sequence numbers are consecutive when nothing is retired"
    );
    assert!(
        s.open_segment_files() <= 1,
        "rollover must release the sealed segment's descriptor"
    );
    s.close().unwrap();

    let s = StoreWAL::open(&base).unwrap();
    for (i, r) in recids.iter().enumerate() {
        assert_eq!(s.get(*r, &L).unwrap(), Some(i as i64));
    }
    s.verify().unwrap();
}

#[test]
fn a_section_larger_than_a_segment_gets_a_segment_of_its_own() {
    let base = tmp();
    let s = StoreWAL::open_segment_bytes(&base, 61).unwrap(); // the minimum
    let big = s.put(&bytes(11, 4000), &R).unwrap();
    s.commit().unwrap();
    // The first segment was empty, so it took the oversize section; the next
    // commit rolls because the threshold is now exceeded.
    let after_first = s.segment_seqs();
    assert_eq!(
        after_first,
        vec![1],
        "an empty segment is never rolled past"
    );
    let small = s.put(&1i64, &L).unwrap();
    s.commit().unwrap();
    assert_eq!(s.segment_seqs(), vec![1, 2]);
    s.close().unwrap();

    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(big, &R).unwrap(), Some(bytes(11, 4000)));
    assert_eq!(s.get(small, &L).unwrap(), Some(1));
    s.verify().unwrap();
}

#[test]
fn a_segment_size_below_the_minimum_is_refused() {
    let base = tmp();
    assert!(matches!(
        StoreWAL::open_segment_bytes(&base, 60),
        Err(DbError::WrongConfiguration(_))
    ));
    assert!(segments(&base).is_empty(), "a refused open creates nothing");
}

// ---------------------------------------------------------------------------
// Torn tail vs mid-log corruption. A damaged section at the end of the ACTIVE
// segment is a crash image: truncate and carry on. One followed by a valid
// section is bit rot: refuse.
// ---------------------------------------------------------------------------

#[test]
fn torn_tail_body_is_truncated_not_fatal() {
    let base = tmp();
    let (r1, len_after_first);
    {
        let s = StoreWAL::open(&base).unwrap();
        r1 = s.put(&1000i64, &L).unwrap();
        s.commit().unwrap();
        len_after_first = log_len(&base);
        s.put(&2000i64, &L).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    // Chop one byte into the second section's body → torn tail.
    let seg = only_segment(&base);
    OpenOptions::new()
        .write(true)
        .open(&seg)
        .unwrap()
        .set_len(len_after_first + 1)
        .unwrap();

    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(r1, &L).unwrap(), Some(1000), "first commit survives");
    // W7: the truncated segment is sealed and a successor is rotated in, so no
    // later append ever reuses the torn segment's checksum domain.
    assert_eq!(s.segment_seqs(), vec![1, 2]);
    assert!(
        s.open_segment_files() <= 1,
        "W7 must not leave the truncated predecessor's descriptor behind"
    );
    s.verify().unwrap();
    let r3 = s.put(&3000i64, &L).unwrap();
    s.commit().unwrap();
    s.close().unwrap();

    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(r1, &L).unwrap(), Some(1000));
    assert_eq!(s.get(r3, &L).unwrap(), Some(3000));
    assert_eq!(s.segment_seqs(), vec![1, 2], "a clean open rotates nothing");
    s.verify().unwrap();
}

#[test]
fn torn_tail_within_section_header_is_truncated() {
    let base = tmp();
    let (r1, len_after_first);
    {
        let s = StoreWAL::open(&base).unwrap();
        r1 = s.put(&11i64, &L).unwrap();
        s.commit().unwrap();
        len_after_first = log_len(&base);
        s.put(&22i64, &L).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    // Chop inside the second section's 25-byte header (5 header bytes present).
    OpenOptions::new()
        .write(true)
        .open(only_segment(&base))
        .unwrap()
        .set_len(len_after_first + 5)
        .unwrap();
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(r1, &L).unwrap(), Some(11));
    s.verify().unwrap();
}

#[test]
fn mid_log_body_corruption_is_fatal() {
    let base = tmp();
    {
        let s = StoreWAL::open(&base).unwrap();
        s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        s.put(&2i64, &L).unwrap();
        s.commit().unwrap(); // a valid section AFTER the one we corrupt
        s.close().unwrap();
    }
    // Flip a byte in the FIRST section's body (segment header 36 + section
    // header 25); the valid second section remains.
    let seg = only_segment(&base);
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&seg)
        .unwrap();
    let mut b = [0u8; 1];
    f.read_exact_at(&mut b, 36 + 25).unwrap();
    b[0] ^= 0xFF;
    f.write_all_at(&b, 36 + 25).unwrap();
    f.sync_all().unwrap();
    drop(f);

    assert!(
        is_corrupt(StoreWAL::open(&base)),
        "a corrupt section followed by a valid one must refuse (not a torn tail)"
    );
}

#[test]
fn mid_log_header_corruption_is_fatal() {
    let base = tmp();
    {
        let s = StoreWAL::open(&base).unwrap();
        s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        s.put(&2i64, &L).unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }
    // Corrupt the FIRST section's tag byte; its declared bodyLen still points at
    // the valid, exactly-next-LSN second section → mid-log corruption.
    let seg = only_segment(&base);
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&seg)
        .unwrap();
    let mut b = [0u8; 1];
    f.read_exact_at(&mut b, 36).unwrap();
    b[0] ^= 0x55;
    f.write_all_at(&b, 36).unwrap();
    f.sync_all().unwrap();
    drop(f);
    assert!(is_corrupt(StoreWAL::open(&base)));
}

#[test]
fn a_damaged_segment_header_refuses_below_the_highest_name() {
    let base = tmp();
    {
        let s = StoreWAL::open_segment_bytes(&base, 100).unwrap();
        for i in 0..10i64 {
            s.put(&i, &L).unwrap();
            s.commit().unwrap();
        }
        assert!(s.segment_seqs().len() > 2);
        s.close().unwrap();
    }
    // Bump the version word of the FIRST segment: a CRC-valid header carrying
    // wrong content is corruption wherever it appears, and the reseal is what
    // makes it a semantic fault rather than a torn create.
    let seg = &segments(&base)[0];
    let mut hdr = std::fs::read(seg).unwrap();
    hdr[8..12].copy_from_slice(&99i32.to_be_bytes());
    let crc = crc32fast::hash(&hdr[..32]) as i32;
    hdr[32..36].copy_from_slice(&crc.to_be_bytes());
    let f = OpenOptions::new().write(true).open(seg).unwrap();
    f.write_all_at(&hdr[..36], 0).unwrap();
    f.sync_all().unwrap();
    drop(f);
    assert!(is_corrupt(StoreWAL::open(&base)));
}

// ---------------------------------------------------------------------------
// The store lock: one writer at a time, across processes and inside this one.
// ---------------------------------------------------------------------------

#[test]
fn a_second_open_of_the_same_store_is_refused() {
    let base = tmp();
    let first = StoreWAL::open(&base).unwrap();
    assert!(matches!(StoreWAL::open(&base), Err(DbError::Locked(_))));
    first.close().unwrap();
    // Released on close: the namespace is available again.
    let second = StoreWAL::open(&base).unwrap();
    second.close().unwrap();
}

// ---------------------------------------------------------------------------
// checkpoint(): the incremental cleaner with its budget set to "everything".
// Roll, re-emit every record the range below still owns as 'C' images, verify
// (W10), write the forced 'K', unlink. The v1 whole-file rewrite is gone.
// ---------------------------------------------------------------------------

#[test]
fn checkpoint_compacts_the_log_and_preserves_state() {
    let base = tmp();
    let mut recids = Vec::new();
    let s = StoreWAL::open_segment_bytes(&base, 200).unwrap();
    for i in 0..60i64 {
        let r = s.put(&i, &L).unwrap();
        s.commit().unwrap();
        recids.push(r);
        // keep rewriting one record so the log holds superseded images
        s.update(recids[0], Some(&i), &L).unwrap();
        s.commit().unwrap();
    }
    let before = log_len(&base);
    let segs_before = s.segment_seqs().len();
    assert!(segs_before > 5, "expected a multi-segment log");
    s.checkpoint().unwrap();
    let after = log_len(&base);
    assert!(
        after < before,
        "checkpoint must compact the log: {before} -> {after}"
    );
    assert!(
        s.segment_seqs().len() < segs_before,
        "and retire segments: {segs_before} -> {}",
        s.segment_seqs().len()
    );
    let (written, retired) = s.cleaner_bytes();
    assert!(
        retired > written,
        "it must pay for itself: {retired} vs {written}"
    );

    // State preserved live, and across a reopen that replays only what is left.
    for (i, r) in recids.iter().enumerate() {
        let want = if i == 0 { 59 } else { i as i64 };
        assert_eq!(s.get(*r, &L).unwrap(), Some(want));
    }
    s.verify().unwrap();
    s.close().unwrap();

    let s = StoreWAL::open(&base).unwrap();
    for (i, r) in recids.iter().enumerate() {
        let want = if i == 0 { 59 } else { i as i64 };
        assert_eq!(s.get(*r, &L).unwrap(), Some(want));
    }
    s.verify().unwrap();
    // still writable after a clean
    let r = s.put(&777i64, &L).unwrap();
    s.commit().unwrap();
    assert_eq!(s.get(r, &L).unwrap(), Some(777));
}

#[test]
fn checkpoint_on_an_empty_log_is_a_no_op() {
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();
    s.checkpoint().unwrap();
    assert_eq!(
        s.segment_seqs(),
        vec![1],
        "nothing to roll, nothing to retire"
    );
    let a = s.put(&5i64, &L).unwrap();
    s.commit().unwrap();
    // One nonempty segment: the roll creates its successor and the cycle then
    // retires the original behind a mark.
    s.checkpoint().unwrap();
    assert_eq!(s.segment_seqs(), vec![2]);
    assert_eq!(s.get(a, &L).unwrap(), Some(5));
    s.close().unwrap();
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(
        s.get(a, &L).unwrap(),
        Some(5),
        "the image replays from the 'C'"
    );
    s.verify().unwrap();
}

#[test]
fn a_deleted_record_is_not_re_emitted_by_a_clean() {
    let base = tmp();
    let s = StoreWAL::open_segment_bytes(&base, 200).unwrap();
    let mut live = Vec::new();
    let mut gone = Vec::new();
    for i in 0..20i64 {
        let r = s.put(&i, &L).unwrap();
        s.commit().unwrap();
        if i % 2 == 0 {
            live.push((r, i));
        } else {
            gone.push(r);
        }
    }
    for r in &gone {
        s.delete(*r).unwrap();
    }
    s.commit().unwrap();
    s.checkpoint().unwrap();
    s.close().unwrap();

    let s = StoreWAL::open(&base).unwrap();
    for (r, v) in &live {
        assert_eq!(s.get(*r, &L).unwrap(), Some(*v));
    }
    for r in &gone {
        assert!(matches!(s.get(*r, &L), Err(DbError::GetVoid(_))));
    }
    s.verify().unwrap();
}

#[test]
fn a_base_in_the_retired_range_is_re_emitted_with_its_delta_folded_in() {
    // The worry a clean has to answer: a delta ABOVE the retiring range whose
    // base image lies INSIDE it. It cannot survive as a dangling reference,
    // because the recid's state entry is in the range too — so it is a
    // candidate, and its image is re-emitted with the delta already folded in.
    let base = tmp();
    let a;
    let mut want;
    {
        let s = StoreWAL::open_segment_bytes(&base, 200).unwrap();
        want = bytes(21, 60);
        a = s.put(&want, &R).unwrap();
        s.commit().unwrap();
        s.update_with_headroom(a, &want, &R, 200).unwrap();
        s.commit().unwrap();
        // deltas, each in a later section (and, at this segment size, later
        // segments) than the image they extend
        for i in 0..6u8 {
            s.append(a, &[i; 8]).unwrap();
            s.commit().unwrap();
            want.extend_from_slice(&[i; 8]);
        }
        assert!(s.segment_seqs().len() > 1);
        s.checkpoint().unwrap();
        assert_eq!(s.get(a, &R).unwrap(), Some(want.clone()));
        s.close().unwrap();
    }
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(a, &R).unwrap(), Some(want));
    s.verify().unwrap();
}

#[test]
fn automatic_cleaning_bounds_log_growth() {
    // A small live set overwritten many times: the log fills with superseded
    // images while the store's footprint stays flat, which is the shape the
    // trigger exists for. The footprint is page-granular (it reports ~2 MiB for
    // a store holding a few hundred bytes), so the amplification term — not the
    // floor — is what decides here; that bias is documented on `cleaning_target`.
    let base = tmp();
    let s = StoreWAL::open_segment_bytes(&base, 64 << 10).unwrap();
    s.set_min_log_bytes(4096).unwrap();
    s.set_space_amplification(1).unwrap();
    let mut recids = Vec::new();
    for i in 0..20u64 {
        recids.push(s.put(&bytes(i, 4000), &R).unwrap());
        s.commit().unwrap();
    }
    for i in 0..1200u64 {
        let victim = recids[(i as usize) % recids.len()];
        s.update(victim, Some(&bytes(10_000 + i, 4000)), &R)
            .unwrap();
        s.commit().unwrap();
    }
    let (written, retired) = s.cleaner_bytes();
    assert!(
        retired > 0,
        "automatic cleaning must have retired segments (wrote {written})"
    );
    let unbounded = 1220 * 4100;
    assert!(
        log_len(&base) < unbounded / 2,
        "the log must stay well below its unbounded size, got {}",
        log_len(&base)
    );
    s.verify().unwrap();
    let snapshot: Vec<Option<Vec<u8>>> = recids.iter().map(|r| s.get(*r, &R).unwrap()).collect();
    s.close().unwrap();

    let s = StoreWAL::open(&base).unwrap();
    for (r, want) in recids.iter().zip(&snapshot) {
        assert_eq!(
            &s.get(*r, &R).unwrap(),
            want,
            "state survives the retirement"
        );
    }
    s.verify().unwrap();
}

// ---------------------------------------------------------------------------
// Lifecycle.
// ---------------------------------------------------------------------------

#[test]
fn write_ops_after_close_return_store_closed() {
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();
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

#[test]
fn double_close_is_ok() {
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();
    let _r = s.put(&7i64, &L).unwrap();
    s.commit().unwrap();
    s.close().unwrap();
    s.close().unwrap();
    assert!(s.is_closed());
}

#[test]
fn delete_on_close_removes_the_whole_namespace() {
    let base = tmp();
    let unrelated = base.parent().unwrap().join("not-ours.txt");
    std::fs::write(&unrelated, b"keep me").unwrap();
    let s = StoreWAL::open_segment_bytes(&base, 100).unwrap();
    s.set_delete_on_close(true);
    for i in 0..10i64 {
        s.put(&i, &L).unwrap();
        s.commit().unwrap();
    }
    assert!(s.segment_seqs().len() > 2);
    s.close().unwrap();

    assert!(segments(&base).is_empty(), "every segment is deleted");
    let mut lock = base.clone().into_os_string();
    lock.push(".lock");
    assert!(!PathBuf::from(lock).exists(), "the lock file goes last");
    assert!(unrelated.exists(), "unrelated names are preserved");
}

// ---------------------------------------------------------------------------
// Records and framing.
// ---------------------------------------------------------------------------

#[test]
fn headroom_past_max_capacity_is_rejected_and_log_stays_reopenable() {
    use mapdb_rust_store::store::index_val::MAX_CAPACITY;
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();
    let small = bytes(1, 10); // fits a plain record
    let r = s.put(&small, &R).unwrap();
    s.commit().unwrap();
    // content fits, but content+headroom rounds past MAX_CAPACITY → RecordTooLarge.
    assert!(matches!(
        s.update_with_headroom(r, &small, &R, MAX_CAPACITY),
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
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(bytes(2, 20)));
    s.verify().unwrap();
}

#[test]
fn an_append_that_overflows_the_headroom_clamps_rather_than_going_linked() {
    // Headroom is a hint; the record is the promise. Appends can push a staged
    // base to the plain maximum, and the requested headroom then overflows it.
    // The capacity clamps to the ceiling — falling to "capacity 0, store it
    // linked" would emit a T_RECORD the decoder rejects as a garbage capacity,
    // i.e. a log that cannot be reopened.
    use mapdb_rust_store::store::index_val::MAX_CAPACITY;
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();
    let content = bytes(4, MAX_CAPACITY - 4096);
    let r = s.put(&content, &R).unwrap();
    // A staged base reports unlimited capacity, so this is accepted.
    let tail = bytes(5, 4000);
    s.append(r, &tail).unwrap();
    s.commit().unwrap();
    let mut want = content.clone();
    want.extend_from_slice(&tail);
    assert_eq!(s.get(r, &R).unwrap(), Some(want.clone()));
    s.close().unwrap();
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(want));
    s.verify().unwrap();
}

#[test]
fn a_refused_append_stages_nothing() {
    // REFUSED is a no-op, so it must leave no staged entry behind: an empty one
    // used to be classified as a T_PREALLOC at commit, which burns an LSN and
    // names a content-live record — exactly what replay rejects.
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();
    let r = s.put(&bytes(6, 100), &R).unwrap();
    s.commit().unwrap();
    let lsn_before = s.next_lsn();
    let refused = s.append(r, &bytes(7, 5000)).unwrap();
    assert_eq!(refused, mapdb_rust_store::store::AppendResult::Refused);
    s.commit().unwrap();
    assert_eq!(s.next_lsn(), lsn_before, "a refused append commits nothing");
    s.close().unwrap();
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(bytes(6, 100)));
    s.verify().unwrap();
}

#[test]
fn a_zero_length_append_is_a_no_op_on_every_record_shape() {
    // The contract says a zero-length append changes nothing. Staging an empty
    // entry for it breaks that in three different ways depending on what the
    // record already is, so all three shapes are pinned here.
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();

    // (1) A committed plain record: no section, so no LSN is burnt.
    let plain = s.put(&bytes(3, 100), &R).unwrap();
    s.commit().unwrap();
    let lsn_before = s.next_lsn();
    assert_eq!(
        s.append(plain, &[]).unwrap(),
        mapdb_rust_store::store::AppendResult::NewSize(100),
        "the unchanged size is returned"
    );
    s.commit().unwrap();
    assert_eq!(s.next_lsn(), lsn_before, "a no-op commits nothing");

    // (2) A committed LINKED record, where the inner store refuses every
    // append: a staged empty append would be emitted, forced, and only THEN
    // refused — on the post-durability path, which fails the store closed.
    let linked = s.put(&bytes(4, 1_200_000), &R).unwrap();
    s.commit().unwrap();
    let lsn_before = s.next_lsn();
    s.append(linked, &[]).unwrap();
    s.commit().unwrap();
    assert_eq!(s.next_lsn(), lsn_before);
    assert!(
        !s.is_closed(),
        "a documented no-op must not close the store"
    );

    // (3) A freshly preallocated recid, which must reopen preallocated rather
    // than as a content-bearing empty record: the empty staged entry used to be
    // classified T_PREALLOC on an untouched record, or T_RECORD with empty
    // content once it carried an append.
    let pre = s.preallocate().unwrap();
    s.append(pre, &[]).unwrap();
    s.commit().unwrap();
    assert_eq!(s.get(pre, &R).unwrap(), None, "still preallocated");

    s.close().unwrap();
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(plain, &R).unwrap(), Some(bytes(3, 100)));
    assert_eq!(s.get(linked, &R).unwrap(), Some(bytes(4, 1_200_000)));
    assert_eq!(s.get(pre, &R).unwrap(), None);
    s.verify().unwrap();
}

#[test]
fn a_zero_length_append_on_an_already_staged_record_keeps_the_staging() {
    // The other half of the rule: it stages nothing NEW, but it must not
    // discard staging that was already there.
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();
    let r = s.put(&bytes(5, 40), &R).unwrap();
    // With headroom, or the pending append below is REFUSED for want of
    // capacity and there is no staging left for the no-op to preserve.
    s.update_with_headroom(r, &bytes(5, 40), &R, 64).unwrap();
    s.commit().unwrap();
    s.append(r, &bytes(6, 10)).unwrap();
    assert_eq!(
        s.append(r, &[]).unwrap(),
        mapdb_rust_store::store::AppendResult::NewSize(50),
        "the pending append is still counted"
    );
    s.commit().unwrap();
    let mut want = bytes(5, 40);
    want.extend_from_slice(&bytes(6, 10));
    assert_eq!(s.get(r, &R).unwrap(), Some(want.clone()));
    s.close().unwrap();
    let s = StoreWAL::open(&base).unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(want));
    s.verify().unwrap();
}

fn running_as_root() -> bool {
    // The permission-dependent tests below prove nothing as root, which ignores
    // the mode bits they turn on.
    unsafe { libc_geteuid() == 0 }
}

extern "C" {
    #[link_name = "geteuid"]
    fn libc_geteuid() -> u32;
}

#[test]
fn a_cleaner_read_failure_fails_the_store_closed() {
    // Automatic cleaning runs INSIDE commit, after the section is forced,
    // applied and the transaction cleared. An I/O error there must fail closed:
    // returning it with the handle open means the store's I/O is broken, it
    // says so once, and then keeps accepting writes. Only the SEMANTIC refusals
    // (W10, identity disagreement) rewind and keep the handle.
    if running_as_root() {
        return;
    }
    let base = tmp();
    // Small segments so the log actually rolls: 40 tiny commits fit inside one
    // 4 KiB segment and would leave nothing to retire.
    let s = StoreWAL::open_segment_bytes(&base, 128).unwrap();
    for i in 0..40i64 {
        s.put(&i, &L).unwrap();
        s.commit().unwrap();
    }
    let seqs = s.segment_seqs();
    assert!(seqs.len() > 2, "need a retiring range: {seqs:?}");
    // Make the lowest retiring segment unreadable, so the cleaner's scan fails
    // when it reopens it on demand.
    let mut p = base.clone().into_os_string();
    p.push(format!(".wal.{:016x}", seqs[0]));
    let victim = PathBuf::from(p);
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&victim, std::fs::Permissions::from_mode(0o000)).unwrap();

    let e = s
        .checkpoint()
        .expect_err("the scan cannot read a retiring segment");
    assert!(
        matches!(e, DbError::Io(_)),
        "an I/O error, not a refusal: {e:?}"
    );
    assert!(
        s.is_closed(),
        "an I/O error in the cleaner fails the store CLOSED"
    );
    assert!(matches!(s.commit(), Err(DbError::StoreClosed)));
    std::fs::set_permissions(&victim, std::fs::Permissions::from_mode(0o600)).unwrap();
}

#[test]
fn delete_on_close_refuses_when_it_cannot_read_the_namespace() {
    // D2 requires propagation. A delete that cannot enumerate must NOT report a
    // clean removal — it would have unlinked the lock, cleared its list and
    // fsynced, while segments are still on disk.
    //
    // This covers the `read_dir` half. The other half — a `file_type()` failure
    // on a name that already matched the segment grammar — is fixed and argued
    // from the source but NOT covered here: on Linux the entry type comes back
    // from `d_type` in the directory read itself, so it answers correctly even
    // with search permission removed, and there is no portable way to make it
    // fail. Verified by probe rather than assumed.
    if running_as_root() {
        return;
    }
    let dir = tmp();
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("s.db");
    let s = StoreWAL::open(&base).unwrap();
    s.put(&7i64, &L).unwrap();
    s.commit().unwrap();
    s.set_delete_on_close(true);
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o300)).unwrap();
    let e = s
        .close()
        .expect_err("an unreadable namespace must not delete silently");
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(matches!(e, DbError::Io(_)), "{e:?}");
    let left: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".wal."))
        .collect();
    assert!(!left.is_empty(), "the segments really are still there");
}

#[test]
fn the_config_setters_refuse_after_close() {
    // This pins the SEQUENTIAL contract only, and says so: the defect it was
    // written for is a race — `close` publishes the flag while holding the
    // write lock, so a check taken before acquiring that lock can be overtaken
    // and the setter then mutates a torn-down state and reports success. The
    // fix is to lock first and re-check under it; that fix is argued from the
    // lock discipline, not proved here, because a deterministic race needs a
    // hook the store does not have. Verified: reverting the fix leaves this
    // test green.
    let base = tmp();
    let s = StoreWAL::open(&base).unwrap();
    s.close().unwrap();
    assert!(matches!(
        s.set_min_log_bytes(1 << 20),
        Err(DbError::StoreClosed)
    ));
    assert!(matches!(
        s.set_space_amplification(4),
        Err(DbError::StoreClosed)
    ));
}

#[test]
fn streaming_replay_with_tiny_window() {
    let base = tmp();
    let mut recids = Vec::new();
    {
        let s = StoreWAL::open_with(&base, true, 8).unwrap();
        for i in 0..30u64 {
            let v = bytes(i, 40 + (i as usize % 17));
            let r = s.put(&v, &R).unwrap();
            s.commit().unwrap();
            recids.push((r, v));
        }
        s.close().unwrap();
    }
    // reopen with an 8-byte replay window: records span many refills.
    let s = StoreWAL::open_with(&base, true, 8).unwrap();
    for (r, v) in &recids {
        assert_eq!(s.get(*r, &R).unwrap(), Some(v.clone()));
    }
    s.verify().unwrap();
}

#[test]
fn a_body_larger_than_the_writers_buffer_is_streamed_whole() {
    // The writer coalesces small framing through a 64 KiB buffer and writes
    // large payloads where they lie; a body that crosses both paths must round
    // trip byte-exactly.
    let base = tmp();
    let mut recids = Vec::new();
    {
        let s = StoreWAL::open(&base).unwrap();
        for i in 0..4u64 {
            recids.push((s.put(&bytes(i, 100_000), &R).unwrap(), bytes(i, 100_000)));
        }
        for i in 0..40u64 {
            recids.push((s.put(&bytes(100 + i, 37), &R).unwrap(), bytes(100 + i, 37)));
        }
        s.commit().unwrap(); // ONE section, ~400 KiB, both paths exercised
        s.close().unwrap();
    }
    let s = StoreWAL::open(&base).unwrap();
    for (r, v) in &recids {
        assert_eq!(s.get(*r, &R).unwrap(), Some(v.clone()));
    }
    s.verify().unwrap();
}
