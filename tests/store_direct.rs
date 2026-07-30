//! StoreDirect-specific tests (spec 02 §5, spec 05 §3): verify() tiling oracle
//! after every op, linked/oversize records, free-space reuse, file reopen +
//! crash detection, and a differential fuzz against the StoreByteArray oracle.

use mapdb_rust_store::error::{DbError, Result};
use mapdb_rust_store::io::{DataInput2, DataOutput2};
use mapdb_rust_store::ser::Serializer;
use mapdb_rust_store::store::{AppendResult, Store, StoreByteArray, StoreDelta, StoreDirect};
use std::cmp::Ordering;
use std::os::unix::fs::FileExt;

/// Raw-bytes serializer: content == value (uses the framed `size`), so a record's
/// on-disk content equals the logical value — ideal for differential testing.
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
    // deterministic LCG-filled buffer
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

#[test]
fn basic_roundtrip_and_verify() {
    let s = StoreDirect::new_heap().unwrap();
    let v = bytes(1, 100);
    let r = s.put(&v, &R).unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(v.clone()));
    // update in place (smaller) keeps capacity, verify tiles
    let v2 = bytes(2, 50);
    s.update(r, Some(&v2), &R).unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(v2));
    // grow beyond capacity: relocates
    let v3 = bytes(3, 5000);
    s.update(r, Some(&v3), &R).unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(v3));
    s.delete(r).unwrap();
    s.verify().unwrap();
    assert!(matches!(s.get(r, &R), Err(DbError::GetVoid(_))));
}

#[test]
fn linked_oversize_record() {
    let s = StoreDirect::new_heap().unwrap();
    // > MAX_CAPACITY (~1 MiB - 48) → stored as a linked chunk chain
    let big = bytes(7, 3_000_000);
    let r = s.put(&big, &R).unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(big.clone()));
    // update a linked record to another linked size
    let big2 = bytes(8, 2_100_000);
    s.update(r, Some(&big2), &R).unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(big2));
    // shrink to a plain record
    let small = bytes(9, 10);
    s.update(r, Some(&small), &R).unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(r, &R).unwrap(), Some(small));
    s.delete(r).unwrap();
    s.verify().unwrap();
}

#[test]
fn free_space_reuse_and_compact() {
    let s = StoreDirect::new_heap().unwrap();
    let mut rs = vec![];
    for i in 0..200u64 {
        rs.push(s.put(&bytes(i, 500), &R).unwrap());
    }
    s.verify().unwrap();
    // delete half — frees extents that later puts should reuse
    for i in (0..200).step_by(2) {
        s.delete(rs[i]).unwrap();
    }
    s.verify().unwrap();
    let size_before = s.get_current_size();
    for i in 0..100u64 {
        s.put(&bytes(1000 + i, 480), &R).unwrap();
    }
    s.verify().unwrap();
    // full compaction rebuilds dense; content of survivors preserved
    let survivor = rs[1];
    let expect = s.get(survivor, &R).unwrap();
    s.compact().unwrap();
    s.verify().unwrap();
    assert_eq!(s.get(survivor, &R).unwrap(), expect);
    // size didn't explode
    assert!(s.get_current_size() <= size_before + 100 * 512 + (1 << 20));
}

#[test]
fn append_and_headroom() {
    // A preallocated record's first append establishes it; capacity is rounded
    // up to the 16-byte allocation unit (capBytesFor), so a small follow-up append
    // still fits until the rounded capacity is exhausted.
    let s = StoreDirect::new_heap().unwrap();
    let r = s.preallocate().unwrap();
    assert_eq!(s.append(r, &[1, 2, 3]).unwrap(), AppendResult::NewSize(3));
    s.verify().unwrap();
    // used=3 + 4-byte counter = 7, capacity rounds to 16 → 9 spare bytes remain.
    assert_eq!(s.capacity_remaining(r).unwrap(), 9);
    assert_eq!(s.append(r, &[4]).unwrap(), AppendResult::NewSize(4));
    // headroom provisions appendable capacity beyond the content. Capacity rounds
    // up to the 16-byte unit: need = 4 + 20 + 16 = 40 → capBytesFor(40) = 48, so
    // remaining = 48 - 4 - 20 = 24 (contract guarantees only >= headroom).
    let r2 = s.put(&bytes(5, 20), &R).unwrap();
    s.update_with_headroom(r2, &bytes(6, 20), &R, 16).unwrap();
    let rem = s.capacity_remaining(r2).unwrap();
    assert!(rem >= 16, "headroom must be honoured, got {rem}");
    assert!(matches!(
        s.append(r2, &[9; 10]).unwrap(),
        AppendResult::NewSize(30)
    ));
    s.verify().unwrap();
    assert_eq!(s.capacity_remaining(r2).unwrap(), rem - 10);
}

#[test]
fn file_reopen_clean() {
    let dir = std::env::temp_dir().join(format!("mapdb5_reopen_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.sd1");

    let mut recids = vec![];
    {
        let s = StoreDirect::open_file(&path).unwrap();
        for i in 0..50u64 {
            recids.push(s.put(&bytes(i, 300 + i as usize), &R).unwrap());
        }
        s.commit().unwrap();
        s.verify().unwrap();
        s.close().unwrap();
    }
    {
        let s = StoreDirect::open_file(&path).unwrap();
        s.verify().unwrap();
        for (i, r) in recids.iter().enumerate() {
            assert_eq!(s.get(*r, &R).unwrap(), Some(bytes(i as u64, 300 + i)));
        }
        s.close().unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn old_file_magic_is_rejected() {
    let dir = std::env::temp_dir().join(format!("mapdb_rust_store_old_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.sd1");
    {
        let s = StoreDirect::open_file(&path).unwrap();
        s.close().unwrap();
    }
    let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.write_all_at(b"MDB5.SD1", 0).unwrap();

    assert!(matches!(
        StoreDirect::open_file(&path),
        Err(DbError::DataCorruption(_))
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn file_reopen_after_uncommitted_change_refuses() {
    let dir = std::env::temp_dir().join(format!("mapdb_rust_store_crash_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.sd1");
    {
        let s = StoreDirect::open_file(&path).unwrap();
        let _r = s.put(&bytes(1, 100), &R).unwrap();
        s.commit().unwrap();
        // mutate again WITHOUT committing, then drop (simulated crash — no close)
        let _r2 = s.put(&bytes(2, 100), &R).unwrap();
        std::mem::forget(s); // don't run close()/Drop stamping
    }
    // reopen must refuse: header checksum no longer matches the mutated header words
    match StoreDirect::open_file(&path) {
        Err(DbError::DataCorruption(_)) => {}
        Err(e) => panic!("expected DataCorruption on reopen, got {e}"),
        Ok(_) => panic!("expected reopen to refuse an uncommitted-crash store, but it opened"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Differential fuzz: apply the same scripted op sequence to StoreDirect and the
/// StoreByteArray oracle, tracking per-store recids, comparing content after
/// every op and full state via verify() at epochs.
/// Headroom whose aligned capacity would exceed MAX_CAPACITY (or overflow the
/// wide sum) must map to RecordTooLarge, never wrap into a small accepted
/// capacity.
#[test]
fn headroom_overflow_is_record_too_large() {
    use mapdb_rust_store::store::index_val::MAX_CAPACITY;
    let s = StoreDirect::new_heap().unwrap();
    let small = bytes(1, 8);
    let r = s.put(&small, &R).unwrap();
    assert!(matches!(
        s.update_with_headroom(r, &small, &R, MAX_CAPACITY),
        Err(DbError::RecordTooLarge)
    ));
    assert!(matches!(
        s.update_with_headroom(r, &small, &R, usize::MAX),
        Err(DbError::RecordTooLarge)
    ));
    // the record is untouched and the store still verifies.
    assert_eq!(s.get(r, &R).unwrap(), Some(small));
    s.verify().unwrap();
}

#[test]
fn differential_vs_oracle() {
    let oracle = StoreByteArray::new(true);
    let direct = StoreDirect::new_heap().unwrap();
    // handles: (oracle_recid, direct_recid, expected_content or None for null)
    let mut handles: Vec<(std::num::NonZeroU64, std::num::NonZeroU64, Option<Vec<u8>>)> =
        Vec::new();

    let mut x: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        x >> 33
    };

    for step in 0..3000u64 {
        let op = next() % 100;
        if op < 45 || handles.is_empty() {
            // put
            let len = (next() % 900) as usize + if next() % 20 == 0 { 1_100_000 } else { 0 };
            let v = bytes(step, len);
            let ro = oracle.put(&v, &R).unwrap();
            let rd = direct.put(&v, &R).unwrap();
            handles.push((ro, rd, Some(v)));
        } else if op < 65 {
            // update (to value or null)
            let i = (next() as usize) % handles.len();
            let (ro, rd, _) = handles[i];
            if next() % 8 == 0 {
                oracle.update(ro, None::<&Vec<u8>>, &R).unwrap();
                direct.update(rd, None::<&Vec<u8>>, &R).unwrap();
                handles[i].2 = None;
            } else {
                let len = (next() % 1500) as usize;
                let v = bytes(step ^ 0xabc, len);
                oracle.update(ro, Some(&v), &R).unwrap();
                direct.update(rd, Some(&v), &R).unwrap();
                handles[i].2 = Some(v);
            }
        } else if op < 80 {
            // delete
            let i = (next() as usize) % handles.len();
            let (ro, rd, _) = handles.remove(i);
            oracle.delete(ro).unwrap();
            direct.delete(rd).unwrap();
        } else if op < 90 {
            // cas
            let i = (next() as usize) % handles.len();
            let (ro, rd, ref expect) = handles[i];
            let newv = bytes(step ^ 0x555, (next() % 400) as usize);
            let ok_o = oracle
                .compare_and_swap(ro, expect.as_ref(), Some(&newv), &R)
                .unwrap();
            let ok_d = direct
                .compare_and_swap(rd, expect.as_ref(), Some(&newv), &R)
                .unwrap();
            assert_eq!(ok_o, ok_d, "cas result diverged at step {step}");
            if ok_o {
                handles[i].2 = Some(newv);
            }
        } else {
            // preallocate then fill
            let ro = oracle.preallocate().unwrap();
            let rd = direct.preallocate().unwrap();
            handles.push((ro, rd, None));
        }

        // compare content of every handle
        for (ro, rd, expect) in &handles {
            let go = oracle.get(*ro, &R).unwrap();
            let gd = direct.get(*rd, &R).unwrap();
            assert_eq!(&go, expect, "oracle content mismatch at step {step}");
            assert_eq!(go, gd, "direct vs oracle mismatch at step {step}");
        }
        // live-record count must match
        assert_eq!(
            oracle.get_all_recids().unwrap().len(),
            direct.get_all_recids().unwrap().len(),
            "live count diverged at step {step}"
        );

        if step % 200 == 0 {
            direct.verify().unwrap();
            oracle.verify().unwrap();
        }
    }
    direct.verify().unwrap();
}

/// Regression: the allocator hot path
/// (free-recid take, free-extent reuse, chunk relink, free put) validates every
/// persisted link/value it dereferences — open() never walks the free-recid
/// stack and never validates decoded free-extent values. Prove a NORMAL store
/// still flows through all of these guarded paths right after a reopen: enough
/// deletes to build multi-chunk stacks, then re-allocation popping through
/// chunk-empty relinks (prev-link validation) and free-extent reuse.
#[test]
fn free_list_reuse_across_reopen_hot_path() {
    let dir = std::env::temp_dir().join(format!("mapdb5_freereuse_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.sd1");
    let survivors = {
        let s = StoreDirect::open_file(&path).unwrap();
        let mut rs = Vec::new();
        for i in 0..400u64 {
            rs.push(s.put(&bytes(i, 64), &R).unwrap());
        }
        // 300 deletes → multi-chunk free-recid stack + populated free-data stacks
        for r in rs.drain(..300) {
            s.delete(r).unwrap();
        }
        s.verify().unwrap();
        s.close().unwrap();
        rs
    };
    {
        let s = StoreDirect::open_file(&path).unwrap();
        s.verify().unwrap();
        // first post-open allocations pop persisted free-recid values and reuse
        // persisted free-data extents — the exact words open() never validated
        let mut fresh = Vec::new();
        for i in 0..300u64 {
            fresh.push(s.put(&bytes(1000 + i, 64), &R).unwrap());
        }
        s.verify().unwrap();
        for (i, r) in fresh.iter().enumerate() {
            assert_eq!(s.get(*r, &R).unwrap(), Some(bytes(1000 + i as u64, 64)));
        }
        // long_stack_put against the persisted stacks (delete after reopen)
        for r in fresh {
            s.delete(r).unwrap();
        }
        s.verify().unwrap();
        for (i, r) in survivors.iter().enumerate() {
            assert_eq!(s.get(*r, &R).unwrap(), Some(bytes(300 + i as u64, 64)));
        }
        s.close().unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Regression: open() does not walk the
/// free-recid stack, so the first post-open allocation is the first look at its
/// persisted chunk words — and the chunk lives in the data area, OUTSIDE the
/// checksummed header page, so reopen cannot catch it. A parity-valid chunk
/// header with an illegal size must yield DataCorruption on the hot path, never
/// a panic. (Format-v1 constants: free-recid master link = u64 BE at offset 64,
/// offset payload masked by `index_val::MOFFSET`.)
#[test]
fn corrupt_free_recid_chunk_header_fails_gracefully() {
    use mapdb_rust_store::store::{index_val, parity};
    use std::os::unix::fs::FileExt;
    let dir = std::env::temp_dir().join(format!("mapdb5_freecorrupt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.sd1");
    {
        let s = StoreDirect::open_file(&path).unwrap();
        let r1 = s.put(&bytes(1, 100), &R).unwrap();
        let r2 = s.put(&bytes(2, 100), &R).unwrap();
        s.delete(r1).unwrap();
        s.delete(r2).unwrap();
        s.close().unwrap(); // stamps the header checksum clean
    }
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let mut w = [0u8; 8];
    f.read_exact_at(&mut w, 64).unwrap();
    let chunk_off = u64::from_be_bytes(w) & index_val::MOFFSET;
    assert!(
        chunk_off >= 1 << 20,
        "free-recid stack should have a chunk in the data area"
    );
    // parity-valid header claiming chunk size 8 (< 16, illegal)
    f.write_all_at(&parity::p4set(8u64 << 48).to_be_bytes(), chunk_off)
        .unwrap();
    f.sync_all().unwrap();
    drop(f);

    let s = StoreDirect::open_file(&path).unwrap();
    match s.preallocate() {
        Err(DbError::DataCorruption(_)) => {}
        other => panic!("expected DataCorruption from the free-recid hot path, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Round-4 crafted-file allocator hardening regressions.
// ---------------------------------------------------------------------------

/// Recompute and rewrite the format-v1 header checksum after patching header
/// words, so a crafted file opens "clean" (mirrors StoreDirect::head_checksum).
/// Constants are format-v1: seed 0x5D1BA5E1, checksum i32 at offset 16, and the
/// mixed region spans [O_DATA_TAIL=24, ZERO_SLOTS_START=524352) in 8-byte steps.
fn restamp_header_checksum(buf: &mut [u8]) {
    const O_HEAD_CHECKSUM: usize = 16;
    const O_DATA_TAIL: usize = 24;
    const ZERO_SLOTS_START: usize = 524352; // O_FREE_DATA_STACKS(72)+8*0xFFFD + 16
    let mut c: i32 = 0x5D1B_A5E1u32 as i32;
    let mut o = O_DATA_TAIL;
    while o < ZERO_SLOTS_START {
        let v = u64::from_be_bytes(buf[o..o + 8].try_into().unwrap());
        c = c.wrapping_mul(31).wrapping_add((v ^ (v >> 32)) as i32);
        o += 8;
    }
    buf[O_HEAD_CHECKSUM..O_HEAD_CHECKSUM + 4].copy_from_slice(&c.to_be_bytes());
}

fn pack_long_size(mut v: u64) -> usize {
    let mut c = 1;
    loop {
        v >>= 7;
        if v == 0 {
            break;
        }
        c += 1;
    }
    c
}

/// Regression (regression): the free-DATA reuse path decodes a
/// persisted packed-long into an allocation offset. A crafted parity-valid value
/// can be near u64::MAX, so `off + cap_bytes` MUST be checked: unchecked it
/// panics in debug and, worse, WRAPS in release (off == u64::MAX-15 in the
/// 16-byte class wraps the sum to 0 and passes the one-page test), returning a
/// huge offset that then indexes a nonexistent slice. Craft exactly that value
/// and prove the first reuse allocation returns DataCorruption in BOTH profiles.
#[test]
fn free_data_reuse_offset_overflow_fails_gracefully() {
    use mapdb_rust_store::store::{index_val, parity};
    let dir = std::env::temp_dir().join(format!("mapdb5_reuse_ovf_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.sd1");
    {
        let s = StoreDirect::open_file(&path).unwrap();
        // a 12-byte record → cap 16 (size class u=1); delete frees it onto the
        // u=1 free-DATA stack, creating a chunk with one packed value.
        let r = s.put(&bytes(1, 12), &R).unwrap();
        s.delete(r).unwrap();
        s.close().unwrap();
    }
    let mut buf = std::fs::read(&path).unwrap();
    // u=1 master link is the first free-DATA stack word at O_FREE_DATA_STACKS=72.
    const MASTER_U1: usize = 72;
    let master = parity::p4get(u64::from_be_bytes(
        buf[MASTER_U1..MASTER_U1 + 8].try_into().unwrap(),
    ))
    .expect("u=1 master link parity");
    let chunk_off = (master & index_val::MOFFSET) as usize;
    assert!(
        chunk_off >= 1 << 20,
        "expected a u=1 free-DATA chunk in the data area"
    );

    // Crafted maximum aligned offset for the 16-byte class: off + 16 overflows.
    let off_target = u64::MAX - 15; // 0xFFFF_FFFF_FFFF_FFF0, 16-aligned
    let raw = parity::p1set(off_target >> 3); // reuse path computes p1get(v) << 3
    let size = pack_long_size(raw);
    // encode the packed long (7-bit groups MSB-first, terminator byte | 0x80)
    let mut enc = Vec::with_capacity(size);
    let mut shift = (size as u32 - 1) * 7;
    while shift > 0 {
        enc.push(((raw >> shift) & 0x7F) as u8);
        shift -= 7;
    }
    enc.push(((raw & 0x7F) | 0x80) as u8);
    assert!(
        enc.iter().all(|&b| b != 0),
        "crafted value must have no zero byte"
    );
    // value area starts at chunk_off+8; set master pos to 8+size so take() reads it.
    buf[chunk_off + 8..chunk_off + 8 + size].copy_from_slice(&enc);
    let new_master = parity::p4set((((8 + size) as u64) << 48) | chunk_off as u64);
    buf[MASTER_U1..MASTER_U1 + 8].copy_from_slice(&new_master.to_be_bytes());
    restamp_header_checksum(&mut buf);
    std::fs::write(&path, &buf).unwrap();

    let s = StoreDirect::open_file(&path).unwrap(); // open only counts free entries
    match s.put(&bytes(2, 12), &R) {
        Err(DbError::DataCorruption(_)) => {}
        other => panic!("expected DataCorruption from the reuse overflow guard, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Regression (regression): init_open must reject a persisted
/// dataTail that violates the geometry the allocator relies on. A parity-valid,
/// checksum-consistent file with fileTail == dataTail == PAGE_SIZE would let the
/// first allocation take the in-page branch, return PAGE_SIZE and write into
/// slice 1 though only slice 0 is mapped → panic. Reopen must fail gracefully.
#[test]
fn reopen_with_invalid_data_tail_refuses() {
    use mapdb_rust_store::store::parity;
    let dir = std::env::temp_dir().join(format!("mapdb5_bad_dtail_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.sd1");
    {
        let s = StoreDirect::open_file(&path).unwrap(); // empty: fileTail=PAGE_SIZE, dataTail=0
        s.close().unwrap();
    }
    const PAGE_SIZE: u64 = 1 << 20;
    const O_DATA_TAIL: usize = 24;
    let mut buf = std::fs::read(&path).unwrap();
    // dataTail == PAGE_SIZE: page-aligned AND == fileTail → illegal geometry.
    buf[O_DATA_TAIL..O_DATA_TAIL + 8].copy_from_slice(&parity::p4set(PAGE_SIZE).to_be_bytes());
    restamp_header_checksum(&mut buf);
    std::fs::write(&path, &buf).unwrap();

    match StoreDirect::open_file(&path) {
        Err(DbError::DataCorruption(_)) => {}
        Err(e) => panic!("expected DataCorruption on reopen with a bad dataTail, got {e}"),
        Ok(_) => panic!("expected reopen to refuse a store with an invalid dataTail"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Regression (regression): init_open must reject a
/// persisted maxRecid that has no index slot in the loaded page mirror. Exploit
/// a checksum-consistent file with NO index pages beyond the zero page, maxRecid
/// == RECIDS_PER_ZERO_PAGE+1 (needs a page that does not exist), and a valid
/// free-recid chunk naming that same recid. Pre-fix: open succeeds (the
/// free-recid stack isn't walked at open), then the first allocation pops the
/// crafted recid and index_set's `expect` panics in BOTH profiles because
/// recid_to_offset is None. Post-fix the store is unusable gracefully.
#[test]
fn reopen_with_max_recid_beyond_index_pages_refuses() {
    use mapdb_rust_store::store::{index_val, parity};
    // recid whose slot would live on the FIRST non-zero index page (absent here)
    const RECIDS_PER_ZERO_PAGE: u64 = 65528; // (PAGE_SIZE - ZERO_SLOTS_START)/8
    const BAD_RECID: u64 = RECIDS_PER_ZERO_PAGE + 1;
    const O_MAX_RECID: usize = 32;
    const O_FREE_RECID_STACK: usize = 64;
    let dir = std::env::temp_dir().join(format!("mapdb5_bad_maxrecid_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.sd1");
    {
        let s = StoreDirect::open_file(&path).unwrap();
        let r = s.put(&bytes(1, 100), &R).unwrap(); // recid 1, maxRecid = 1
        s.delete(r).unwrap(); // free-recid stack: one chunk, one value = p1set(1<<1)
        s.close().unwrap();
    }
    let mut buf = std::fs::read(&path).unwrap();
    // 1) crank maxRecid up to a recid with no index page (stored as p4set(v<<4)).
    buf[O_MAX_RECID..O_MAX_RECID + 8].copy_from_slice(&parity::p4set(BAD_RECID << 4).to_be_bytes());
    // 2) overwrite the single free-recid value with the crafted recid (p1set(recid<<1)).
    let master = parity::p4get(u64::from_be_bytes(
        buf[O_FREE_RECID_STACK..O_FREE_RECID_STACK + 8]
            .try_into()
            .unwrap(),
    ))
    .expect("free-recid master parity");
    let chunk_off = (master & index_val::MOFFSET) as usize;
    assert!(
        chunk_off >= 1 << 20,
        "expected a free-recid chunk in the data area"
    );
    let raw = parity::p1set(BAD_RECID << 1);
    let size = pack_long_size(raw);
    let mut enc = Vec::with_capacity(size);
    let mut shift = (size as u32 - 1) * 7;
    while shift > 0 {
        enc.push(((raw >> shift) & 0x7F) as u8);
        shift -= 7;
    }
    enc.push(((raw & 0x7F) | 0x80) as u8);
    buf[chunk_off + 8..chunk_off + 8 + size].copy_from_slice(&enc);
    let new_master = parity::p4set((((8 + size) as u64) << 48) | chunk_off as u64);
    buf[O_FREE_RECID_STACK..O_FREE_RECID_STACK + 8].copy_from_slice(&new_master.to_be_bytes());
    restamp_header_checksum(&mut buf);
    std::fs::write(&path, &buf).unwrap();

    // The store must be unusable gracefully: open rejects the bad maxRecid, or
    // (belt-and-suspenders) the first allocation's reuse backstop does — never a
    // panic in index_set.
    match StoreDirect::open_file(&path) {
        Err(DbError::DataCorruption(_)) => {}
        Err(e) => panic!("expected DataCorruption on reopen with a bad maxRecid, got {e}"),
        Ok(s) => match s.preallocate() {
            Err(DbError::DataCorruption(_)) => {}
            other => panic!("expected DataCorruption from the reuse backstop, got {other:?}"),
        },
    }
    let _ = std::fs::remove_dir_all(&dir);
}
