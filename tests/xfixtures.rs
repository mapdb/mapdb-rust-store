//! Cross-port conformance fixture GENERATOR (Stage 1, D workload).
//!
//! These fixtures pin the CURRENT state of an UNSTABLE on-disk format for
//! divergence detection between the engines. Cross-engine openability is an
//! implementation fact, not a supported feature; any format change regenerates
//! the fixtures as part of that change.
//!
//! Run (ignored by default; public API only, deterministic output):
//!
//! ```text
//! XFIXTURES_OUT=<dir> cargo test --locked --test xfixtures write_fixtures -- --ignored --exact
//! ```
//!
//! Refuses a nonempty output dir unless `XFIXTURES_FORCE=1`. Writes
//! `direct-v1-rust.db` plus `fragment.tsv` (fixture/file/recid/recidrange rows
//! for the sync script; the file row's gzSha256 column is left empty for the
//! script to fill).

use mapdb_rust_store::error::Result;
use mapdb_rust_store::store::{Recid, Store, StoreDirect};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::io::Write as _;
use std::os::unix::fs::FileExt;
use std::path::Path;

use mapdb_rust_store::io::{DataInput2, DataOutput2};
use mapdb_rust_store::ser::Serializer;

/// Raw-bytes serializer (same shape as the TCK's `RawSer`): record content ==
/// logical value, so the on-disk bytes are exactly the contract's payload
/// function — no string/compression serializers per the Stage-1 contract.
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

const FIXTURE_ID: &str = "direct-v1-rust";
const DB_NAME: &str = "direct-v1-rust.db";

/// Contract payload function: `payload(payloadId, len)[i] = (i*131 + payloadId) & 0xff`.
fn payload(payload_id: u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i as u64).wrapping_mul(131).wrapping_add(payload_id) & 0xff) as u8)
        .collect()
}

// ---------------------------------------------------------------------------
// Minimal index-slot decoder over raw FILE bytes (generator self-check only;
// conformance consumers stay on the public API). Constants and bit ops are
// copied verbatim from the engine, as the contract requires citing them:
//   - slot location: `recid_to_offset` in src/store/direct.rs
//     (ZERO_SLOTS_START = 524336 + 16 = 524352; slot(r) = 524352 + (r-1)*8,
//     valid for recids on the zero index page, r <= 65528);
//   - parity1 strip: `p1get` in src/store/parity.rs;
//   - field extraction: `MOFFSET` / `cap_units` / `offset` in
//     src/store/index_val.rs.
// ---------------------------------------------------------------------------

const ZERO_SLOTS_START: u64 = 524_352;
const RECIDS_PER_ZERO_PAGE: u64 = 65_528;
/// src/store/index_val.rs `MOFFSET`: 44-bit, 16-aligned offset field.
const MOFFSET: u64 = 0x0000_FFFF_FFFF_FFF0;
/// src/store/index_val.rs `CAP_DELETED`: capacityUnits tombstone sentinel.
const CAP_DELETED: u32 = 0xFFFE;

/// src/store/parity.rs `p1get`: validate and strip parity1.
fn p1get(v: u64) -> u64 {
    assert_eq!(v.count_ones() & 1, 1, "parity1 broken in index slot {v:#x}");
    v & !1
}

/// Read the recid's raw big-endian index slot straight from the file bytes.
fn read_index_slot(db: &Path, recid: Recid) -> u64 {
    let r = recid.get();
    assert!(
        (1..=RECIDS_PER_ZERO_PAGE).contains(&r),
        "decoder only handles the zero index page, got recid {r}"
    );
    let f = std::fs::File::open(db).expect("open db for slot decode");
    let mut b = [0u8; 8];
    f.read_exact_at(&mut b, ZERO_SLOTS_START + (r - 1) * 8)
        .expect("read index slot");
    u64::from_be_bytes(b)
}

/// src/store/index_val.rs `offset`: `iv & MOFFSET` on the parity-stripped slot.
fn slot_offset(db: &Path, recid: Recid) -> u64 {
    p1get(read_index_slot(db, recid)) & MOFFSET
}

/// src/store/index_val.rs `cap_units`: `(iv >> 48) as u32`.
fn slot_cap_units(db: &Path, recid: Recid) -> u32 {
    (p1get(read_index_slot(db, recid)) >> 48) as u32
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn generator_commit() -> String {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

/// Full contract reader assertions against a reopened store (same checks the
/// cross-engine conformance harness runs; the generator must pass its own).
#[allow(clippy::too_many_arguments)]
fn assert_reader_contract(
    s: &StoreDirect,
    a: Recid,
    b: Recid,
    c: Recid,
    d: Recid,
    e: Recid,
    f: Recid,
    g: Recid,
    churn: &[Recid],
) {
    s.verify().expect("verify() on reopened fixture");
    assert_eq!(s.get(a, &R).unwrap(), Some(payload(1, 100)), "A content");
    assert_eq!(
        s.get(b, &R).unwrap(),
        Some(Vec::new()),
        "B is present and zero-length, NOT null"
    );
    assert_eq!(s.get(c, &R).unwrap(), None, "C is explicit null");
    assert_eq!(s.get(d, &R).unwrap(), None, "D prealloc reads as None");
    assert_eq!(
        s.get(f, &R).unwrap(),
        Some(payload(6, 1_048_525)),
        "F linked content"
    );
    assert_eq!(s.get(g, &R).unwrap(), Some(payload(7, 256)), "G content");
    for &r in churn.iter().chain(std::iter::once(&e)) {
        assert!(
            matches!(
                s.get(r, &R),
                Err(mapdb_rust_store::DbError::GetVoid(x)) if x == r.get()
            ),
            "recid {r} must be deleted (GetVoid)"
        );
    }
    let all: std::collections::BTreeSet<Recid> = s.get_all_recids().unwrap().into_iter().collect();
    let want: std::collections::BTreeSet<Recid> = [a, b, c, f, g].into_iter().collect();
    assert_eq!(all, want, "getAllRecids must be exactly {{A,B,C,F,G}}");
}

/// D-workload fixture generator (contract `write_fixtures`). `#[ignore]`d: run
/// explicitly with `XFIXTURES_OUT` set, see the module header.
#[test]
#[ignore]
fn write_fixtures() {
    let out = std::env::var("XFIXTURES_OUT")
        .expect("set XFIXTURES_OUT=<dir> to run the fixture generator");
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("create output dir");
    let force = std::env::var("XFIXTURES_FORCE").as_deref() == Ok("1");
    let nonempty = std::fs::read_dir(&out).expect("read output dir").count() > 0;
    if nonempty && !force {
        panic!(
            "output dir {} is not empty; set XFIXTURES_FORCE=1 to overwrite",
            out.display()
        );
    }
    let db = out.join(DB_NAME);
    if db.exists() {
        std::fs::remove_file(&db).expect("remove stale db");
    }

    // ---- D workload, EXACT contract order (public API only) ----
    let s = StoreDirect::open_file(&db).expect("create fixture store");
    let a = s.put(&payload(1, 100), &R).unwrap(); // 1
    let b = s.put(&payload(2, 0), &R).unwrap(); // 2: zero-length live
    let c = s.put(&payload(3, 40), &R).unwrap(); // 3: explicit null via update(recid, None)
    s.update::<Vec<u8>>(c, None, &R).unwrap();
    let d = s.preallocate().unwrap(); // 4: never written
    let f = s.put(&payload(6, 1_048_525), &R).unwrap(); // 5: first-linked boundary
    let g = s.preallocate().unwrap(); // 6: updated in step 10
    let e = s.put(&payload(5, 256), &R).unwrap(); // 7
    let churn: Vec<Recid> = (0..200)
        .map(|j| s.put(&payload(1000 + j, 256), &R).unwrap())
        .collect(); // 8: same capacity class as E
    for w in churn.windows(2) {
        assert_eq!(
            w[1].get(),
            w[0].get() + 1,
            "churn recids must be contiguous (manifest uses recidrange)"
        );
    }
    // Extra commit between contract steps 8 and 9 (explicitly ALLOWED by the
    // contract; commit is not an allocation): flush so E's pre-delete data
    // offset can be captured from the file bytes for the E->G reuse check.
    s.commit().unwrap();
    let e_offset = slot_offset(&db, e);
    assert_ne!(e_offset, 0, "E must have a data extent before deletion");
    for &r in &churn {
        s.delete(r).unwrap(); // 9: churn in creation order...
    }
    s.delete(e).unwrap(); // ...then E LAST
    s.update(g, Some(&payload(7, 256)), &R).unwrap(); // 10: must reuse E's extent
    s.commit().unwrap(); // 11
    s.close().unwrap();

    // ---- self-check: reopen and run the full reader contract ----
    let pre = std::fs::read(&db).expect("read fixture bytes");
    let s2 = StoreDirect::open_file(&db).expect("reopen fixture");
    assert_reader_contract(&s2, a, b, c, d, e, f, g, &churn);
    s2.close().unwrap();
    let post = std::fs::read(&db).expect("re-read fixture bytes");
    assert_eq!(
        pre, post,
        "read-only reopen must leave the fixture bytes unchanged"
    );

    // ---- self-check: E->G extent reuse + E tombstone, from raw file bytes ----
    assert_eq!(
        slot_offset(&db, g),
        e_offset,
        "G must reuse E's freed data extent (same-capacity free stack is LIFO)"
    );
    assert_eq!(
        slot_cap_units(&db, e),
        CAP_DELETED,
        "E's index slot must carry the deleted tombstone"
    );

    // ---- fragment.tsv for the sync script (gzSha256 column left empty) ----
    let sha = sha256_hex(&post);
    let mut t = String::new();
    t.push_str(
        "# xfixtures generator fragment (rust). Merged into MANIFEST.tsv by the sync script.\n",
    );
    t.push_str(&format!(
        "fixture\t{FIXTURE_ID}\tdirect\trust\t{}\n",
        generator_commit()
    ));
    t.push_str(&format!(
        "file\t{FIXTURE_ID}\t{DB_NAME}\t{}\t{sha}\t\n",
        post.len()
    ));
    for (label, recid, state, pid, len) in [
        ("A", a, "live", 1u64, 100usize),
        ("B", b, "live", 2, 0),
        ("C", c, "null", 3, 40),
        ("D", d, "prealloc", 0, 0),
        ("E", e, "deleted", 5, 256),
        ("F", f, "live", 6, 1_048_525),
        ("G", g, "live", 7, 256),
    ] {
        t.push_str(&format!(
            "recid\t{FIXTURE_ID}\t{label}\t{recid}\t{state}\t{pid}\t{len}\n"
        ));
    }
    t.push_str(&format!(
        "recidrange\t{FIXTURE_ID}\tchurn\t{}\t{}\tdeleted\t1000\t256\n",
        churn.first().unwrap(),
        churn.last().unwrap()
    ));
    let mut fr = std::fs::File::create(out.join("fragment.tsv")).expect("create fragment.tsv");
    fr.write_all(t.as_bytes()).expect("write fragment.tsv");
    fr.sync_all().expect("sync fragment.tsv");
}
