//! Cross-port conformance fixture GENERATOR (Stage 1 D workload + Stage 2 W
//! workloads).
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
//! `direct-v1-rust.db`, `wal-v1-rust-tail.wal`, `wal-v1-rust-ckpt.wal` plus
//! `fragment.tsv` (fixture/file/recid/recidrange rows for the sync script; the
//! file rows' gzSha256 column is left empty for the script to fill).

use mapdb_rust_store::error::Result;
use mapdb_rust_store::store::{Recid, Store, StoreDirect, StoreTx, StoreWAL};
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

// ---------------------------------------------------------------------------
// Stage 2: W (WAL v1) fixtures — `wal-v1-rust-tail.wal` / `wal-v1-rust-ckpt.wal`
// ---------------------------------------------------------------------------

/// Recids allocated by one W-workload run (labels A..F per the contract; the
/// tail namespace additionally records the rolled-back put's recid, which must
/// stay invisible everywhere — no fragment row, absent after reopen).
#[derive(PartialEq, Eq, Debug)]
struct WalRecids {
    a: Recid,
    b: Recid,
    c: Recid,
    d: Recid,
    e: Recid,
    f: Recid,
    rolled: Option<Recid>,
}

/// Stage-2 W workload (contract §2) against a fresh `StoreWAL` at `path`.
/// `ckpt == false` (tail): T1..T4 committed, then a rollback-only T5 LAST.
/// `ckpt == true`  (ckpt): T1..T3, public `checkpoint()`, then T4. No rollback.
fn build_wal_fixture(path: &Path, base: u64, ckpt: bool) -> WalRecids {
    if path.exists() {
        std::fs::remove_file(path).expect("remove stale wal fixture");
    }
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".ckpt");
    let tmp = std::path::PathBuf::from(tmp);
    if tmp.exists() {
        std::fs::remove_file(&tmp).expect("remove stale wal checkpoint temp");
    }
    let s = StoreWAL::open(path).expect("create wal fixture store");
    // T1
    let a = s.put(&payload(base, 100), &R).unwrap();
    let b = s.put(&payload(base + 1, 0), &R).unwrap();
    let c = s.put(&payload(base + 2, 40), &R).unwrap();
    s.commit().unwrap();
    // T2: explicit null + committed prealloc
    s.update::<Vec<u8>>(c, None, &R).unwrap();
    let d = s.preallocate().unwrap();
    s.commit().unwrap();
    // T3: E plain + F oversize (1_200_000 B spans the ~1 MiB replay buffering)
    let e = s.put(&payload(base + 3, 256), &R).unwrap();
    let f = s.put(&payload(base + 4, 1_200_000), &R).unwrap();
    s.commit().unwrap();
    if ckpt {
        // snapshot 'C' section; the T4 'S' section follows it in the log.
        s.checkpoint().unwrap();
    }
    // T4
    s.delete(e).unwrap();
    s.update(a, Some(&payload(base + 5, 120)), &R).unwrap();
    s.commit().unwrap();
    // T5 (tail only): rollback LAST — writes nothing; the put must be invisible.
    let rolled = if ckpt {
        None
    } else {
        let r = s.put(&payload(base + 6, 64), &R).unwrap();
        s.rollback().unwrap();
        Some(r)
    };
    s.close().unwrap();
    WalRecids {
        a,
        b,
        c,
        d,
        e,
        f,
        rolled,
    }
}

/// Local scan of the raw v1 WAL bytes returning the section tags in file
/// order. Format per the `src/store/wal.rs` module comment: 16-byte file
/// header (magic "MDBS.WAL" | version i32 BE | flags i32 BE), then sections
/// `tag u8 | lsn i64 BE | bodyLen i64 BE | hdrCrc i32 | bodyCrc i32 | body` —
/// the scan skips `bodyLen` body bytes after each 25-byte section header.
fn wal_section_tags(bytes: &[u8]) -> Vec<u8> {
    assert_eq!(&bytes[..8], b"MDBS.WAL", "fixture must carry the v1 magic");
    let mut tags = Vec::new();
    let mut pos = 16usize;
    while pos < bytes.len() {
        assert!(
            pos + 25 <= bytes.len(),
            "torn section header in generated fixture at offset {pos}"
        );
        let body_len = i64::from_be_bytes(bytes[pos + 9..pos + 17].try_into().unwrap());
        assert!(body_len >= 0, "negative bodyLen in generated fixture");
        tags.push(bytes[pos]);
        pos += 25 + body_len as usize;
    }
    assert_eq!(
        pos,
        bytes.len(),
        "sections must tile the generated fixture exactly"
    );
    tags
}

/// Full W reader contract against a reopened `StoreWAL` (same checks the
/// cross-engine conformance harness runs; the generator must pass its own).
fn assert_wal_reader_contract(s: &StoreWAL, base: u64, r: &WalRecids) {
    s.verify().expect("verify() on reopened wal fixture");
    assert_eq!(
        s.get(r.a, &R).unwrap(),
        Some(payload(base + 5, 120)),
        "A content (updated in T4)"
    );
    assert_eq!(
        s.get(r.b, &R).unwrap(),
        Some(Vec::new()),
        "B is present and zero-length, NOT null"
    );
    assert_eq!(s.get(r.c, &R).unwrap(), None, "C is explicit null");
    assert_eq!(s.get(r.d, &R).unwrap(), None, "D prealloc reads as None");
    assert!(
        matches!(
            s.get(r.e, &R),
            Err(mapdb_rust_store::DbError::GetVoid(x)) if x == r.e.get()
        ),
        "E must be deleted (GetVoid)"
    );
    assert_eq!(
        s.get(r.f, &R).unwrap(),
        Some(payload(base + 4, 1_200_000)),
        "F oversize content"
    );
    if let Some(rolled) = r.rolled {
        // leak detector: the rolled-back put must be void after replay...
        assert!(
            matches!(
                s.get(rolled, &R),
                Err(mapdb_rust_store::DbError::GetVoid(x)) if x == rolled.get()
            ),
            "rolled-back put must be invisible after reopen"
        );
    }
    // ...and the recid set must be EXACTLY {A,B,C,F} — plus nothing.
    let all: std::collections::BTreeSet<Recid> = s.get_all_recids().unwrap().into_iter().collect();
    let want: std::collections::BTreeSet<Recid> = [r.a, r.b, r.c, r.f].into_iter().collect();
    assert_eq!(all, want, "getAllRecids must be exactly {{A,B,C,F}}");
}

/// Refuses to run this generator, before it touches the output directory.
///
/// The W half became unrunnable at the WAL v3 cutover (A2): `StoreWAL` writes a
/// segment NAMESPACE, not the single `.wal` file the `wal-v1-*` fixture rows
/// publish, so `build_wal_fixture` creates `<name>.wal.wal.<seq>` segments and
/// the read-back dies with a bare `NotFound` — after the D fixture has already
/// been overwritten and a stray namespace left behind that `XFIXTURES_FORCE`
/// does not know how to clean.
///
/// Zig's peer generator got this refusal at B2 part 2; rust's did not, and the
/// omission survived because the entry point is `#[ignore]`d and so no gate
/// ever ran it. The refusal is therefore a FUNCTION with a test, not a comment:
/// the whole reason it was missing is that nothing executed the thing.
///
/// Stage C retires these cells (plan §5, contract §9); C7r removes this.
fn refuse_stale_v1_generator() -> ! {
    panic!(
        "the xfixtures generator is STALE at the WAL v3 cutover (A2): the \
         wal-v1-rust-* fixtures cannot be regenerated by a v3 store, which \
         writes a segment namespace rather than one .wal file. Refusing before \
         touching the output directory — a partial run overwrites \
         direct-v1-rust.db and then dies. The WAL v3 accept bundles are \
         tests/wal3_fixtures.rs (slice C2r); these cells retire at Stage C."
    )
}

/// The refusal fires, and says why. Not `#[ignore]`d on purpose: an
/// unexecutable guard on an unexecutable generator is two dead things.
#[test]
fn the_v1_generator_refuses_to_run() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let payload = std::panic::catch_unwind(refuse_stale_v1_generator).unwrap_err();
    std::panic::set_hook(prev);
    // A `panic!` with no format arguments carries a `&str`, one with arguments a
    // `String`. Accept either rather than pinning which this message happens to
    // be today: the test is about what the refusal SAYS.
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a string panic payload");
    assert!(
        msg.contains("STALE at the WAL v3 cutover") && msg.contains("wal3_fixtures.rs"),
        "the refusal must name both the cause and the successor: {msg}"
    );
}

/// D-workload fixture generator (contract `write_fixtures`). `#[ignore]`d: run
/// explicitly with `XFIXTURES_OUT` set, see the module header.
#[test]
#[ignore]
fn write_fixtures() {
    refuse_stale_v1_generator();
    #[allow(unreachable_code)]
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

    // ---- Stage 2: W fixtures (payload-id bases: rust tail = 11, ckpt = 21) ----
    let scratch = std::env::temp_dir().join(format!("mapdb5_xfix_gen_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).expect("create determinism scratch dir");
    for (fixture_id, base, ckpt) in [
        ("wal-v1-rust-tail", 11u64, false),
        ("wal-v1-rust-ckpt", 21, true),
    ] {
        let file_name = format!("{fixture_id}.wal");
        let wal_path = out.join(&file_name);
        let recids = build_wal_fixture(&wal_path, base, ckpt);
        let pre = std::fs::read(&wal_path).expect("read wal fixture bytes");

        // self-check: determinism — a second run must produce identical bytes
        // and identical recids (the sync script's two-run check covers this
        // cross-repo; the generator asserts it locally too).
        let twin = scratch.join(&file_name);
        let recids2 = build_wal_fixture(&twin, base, ckpt);
        assert_eq!(recids, recids2, "{fixture_id}: recids differ across runs");
        assert_eq!(
            pre,
            std::fs::read(&twin).expect("read twin wal bytes"),
            "{fixture_id}: bytes differ across two generator runs"
        );

        // self-check: section-tag scan over the raw bytes.
        let tags = wal_section_tags(&pre);
        if ckpt {
            assert_eq!(tags[0], b'C', "{fixture_id}: first section must be 'C'");
            assert!(
                tags[1..].contains(&b'S'),
                "{fixture_id}: at least one 'S' section must follow the checkpoint"
            );
            assert!(
                tags[1..].iter().all(|&x| x == b'S'),
                "{fixture_id}: unexpected non-'S' tag after the checkpoint: {tags:?}"
            );
        } else {
            assert!(
                !tags.is_empty() && tags.iter().all(|&x| x == b'S'),
                "{fixture_id}: every section tag must be 'S' (no 'C'), got {tags:?}"
            );
        }

        // self-check: reopen and run the full W reader contract; byte-stable.
        let s = StoreWAL::open(&wal_path).expect("reopen wal fixture");
        assert_wal_reader_contract(&s, base, &recids);
        s.close().unwrap();
        assert_eq!(
            pre,
            std::fs::read(&wal_path).expect("re-read wal fixture bytes"),
            "{fixture_id}: verification reopen must leave the bytes unchanged"
        );
        assert!(
            !out.join(format!("{file_name}.ckpt")).exists(),
            "{fixture_id}: no .ckpt companion may remain after a clean close"
        );

        // fragment rows: labels A..F only — NO row for the rolled-back put.
        t.push_str(&format!(
            "fixture\t{fixture_id}\tport-wal\trust\t{}\n",
            generator_commit()
        ));
        t.push_str(&format!(
            "file\t{fixture_id}\t{file_name}\t{}\t{}\t\n",
            pre.len(),
            sha256_hex(&pre)
        ));
        for (label, recid, state, pid, len) in [
            ("A", recids.a, "live", base + 5, 120usize),
            ("B", recids.b, "live", base + 1, 0),
            ("C", recids.c, "null", base + 2, 40),
            ("D", recids.d, "prealloc", 0, 0),
            ("E", recids.e, "deleted", base + 3, 256),
            ("F", recids.f, "live", base + 4, 1_200_000),
        ] {
            t.push_str(&format!(
                "recid\t{fixture_id}\t{label}\t{recid}\t{state}\t{pid}\t{len}\n"
            ));
        }
    }
    std::fs::remove_dir_all(&scratch).expect("remove determinism scratch dir");

    let mut fr = std::fs::File::create(out.join("fragment.tsv")).expect("create fragment.tsv");
    fr.write_all(t.as_bytes()).expect("write fragment.tsv");
    fr.sync_all().expect("sync fragment.tsv");
}
