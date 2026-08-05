//! Stage C slice **C2r**: the rust deterministic generator for the two WAL v3
//! accept bundles, `wal3-rust-tail` and `wal3-rust-cleaned`
//! (`todo/store-cross/impl-contract-stage3.md` §5.2, §5.3, §5.3.1, §5.4).
//!
//! Run it (ignored by default, like this repo's other fixture generator):
//!
//! ```text
//! XFIXTURES_OUT=<dir> [XFIXTURES_FORCE=1] \
//!   cargo test --locked --test wal3_fixtures write_wal3_fixtures -- --ignored --exact
//! ```
//!
//! Writes `<out>/wal3-rust-tail/` and `<out>/wal3-rust-cleaned/` (segments
//! only, base `x`), plus `fragment.tsv` and `layout.tsv`. Assembly,
//! compression and the `expect`/`post` rows are the sync script's job (C4).
//!
//! # This is the rust peer of C2j, and the bar is NOT java's bytes
//!
//! D6 permits writers to diverge in how they group entries into sections, so
//! this generator is not required to reproduce `Wal3FixtureWriter`'s output
//! byte for byte. What it IS required to reach is the structure: every §5.3.1
//! witness row, and all 24 adopted recipes deriving against the published
//! bytes (`python3 todo/store-cross/xcheck_bundles.py <out>`). A green
//! generator is not sufficient evidence — row 5 is invisible to any generator
//! (see [`row_five_is_invisible_to_this_generator`]).
//!
//! # Why the `cleaned` workload is not §5.3's literal one
//!
//! §5.3.1 wants three retained segments, a middle one whose first two sections
//! are entry-bearing, and an active one holding exactly one section under
//! `segmentBytes`. "T1–T3 then `checkpoint()`" cannot produce that. Rollover is
//! `active.file_len >= segment_bytes && !active.is_empty()` tested BEFORE a
//! section is appended (`src/store/wal_write.rs`, `StoreWAL.java:1688`), so
//! sections pile into a segment until it crosses the limit and only the NEXT
//! one rotates. Checkpointing after T3 makes the cleaner's image cover F's
//! 1.2 MB of live data, which overflows the segment holding it, so the forced
//! `'K'` mark lands as the FIRST section of the next segment — exactly where
//! §5.3.1 row 2 forbids it.
//!
//! Measured, not inherited: [`witnesses_depend_on_the_shaping`] runs the same
//! candidate variants against THIS engine and pins what each one produces. The
//! adopted workload is §5.3's amended table, and the three rejected variants
//! fail here for the same reasons they fail in java — which is a measurement,
//! not an assumption, and it is the evidence for §5.3 binding all three
//! engines rather than describing java.
//!
//! # Self-checks
//!
//! The published bytes are re-parsed by [`Seg`], a local minimal v3 decoder
//! written from the format description rather than from `src/store/`. A
//! generator that self-checks with the code that wrote the bytes checks
//! nothing; `walfmt.py` is the other half of a cross-check that only works if
//! the halves are independent. §5.3.1 row 6 (`file_len < segment_bytes`) is
//! asserted HERE and nowhere else — `segment_bytes` is a generator setting and
//! leaves no trace in the published bytes.

use mapdb_rust_store::error::Result;
use mapdb_rust_store::io::{DataInput2, DataOutput2};
use mapdb_rust_store::ser::Serializer;
use mapdb_rust_store::store::{Recid, Store, StoreTx, StoreWAL};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// §5.1 common configuration
// ---------------------------------------------------------------------------

/// §5.1, pinned: rotates deterministically without 64 MiB fixtures. The
/// setter's floor is `SEG_HDR + SEC_HDR` = 61.
const SEGMENT_BYTES: u64 = 65_536;
/// §5.1: above the whole workload's byte total, so the budgeted auto-cleaner —
/// whose foreground slice has a WALL-CLOCK arm and would therefore stop at a
/// machine-speed-dependent point — never starts. Asserted after the fact via
/// [`StoreWAL::cleaner_bytes`].
const MIN_LOG_BYTES: u64 = 64 * 1024 * 1024;

const TAIL_ID: &str = "wal3-rust-tail";
const CLEANED_ID: &str = "wal3-rust-cleaned";
/// §5: distinct per bundle so content differs even where recids coincide.
const TAIL_BASE: u64 = 121;
const CLEANED_BASE: u64 = 131;
const BASE_NAME: &str = "x";

/// Contract payload function: `payload(id, len)[i] = (i*131 + id) & 0xff`.
fn payload(payload_id: u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i as u64).wrapping_mul(131).wrapping_add(payload_id) & 0xff) as u8)
        .collect()
}

/// Raw-bytes serializer: record content == logical value, so the on-disk bytes
/// are exactly the contract's payload function.
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

// ---------------------------------------------------------------------------
// the workloads
// ---------------------------------------------------------------------------

/// Recids the workload allocated. Both shapes allocate the same six, in the
/// same order.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct Recids {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    f: u64,
}

impl Recids {
    fn a(&self) -> Recid {
        Recid::new(self.a).unwrap()
    }
    fn c(&self) -> Recid {
        Recid::new(self.c).unwrap()
    }
    fn e(&self) -> Recid {
        Recid::new(self.e).unwrap()
    }
}

fn open(base: &Path) -> StoreWAL {
    let s = StoreWAL::open(base).expect("open the fixture namespace");
    s.set_segment_bytes(SEGMENT_BYTES)
        .expect("§5.1 segmentBytes");
    s.set_min_log_bytes(MIN_LOG_BYTES)
        .expect("§5.1 minLogBytes");
    s
}

fn t1(s: &StoreWAL, r: &mut Recids, base: u64) {
    r.a = s.put(&payload(base, 100), &R).unwrap().get();
    r.b = s.put(&payload(base + 1, 0), &R).unwrap().get();
    r.c = s.put(&payload(base + 2, 40), &R).unwrap().get();
    s.commit().unwrap();
}

fn t2(s: &StoreWAL, r: &mut Recids) {
    s.update::<Vec<u8>>(r.c(), None, &R).unwrap();
    r.d = s.preallocate().unwrap().get();
    s.commit().unwrap();
}

fn t3(s: &StoreWAL, r: &mut Recids, base: u64) {
    r.e = s.put(&payload(base + 3, 256), &R).unwrap().get();
    r.f = s.put(&payload(base + 4, 1_200_000), &R).unwrap().get();
    s.commit().unwrap();
}

fn t4(s: &StoreWAL, r: &Recids, base: u64) {
    s.delete(r.e()).unwrap();
    s.update(r.a(), Some(&payload(base + 5, 120)), &R).unwrap();
    s.commit().unwrap();
}

/// Gives C content, then takes it away again: two sections, no new recid, and C
/// is null before and after. The SECOND one is §5.3.1 row 5's size-preserving
/// `T_APPEND` candidate — a null `T_RECORD` and a payload-free `T_APPEND` are
/// the same four bytes — and row 5 reads section index 1, so this pair must be
/// the first two sections of the middle segment and in this order.
fn shape_c(s: &StoreWAL, r: &Recids, base: u64) {
    s.update(r.c(), Some(&payload(base + 6, 48)), &R).unwrap();
    s.commit().unwrap();
    s.update::<Vec<u8>>(r.c(), None, &R).unwrap();
    s.commit().unwrap();
}

/// Pushes the active segment past `segment_bytes` and then commits once more,
/// so the LAST commit lands alone in a fresh segment. Rollover is tested BEFORE
/// a section is appended, so an oversized section joins the segment it
/// overflows and only its successor rotates: one commit cannot do this, and the
/// Stage C plan predicted otherwise. Both halves rewrite A and the second
/// restores A's §5.2 content, so the final logical state is untouched.
///
/// "Only its successor rotates" is about ORDINARY appends. Cleaning rotates too,
/// at two unconditional episode seals; neither is reachable here, because the
/// sole checkpoint is already behind us and `min_log_bytes` keeps auto-clean
/// from ever firing (§5.1). Rotating with one of those instead would change the
/// cleaning history the whole shape is built around.
fn shape_rotate(s: &StoreWAL, r: &Recids, base: u64) {
    s.update(r.a(), Some(&payload(base + 7, SEGMENT_BYTES as usize)), &R)
        .unwrap();
    s.commit().unwrap();
    s.update(r.a(), Some(&payload(base + 5, 120)), &R).unwrap();
    s.commit().unwrap();
}

/// §5.2's T1–T5: no cleaning ever runs, and T5 rolls back.
fn tail_workload(base: &Path, r: &mut Recids) {
    let s = open(base);
    t1(&s, r, TAIL_BASE);
    t2(&s, r);
    t3(&s, r, TAIL_BASE);
    t4(&s, r, TAIL_BASE);
    s.put(&payload(TAIL_BASE + 6, 64), &R).unwrap(); // T5: must leave no trace
    s.rollback().unwrap();
    assert_eq!(
        s.cleaner_bytes().0,
        0,
        "the tail shape must contain no cleaner output"
    );
    assert_final_state(&s, r, TAIL_ID, TAIL_BASE);
    s.close().unwrap();
}

/// §5.3's amended workload: the checkpoint after T2, then the two
/// state-preserving shaping pairs. See the module docs for the measurement.
fn cleaned_workload(base: &Path, r: &mut Recids) {
    let s = open(base);
    t1(&s, r, CLEANED_BASE);
    t2(&s, r);
    assert_eq!(
        s.cleaner_bytes().0,
        0,
        "auto-clean began before the explicit checkpoint: the bundle would stop \
         at a machine-speed-dependent point (§5.1, §5.4 obligation 3)"
    );
    s.checkpoint().unwrap(); // the ONLY cleaning, and it runs unbudgeted
    let after_checkpoint = s.cleaner_bytes().0;
    assert!(
        after_checkpoint > 0,
        "checkpoint() wrote no image: this is not a cleaned shape"
    );

    t3(&s, r, CLEANED_BASE); // 1.2 MB: oversizes the LOWEST retained segment
    shape_c(&s, r, CLEANED_BASE); // the middle segment's first two sections
    t4(&s, r, CLEANED_BASE);
    shape_rotate(&s, r, CLEANED_BASE); // one commit to cross, one to land alone

    assert_eq!(
        s.cleaner_bytes().0,
        after_checkpoint,
        "cleaning ran a second time after the checkpoint: §5.4 obligation 4 \
         allows exactly one episode, at one prescribed boundary"
    );
    assert_final_state(&s, r, CLEANED_ID, CLEANED_BASE);
    s.close().unwrap();
}

/// The final logical state §5.2 pins, which §5.3 shares verbatim.
fn assert_final_state(s: &StoreWAL, r: &Recids, ctx: &str, base: u64) {
    assert_state_with_a(s, r, ctx, base, &payload(base + 5, 120));
}

/// §5.2's final logical state with A's content named by the caller.
///
/// Every adopted workload ends with A holding `p(base+5, 120)`; the one probe
/// variant that deliberately stops mid-pair ends with A holding the oversized
/// payload instead. Naming A's expectation rather than skipping the check is
/// what keeps that variant's exception scoped to the ONE record it is about —
/// an unrelated state defect in it would otherwise be exempt too.
fn assert_state_with_a(s: &StoreWAL, r: &Recids, ctx: &str, base: u64, a_expect: &[u8]) {
    s.verify()
        .unwrap_or_else(|e| panic!("{ctx}: verify(): {e:?}"));
    assert_eq!(
        s.get(r.a(), &R).unwrap().as_deref(),
        Some(a_expect),
        "{ctx}: A content"
    );
    assert_eq!(
        s.get(Recid::new(r.b).unwrap(), &R).unwrap(),
        Some(Vec::new()),
        "{ctx}: B is present and zero-length, NOT null"
    );
    assert_eq!(s.get(r.c(), &R).unwrap(), None, "{ctx}: C is explicit null");
    assert_eq!(
        s.get(Recid::new(r.d).unwrap(), &R).unwrap(),
        None,
        "{ctx}: D prealloc reads as None"
    );
    assert!(
        matches!(
            s.get(r.e(), &R),
            Err(mapdb_rust_store::DbError::GetVoid(x)) if x == r.e
        ),
        "{ctx}: E must be deleted (GetVoid)"
    );
    assert_eq!(
        s.get(Recid::new(r.f).unwrap(), &R).unwrap(),
        Some(payload(base + 4, 1_200_000)),
        "{ctx}: F linked content"
    );
    // The leak detector: the rolled-back put of §5.2's T5 has no recid row and
    // must not be reachable, so the recid SET is compared exactly.
    let all: BTreeSet<u64> = s
        .get_all_recids()
        .unwrap()
        .into_iter()
        .map(|x| x.get())
        .collect();
    let want: BTreeSet<u64> = [r.a, r.b, r.c, r.f].into_iter().collect();
    assert_eq!(all, want, "{ctx}: getAllRecids must be exactly {{A,B,C,F}}");
}

/// The `recid` rows §5.2 pins, in fragment order.
fn expects(r: &Recids, base: u64) -> Vec<(&'static str, u64, &'static str, u64, usize)> {
    vec![
        ("A", r.a, "live", base + 5, 120),
        ("B", r.b, "live", base + 1, 0),
        ("C", r.c, "null", base + 2, 40),
        ("D", r.d, "prealloc", 0, 0),
        ("E", r.e, "deleted", base + 3, 256),
        ("F", r.f, "live", base + 4, 1_200_000),
    ]
}

// ---------------------------------------------------------------------------
// a local, independent v3 decoder (self-check only)
//
// Layout, from the format description in `todo/store-wal3/wal-v3-adoption.md`:
// a 36-byte segment header magic[8] | version(4) | flags(4) | seq(8) |
// firstLsn(8) | crc32(4) with the CRC over the first 32 bytes; then 25-byte
// section headers tag(1) | lsn(8) | bodyLen(8) | hdrCrc(4) | bodyCrc(4), each
// CRC primed with the 36 header bytes and be64(sectionOffset) BEFORE the
// section's own bytes — the offset is in the domain, which is what makes a
// section un-relocatable.
//
// Deliberately not `src/store/wal_segments.rs`'s parser and not `walfmt.py`.
// ---------------------------------------------------------------------------

const SEG_HDR: usize = 36;
const SEG_HDR_CRC_LEN: usize = 32;
const SEC_HDR: usize = 25;
const SEC_HDR_CRC_LEN: usize = 17;
const MAGIC: &[u8; 8] = b"MDBS.WAL";
const FORMAT_VERSION: u32 = 3;
const MARK_BODY_LEN: i64 = 16;

/// One decoded section: its offset, tag, LSN and body length.
struct Sec {
    off: usize,
    tag: u8,
    lsn: i64,
}

/// One decoded segment: both CRCs verified, every section walked.
struct Seg {
    rel_name: String,
    raw: Vec<u8>,
    seq: i64,
    first_lsn: i64,
    sections: Vec<Sec>,
    /// `(index, cleanedThroughSeq, logStartLsn)` of this segment's `'K'` mark.
    mark: Option<(usize, i64, i64)>,
}

fn crc32(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}

/// A hasher primed with a section's domain: all 36 header bytes then
/// `be64(sectionOffset)`, fed BEFORE the section's own bytes. Getting this
/// order wrong is why the decoder is written out rather than shared.
fn domain(raw: &[u8], section_off: usize) -> crc32fast::Hasher {
    let mut h = crc32fast::Hasher::new();
    h.update(&raw[..SEG_HDR]);
    h.update(&(section_off as u64).to_be_bytes());
    h
}

fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(b[off..off + 4].try_into().unwrap())
}

fn be64(b: &[u8], off: usize) -> i64 {
    i64::from_be_bytes(b[off..off + 8].try_into().unwrap())
}

/// `format!("{seq:016x}")` — hex, NOT decimal (§3).
fn segment_name(seq: i64) -> String {
    format!("{BASE_NAME}.wal.{seq:016x}")
}

impl Seg {
    fn parse(rel_name: &str, raw: Vec<u8>) -> Seg {
        assert!(
            raw.len() >= SEG_HDR,
            "{rel_name}: shorter than a segment header"
        );
        assert_eq!(&raw[..8], MAGIC, "{rel_name}: bad magic");
        assert_eq!(
            be32(&raw, 8),
            FORMAT_VERSION,
            "{rel_name}: wrong format version"
        );
        assert_eq!(be32(&raw, 12), 0, "{rel_name}: nonzero header flags");
        let seq = be64(&raw, 16);
        let first_lsn = be64(&raw, 24);
        assert_eq!(
            crc32(&raw[..SEG_HDR_CRC_LEN]),
            be32(&raw, SEG_HDR_CRC_LEN),
            "{rel_name}: segment header CRC mismatch"
        );
        let mut sections = Vec::new();
        let mut mark: Option<(usize, i64, i64)> = None;
        let mut off = SEG_HDR;
        while off < raw.len() {
            assert!(
                off + SEC_HDR <= raw.len(),
                "{rel_name}: truncated section header at {off}"
            );
            let tag = raw[off];
            assert!(
                tag == b'S' || tag == b'C' || tag == b'K',
                "{rel_name}: unknown section tag {:?} at {off}",
                tag as char
            );
            let lsn = be64(&raw, off + 1);
            let body_len = be64(&raw, off + 9);
            let mut h = domain(&raw, off);
            h.update(&raw[off..off + SEC_HDR_CRC_LEN]);
            assert_eq!(
                h.finalize(),
                be32(&raw, off + 17),
                "{rel_name}: section header CRC mismatch at {off}"
            );
            // Subtract rather than add: `off + SEC_HDR + body_len <= raw.len()`
            // reads more naturally and OVERFLOWS for a body_len near i64::MAX,
            // wrapping negative and passing the very check it is. Both operands
            // of the remaining-bytes form are already bounded by the file length.
            assert!(
                body_len >= 0 && body_len <= (raw.len() - off - SEC_HDR) as i64,
                "{rel_name}: section body at {off} claims {body_len} bytes, past \
                 the end of a {}-byte file",
                raw.len()
            );
            let body_len_usize = body_len as usize;
            let mut hb = domain(&raw, off);
            hb.update(&raw[off + SEC_HDR..off + SEC_HDR + body_len_usize]);
            assert_eq!(
                hb.finalize(),
                be32(&raw, off + 21),
                "{rel_name}: section body CRC mismatch at {off}"
            );
            if tag == b'K' {
                assert_eq!(
                    body_len, MARK_BODY_LEN,
                    "{rel_name}: a 'K' body is {body_len} bytes, not {MARK_BODY_LEN}"
                );
                assert!(mark.is_none(), "{rel_name}: two 'K' marks in one segment");
                mark = Some((
                    sections.len(),
                    be64(&raw, off + SEC_HDR),
                    be64(&raw, off + SEC_HDR + 8),
                ));
            }
            sections.push(Sec { off, tag, lsn });
            off += SEC_HDR + body_len_usize;
        }
        assert_eq!(
            off,
            raw.len(),
            "{rel_name}: trailing bytes after the last section"
        );
        assert!(
            !sections.is_empty(),
            "{rel_name}: a published segment with no sections"
        );
        Seg {
            rel_name: rel_name.to_string(),
            raw,
            seq,
            first_lsn,
            sections,
            mark,
        }
    }

    fn first(&self) -> &Sec {
        &self.sections[0]
    }
    fn last(&self) -> &Sec {
        self.sections.last().unwrap()
    }
}

// ---------------------------------------------------------------------------
// enumeration and structural self-checks
// ---------------------------------------------------------------------------

fn drop_lock(dir: &Path) {
    let _ = std::fs::remove_file(dir.join(format!("{BASE_NAME}.lock")));
}

/// The published namespace, decoded, ordered by sequence. Refuses anything but
/// segments (§5.4 obligation 7).
fn read_namespace(dir: &Path, ctx: &str) -> Vec<Seg> {
    let mut segs = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{ctx}: {}: {e}", dir.display()))
    {
        let entry = entry.expect("read a namespace entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        assert!(
            entry.file_type().unwrap().is_file(),
            "{ctx}: unexpected non-file in the namespace: {name}"
        );
        assert_ne!(
            name,
            format!("{BASE_NAME}.lock"),
            "{ctx}: the lock sidecar must be removed before enumeration (§2.2)"
        );
        assert!(
            is_segment_name(&name),
            "{ctx}: scratch or foreign file in the namespace: {name} \
             (§5.4 obligation 7)"
        );
        segs.push(Seg::parse(&name, std::fs::read(entry.path()).unwrap()));
    }
    segs.sort_by_key(|g| g.seq);
    for g in &segs {
        assert_eq!(
            g.rel_name,
            segment_name(g.seq),
            "{ctx}: {} does not match {{:016x}} of its own sequence {}",
            g.rel_name,
            g.seq
        );
    }
    segs
}

/// `x.wal.<16 lowercase hex>`, written out rather than regex'd (no regex crate
/// in this repo's dependency set, and the shape is three lines).
fn is_segment_name(name: &str) -> bool {
    let prefix = format!("{BASE_NAME}.wal.");
    match name.strip_prefix(&prefix) {
        Some(hex) => {
            hex.len() == 16
                && hex
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }
        None => false,
    }
}

/// §5.4 obligation 7, shared by both shapes: count, `firstLsn` equalities,
/// contiguous sequences, LSN order.
fn check_common(segs: &[Seg], ctx: &str) {
    assert!(
        segs.len() >= 2,
        "{ctx}: {} segment(s); both shapes need >= 2",
        segs.len()
    );
    let mut prev_lsn = i64::MIN;
    for (i, g) in segs.iter().enumerate() {
        assert_eq!(
            g.first_lsn,
            g.first().lsn,
            "{ctx}: {} states firstLsn {}, its first section holds {}",
            g.rel_name,
            g.first_lsn,
            g.first().lsn
        );
        if i > 0 {
            assert_eq!(
                g.seq,
                segs[i - 1].seq + 1,
                "{ctx}: sequences are not contiguous: {} then {}",
                segs[i - 1].seq,
                g.seq
            );
        }
        for sec in &g.sections {
            assert!(
                sec.lsn > prev_lsn,
                "{ctx}: LSN {} at {}:{} does not follow {prev_lsn}",
                sec.lsn,
                g.rel_name,
                sec.off
            );
            prev_lsn = sec.lsn;
        }
    }
}

/// §5.2's generator self-check.
fn check_tail(segs: &[Seg]) {
    check_common(segs, TAIL_ID);
    assert_eq!(
        segs[0].seq, 1,
        "{TAIL_ID}: sequences must start at 1, not {}",
        segs[0].seq
    );
    for g in segs {
        for sec in &g.sections {
            assert_eq!(
                sec.tag, b'S',
                "{TAIL_ID}: {}:{} is tag {:?}; an uncleaned log carries only 'S'",
                g.rel_name, sec.off, sec.tag as char
            );
        }
    }
}

/// §5.3's self-check and FIVE of §5.3.1's six witness rows, returning the
/// layout index §5.3.1 asks the generator to publish.
///
/// Which five, stated precisely because "checks §5.3.1" would be a false claim.
/// Rows 1, 2, 3, 4 and 6 are checked here. **Row 5 is not** — it asks whether
/// the middle segment's second section holds an entry admitting a
/// size-preserving `T_APPEND` rewrite, which means decoding the entry stream
/// and searching for a replacement encoding of exactly the same length. That is
/// `derive._stranded_append`'s job, and reimplementing it here would be a
/// second entry codec to keep in step with the first. So the shaping pair that
/// exists for row 5 ([`shape_c`]) has its necessity established by
/// `xcheck_bundles.py`, not by this file or by `ci/check.sh`.
///
/// Rows 1-5 are also re-derived independently by `derive.check_witnesses` from
/// the same bytes, so the four checked in both places are checked by two
/// codecs. Row 6 exists ONLY here: `segment_bytes` is a generator setting and
/// leaves no trace in the bytes.
fn check_cleaned(segs: &[Seg]) -> Vec<(&'static str, i64)> {
    check_common(segs, CLEANED_ID);

    // exactly one valid mark, and K4
    let mut mark_seg: Option<&Seg> = None;
    for g in segs {
        if g.mark.is_some() {
            assert!(
                mark_seg.is_none(),
                "{CLEANED_ID}: two segments carry a 'K' mark"
            );
            mark_seg = Some(g);
        }
    }
    let mark_seg = mark_seg
        .unwrap_or_else(|| panic!("{CLEANED_ID}: no 'K' mark; checkpoint() must force one"));
    let (mark_index, through, log_start) = mark_seg.mark.unwrap();
    let mark_lsn = mark_seg.sections[mark_index].lsn;
    assert!(
        through > 0 && through < mark_seg.seq,
        "{CLEANED_ID}: K4 violated: cleanedThroughSeq {through} is not in (0, {})",
        mark_seg.seq
    );
    assert!(
        log_start > 0 && log_start <= mark_lsn,
        "{CLEANED_ID}: logStartLsn {log_start} is not in (0, {mark_lsn}]"
    );

    let retained: Vec<&Seg> = segs.iter().filter(|g| g.seq > through).collect();
    // Cardinality BEFORE indexing: an image retaining nothing is refused for row 1,
    // which is what is wrong with it, and not for an index that happened to be out
    // of bounds on the way to saying so.
    assert_eq!(
        retained.len(),
        3,
        "{CLEANED_ID}: §5.3.1 row 1 requires exactly three retained segments; \
         this bundle retains {}",
        retained.len()
    );
    assert!(
        retained[0].seq > 1,
        "{CLEANED_ID}: the retained floor must be above segment 1 (§5.3)"
    );
    let (lowest, middle, active) = (retained[0], retained[1], retained[2]);
    assert_eq!(
        active.seq,
        segs.last().unwrap().seq,
        "{CLEANED_ID}: the highest retained segment is not the highest segment"
    );

    // §5.3: a 'C' image before the mark and an 'S' after it
    assert_eq!(
        mark_seg.seq, lowest.seq,
        "{CLEANED_ID}: the mark sits in segment {}; §5.3.1 row 2 forbids it in \
         the middle retained segment, so it must be in the lowest one ({}) \
         beside the 'C' image",
        mark_seg.seq, lowest.seq
    );
    assert!(
        mark_seg.sections[..mark_index]
            .iter()
            .any(|s| s.tag == b'C'),
        "{CLEANED_ID}: no 'C' image precedes the mark"
    );
    assert_eq!(
        active.last().tag,
        b'S',
        "{CLEANED_ID}: no 'S' section follows the mark"
    );

    // row 4: the lowest retained segment's stated firstLsn IS the mark's floor,
    // and the retained set is dense — check_common proved ascent, this proves
    // there are no gaps.
    assert_eq!(
        lowest.first_lsn, log_start,
        "{CLEANED_ID}: §5.3.1 row 4: the lowest retained segment states firstLsn \
         {}, the mark attests {log_start}",
        lowest.first_lsn
    );
    let mut expect = log_start;
    for g in &retained {
        for sec in &g.sections {
            assert_eq!(
                sec.lsn, expect,
                "{CLEANED_ID}: §5.3.1 row 4: LSNs are not dense across the \
                 retained set: expected {expect} at {}:{}, found {}",
                g.rel_name, sec.off, sec.lsn
            );
            expect += 1;
        }
    }

    // row 2
    assert!(
        middle.sections.len() >= 2,
        "{CLEANED_ID}: §5.3.1 row 2: the middle retained segment carries {} \
         section(s), fewer than two",
        middle.sections.len()
    );
    for i in 0..2 {
        assert_ne!(
            middle.sections[i].tag, b'K',
            "{CLEANED_ID}: §5.3.1 row 2: section {i} of the middle retained \
             segment is the mark; both must be entry-bearing"
        );
    }

    // row 3
    assert_eq!(
        active.sections.len(),
        1,
        "{CLEANED_ID}: §5.3.1 row 3: the active segment carries {} sections, not one",
        active.sections.len()
    );

    // row 6 — checkable HERE ONLY: segment_bytes is a generator setting, not a
    // published byte.
    assert!(
        (active.raw.len() as u64) < SEGMENT_BYTES,
        "{CLEANED_ID}: §5.3.1 row 6: the active segment is {} bytes, not under \
         segmentBytes {SEGMENT_BYTES}; Q8's appended record would force a \
         rollover and there would be no section to assert",
        active.raw.len()
    );

    selector_index(segs, through)
}

/// §5.3.1's layout index: every segment selector in
/// `catalogue.SEGMENT_SELECTORS` that resolves to EXACTLY ONE segment, and what
/// it resolved to.
///
/// Exactly one is the whole point. A recipe addresses its target by selector
/// and the deriver refuses to pick between candidates, so a selector resolving
/// to zero or to two makes the cell that uses it unbuildable — and a selector
/// resolving to the WRONG single segment produces a cell labelled `reject` that
/// is really an accept. The index records resolution as a SET: a selector
/// missing here is a selector this bundle cannot host, which is as much a fact
/// as the ones present, and `xcheck_bundles.py` compares both directions.
///
/// Mirrors `derive._segment_candidates` deliberately and independently.
fn selector_index(segs: &[Seg], through: i64) -> Vec<(&'static str, i64)> {
    let retained: Vec<&Seg> = segs.iter().filter(|g| g.seq > through).collect();
    let highest = segs.last().unwrap().seq;

    let cand: Vec<(&'static str, Vec<i64>)> = vec![
        (
            "lowest_retained",
            retained.first().map(|g| g.seq).into_iter().collect(),
        ),
        (
            "middle_retained",
            retained
                .iter()
                .skip(1)
                .filter(|g| g.seq != highest)
                .map(|g| g.seq)
                .collect(),
        ),
        (
            "single_section_retained",
            retained
                .iter()
                .skip(1)
                .filter(|g| g.sections.len() == 1)
                .map(|g| g.seq)
                .collect(),
        ),
        ("highest", vec![highest]),
        (
            "mark",
            segs.iter()
                .filter(|g| g.mark.is_some())
                .map(|g| g.seq)
                .collect(),
        ),
    ];
    cand.into_iter()
        .filter(|(_, v)| v.len() == 1)
        .map(|(k, v)| (k, v[0]))
        .collect()
}

/// Grades an arbitrary namespace directory against §5.3's self-check and the
/// FIVE §5.3.1 rows [`check_cleaned`] can decide — 1, 2, 3, 4 and 6, never 5 —
/// exactly as the generator grades its own output. Panics naming the first row
/// that fails.
///
/// Exists so a candidate workload can be FALSIFIED rather than only confirmed:
/// the gate runs the rejected probe variants through this and requires each to
/// be refused for the reason claimed. Without it, a generator whose shaping is
/// unnecessary and one whose shaping is essential look identical.
fn grade_cleaned(dir: &Path) -> Vec<(&'static str, i64)> {
    drop_lock(dir);
    check_cleaned(&read_namespace(dir, CLEANED_ID))
}

/// A one-line structural summary of any namespace: where the mark sits, and how
/// the retained set is shaped around it.
///
/// The companion to [`grade_cleaned`]. Grading reports the FIRST row a
/// candidate violates, which is not always the row that candidate is
/// interesting for — §5.3's literal workload loses on row 1 long before
/// anything looks at row 2, even though row 2 is the reason it can never be
/// repaired by adding a segment.
fn describe_shape(dir: &Path) -> String {
    drop_lock(dir);
    let segs = read_namespace(dir, "describe_shape");
    let mut through = 0i64;
    let mut mark = "none".to_string();
    for g in &segs {
        if let Some((idx, t, _)) = g.mark {
            through = t;
            mark = format!("{}:{}", g.seq, idx);
        }
    }
    let retained: Vec<i64> = segs
        .iter()
        .filter(|g| g.seq > through)
        .map(|g| g.seq)
        .collect();
    format!(
        "mark={mark} retained={retained:?} activeSections={}",
        segs.last().unwrap().sections.len()
    )
}

// ---------------------------------------------------------------------------
// emission
// ---------------------------------------------------------------------------

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

/// One shape's finished product: its published bytes (in sequence order), its
/// recids and its layout index.
struct Bundle {
    id: &'static str,
    image: Vec<(String, Vec<u8>)>,
    recids: Recids,
    layout: Vec<(&'static str, i64)>,
}

/// relName, length and sha of every segment — the comparison §5.4 obligation 8
/// makes across runs, over the complete map rather than file by file.
fn describe(image: &[(String, Vec<u8>)]) -> String {
    image
        .iter()
        .map(|(n, b)| format!("{n}\t{}\t{}\n", b.len(), sha256_hex(b)))
        .collect()
}

fn image_of(segs: &[Seg]) -> Vec<(String, Vec<u8>)> {
    segs.iter()
        .map(|g| (g.rel_name.clone(), g.raw.clone()))
        .collect()
}

fn wipe(dir: &Path) {
    if dir.exists() {
        std::fs::remove_dir_all(dir).expect("wipe a scratch directory");
    }
}

/// Runs one shape into a scratch directory, self-checks it, and returns it.
///
/// §5.4 obligation 1: the base is EMPTY, because the directory is wiped and
/// recreated here. Obligation 5: the store is CLOSED by the workload before
/// anything reads the files — a snapshot of an open store is not the published
/// image.
fn produce(scratch: &Path, id: &'static str, cleaned: bool) -> Bundle {
    wipe(scratch);
    std::fs::create_dir_all(scratch).expect("create the scratch namespace");
    let mut r = Recids::default();
    let base = scratch.join(BASE_NAME);
    if cleaned {
        cleaned_workload(&base, &mut r);
    } else {
        tail_workload(&base, &mut r);
    }
    drop_lock(scratch);
    let segs = read_namespace(scratch, id);
    let layout = if cleaned {
        check_cleaned(&segs)
    } else {
        check_tail(&segs);
        // no mark, so nothing is superseded
        selector_index(&segs, 0)
    };
    let image = image_of(&segs);
    // Reopen: the same reader contract every accept cell will run, then prove
    // the reopen published nothing new — §5.5 says no accept-bundle cell
    // mutates, and all 12 of this bundle's accept cells rest on that.
    let s = open(&base);
    assert_final_state(
        &s,
        &r,
        &format!("{id} reopen"),
        if cleaned { CLEANED_BASE } else { TAIL_BASE },
    );
    s.close().unwrap();
    drop_lock(scratch);
    assert_eq!(
        describe(&image_of(&read_namespace(scratch, id))),
        describe(&image),
        "{id}: a clean reopen changed the published bytes, so §5.5's \"no ACCEPT \
         bundle cell mutates\" is false for this bundle"
    );
    Bundle {
        id,
        image,
        recids: r,
        layout,
    }
}

/// Produces one shape TWICE into separate scratch directories and refuses to
/// publish unless the complete relName->bytes maps agree (§5.4 obligation 8).
///
/// Necessary and not sufficient, and the difference matters: two runs in ONE
/// process share every process-wide seed, so this catches a workload that reads
/// a clock or a directory listing and would NOT catch one that depends on an
/// address. The sync script's separate obligation to invoke each generator
/// twice, in two processes, is what covers that; this is the cheap half that
/// fails at the generator rather than three slices later.
fn produce_twice(root: &Path, id: &'static str, cleaned: bool) -> Bundle {
    let b1 = produce(&root.join(format!(".run1-{id}")), id, cleaned);
    let b2 = produce(&root.join(format!(".run2-{id}")), id, cleaned);
    assert_eq!(
        describe(&b1.image),
        describe(&b2.image),
        "{id} is NOT deterministic across two runs"
    );
    assert_eq!(
        b1.recids, b2.recids,
        "{id}: recid allocation differs across two runs"
    );
    assert_eq!(
        b1.layout, b2.layout,
        "{id}: the layout index differs across two runs"
    );
    b1
}

fn publish(out: &Path, b: &Bundle) {
    let dest = out.join(b.id);
    wipe(&dest);
    std::fs::create_dir_all(&dest).expect("create the bundle directory");
    for (name, bytes) in &b.image {
        std::fs::write(dest.join(name), bytes).expect("publish a segment");
    }
}

// ---------------------------------------------------------------------------
// fragment.tsv and layout.tsv
// ---------------------------------------------------------------------------

fn fragment(tail: &Bundle, cleaned: &Bundle) -> String {
    let commit = generator_commit();
    let mut t = String::new();
    t.push_str("# xfixtures fragment written by tests/wal3_fixtures.rs (C2r).\n");
    t.push_str("# The sync script merges fragments, appends the gzSha256 column to file rows\n");
    t.push_str("# and adds the expect/post rows from catalogue.py.\n");
    for b in [tail, cleaned] {
        t.push_str(&format!(
            "fixture\t{}\twal3-namespace\trust\t{commit}\n",
            b.id
        ));
        // §2: file rows sorted numerically by segment sequence — image_of
        // preserves that order.
        for (name, bytes) in &b.image {
            t.push_str(&format!(
                "file\t{}\t{name}\t{}\t{}\n",
                b.id,
                bytes.len(),
                sha256_hex(bytes)
            ));
        }
        let base = if std::ptr::eq(b, tail) {
            TAIL_BASE
        } else {
            CLEANED_BASE
        };
        for (label, recid, state, pid, len) in expects(&b.recids, base) {
            t.push_str(&format!(
                "recid\t{}\t{label}\t{recid}\t{state}\t{pid}\t{len}\n",
                b.id
            ));
        }
    }
    t
}

/// §5.3.1's layout index, as `symbol` rows in §10.1's shape.
///
/// What makes it more than decoration: these values come from THIS generator's
/// own decoder and its own knowledge of the workload, and
/// `derive.resolve_symbols` resolves the same names independently from the
/// published bytes. The gate compares them. A row nothing reads is a claim
/// nothing checked (§10.1), so this file is written to be read.
fn layout(tail: &Bundle, cleaned: &Bundle) -> String {
    let mut t = String::new();
    t.push_str("# §5.3.1 layout index written by tests/wal3_fixtures.rs (C2r).\n");
    t.push_str("# symbol <fixtureId> <@segmentSelector> <relName>, one row per selector that\n");
    t.push_str("# resolves to exactly one segment. An ABSENT selector is a claim too: this\n");
    t.push_str("# bundle cannot host a recipe that addresses it. Cross-checked against\n");
    t.push_str("# derive._segment_candidates, both directions, by xcheck_bundles.py.\n");
    for b in [tail, cleaned] {
        for (sym, seq) in &b.layout {
            t.push_str(&format!(
                "symbol\t{}\t@{sym}\t{}\n",
                b.id,
                segment_name(*seq)
            ));
        }
    }
    t
}

/// The whole generator: both shapes, twice each, published with their sidecars.
fn generate(out: &Path, force: bool) {
    if out.exists() {
        assert!(out.is_dir(), "--out is not a directory: {}", out.display());
        let nonempty = std::fs::read_dir(out).expect("read the output dir").count() > 0;
        assert!(
            !nonempty || force,
            "output directory not empty (set XFIXTURES_FORCE=1 to overwrite): {}",
            out.display()
        );
    }
    std::fs::create_dir_all(out).expect("create the output dir");

    let tail = produce_twice(out, TAIL_ID, false);
    let cleaned = produce_twice(out, CLEANED_ID, true);
    for id in [TAIL_ID, CLEANED_ID] {
        wipe(&out.join(format!(".run1-{id}")));
        wipe(&out.join(format!(".run2-{id}")));
    }
    publish(out, &tail);
    publish(out, &cleaned);
    write_synced(
        &out.join("fragment.tsv"),
        fragment(&tail, &cleaned).as_bytes(),
    );
    write_synced(&out.join("layout.tsv"), layout(&tail, &cleaned).as_bytes());

    // §5.4 obligation 7 applied to the OUTPUT directory, not just to the
    // scratch namespaces. A forced rerun over a directory holding anything else
    // republishes the two bundles and leaves the rest in place, and the sync
    // script would then consume whatever was there. Checked rather than
    // deleted: `--out` is a path the caller chose, and silently emptying it is
    // a bigger risk than refusing to publish into it.
    let mut stray: Vec<String> = std::fs::read_dir(out)
        .expect("read the output dir")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| {
            !matches!(
                n.as_str(),
                TAIL_ID | CLEANED_ID | "fragment.tsv" | "layout.tsv"
            )
        })
        .collect();
    stray.sort();
    assert!(
        stray.is_empty(),
        "the output directory also holds {stray:?}; a generator's output \
         directory must contain exactly the two bundles and the two sidecars, \
         because everything in it is what the sync script will consume"
    );
}

fn write_synced(path: &Path, bytes: &[u8]) {
    let mut f = std::fs::File::create(path).expect("create a sidecar");
    f.write_all(bytes).expect("write a sidecar");
    f.sync_all().expect("sync a sidecar");
}

// ---------------------------------------------------------------------------
// the CLI entry point (ignored; the sync script runs this)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn write_wal3_fixtures() {
    let out = std::env::var("XFIXTURES_OUT")
        .expect("set XFIXTURES_OUT=<dir> to run the fixture generator");
    let force = std::env::var("XFIXTURES_FORCE").as_deref() == Ok("1");
    generate(Path::new(&out), force);
}

// ---------------------------------------------------------------------------
// the shape probe: candidate `cleaned` workloads, measured against THIS engine
// ---------------------------------------------------------------------------

/// Runs one named candidate `cleaned` workload and leaves its segments on disk.
///
/// The Stage C plan predicted, from the rollover rule alone, that §5.3's
/// literal workload cannot produce §5.3.1's shape — and its proposed repair was
/// wrong in its second half. C2j measured both against java. This probe
/// measures them against RUST, because "java behaves this way" is not evidence
/// about a port: it is the hypothesis the port is supposed to test.
///
/// Every variant that could become the generator's workload ends in the final
/// logical state §5.3 pins and asserts it before closing, so a variant that
/// reaches the shape by changing what the fixture MEANS fails here rather than
/// in review. The one exception is `shaped-half-rotate`, which exists precisely
/// to show that half of a state-preserving PAIR is not state-preserving. It
/// asserts the state it DOES reach — A holding the oversized payload, the rest
/// of §5.2 unchanged — rather than skipping the check, so the exception covers
/// the one record it is about and nothing else.
fn probe_variant(variant: &str, dir: &Path) {
    wipe(dir);
    std::fs::create_dir_all(dir).expect("create the probe namespace");
    let mut r = Recids::default();
    let base = dir.join(BASE_NAME);
    let s = open(&base);
    match variant {
        // §5.3 exactly as revision 1 wrote it
        "spec" => {
            t1(&s, &mut r, CLEANED_BASE);
            t2(&s, &mut r);
            t3(&s, &mut r, CLEANED_BASE);
            s.checkpoint().unwrap();
            t4(&s, &r, CLEANED_BASE);
        }
        // the plan's proposed move, on its own
        "ckpt-after-t2" => {
            t1(&s, &mut r, CLEANED_BASE);
            t2(&s, &mut r);
            s.checkpoint().unwrap();
            t3(&s, &mut r, CLEANED_BASE);
            t4(&s, &r, CLEANED_BASE);
        }
        // the plan's proposal in full, as written there: one oversized commit
        // was expected to "force the rotation into a single-section active
        // segment". It does not.
        "ckpt-after-t2-shaped" => {
            t1(&s, &mut r, CLEANED_BASE);
            t2(&s, &mut r);
            s.checkpoint().unwrap();
            t3(&s, &mut r, CLEANED_BASE);
            t4(&s, &r, CLEANED_BASE);
            shape_c(&s, &r, CLEANED_BASE);
            s.update(
                Recid::new(r.f).unwrap(),
                Some(&payload(CLEANED_BASE + 4, 1_200_000)),
                &R,
            )
            .unwrap();
            s.commit().unwrap();
        }
        // the adopted workload MINUS shape_c, to show shape_c matters
        "shaped-no-C" => {
            t1(&s, &mut r, CLEANED_BASE);
            t2(&s, &mut r);
            s.checkpoint().unwrap();
            t3(&s, &mut r, CLEANED_BASE);
            t4(&s, &r, CLEANED_BASE);
            shape_rotate(&s, &r, CLEANED_BASE);
        }
        // the adopted workload MINUS shape_rotate: the other shaping decision,
        // removed on its own.
        "shaped-no-rotate" => {
            t1(&s, &mut r, CLEANED_BASE);
            t2(&s, &mut r);
            s.checkpoint().unwrap();
            t3(&s, &mut r, CLEANED_BASE);
            shape_c(&s, &r, CLEANED_BASE);
            t4(&s, &r, CLEANED_BASE);
        }
        // the adopted workload with only the FIRST half of shape_rotate: the
        // commit that crosses `segment_bytes`, without the one that lands alone
        // in the segment it opens. This is the case the pair exists for, and it
        // ends in the WRONG final state (A holds the oversized payload), so it
        // is measured for shape only and must not assert §5.3's state.
        "shaped-half-rotate" => {
            t1(&s, &mut r, CLEANED_BASE);
            t2(&s, &mut r);
            s.checkpoint().unwrap();
            t3(&s, &mut r, CLEANED_BASE);
            shape_c(&s, &r, CLEANED_BASE);
            t4(&s, &r, CLEANED_BASE);
            s.update(
                r.a(),
                Some(&payload(CLEANED_BASE + 7, SEGMENT_BYTES as usize)),
                &R,
            )
            .unwrap();
            s.commit().unwrap();
            assert!(
                s.cleaner_bytes().0 > 0,
                "{variant}: the checkpoint wrote no image"
            );
            // Not §5.3's final state — that is the POINT of this variant — but
            // not unchecked either: A holds the oversized payload and every
            // other record is where §5.2 leaves it. The exception this variant
            // is granted is exactly one record wide.
            assert_state_with_a(
                &s,
                &r,
                variant,
                CLEANED_BASE,
                &payload(CLEANED_BASE + 7, SEGMENT_BYTES as usize),
            );
            s.close().unwrap();
            drop_lock(dir);
            return;
        }
        other => panic!("unknown probe variant: {other}"),
    }
    assert!(
        s.cleaner_bytes().0 > 0,
        "{variant}: the checkpoint wrote no image; this variant is not a CLEANED shape at all"
    );
    assert_final_state(&s, &r, variant, CLEANED_BASE);
    s.close().unwrap();
    drop_lock(dir);
}

// ---------------------------------------------------------------------------
// the gate
// ---------------------------------------------------------------------------
//
// The generator asserts §5.2/§5.3/§5.3.1/§5.4 about its own output while it
// runs, so the gate's first job is simply to RUN it: an assertion in a program
// nobody invokes is not a check, and a generator invoked only by the sync
// script, in the planning repo, on the day of the cutover, is that defect one
// slice later.
//
// Its second job is the part the generator cannot do for itself, because a
// program cannot catch its own blind spot by re-reading its own output: that
// the two shapes really are DIFFERENT shapes, and that each §5.3.1 witness
// depends on something — falsify the workload and the witness must fail.

/// A scratch directory unique to one test, removed and recreated on entry.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mapdb5_wal3_c2r_{}_{tag}", std::process::id()));
    wipe(&dir);
    std::fs::create_dir_all(&dir).expect("create a test scratch directory");
    dir
}

/// The panic message a closure produced, or `None` if it did not panic. The
/// hook is silenced for the duration so an EXPECTED refusal does not print a
/// backtrace that reads like a failure.
fn refusal_of(f: impl FnOnce() + std::panic::UnwindSafe) -> Option<String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    match out {
        Ok(()) => None,
        Err(e) => Some(
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".to_string()),
        ),
    }
}

/// The variant must be REFUSED by the generator's own §5.3.1 grading, for the
/// stated reason.
fn expect_refusal(dir: &Path, variant: &str, expected: &str) {
    let owned = dir.to_path_buf();
    match refusal_of(move || {
        grade_cleaned(&owned);
    }) {
        None => panic!(
            "variant {variant} was expected to violate §5.3.1 ({expected}) and it \
             satisfied every row instead — either the shaping this generator does \
             is unnecessary, or this expectation is stale"
        ),
        Some(msg) => assert!(
            msg.contains(expected),
            "variant {variant} was refused, but not for the reason claimed.\n  \
             expected a message containing: {expected}\n  got: {msg}"
        ),
    }
}

fn describe_dir(dir: &Path) -> String {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read a bundle directory")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
        .iter()
        .map(|n| {
            let raw = std::fs::read(dir.join(n)).unwrap();
            format!("{n}\t{}\t{}\n", raw.len(), sha256_hex(&raw))
        })
        .collect()
}

fn read_to_string(path: &Path) -> String {
    std::fs::read_to_string(path).expect("read a sidecar")
}

/// `"<fixtureId> @<selector>" -> relName` from the published `layout.tsv`.
fn layout_rows(dir: &Path) -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    for line in read_to_string(&dir.join("layout.tsv")).lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 4, "layout.tsv row arity: {line}");
        assert_eq!(f[0], "symbol", "layout.tsv row type: {line}");
        assert!(
            out.insert(format!("{} {}", f[1], f[2]), f[3].to_string())
                .is_none(),
            "layout.tsv claims {} {} twice",
            f[1],
            f[2]
        );
    }
    out
}

/// The whole generator, end to end. Every §5.2/§5.3/§5.3.1/§5.4 self-check
/// inside it is exercised by this one call; the assertions here are about the
/// PRODUCT.
#[test]
fn generator_produces_both_bundles() {
    let dir = scratch("both");
    generate(&dir, true);

    for id in [TAIL_ID, CLEANED_ID] {
        let b = dir.join(id);
        assert!(b.is_dir(), "{id} was not published");
        let names: Vec<String> = std::fs::read_dir(&b)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.len() >= 2,
            "{id} published {names:?}; both shapes need >= 2 segments"
        );
        for n in &names {
            assert!(
                is_segment_name(n),
                "{id}: {n} is not a {{:016x}} segment name"
            );
        }
    }
    // §5.3: the cleaned bundle's retained floor is above segment 1, which is the
    // shape v1 could not express — and the reason this bundle exists at all.
    assert!(
        !dir.join(CLEANED_ID).join(segment_name(1)).exists(),
        "the cleaned bundle must not contain segment 1"
    );

    // §5.4 obligation 7: no scratch survives, and nothing but the two bundles
    // and the two sidecars is published.
    let mut published: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    published.sort();
    assert_eq!(
        published,
        vec![
            "fragment.tsv".to_string(),
            "layout.tsv".to_string(),
            CLEANED_ID.to_string(),
            TAIL_ID.to_string(),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>(),
        "unexpected files in the output directory"
    );
    wipe(&dir);
}

/// §5.4 obligation 7 for the output directory: a forced rerun over a directory
/// holding anything else must be REFUSED, not quietly republished around.
#[test]
fn refuses_to_publish_beside_stray_files() {
    let dir = scratch("stray");
    generate(&dir, true);
    std::fs::write(dir.join("leftover.tsv"), [1u8]).unwrap();
    let owned = dir.clone();
    let msg = refusal_of(move || generate(&owned, true)).unwrap_or_else(|| {
        panic!(
            "the generator published into a directory holding a stray file; the \
             sync script consumes everything in that directory"
        )
    });
    assert!(
        msg.contains("leftover.tsv"),
        "refused, but not for the stray file: {msg}"
    );
    wipe(&dir);
}

/// §5.4 obligation 1: a non-empty output directory is refused without `force`.
#[test]
fn refuses_a_nonempty_output_directory_without_force() {
    let dir = scratch("noforce");
    std::fs::write(dir.join("something"), [1u8]).unwrap();
    let owned = dir.clone();
    let msg = refusal_of(move || generate(&owned, false))
        .expect("a non-empty output directory must be refused without force");
    assert!(
        msg.contains("output directory not empty"),
        "refused, but not for the non-empty directory: {msg}"
    );
    wipe(&dir);
}

/// Runs the documented generator CLI in a SEPARATE OS PROCESS and returns its
/// output directory.
///
/// `current_exe()` is this test binary, and `write_wal3_fixtures` is the
/// `#[ignore]`d entry point the sync script invokes, so the child runs exactly
/// the published command path — not a function call dressed up as one.
fn generate_in_a_child_process(tag: &str) -> PathBuf {
    let out = scratch(tag);
    let exe = std::env::current_exe().expect("locate this test binary");
    let status = std::process::Command::new(&exe)
        .args([
            "write_wal3_fixtures",
            "--ignored",
            "--exact",
            "--test-threads=1",
        ])
        .env("XFIXTURES_OUT", &out)
        .env("XFIXTURES_FORCE", "1")
        .status()
        .expect("spawn the generator in a child process");
    assert!(
        status.success(),
        "the generator child process failed: {status} ({})",
        exe.display()
    );
    out
}

/// §5.4 obligation 8, ACROSS PROCESSES, and this time literally.
///
/// The generator compares two runs internally, but both live in ONE process and
/// share every process-wide seed, so an output depending on an address, a
/// hash-map iteration order or a lazily initialised global would agree with
/// itself and still differ between runs. The comparison here is over the
/// COMPLETE published tree — both bundles and both sidecars — because
/// `fragment.tsv` does not exist yet when `produce_twice` runs and is therefore
/// the one obligation the generator structurally cannot assert about itself.
#[test]
fn two_processes_agree_byte_for_byte() {
    let (a, b) = (
        generate_in_a_child_process("proc-a"),
        generate_in_a_child_process("proc-b"),
    );
    for id in [TAIL_ID, CLEANED_ID] {
        assert_eq!(
            describe_dir(&a.join(id)),
            describe_dir(&b.join(id)),
            "{id} is not deterministic across two processes"
        );
    }
    for side in ["fragment.tsv", "layout.tsv"] {
        assert_eq!(
            read_to_string(&a.join(side)),
            read_to_string(&b.join(side)),
            "{side} is not deterministic across two processes"
        );
    }
    // The whole tree, not just the parts named above: a generator that grew a
    // third output would otherwise be compared by nothing.
    assert_eq!(
        describe_tree(&a),
        describe_tree(&b),
        "the published trees differ"
    );
    wipe(&a);
    wipe(&b);
}

/// Every published path under `root`, with its length and sha — the complete
/// tree §5.4 obligation 8 compares, rather than an enumerated list of the files
/// this generator happens to write today.
fn describe_tree(root: &Path) -> String {
    let mut rows = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).expect("walk the published tree") {
            let e = e.unwrap();
            if e.file_type().unwrap().is_dir() {
                stack.push(e.path());
                continue;
            }
            let rel = e.path().strip_prefix(root).unwrap().to_owned();
            let raw = std::fs::read(e.path()).unwrap();
            rows.push(format!(
                "{}\t{}\t{}\n",
                rel.display(),
                raw.len(),
                sha256_hex(&raw)
            ));
        }
    }
    rows.sort();
    rows.concat()
}

/// The two shapes must be genuinely different, or one of them is testing
/// nothing — restated as the layout index each produces.
#[test]
fn the_two_shapes_differ() {
    let dir = scratch("differ");
    generate(&dir, true);
    assert_ne!(
        describe_dir(&dir.join(TAIL_ID)),
        describe_dir(&dir.join(CLEANED_ID)),
        "the tail and cleaned bundles are byte-identical, so one is redundant"
    );
    let layout = layout_rows(&dir);
    // §5.3.1 rows 1-3 as the index they produce. The tail shape has no middle
    // retained segment because it has no mark and therefore only two segments;
    // the cleaned shape must have all three positions.
    assert_eq!(
        layout
            .get(&format!("{CLEANED_ID} @middle_retained"))
            .map(String::as_str),
        Some(segment_name(3).as_str())
    );
    assert_eq!(
        layout
            .get(&format!("{CLEANED_ID} @single_section_retained"))
            .map(String::as_str),
        Some(segment_name(4).as_str())
    );
    assert_eq!(
        layout
            .get(&format!("{CLEANED_ID} @mark"))
            .map(String::as_str),
        Some(segment_name(2).as_str())
    );
    assert!(
        !layout.contains_key(&format!("{TAIL_ID} @mark")),
        "the tail shape must host no `mark` selector"
    );
    wipe(&dir);
}

/// The published bundles open cleanly in a fresh directory and mutate nothing.
///
/// This is §5.5's claim, and all 36 accept cells rest on it: "no ACCEPT bundle
/// cell mutates", so no cell carries a `created:`/`truncated:` override. The
/// generator checks it for its scratch copy; this checks the copy that was
/// actually published, and opens it the way a cell does — plain
/// `StoreWAL::open`, no generator settings.
#[test]
fn published_bundles_open_without_mutating() {
    let dir = scratch("nomutate");
    generate(&dir, true);
    for id in [TAIL_ID, CLEANED_ID] {
        let cell = scratch(&format!("cell-{id}"));
        for e in std::fs::read_dir(dir.join(id)).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), cell.join(e.file_name())).unwrap();
        }
        let before = describe_dir(&cell);
        let s = StoreWAL::open(&cell.join(BASE_NAME)).expect("open a published bundle");
        s.verify().expect("verify a published bundle");
        s.close().unwrap();
        drop_lock(&cell);
        assert_eq!(
            before,
            describe_dir(&cell),
            "{id}: a clean rw open mutated the published bundle, so §5.5 is false \
             for it and its accept cells need file-set overrides"
        );
        wipe(&cell);
    }
    wipe(&dir);
}

/// The three REJECTED candidate workloads, measured against this engine.
///
/// What this pins, stated narrowly because the name it first carried
/// ("witnesses depend on the shaping") claimed more than the inputs deliver:
/// all three of these lose §5.3.1 **row 1** — they are the history of how §5.3's
/// table was arrived at, not a per-witness falsification. Two of them change
/// the checkpoint's position and the third tests a rewrite that was proposed
/// and rejected; none of them removes a single adopted shaping step. The
/// per-step falsifiers are [`each_shaping_step_is_load_bearing`] and
/// [`row_five_is_invisible_to_this_generator`], and the per-WITNESS-ROW
/// falsifiers are `derive.py --self-test`'s, which mutates the model once per
/// row and refuses to let a row exist without a case.
///
/// The three shape strings are the evidence that §5.3's amended table binds
/// rust and not only java: they are the numbers C2j measured, reproduced by an
/// independent implementation of the same rules against a different writer.
#[test]
fn rejected_candidate_workloads_measured() {
    // §5.3 as revision 1 literally wrote it: checkpoint after T3, no shaping.
    // The cleaner's image covers F's 1.2 MB, which overflows the segment
    // holding it, so the forced mark lands as section 0 of the NEXT segment.
    // That is the finding that moved the checkpoint: adding segments cannot
    // repair it, because whichever segment then becomes the middle one opens
    // with the 'K' that §5.3.1 row 2 forbids there.
    let d = scratch("v-spec");
    probe_variant("spec", &d);
    assert_eq!(
        describe_shape(&d),
        "mark=3:0 retained=[2, 3] activeSections=2",
        "§5.3's literal workload no longer puts the mark where C2j measured it"
    );
    expect_refusal(&d, "spec", "row 1 requires exactly three retained segments");
    wipe(&d);

    // The checkpoint moved: the mark is now section 1 of the LOWEST retained
    // segment, beside the 'C' image, which is what row 2 needs. Two retained
    // segments is what is left to fix.
    let d = scratch("v-ckpt");
    probe_variant("ckpt-after-t2", &d);
    assert_eq!(
        describe_shape(&d),
        "mark=2:1 retained=[2, 3] activeSections=1",
        "moving the checkpoint no longer puts the mark beside the 'C' image"
    );
    expect_refusal(
        &d,
        "ckpt-after-t2",
        "row 1 requires exactly three retained segments",
    );
    wipe(&d);

    // The plan's own first proposal for the third segment: one oversized
    // `update(F, own content)` was expected to "force the rotation into a
    // single-section active segment". It does not. Rollover is tested BEFORE
    // the append, so the 1.2 MB section joins the segment it overflows and
    // nothing rotates — the active segment ends with FOUR sections.
    let d = scratch("v-ckpt-shaped");
    probe_variant("ckpt-after-t2-shaped", &d);
    assert_eq!(
        describe_shape(&d),
        "mark=2:1 retained=[2, 3] activeSections=4",
        "the plan's rejected proposal no longer fails the way C2j measured it"
    );
    expect_refusal(
        &d,
        "ckpt-after-t2-shaped",
        "row 1 requires exactly three retained segments",
    );
    wipe(&d);
}

/// The one shaping step this file CANNOT justify, made executable rather than
/// argued.
///
/// [`shape_c`] exists for §5.3.1 row 5, and row 5 is the row [`check_cleaned`]
/// does not check — deciding it means decoding the entry stream and searching
/// for a size-preserving replacement encoding. So dropping `shape_c` produces a
/// bundle this generator's own grading ACCEPTS, and only
/// `derive.check_witnesses` refuses it:
///
/// ```text
/// FAIL stranded-append-candidate: §5.3.1 row 5: no entry in the selected
/// section admits a stranded T_APPEND
/// ```
///
/// This test asserts the rust side's BLINDNESS, which is the only part of it
/// this file can own. If it ever starts failing, the generator has grown a
/// row-5 check and this test should become an `expect_refusal` instead — which
/// is a better outcome, not a regression.
#[test]
fn row_five_is_invisible_to_this_generator() {
    let d = scratch("v-no-c");
    probe_variant("shaped-no-C", &d);
    assert_eq!(
        describe_shape(&d),
        "mark=2:1 retained=[2, 3, 4] activeSections=1",
        "dropping shape_c no longer produces the shape this test is about"
    );
    grade_cleaned(&d); // accepted: rows 1,2,3,4,6 all hold without shape_c
    wipe(&d);
}

/// Each ADOPTED shaping step, removed on its own, and what it costs.
///
/// This is the test [`rejected_candidate_workloads_measured`] was mistakenly
/// named for. The rotation pair is two commits because rollover is tested
/// BEFORE a section is appended, and both halves are load-bearing in different
/// ways — so each half is removed separately rather than the pair being removed
/// as a unit, which would have shown only that "some rotation is needed".
///
/// The remaining adopted step, [`shape_c`], is falsified in
/// [`row_five_is_invisible_to_this_generator`] — it cannot be falsified here,
/// because the row it exists for is the one this generator cannot see.
#[test]
fn each_shaping_step_is_load_bearing() {
    // No rotation at all: the log never opens a third segment, so `middle` and
    // `active` are the same file and row 1 is lost.
    let d = scratch("v-no-rot");
    probe_variant("shaped-no-rotate", &d);
    assert_eq!(
        describe_shape(&d),
        "mark=2:1 retained=[2, 3] activeSections=3",
        "dropping the rotation pair no longer produces the shape this test pins"
    );
    expect_refusal(
        &d,
        "shaped-no-rotate",
        "row 1 requires exactly three retained segments",
    );
    wipe(&d);

    // Only the half that CROSSES `segment_bytes`. This is the case the pair
    // exists for: the oversized section joins the segment it overflows, so
    // nothing has rotated yet and the third segment never opens.
    let d = scratch("v-half-rot");
    probe_variant("shaped-half-rotate", &d);
    assert_eq!(
        describe_shape(&d),
        "mark=2:1 retained=[2, 3] activeSections=4",
        "the oversized commit now rotates on its own, which would make the \
         second half of the rotation pair unnecessary — read StoreWAL.java:1688 \
         and rust's `rollover` before changing this expectation"
    );
    expect_refusal(
        &d,
        "shaped-half-rotate",
        "row 1 requires exactly three retained segments",
    );
    wipe(&d);
}
