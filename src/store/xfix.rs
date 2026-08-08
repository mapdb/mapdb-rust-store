//! Cross-port fixture reader: the **v1/v2 manifest dispatch**, the WAL v3
//! segment decoder, and the assertion helpers both halves of the suite share
//! (Stage C slice **C3r**, `todo/store-wal3/wal3-c3-plan.md`).
//!
//! # Why this file lives in `src/` and is included from `tests/`
//!
//! Decision **C-D3**: the schema-v2 `ro` cells need a read-only opener, and
//! `WalOptions`/`open_cfg` are `pub(crate)`. Revision 1 of the Stage C plan
//! proposed a `#[doc(hidden)] pub fn open_read_only` and that was refused —
//! `#[doc(hidden)]` hides an item from rustdoc and from nothing else, so it
//! would still be callable by any downstream crate and still semver-visible,
//! contradicting D7 ("no public read-only DB surface in this workstream").
//! So the `ro` executor is an in-crate `#[cfg(test)]` module
//! ([`super::xfix_ro`]) and this module — parser, decoder, assertions — is
//! **compiled twice**: once inside the lib for that executor, and once into
//! `tests/xfixture_conformance.rs` and `tests/wal3_decode.rs` through
//! `#[path]`. Rust gains no public API.
//!
//! Being compiled in two crates has one consequence worth stating: this file
//! must never say `crate::`, because `crate` is the lib in one build and a
//! test binary in the other. It names the engine as `mapdb_rust_store::…`,
//! which the lib build resolves through the `extern crate self as` alias in
//! `src/lib.rs`. It therefore reaches only the engine's **public** API, and
//! the format constants below are transcribed rather than imported —
//! [`super::xfix_ro::the_transcribed_constants_match_the_engine`] is the check
//! that keeps the transcription honest, and it can only live in-crate.
//!
//! # What the decoder validates, and what it does not
//!
//! [`decode`] checks the magic, the format version, the header CRC, both
//! section CRCs, the section tag vocabulary and every entry-stream bound. It
//! does **not** re-implement the engine's rules: `flags == 0`, seq/filename
//! agreement, dense increasing LSNs, cross-segment linkage, mark ranges,
//! `cap_valid`, one-entry-per-recid. The engine does all of that, and the C3
//! suite opens these same bundles through the engine in the same run.
#![allow(dead_code)]

use flate2::read::GzDecoder;
use mapdb_rust_store::error::{DbError, Result};
use mapdb_rust_store::io::{DataInput2, DataOutput2};
use mapdb_rust_store::ser::Serializer;
use mapdb_rust_store::store::{Recid, Store, StoreDirect, StoreWAL};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as _;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// vocabularies (shared with the java reader, org.mapdb.xfixtures.XFixtureManifest)
// ---------------------------------------------------------------------------

pub const ENGINE: &str = "rust";
pub const ENGINES: [&str; 3] = ["java", "rust", "zig"];
pub const MODES: [&str; 2] = ["ro", "rw"];
pub const STATES: [&str; 4] = ["live", "null", "prealloc", "deleted"];
pub const VERDICTS: [&str; 2] = ["accept", "reject"];

/// Contract §2's `kind` vocabulary. D6 fixed the v2 set as "all v1 kinds +
/// `wal3-namespace`", and `port-wal`/`java-wal-namespace` are **retained as
/// valid tokens** though no v2 fixture uses them — retiring a fixture family is
/// not a reason to make a version-dispatch parser reject the token.
pub const V2_KINDS: [&str; 5] = [
    "direct",
    "reject",
    "wal3-namespace",
    "port-wal",
    "java-wal-namespace",
];

/// A v2 fixture no engine wrote records `derived` here, and then owes exactly
/// one `derived` row (contract §2, amendment 3).
pub const V2_GENERATORS: [&str; 4] = ["java", "rust", "zig", "derived"];

pub const V2_OPENERS: [&str; 2] = ["direct", "wal3"];

/// sha256 of the empty byte string — the zero-length-content marker that has
/// to stay distinguishable from NULL content.
pub const EMPTY_SHA: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

// ---------------------------------------------------------------------------
// small utilities
// ---------------------------------------------------------------------------

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

pub fn gunzip(gz: &[u8], what: &str) -> Vec<u8> {
    let mut raw = Vec::new();
    GzDecoder::new(gz)
        .read_to_end(&mut raw)
        .unwrap_or_else(|e| panic!("gunzip {what} failed: {e}"));
    raw
}

/// The corpus payload function: `payload(id, len)[i] == (i*131 + id) & 0xff`.
/// Invertible from its first byte, which is what makes it usable as a witness
/// that an entry stream was framed the way the writer wrote it.
pub fn payload(payload_id: u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i as u64).wrapping_mul(131).wrapping_add(payload_id) & 0xff) as u8)
        .collect()
}

pub fn hex32(v: u32) -> String {
    format!("{v:08x}")
}

/// The comment block `GOLDEN-BODY.tsv` carries, pinned verbatim.
///
/// Rust compares the two golden files by ROW, dropping comments — but for this
/// file that loses something java's whole-text comparison keeps. The block is
/// the file's PROVENANCE: it states that the frozen java reader authored it,
/// that `lenPlus` is raw, and how to regenerate it. Without a pin, that header
/// could be deleted or rewritten to claim python authorship while every test
/// stayed green, and the next reader would be graded against a file whose
/// authority claim nobody was checking. (`GOLDEN-DECODE.tsv` needs no
/// equivalent: java compares it by row too.)
pub const GOLDEN_BODY_HEADER: &[&str] = &[
    "# The DECODED BODIES of every pinned schema-v2 sample section, as the FROZEN JAVA",
    "# READER reads them — contract §11.2's engine-against-engine half.",
    "#",
    "#   sec  <bundle> <relName> <index> <tag> <entryCount>",
    "#   ent  <bundle> <relName> <index> <ord> <kind> <recid> <cap> <lenPlus> <contentSha256>",
    "#   mark <bundle> <relName> <index> <cleanedThroughSeq> <logStartLsn>",
    "#",
    "# GOLDEN-DECODE.tsv pins FRAMING and deliberately stops there: walfmt.py is a",
    "# structural codec, and store record semantics written in python would be a fifth",
    "# implementation no one reviews. This file is the other half, and Java authors it",
    "# because Java is the reference for what a body MEANS.",
    "#",
    "# lenPlus IS RAW, NOT A LENGTH. `lenPlus == 0` is NULL content; `lenPlus == 1` is",
    "# ZERO-LENGTH content (StoreWAL.applySection). A reader that decodes lenPlus into a",
    "# length collapses the two, and two readers that both collapse it agree forever.",
    "# contentSha256 is `-` for NULL and the empty-string sha for zero-length, so the two",
    "# differ in both columns. The sample contains one of each: recid 12 and recid 11.",
    "#",
    "# `-` means the column does not apply to that entry kind. cap is emitted because a",
    "# reader must decode it to find the next entry at all; leaving it out would be a",
    "# field the comparison never reaches.",
    "#",
    "# Regenerate with mapdb-java-store's org.mapdb.xfixtures.Wal3BodyDump; the java suite",
    "# re-derives it and fails on drift.",
];

/// Data lines of a golden `.tsv`: comments and blank lines dropped. The
/// comment block is prose the authoring engine wrote; only the rows are a
/// decode, and only the rows are compared.
pub fn golden_rows(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

/// Compares two row lists line by line, reporting the FIRST disagreement with
/// its row number before falling back to the length check — a diff that only
/// says "1042 lines != 1041" makes a one-row drift expensive to find.
pub fn assert_rows_equal(what: &str, want: &[&str], got: &[String]) {
    for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
        assert_eq!(*w, g.as_str(), "{what} row {}", i + 1);
    }
    assert_eq!(want.len(), got.len(), "{what}: row count");
}

fn dir_entries(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// the WAL v3 codec constants, transcribed
// ---------------------------------------------------------------------------

pub const SEG_HDR: usize = 36;
pub const SEG_HDR_CRC_LEN: usize = 32;
pub const SEC_HDR: usize = 25;
pub const SEC_HDR_CRC_LEN: usize = 17;
pub const MAGIC: &[u8; 8] = b"MDBS.WAL";
pub const FORMAT_VERSION: u32 = 3;
pub const MARK_BODY_LEN: i64 = 16;

pub const TAG_SECTION: u8 = b'S';
pub const TAG_IMAGE: u8 = b'C';
pub const TAG_MARK: u8 = b'K';

/// `StoreDirect`'s plain-record capacity ceiling, transcribed from
/// `index_val::MAX_CAPACITY` (checked in [`super::xfix_ro`]). It is half of
/// `cap_valid`'s rule and the C3r review found it missing: without the ceiling,
/// [`check_cap`] accepts capacities recovery rejects, and — worse — accepts
/// `cap == 0` for content that is not oversize at all.
pub const MAX_CAPACITY: i64 = 0xFFFD * 16;

pub const T_PREALLOC: u8 = 1;
pub const T_RECORD: u8 = 2;
pub const T_APPEND: u8 = 3;
pub const T_DELETE: u8 = 4;

fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(b[off..off + 4].try_into().unwrap())
}

fn be64(b: &[u8], off: usize) -> i64 {
    i64::from_be_bytes(b[off..off + 8].try_into().unwrap())
}

/// A hasher primed with a section's CRC domain: all 36 header bytes then
/// `be64(sectionOffset)`, fed BEFORE the section's own bytes. The offset being
/// in the domain is what makes a section un-relocatable, and getting the order
/// wrong is the reason this is written out rather than reasoned about.
fn domain(raw: &[u8], section_off: usize) -> crc32fast::Hasher {
    let mut h = crc32fast::Hasher::new();
    h.update(&raw[..SEG_HDR]);
    h.update(&(section_off as u64).to_be_bytes());
    h
}

// ---------------------------------------------------------------------------
// decoded shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    pub flags: u32,
    pub seq: i64,
    pub first_lsn: i64,
    pub header_crc: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub index: usize,
    pub offset: usize,
    pub tag: u8,
    pub lsn: i64,
    pub body_len: i64,
    pub hdr_crc: u32,
    pub body_crc: u32,
    pub body: Vec<u8>,
}

/// One entry of an `'S'`/`'C'` body.
///
/// `cap` and `len_plus` are `None` for the kinds that do not carry them, and
/// **`len_plus` is RAW**: `Some(0)` is NULL content and `Some(1)` is
/// zero-length content. Decoding it into a length collapses the two, and two
/// readers that both collapse it agree forever.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub tag: u8,
    pub recid: i64,
    pub cap: Option<i64>,
    pub len_plus: Option<i64>,
    pub content: Option<Vec<u8>>,
}

impl Entry {
    pub fn kind(&self) -> &'static str {
        match self.tag {
            T_PREALLOC => "PREALLOC",
            T_RECORD => "RECORD",
            T_DELETE => "DELETE",
            T_APPEND => "APPEND",
            other => panic!("entry tag {other} has no name"),
        }
    }

    pub fn is_record(&self) -> bool {
        self.tag == T_RECORD
    }
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub header: Header,
    pub sections: Vec<Section>,
    /// Bytes after the last complete section — a torn tail, reported rather
    /// than silently accepted so a caller can decide whether it is legal here.
    pub trailing: usize,
}

/// Decodes one segment file, validating both CRCs of every section.
pub fn decode(raw: &[u8], where_: &str) -> Segment {
    assert!(
        raw.len() >= SEG_HDR,
        "{where_}: {} bytes is shorter than a {SEG_HDR}-byte segment header",
        raw.len()
    );
    assert_eq!(&raw[..8], MAGIC, "{where_}: bad segment magic");
    let version = be32(raw, 8);
    assert_eq!(version, FORMAT_VERSION, "{where_}: format version");
    let header_crc = be32(raw, SEG_HDR_CRC_LEN);
    assert_eq!(
        crc32fast::hash(&raw[..SEG_HDR_CRC_LEN]),
        header_crc,
        "{where_}: segment header CRC"
    );
    let header = Header {
        version,
        flags: be32(raw, 12),
        seq: be64(raw, 16),
        first_lsn: be64(raw, 24),
        header_crc,
    };

    let mut sections = Vec::new();
    let mut off = SEG_HDR;
    while off < raw.len() {
        if off + SEC_HDR > raw.len() {
            break; // torn: not even a whole section header left
        }
        let tag = raw[off];
        assert!(
            tag == TAG_SECTION || tag == TAG_IMAGE || tag == TAG_MARK,
            "{where_}: unknown section tag {:?} at offset {off}",
            tag as char
        );
        let lsn = be64(raw, off + 1);
        let body_len = be64(raw, off + 9);
        let hdr_crc = be32(raw, off + 17);
        let mut h = domain(raw, off);
        h.update(&raw[off..off + SEC_HDR_CRC_LEN]);
        assert_eq!(
            h.finalize(),
            hdr_crc,
            "{where_}: section header CRC at offset {off}"
        );
        // Subtract rather than add: `off + SEC_HDR + body_len <= raw.len()`
        // overflows for a body_len near i64::MAX and wraps negative, passing
        // the very check it is. Both operands here are bounded by the file.
        assert!(
            body_len >= 0,
            "{where_}: negative bodyLen {body_len} at {off}"
        );
        if body_len > (raw.len() - off - SEC_HDR) as i64 {
            break; // torn: the body is not all here
        }
        let body = raw[off + SEC_HDR..off + SEC_HDR + body_len as usize].to_vec();
        let body_crc = be32(raw, off + 21);
        let mut hb = domain(raw, off);
        hb.update(&body);
        assert_eq!(
            hb.finalize(),
            body_crc,
            "{where_}: section body CRC at offset {off}"
        );
        if tag == TAG_MARK {
            assert_eq!(
                body_len, MARK_BODY_LEN,
                "{where_}: a 'K' body at {off} is {body_len} bytes, not {MARK_BODY_LEN}"
            );
        }
        sections.push(Section {
            index: sections.len(),
            offset: off,
            tag,
            lsn,
            body_len,
            hdr_crc,
            body_crc,
            body,
        });
        off += SEC_HDR + body_len as usize;
    }
    Segment {
        header,
        trailing: raw.len() - off,
        sections,
    }
}

/// Reads the packed longs of an entry stream. **The high bit marks the LAST
/// byte**, which is the inverse of the usual varint convention and the single
/// likeliest thing for a port to get backwards.
struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
    where_: &'a str,
}

impl<'a> Cursor<'a> {
    fn byte(&mut self) -> u8 {
        assert!(
            self.pos < self.b.len(),
            "{}: entry stream ends mid-value at {}",
            self.where_,
            self.pos
        );
        let v = self.b[self.pos];
        self.pos += 1;
        v
    }

    fn packed(&mut self) -> i64 {
        let mut ret: i64 = 0;
        loop {
            let v = self.byte();
            ret = (ret << 7) | (v & 0x7F) as i64;
            if v & 0x80 != 0 {
                return ret;
            }
        }
    }

    fn take(&mut self, n: usize) -> Vec<u8> {
        assert!(
            self.pos + n <= self.b.len(),
            "{}: entry content of {n} bytes runs past the {}-byte body",
            self.where_,
            self.b.len()
        );
        let out = self.b[self.pos..self.pos + n].to_vec();
        self.pos += n;
        out
    }
}

/// Decodes the ordered entry stream of an `'S'` or `'C'` section.
///
/// `'C'` is decoded exactly like `'S'`: the two tags differ in what recovery
/// does with the section, not in how a body is framed (`StoreWAL.java:850`).
/// Only `'K'` is withheld — its body is a mark, see [`mark`].
pub fn entries(s: &Section, where_: &str) -> Vec<Entry> {
    assert!(
        s.tag == TAG_SECTION || s.tag == TAG_IMAGE,
        "{where_} section {}: tag {:?} carries no entry stream",
        s.index,
        s.tag as char
    );
    let ctx = format!("{where_} section {}", s.index);
    let mut cur = Cursor {
        b: &s.body,
        pos: 0,
        where_: &ctx,
    };
    let mut out = Vec::new();
    while cur.pos < s.body.len() {
        let tag = cur.byte();
        let mut e = Entry {
            tag,
            recid: 0,
            cap: None,
            len_plus: None,
            content: None,
        };
        e.recid = cur.packed();
        match tag {
            T_PREALLOC | T_DELETE => {}
            T_RECORD => {
                e.cap = Some(cur.packed());
                let len_plus = cur.packed();
                e.len_plus = Some(len_plus);
                if len_plus > 0 {
                    e.content = Some(cur.take((len_plus - 1) as usize));
                }
            }
            T_APPEND => panic!(
                "{ctx}: T_APPEND is not decoded here — the C3 body dump has no columns for it \
                 and no fixture exercises it; extend both together"
            ),
            other => panic!("{ctx}: unknown entry tag {other} at {}", cur.pos - 1),
        }
        out.push(e);
    }
    out
}

/// The two fields of a `'K'` mark body, in wire order:
/// `(cleanedThroughSeq, logStartLsn)`.
pub fn mark(s: &Section, where_: &str) -> (i64, i64) {
    assert_eq!(
        s.tag, TAG_MARK,
        "{where_} section {}: not a mark section",
        s.index
    );
    assert_eq!(
        s.body.len() as i64,
        MARK_BODY_LEN,
        "{where_} section {}: mark body length",
        s.index
    );
    (be64(&s.body, 0), be64(&s.body, 8))
}

// ---------------------------------------------------------------------------
// MANIFEST.tsv — the v1/v2 dispatch
// ---------------------------------------------------------------------------

fn check(cond: bool, msg: impl FnOnce() -> String) {
    if !cond {
        panic!("{}", msg());
    }
}

/// Field-count plus emptiness. A TSV row with a blank column parses "fine" and
/// then every consumer of that column sees `""`, so the emptiness half is not
/// decoration.
fn arity(t: &[&str], want: usize, line: &str) {
    check(t.len() == want, || {
        format!(
            "bad {} row: expected {want} fields, got {}: {line}",
            t[0],
            t.len()
        )
    });
    for (i, f) in t.iter().enumerate() {
        check(!f.is_empty(), || {
            format!("bad {} row: field {i} is empty: {line}", t[0])
        });
    }
}

/// A canonical decimal non-negative integer: no sign, no leading zero, no
/// whitespace, in range. `str::parse` accepts `+7` and `007`, both of which
/// would make two manifests that differ textually mean the same thing.
fn nat(s: &str, line: &str) -> u64 {
    check(
        !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()) && (s == "0" || !s.starts_with('0')),
        || format!("not a canonical decimal non-negative integer: {s} in: {line}"),
    );
    s.parse()
        .unwrap_or_else(|_| panic!("integer out of range: {s} in: {line}"))
}

/// A single path component that is safe to join onto a cell directory. The
/// manifest is data, and a `relName` of `../../x` would be executed as written.
fn rel_name(s: &str, line: &str) -> String {
    check(
        !s.is_empty()
            && !s.contains('/')
            && !s.contains('\\')
            && !s.contains('\0')
            && s != "."
            && s != ".."
            && !s.starts_with('-')
            && !Path::new(s).is_absolute(),
        || format!("unsafe relName {s} in: {line}"),
    );
    s.to_string()
}

fn one_of(s: &str, allowed: &[&str], what: &str, line: &str) -> String {
    check(allowed.contains(&s), || {
        format!("unknown {what} {s:?} (want one of {allowed:?}) in: {line}")
    });
    s.to_string()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RecidState {
    Live,
    Null,
    Prealloc,
    Deleted,
}

fn parse_state(s: &str, line: &str) -> RecidState {
    match s {
        "live" => RecidState::Live,
        "null" => RecidState::Null,
        "prealloc" => RecidState::Prealloc,
        "deleted" => RecidState::Deleted,
        other => panic!("unknown recid state {other:?} in manifest line: {line}"),
    }
}

#[derive(Clone, Debug)]
pub struct RecidRow {
    pub fixture: String,
    pub label: String,
    pub recid: u64,
    pub state: RecidState,
    pub payload_id: u64,
    pub len: usize,
}

/// A `recidrange` may not be unbounded: the reader materialises one row per
/// recid, so `0 4294967295` is a manifest that hangs the suite.
const MAX_RANGE_SPAN: u64 = 1 << 20;

fn add_recid(into: &mut Vec<RecidRow>, r: RecidRow, line: &str) {
    for prior in into.iter() {
        check(
            !(prior.fixture == r.fixture && prior.recid == r.recid),
            || {
                format!(
                    "duplicate recid {} in fixture {}: {line}",
                    r.recid, r.fixture
                )
            },
        );
    }
    into.push(r);
}

fn push_range(into: &mut Vec<RecidRow>, fixture: &str, t: &[&str], line: &str) {
    let from = nat(t[3], line);
    let to = nat(t[4], line);
    let state = parse_state(t[5], line);
    let base = nat(t[6], line);
    let len = nat(t[7], line) as usize;
    check(from <= to, || format!("empty recidrange: {line}"));
    check(to - from < MAX_RANGE_SPAN, || {
        format!("recidrange spans {} recids: {line}", to - from + 1)
    });
    for r in from..=to {
        add_recid(
            into,
            RecidRow {
                fixture: fixture.to_string(),
                label: format!("{}[{}]", t[2], r - from),
                recid: r,
                state,
                payload_id: base + (r - from),
                len,
            },
            line,
        );
    }
}

// --- schema v2 ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct V2File {
    pub fixture: String,
    pub rel: String,
    pub raw_len: u64,
    pub raw_sha: String,
    pub gz_sha: String,
}

impl V2File {
    /// v2 blobs are namespaced by fixture (`<fixtureId>.<relName>.gz`) because
    /// a bundle has many files and all three bundles name their first segment
    /// `x.wal.0000000000000001`.
    pub fn blob_name(&self) -> String {
        format!("{}.{}.gz", self.fixture, self.rel)
    }
}

/// `expect <fid> <engine> <mode> <verdict> <opener> <openArg>`.
#[derive(Clone, Debug)]
pub struct V2Expect {
    pub fixture: String,
    pub engine: String,
    pub mode: String,
    pub verdict: String,
    pub opener: String,
    pub open_arg: String,
}

/// `post <fid> <engine> <mode> <relName> <disposition>`.
#[derive(Clone, Debug)]
pub struct V2Post {
    pub fixture: String,
    pub engine: String,
    pub mode: String,
    pub rel: String,
    pub verb: String,
    pub len: Option<u64>,
    pub sha: Option<String>,
}

/// `applies <fid> <engine> <mode>` — a cell this corpus actually contains
/// (contract §2.3, slice C5).
///
/// The corpus's cell set is legitimately partial, so "every fixture × every
/// mode" is the wrong cardinality rule for it; `applies` says which cells
/// exist, and the executor's rule becomes "the cells I ran are exactly the
/// `applies` rows addressed to me".
#[derive(Clone, Debug)]
pub struct V2Applies {
    pub fixture: String,
    pub engine: String,
    pub mode: String,
}

/// `action <fid> <engine> <mode> <verb> <args>` — a post-open executor step.
#[derive(Clone, Debug)]
pub struct V2Action {
    pub fixture: String,
    pub engine: String,
    pub mode: String,
    pub verb: String,
    /// The rendered `k=v,…` spec, kept as the STRING the row carried: it is
    /// compared against `catalogue.render_action_args`'s single rendering in
    /// todo's gate, so re-rendering it here would author a second authority.
    pub arg_spec: String,
}

/// `bytes <fid> <engine> <mode> <relName> <offset> <hex>` — an assertion
/// against the CAPTURED POST bytes, never a pre-open patch (contract §2.3).
#[derive(Clone, Debug)]
pub struct V2Bytes {
    pub fixture: String,
    pub engine: String,
    pub mode: String,
    pub rel: String,
    pub offset: u64,
    pub hex: String,
}

/// `reopen <fid> <engine> <mode> <family>` — after the cell's actions have run
/// and the store has been closed, a SECOND open must fail with this family.
///
/// Stability / second-open grade only (C8f f1). First-open family grading is
/// carried by [`V2Family`], including the mutating R6-audit/rw arms that have
/// no `reopen` row.
#[derive(Clone, Debug)]
pub struct V2Reopen {
    pub fixture: String,
    pub engine: String,
    pub mode: String,
    pub family: String,
}

/// `family <fid> <engine> <mode> <familyName>` — first-open refusal grade for a
/// reject arm (C8f f1 / plan §4.2).
///
/// Fail-closed bijection with reject arms: every reject cell this engine runs
/// owes exactly one row; an accept cell carrying one fails as unconsumed.
/// Mutating R6-audit/rw carries `family` only (no `reopen`).
#[derive(Clone, Debug)]
pub struct V2Family {
    pub fixture: String,
    pub engine: String,
    pub mode: String,
    pub family: String,
}

#[derive(Default, Debug)]
pub struct V2 {
    pub fixture_kinds: BTreeMap<String, String>,
    pub files: Vec<V2File>,
    pub applies: Vec<V2Applies>,
    pub expects: Vec<V2Expect>,
    pub posts: Vec<V2Post>,
    pub actions: Vec<V2Action>,
    pub bytes: Vec<V2Bytes>,
    pub reopens: Vec<V2Reopen>,
    pub families: Vec<V2Family>,
    pub recids: Vec<RecidRow>,
}

impl V2 {
    pub fn files_of(&self, fixture: &str) -> Vec<&V2File> {
        self.files.iter().filter(|f| f.fixture == fixture).collect()
    }

    pub fn actions_of(&self, fixture: &str, engine: &str, mode: &str) -> Vec<&V2Action> {
        self.actions
            .iter()
            .filter(|a| a.fixture == fixture && a.engine == engine && a.mode == mode)
            .collect()
    }

    pub fn bytes_of(&self, fixture: &str, engine: &str, mode: &str) -> Vec<&V2Bytes> {
        self.bytes
            .iter()
            .filter(|b| b.fixture == fixture && b.engine == engine && b.mode == mode)
            .collect()
    }

    pub fn reopens_of(&self, fixture: &str, engine: &str, mode: &str) -> Vec<&V2Reopen> {
        self.reopens
            .iter()
            .filter(|r| r.fixture == fixture && r.engine == engine && r.mode == mode)
            .collect()
    }

    pub fn families_of(&self, fixture: &str, engine: &str, mode: &str) -> Vec<&V2Family> {
        self.families
            .iter()
            .filter(|r| r.fixture == fixture && r.engine == engine && r.mode == mode)
            .collect()
    }

    pub fn recids_of(&self, fixture: &str) -> Vec<&RecidRow> {
        self.recids
            .iter()
            .filter(|r| r.fixture == fixture)
            .collect()
    }

    pub fn posts_of(&self, fixture: &str, engine: &str, mode: &str) -> Vec<&V2Post> {
        self.posts
            .iter()
            .filter(|p| p.fixture == fixture && p.engine == engine && p.mode == mode)
            .collect()
    }

    /// **The expected cell set, derived from the `fixture` rows** — not from
    /// the `expect` rows the executor is about to run.
    ///
    /// A count, or the set of modes actually seen, is a projection of the
    /// already-truncated input: the C3j review deleted one `expect` row and
    /// the java suite stayed green, because another fixture still supplied
    /// that mode. Every declared fixture owes one cell per engine per mode, so
    /// a missing `expect` row now contradicts the `fixture` row still sitting
    /// next to it.
    pub fn declared_fixtures(&self) -> BTreeSet<String> {
        self.fixture_kinds.keys().cloned().collect()
    }
}

/// A parsed schema-v2 manifest. Version is always 2 after C7r.
#[derive(Debug)]
pub struct Loaded {
    pub version: u32,
    pub v2: V2,
}

impl Loaded {
    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn v2(&self) -> &V2 {
        &self.v2
    }
}

/// Parses a schema-v2 manifest. Schema version 1 is retired (Stage C, C7r).
///
/// The version line remains a hard gate: v1 and v2 `expect` rows shared arity
/// with different columns, so guessing the schema from row shape would misread
/// fields without a single arity check firing. An unknown or retired version is
/// refused rather than assumed.
pub fn parse(text: &str) -> Loaded {
    let mut lines = text.lines();
    let head = loop {
        match lines.next() {
            None => panic!("MANIFEST.tsv has no version line"),
            Some(l) if l.is_empty() || l.starts_with('#') => continue,
            Some(l) => break l,
        }
    };
    let t: Vec<&str> = head.split('\t').collect();
    check(t.len() == 2 && t[0] == "version", || {
        format!("the first data line must be `version<TAB><n>`, not: {head}")
    });
    let rest: Vec<&str> = lines.collect();
    match t[1] {
        "1" => panic!(
            "manifest schema version 1 is retired (Stage C, C7r) — this reader speaks only \
             schema 2; the dual v1/v2 dispatch is gone"
        ),
        "2" => Loaded {
            version: 2,
            v2: parse_v2(&rest),
        },
        other => panic!(
            "unsupported manifest schema version {other} — this reader speaks only schema 2, and \
             refuses rather than guessing at the columns"
        ),
    }
}

fn parse_v2(lines: &[&str]) -> V2 {
    let mut m = V2::default();
    // Contract §2, amendment 3: a fixture whose generatorEngine is `derived`
    // MUST have exactly one `derived` row, and no other fixture may have one.
    let mut wants_derived: BTreeSet<String> = BTreeSet::new();
    let mut has_derived: BTreeSet<String> = BTreeSet::new();
    for line in lines {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "version" => panic!("a second version row: {line}"),
            "fixture" => {
                arity(&t, 5, line);
                one_of(t[2], &V2_KINDS, "fixture kind", line);
                if one_of(t[3], &V2_GENERATORS, "generatorEngine", line) == "derived" {
                    wants_derived.insert(t[1].to_string());
                }
                check(
                    m.fixture_kinds
                        .insert(t[1].to_string(), t[2].to_string())
                        .is_none(),
                    || format!("duplicate fixture row for {}: {line}", t[1]),
                );
            }
            "derived" => {
                arity(&t, 5, line);
                // `derived <fid> <src> <deriverVersion> <recipe>` — PROVENANCE
                // for a fixture that also has its own `fixture` row, not a
                // second way to declare one. Reading t[2] as a kind (the shape
                // of the `fixture` row above) would file a fixture id in the
                // kind column and never be noticed, because nothing here
                // consumes kinds.
                nat(t[3], line);
                check(has_derived.insert(t[1].to_string()), || {
                    format!("two derived rows for {}: {line}", t[1])
                });
            }
            "file" => {
                arity(&t, 6, line);
                let f = V2File {
                    fixture: t[1].to_string(),
                    rel: rel_name(t[2], line),
                    raw_len: nat(t[3], line),
                    raw_sha: t[4].to_string(),
                    gz_sha: t[5].to_string(),
                };
                for prior in &m.files {
                    check(!(prior.fixture == f.fixture && prior.rel == f.rel), || {
                        format!("duplicate file row for {}/{}: {line}", f.fixture, f.rel)
                    });
                }
                m.files.push(f);
            }
            "applies" => {
                arity(&t, 4, line);
                let ap = V2Applies {
                    fixture: t[1].to_string(),
                    engine: one_of(t[2], &ENGINES, "engine", line),
                    mode: one_of(t[3], &MODES, "mode", line),
                };
                for prior in &m.applies {
                    check(
                        !cell_eq(
                            (&prior.fixture, &prior.engine, &prior.mode),
                            (&ap.fixture, &ap.engine, &ap.mode),
                        ),
                        || format!("duplicate applies row: {line}"),
                    );
                }
                m.applies.push(ap);
            }
            "expect" => {
                arity(&t, 7, line);
                let e = V2Expect {
                    fixture: t[1].to_string(),
                    engine: one_of(t[2], &ENGINES, "engine", line),
                    mode: one_of(t[3], &MODES, "mode", line),
                    verdict: one_of(t[4], &VERDICTS, "verdict", line),
                    opener: one_of(t[5], &V2_OPENERS, "opener", line),
                    open_arg: rel_name(t[6], line),
                };
                for prior in &m.expects {
                    check(
                        !(prior.fixture == e.fixture
                            && prior.engine == e.engine
                            && prior.mode == e.mode),
                        || {
                            format!(
                                "duplicate expect row for {}/{}/{}: {line}",
                                e.fixture, e.engine, e.mode
                            )
                        },
                    );
                }
                m.expects.push(e);
            }
            "post" => {
                arity(&t, 6, line);
                let (verb, len, sha) = parse_disposition(t[5], line);
                let p = V2Post {
                    fixture: t[1].to_string(),
                    engine: one_of(t[2], &ENGINES, "engine", line),
                    mode: one_of(t[3], &MODES, "mode", line),
                    rel: rel_name(t[4], line),
                    verb,
                    len,
                    sha,
                };
                for prior in &m.posts {
                    check(
                        !(prior.fixture == p.fixture
                            && prior.engine == p.engine
                            && prior.mode == p.mode
                            && prior.rel == p.rel),
                        || {
                            format!(
                                "duplicate post row for {}/{}/{}/{}: {line}",
                                p.fixture, p.engine, p.mode, p.rel
                            )
                        },
                    );
                }
                m.posts.push(p);
            }
            "recid" => {
                arity(&t, 7, line);
                add_recid(
                    &mut m.recids,
                    RecidRow {
                        fixture: t[1].to_string(),
                        label: t[2].to_string(),
                        recid: nat(t[3], line),
                        state: parse_state(t[4], line),
                        payload_id: nat(t[5], line),
                        len: nat(t[6], line) as usize,
                    },
                    line,
                );
            }
            "recidrange" => {
                arity(&t, 8, line);
                let fixture = t[1].to_string();
                push_range(&mut m.recids, &fixture, &t, line);
            }
            "edit" => {
                arity(&t, 6, line);
            }
            "action" => {
                arity(&t, 6, line);
                let a = V2Action {
                    fixture: t[1].to_string(),
                    engine: one_of(t[2], &ENGINES, "engine", line),
                    mode: one_of(t[3], &MODES, "mode", line),
                    // The VERB is not vocabulary-checked here, for the reason
                    // the `reopen` family is not: `catalogue.ACTION_VERBS` is
                    // the authority, this engine implements a subset, and a
                    // parser list would accept a verb the executor then cannot
                    // run while going stale on its own. `run_action` refuses an
                    // unimplemented verb, which is both stricter and the
                    // refusal that matters.
                    verb: t[4].to_string(),
                    arg_spec: action_args(t[5], line),
                };
                // One row per cell per VERB, not per cell: `catalogue.actions`
                // holds a list, so a second verb on one cell is a legal future
                // shape and refusing it would refuse the corpus, not a defect.
                for prior in &m.actions {
                    check(
                        !(cell_eq(
                            (&prior.fixture, &prior.engine, &prior.mode),
                            (&a.fixture, &a.engine, &a.mode),
                        ) && prior.verb == a.verb),
                        || format!("duplicate action row: {line}"),
                    );
                }
                m.actions.push(a);
            }
            "reopen" => {
                arity(&t, 5, line);
                let r = V2Reopen {
                    fixture: t[1].to_string(),
                    engine: one_of(t[2], &ENGINES, "engine", line),
                    mode: one_of(t[3], &MODES, "mode", line),
                    // Not vocabulary-checked: `catalogue.FAMILIES` has eighteen
                    // members and this engine has a predicate for a handful, so
                    // a list here would accept a family `assert_family` then
                    // refuses — two lists, one of which goes stale.
                    family: t[4].to_string(),
                };
                for prior in &m.reopens {
                    check(
                        !cell_eq(
                            (&prior.fixture, &prior.engine, &prior.mode),
                            (&r.fixture, &r.engine, &r.mode),
                        ),
                        || format!("duplicate reopen row: {line}"),
                    );
                }
                m.reopens.push(r);
            }
            "family" => {
                // C8f f1 / plan §4.2 — same arity and key shape as `reopen`,
                // but oracle-profile first-open grade (not the second open).
                arity(&t, 5, line);
                let r = V2Family {
                    fixture: t[1].to_string(),
                    engine: one_of(t[2], &ENGINES, "engine", line),
                    mode: one_of(t[3], &MODES, "mode", line),
                    // Same non-vocabulary reason as `reopen`: predicates live
                    // in `assert_family`, not a second list here.
                    family: t[4].to_string(),
                };
                for prior in &m.families {
                    check(
                        !cell_eq(
                            (&prior.fixture, &prior.engine, &prior.mode),
                            (&r.fixture, &r.engine, &r.mode),
                        ),
                        || format!("duplicate family row: {line}"),
                    );
                }
                m.families.push(r);
            }
            "bytes" => {
                arity(&t, 7, line);
                let b = V2Bytes {
                    fixture: t[1].to_string(),
                    engine: one_of(t[2], &ENGINES, "engine", line),
                    mode: one_of(t[3], &MODES, "mode", line),
                    rel: rel_name(t[4], line),
                    offset: nat(t[5], line),
                    hex: hex_blob(t[6], line),
                };
                // Keyed by cell AND (file, offset): a cell may assert several
                // ranges, and two rows for the same range are a contradiction.
                for prior in &m.bytes {
                    check(
                        !(cell_eq(
                            (&prior.fixture, &prior.engine, &prior.mode),
                            (&b.fixture, &b.engine, &b.mode),
                        ) && prior.rel == b.rel
                            && prior.offset == b.offset),
                        || format!("duplicate bytes row: {line}"),
                    );
                }
                m.bytes.push(b);
            }
            other => panic!("unknown v2 manifest row type {other:?}: {line}"),
        }
    }
    check(!m.files.is_empty(), || {
        "a v2 manifest with no file rows".to_string()
    });
    check(wants_derived == has_derived, || {
        format!(
            "the fixtures declaring generatorEngine=derived are {wants_derived:?} but the \
             fixtures carrying a derived row are {has_derived:?}"
        )
    });
    referential_integrity(&m.fixture_kinds, &referenced_v2(&m));
    m
}

/// Every fixture id a row REFERS to must be DECLARED by exactly one `fixture`
/// row, and every declared fixture must be referred to.
///
/// Without this, one coordinated deletion defeats the exact-cell-set rule that
/// §6.1 exists to enforce: drop a `fixture` row together with this engine's two
/// `expect` rows and both halves of the executor see a consistently smaller
/// world — `want` shrinks by the same fixture that `ran` lost. The `file` and
/// `recid` rows stay behind, the golden comparisons still decode them, and the
/// resource inventory is unchanged. The C3r review found that one; the fix is
/// to make the declaration load-bearing for rows that are not being deleted.
fn referential_integrity(declared: &BTreeMap<String, String>, referenced: &BTreeSet<String>) {
    let known: BTreeSet<String> = declared.keys().cloned().collect();
    let undeclared: Vec<&String> = referenced.difference(&known).collect();
    check(undeclared.is_empty(), || {
        format!("rows refer to fixtures with no `fixture` row: {undeclared:?}")
    });
    let unused: Vec<&String> = known.difference(referenced).collect();
    check(unused.is_empty(), || {
        format!("fixtures are declared but no row refers to them: {unused:?}")
    });
}

fn referenced_v2(m: &V2) -> BTreeSet<String> {
    let mut r = BTreeSet::new();
    r.extend(m.files.iter().map(|x| x.fixture.clone()));
    r.extend(m.applies.iter().map(|x| x.fixture.clone()));
    r.extend(m.expects.iter().map(|x| x.fixture.clone()));
    r.extend(m.posts.iter().map(|x| x.fixture.clone()));
    r.extend(m.actions.iter().map(|x| x.fixture.clone()));
    r.extend(m.bytes.iter().map(|x| x.fixture.clone()));
    r.extend(m.reopens.iter().map(|x| x.fixture.clone()));
    r.extend(m.families.iter().map(|x| x.fixture.clone()));
    r.extend(m.recids.iter().map(|x| x.fixture.clone()));
    r
}

/// The five cell-keyed v2 row types share one identity — `(fixture, engine,
/// mode)` — and it is written once so five copies cannot drift.
fn cell_eq(a: (&str, &str, &str), b: (&str, &str, &str)) -> bool {
    a == b
}

/// `catalogue.ARG_VALUE_CHARS`, transcribed. TAB, `,` and `=` are absent by
/// construction: they are the three separators the row is re-split on.
const ARG_VALUE_CHARS: &str =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._@:/+-";

/// An `action` row's argument spec, matching `catalogue.render_action_args`:
/// `k=v` pairs joined by `,`, **keys in sorted order**, each key
/// `[a-z][a-z0-9_]*`, each value nonempty and drawn from [`ARG_VALUE_CHARS`].
///
/// The sort order is CHECKED rather than normalised. The spec travels to
/// `run_action` as a string and is compared, in todo's gate, against one
/// rendering authority; a reader that accepted any order would accept a
/// manifest python refuses, and the two roots would then disagree about what
/// the same cell says.
fn action_args(s: &str, line: &str) -> String {
    let mut prev: Option<&str> = None;
    for pair in s.split(',') {
        let eq = pair.find('=');
        check(
            eq.is_some_and(|i| i > 0) && pair.matches('=').count() == 1,
            || format!("action argument {pair:?} is not one k=v pair in: {line}"),
        );
        let (k, v) = pair.split_at(eq.unwrap());
        let v = &v[1..];
        check(
            k.starts_with(|c: char| c.is_ascii_lowercase())
                && k.bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_'),
            || format!("action argument key {k:?} is not [a-z][a-z0-9_]* in: {line}"),
        );
        check(prev.is_none_or(|p| p < k), || {
            format!("action argument keys must be sorted and distinct: {k:?} follows {prev:?} in: {line}")
        });
        prev = Some(k);
        check(
            !v.is_empty() && v.chars().all(|c| ARG_VALUE_CHARS.contains(c)),
            || {
                format!(
                    "action argument {k}={v}: the value must be nonempty and drawn from the \
                     pinned character class in: {line}"
                )
            },
        );
    }
    s.to_string()
}

/// A `bytes` row's asserted value: a nonempty, even-length run of lowercase hex.
fn hex_blob(s: &str, line: &str) -> String {
    check(
        !s.is_empty()
            && s.len().is_multiple_of(2)
            && s.bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        || format!("not a nonempty even-length lowercase hex blob: {s:?} in: {line}"),
    );
    s.to_string()
}

/// `unchanged` | `deleted` | `truncated:<len>:<sha>` | `created:<len>:<sha>` |
/// `modified:<len>:<sha>` — the sized verbs carry exactly two arguments.
fn parse_disposition(s: &str, line: &str) -> (String, Option<u64>, Option<String>) {
    let parts: Vec<&str> = s.split(':').collect();
    let want_args = match parts[0] {
        "unchanged" | "deleted" => 0usize,
        "truncated" | "created" | "modified" => 2usize,
        other => panic!("unknown post disposition verb {other:?} in: {line}"),
    };
    check(parts.len() == want_args + 1, || {
        format!(
            "post disposition {} takes {want_args} argument(s), got {}: {line}",
            parts[0],
            parts.len() - 1
        )
    });
    if want_args == 0 {
        (parts[0].to_string(), None, None)
    } else {
        let len = nat(parts[1], line);
        let sha = parts[2].to_string();
        check(
            sha.len() == 64
                && sha
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            || format!("not a lowercase sha256 hex digest: {sha} in: {line}"),
        );
        (parts[0].to_string(), Some(len), Some(sha))
    }
}

// ---------------------------------------------------------------------------
// loading the distributed sample
// ---------------------------------------------------------------------------

/// A schema-v2 sample root: the parsed manifest plus every file's verified raw
/// bytes, keyed `(fixtureId, relName)`.
pub struct SampleV2 {
    pub manifest: V2,
    pub raw: BTreeMap<(String, String), Vec<u8>>,
}

impl SampleV2 {
    /// Files in the canonical dump order — `(fixtureId, relName)`, which is
    /// what `GOLDEN-BODY.tsv` and `GOLDEN-DECODE.tsv` are sorted by.
    pub fn ordered(&self) -> Vec<&V2File> {
        let mut v: Vec<&V2File> = self.manifest.files.iter().collect();
        v.sort_by(|a, b| (&a.fixture, &a.rel).cmp(&(&b.fixture, &b.rel)));
        v
    }

    pub fn bytes_of(&self, f: &V2File) -> &[u8] {
        &self.raw[&(f.fixture.clone(), f.rel.clone())]
    }
}

pub fn v2_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/xfixtures-v2")
}

/// The schema-v2 **preflight corpus** root — a byte-identical copy of the
/// `root`-marked files of `todo/store-cross/preflight-v2/`, pinned by
/// `freeze_v2.PREFLIGHT_DIST_SEALS["rust"]` (slice C5r).
pub fn v2_corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/xfixtures-v2-corpus")
}

pub fn read_root_file(root: &Path, name: &str) -> Vec<u8> {
    let p = root.join(name);
    std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "{} is missing or unreadable ({e}); run the xfixtures sync step",
            p.display()
        )
    })
}

pub fn read_root_text(root: &Path, name: &str) -> String {
    String::from_utf8(read_root_file(root, name)).expect("fixture tsv must be UTF-8")
}

/// Loads the v2 sample, verifying every blob's gz sha, raw length and raw sha
/// BEFORE anything decodes or opens it. A dump taken from bytes that were
/// never checked against their pins describes whatever happened to be on disk.
pub fn load_sample_v2(root: &Path) -> SampleV2 {
    load_sample_v2_text(root, &read_root_text(root, "MANIFEST.tsv"))
}

/// Pre-f2 bridge: inject catalogue-derived `family` rows for every rust reject
/// arm in `text`.
///
/// C8f f1 lands the consumer before f2 freezes `family` into the corpus
/// MANIFEST. Full-suite paths call this so first-open grading is measurable;
/// bare frozen loads stay fail-closed (no family row → red).
///
/// **Self-expiring:** refuses any input that already carries a rust `family`
/// row (including a partial/mixed set). That stops the hand-pinned table from
/// overwriting a sealed freeze after f2 and reporting green on wrong/missing
/// oracle rows. At f2: drop every call site of this inject and load the raw
/// frozen MANIFEST.
pub fn inject_rust_family_rows(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut reject_arms: Vec<(String, String)> = Vec::new();
    for line in text.split('\n') {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 3 && cols[0] == "family" && cols[2] == "rust" {
            panic!(
                "inject_rust_family_rows refuses input that already has rust family rows \
                 (found `{line}`); drop the pre-f2 bridge after freeze — it must not \
                 overwrite sealed family oracle rows"
            );
        }
        if cols.len() >= 5 && cols[0] == "expect" && cols[2] == "rust" && cols[4] == "reject" {
            reject_arms.push((cols[1].to_string(), cols[3].to_string()));
        }
        out.push(line.to_string());
    }
    assert!(
        !reject_arms.is_empty(),
        "inject_rust_family_rows found no rust reject arms to equip"
    );
    for (fid, mode) in reject_arms {
        let fam = rust_reject_family_for_inject(&fid)
            .unwrap_or_else(|| panic!("no catalogue family pinned for rust reject fixture {fid}"));
        out.push(format!("family\t{fid}\trust\t{mode}\t{fam}"));
    }
    // Always end with a newline so a caller that does `format!("{injected}row\n")`
    // cannot glue the new row onto the last family line.
    let mut s = out.join("\n");
    if !s.ends_with('\n') {
        s.push('\n');
    }
    s
}

/// Loads the corpus root with [`inject_rust_family_rows`] applied (pre-f2).
///
/// Panics once the frozen root already carries rust `family` rows — that is
/// the signal to delete this loader and use [`load_sample_v2`] instead.
pub fn load_corpus_v2_with_family_rows(root: &Path) -> SampleV2 {
    let text = read_root_text(root, "MANIFEST.tsv");
    let injected = inject_rust_family_rows(&text);
    load_sample_v2_text(root, &injected)
}

/// `catalogue.cell.family_for("rust")` for every reject fixture the frozen
/// corpus still addresses to rust. Kept in lockstep with T1's fill until f2.
fn rust_reject_family_for_inject(fixture: &str) -> Option<&'static str> {
    Some(match fixture {
        "reject-wal3-n6-barewal" => "N6",
        "reject-wal3-d1-barebase" | "reject-wal3-d1-ckpt" => "D1",
        "reject-wal3-h5-version" => "H5",
        "reject-wal3-h6-flags" => "H6",
        "reject-wal3-h7-seq" => "H7",
        "reject-wal3-h9-firstlsn" => "H9",
        "reject-wal3-k4-through" => "K4",
        "reject-wal3-k-through0" | "reject-wal3-k-logstart0" | "reject-wal3-k-logstart-hi" => {
            "S8/K-bounds"
        }
        "reject-wal3-s2-lsn-regress" => "S2",
        "reject-wal3-s9-gap" => "S9",
        "reject-wal3-s4-midlog-crc" => "S4/mid-log",
        "reject-wal3-r4-floor" => "R4-floor",
        "reject-wal3-r4-chain" => "R4-chain",
        "reject-wal3-r4-self" => "R4-self",
        "reject-wal3-segment-at-direct" => "direct-magic",
        "mut-wal3-mark-then-refusal" => "R6-audit",
        "div-wal3-lsn-exhausted" => "StoreFull",
        "div-wal3-entry-recid0" | "div-wal3-packlong-overlong" => "DataCorruption",
        _ => return None,
    })
}

/// [`load_sample_v2`] over manifest text supplied by the caller, so a DOCTORED
/// manifest runs against the root's real bytes through the production path.
///
/// Doctoring the text rather than the parsed structure is deliberate: every
/// case then exercises the shipped parser, and a case that meant to add a row
/// the grammar refuses fails loudly instead of constructing something the
/// reader would never have accepted.
pub fn load_sample_v2_text(root: &Path, text: &str) -> SampleV2 {
    let loaded = parse(text);
    let manifest = loaded.v2;
    let mut raw = BTreeMap::new();
    for f in &manifest.files {
        let gz = read_root_file(root, &f.blob_name());
        assert_eq!(
            sha256_hex(&gz),
            f.gz_sha,
            "gzSha256 mismatch for {}",
            f.blob_name()
        );
        let bytes = gunzip(&gz, &f.blob_name());
        assert_eq!(
            bytes.len() as u64,
            f.raw_len,
            "rawLen mismatch for {}",
            f.blob_name()
        );
        assert_eq!(
            sha256_hex(&bytes),
            f.raw_sha,
            "rawSha256 mismatch for {}",
            f.blob_name()
        );
        raw.insert((f.fixture.clone(), f.rel.clone()), bytes);
    }
    SampleV2 { manifest, raw }
}

// ---------------------------------------------------------------------------
// the two §11.2 comparisons, rendered as rows
// ---------------------------------------------------------------------------

/// `GOLDEN-DECODE.tsv`'s rows, re-derived by THIS reader.
///
/// This is the half `GOLDEN.tsv` cannot do: a raw sha attests which bytes were
/// read and says nothing about the parse. Note in particular that the section
/// COUNT is not CRC-protected — both section CRCs bind a section's own bytes to
/// its offset, so a reader that stops one section early still validates every
/// section it did read. Only this comparison reaches that.
pub fn render_framing(sample: &SampleV2) -> Vec<String> {
    let mut out = Vec::new();
    for f in sample.ordered() {
        let where_ = format!("{}/{}", f.fixture, f.rel);
        let seg = decode(sample.bytes_of(f), &where_);
        assert_eq!(
            seg.trailing, 0,
            "{where_}: {} bytes follow the last section",
            seg.trailing
        );
        out.push(format!(
            "hdr\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            f.fixture,
            f.rel,
            seg.header.version,
            seg.header.flags,
            seg.header.seq,
            seg.header.first_lsn,
            hex32(seg.header.header_crc)
        ));
        for s in &seg.sections {
            out.push(format!(
                "sec\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                f.fixture,
                f.rel,
                s.index,
                s.offset,
                s.tag as char,
                s.lsn,
                s.body_len,
                hex32(s.hdr_crc),
                hex32(s.body_crc)
            ));
        }
    }
    out
}

/// `GOLDEN-BODY.tsv`'s rows, re-derived by THIS reader.
///
/// Java authored that file with the frozen reader; this is the engine-against-
/// engine half of contract §11.2, and Java is authoritative by construction.
pub fn render_body(sample: &SampleV2) -> Vec<String> {
    let mut out = Vec::new();
    let mut bundle: Option<String> = None;
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for f in sample.ordered() {
        if bundle.as_deref() != Some(f.fixture.as_str()) {
            if let Some(b) = &bundle {
                check_recids_against_manifest(&sample.manifest, b, &seen);
            }
            bundle = Some(f.fixture.clone());
            seen = BTreeSet::new();
        }
        let where_ = format!("{}/{}", f.fixture, f.rel);
        let seg = decode(sample.bytes_of(f), &where_);
        for s in &seg.sections {
            if s.tag == TAG_MARK {
                let (through, log_start) = mark(s, &where_);
                check_mark(
                    (through, log_start),
                    seg.header.seq,
                    s.lsn,
                    &format!("{where_} section {}", s.index),
                );
                out.push(format!(
                    "sec\t{}\t{}\t{}\t{}\t-",
                    f.fixture, f.rel, s.index, s.tag as char
                ));
                out.push(format!(
                    "mark\t{}\t{}\t{}\t{through}\t{log_start}",
                    f.fixture, f.rel, s.index
                ));
                continue;
            }
            let es = entries(s, &where_);
            out.push(format!(
                "sec\t{}\t{}\t{}\t{}\t{}",
                f.fixture,
                f.rel,
                s.index,
                s.tag as char,
                es.len()
            ));
            for (i, e) in es.iter().enumerate() {
                assert!(
                    e.recid > 0,
                    "{where_} section {}: recid {}",
                    s.index,
                    e.recid
                );
                seen.insert(e.recid as u64);
                out.push(format!(
                    "ent\t{}\t{}\t{}\t{i}\t{}\t{}\t{}\t{}\t{}",
                    f.fixture,
                    f.rel,
                    s.index,
                    e.kind(),
                    e.recid,
                    e.cap.map_or("-".to_string(), |c| c.to_string()),
                    e.len_plus.map_or("-".to_string(), |l| l.to_string()),
                    content_sha(e, &format!("{where_} section {} entry {i}", s.index))
                ));
            }
        }
    }
    let bundle = bundle.expect("the sample has no file rows");
    check_recids_against_manifest(&sample.manifest, &bundle, &seen);
    out
}

/// The content column, and the decode's own self-check.
///
/// Three independent things are asserted before a sha is emitted: the content
/// length agrees with `lenPlus`, the capacity satisfies the engine's
/// `cap_valid` rule ([`check_cap`]), and the bytes lie in the payload language
/// ([`check_payload`]). None of the three is a corpus-membership proof on its
/// own; together they refuse the streams a mis-framed decode actually produces.
fn content_sha(e: &Entry, where_: &str) -> String {
    if !e.is_record() || e.len_plus == Some(0) {
        assert!(
            e.content.is_none(),
            "{where_}: a non-record or NULL entry carries content"
        );
        if e.is_record() {
            assert_eq!(e.cap, Some(0), "{where_}: a NULL record's cap must be 0");
        }
        return "-".to_string();
    }
    let c = e.content.as_ref().expect("a sized record carries content");
    assert_eq!(
        c.len() as i64,
        e.len_plus.unwrap() - 1,
        "{where_}: content length disagrees with lenPlus"
    );
    check_cap(e.cap.unwrap(), c.len(), where_);
    check_payload(c, where_);
    sha256_hex(c)
}

/// The witness that these content bytes lie in the payload LANGUAGE.
///
/// `payload(id, len)[i] == (i*131 + id) & 0xff` is invertible from its first
/// byte, so rebuilding it from the recovered id and comparing catches a run
/// that is not a payload at all — which is what a decoder that read the
/// packed-long continuation bit the wrong way round produces. It is **not** a
/// check that this bundle issued this payload: it consults no fixture history,
/// and since `payload` is an arithmetic progression, every suffix of a payload
/// is another payload. `lenPlus` and the golden sha column cover that gap.
/// Zero-length content carries no id and is vacuously fine.
pub fn check_payload(c: &[u8], where_: &str) {
    if c.is_empty() {
        return;
    }
    let id = c[0] as u64;
    assert_eq!(
        c,
        payload(id, c.len()),
        "{where_}: the {} content bytes are not payload({id}, {}) — this entry stream was \
         not framed the way the writer wrote it",
        c.len(),
        c.len()
    );
}

/// The independent witness for the emitted `cap` column.
///
/// Nothing else in the slice observes `cap`: replay consumes it and exposes
/// only the resulting record, and the golden comparison grades the column
/// against a file another engine wrote. An emitter that consumed the varint
/// correctly and printed a fabricated number would pass both. This is
/// `StoreWAL.cap_valid`'s rule restated over what the dump can see — a plain
/// capacity is 16-aligned and leaves room for the 4-byte header; 0 means
/// oversize content stored linked.
pub fn check_cap(cap: i64, len: usize, where_: &str) {
    let need = 4 + len as i64;
    if cap == 0 {
        // `cap == 0` is how the writer encodes OVERSIZE content, stored linked.
        // Accepting it unconditionally — which this did until the C3r review —
        // blesses a zero capacity on content that fits a plain record, which is
        // exactly the value recovery refuses.
        assert!(
            need > MAX_CAPACITY,
            "{where_}: cap 0 means content stored linked because it is oversize, but {len} \
             content bytes need only {need} and the plain-record ceiling is {MAX_CAPACITY}"
        );
        return;
    }
    assert!(
        cap >= need && cap <= MAX_CAPACITY && cap & 15 == 0,
        "{where_}: cap {cap} is not a valid capacity for {len} content bytes (must be \
         16-aligned, at least {need}, and at most {MAX_CAPACITY})"
    );
}

/// The independent witness for the two `'K'` mark longs, which are otherwise
/// indistinguishable once decoded — both are longs in one 16-byte body, so a
/// decoder returning them in the other order emits a self-consistent file.
/// These are the engine's own S8/K4 rules; the sample's mark is
/// `(through=2, logStart=9)` in segment 4, so a swap makes `through` 9 and
/// trips K4 at once.
pub fn check_mark(mark: (i64, i64), seg_seq: i64, lsn: i64, where_: &str) {
    let (through, log_start) = mark;
    assert!(through > 0, "{where_}: cleanedThroughSeq is {through}");
    assert!(
        through < seg_seq,
        "{where_}: a mark in segment {seg_seq} authorizes removing segment {through}, \
         including itself (K4)"
    );
    assert!(
        log_start > 0 && log_start <= lsn,
        "{where_}: logStartLsn {log_start} is not an LSN at or below the mark's own {lsn} (S8)"
    );
}

/// Cross-checks the recids the entry stream mentions against the ones the
/// manifest names. The manifest's rows were folded out of these same bytes by
/// an independent (python) reader, so the two agree only if both unpacked the
/// same varints.
///
/// **The relation is ONE-WAY, and that is not laxness.** Plan §5 forbids
/// asserting that a log contains only the recids the manifest names — a
/// rolled-back put need only be invisible through the API, and
/// `wal3-java-tail` already carries recids beyond the ones §5.2 describes. Set
/// equality would quietly assert the forbidden direction and pass only because
/// these three bundles happen to have equal sets.
pub fn check_recids_against_manifest(m: &V2, fixture: &str, seen: &BTreeSet<u64>) {
    let rows = m.recids_of(fixture);
    assert!(
        !rows.is_empty(),
        "{fixture}: no recid rows to cross-check against"
    );
    let missing: BTreeSet<u64> = rows
        .iter()
        .map(|r| r.recid)
        .filter(|r| !seen.contains(r))
        .collect();
    assert!(
        missing.is_empty(),
        "{fixture}: the manifest names recids {missing:?} that the decoded entry stream never \
         mentions"
    );
}

// ---------------------------------------------------------------------------
// running cells
// ---------------------------------------------------------------------------

/// Raw-bytes serializer: record content == logical value, so gets compare
/// directly against the contract payloads.
pub struct RawSer;

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

pub const R: RawSer = RawSer;

// ---------------------------------------------------------------------------
// the `action` verbs
// ---------------------------------------------------------------------------

/// Runs one catalogue `action` against an OPEN store — this engine's side of
/// contract §2.3's post-open oracle step.
///
/// **Every argument the catalogue pins is required and is checked.** An unknown
/// verb, a missing argument, an unrecognised argument, or an argument whose
/// value this engine does not implement is a hard failure. Not defensiveness:
/// the caller passes the catalogue's arguments through verbatim, so a catalogue
/// edit this function cannot honour must STOP the run rather than silently
/// execute the old behaviour and author a post state for it. Contract §2.3 says
/// so in one line — *"an executor MUST refuse an action row whose verb it does
/// not implement, and MUST refuse an argument it does not implement. Skipping
/// it is forbidden: an oracle silently not run is a green cell that checked
/// nothing."*
///
/// **Every argument means one INDEPENDENT branch, and each owes its own input.**
/// The round-1 review found the cost of treating one doctored case as covering
/// several: `op` and `serializer` are two separate value checks, only `op` had a
/// doctored input, and deleting the `serializer` assertion left the entire gate
/// green. Every branch below now has a production-path case in
/// `the_action_row_is_executed` and a named mutant beside it.
///
/// Returns the one-line machine-readable description java's `Wal3Actions.run`
/// returns, in the same shape, so a future cross-engine comparison of what an
/// action DID has something to compare.
///
/// **`recid_label` is required, present-checked, and its VALUE is unobserved.**
/// It reaches only that returned description, which every caller here discards,
/// and no corpus cell varies it. Java is in the same position. Recorded rather
/// than dressed up: a required argument nothing reads is a branch a future
/// reviewer should not have to rediscover.
pub fn run_action(s: &StoreWAL, verb: &str, arg_spec: &str) -> String {
    assert_eq!(verb, "commit_one_record", "unknown action verb: {verb}");
    let known = [
        "op",
        "payload_id",
        "payload_len",
        "recid_label",
        "serializer",
    ];
    let mut args: BTreeMap<&str, &str> = BTreeMap::new();
    for pair in arg_spec.split(',') {
        let (k, v) = pair
            .split_once('=')
            .unwrap_or_else(|| panic!("action argument is not k=v: {pair}"));
        assert!(args.insert(k, v).is_none(), "action argument repeated: {k}");
    }
    for k in args.keys() {
        assert!(
            known.contains(k),
            "unknown argument {k} for {verb}; it takes {known:?}"
        );
    }
    let need = |k: &str| -> &str {
        args.get(k).copied().unwrap_or_else(|| {
            panic!(
                "action argument {k} is required; got {:?}, the verb takes {known:?}",
                args.keys().collect::<Vec<_>>()
            )
        })
    };
    let op = need("op");
    let label = need("recid_label");
    let payload_id: u64 = need("payload_id")
        .parse()
        .unwrap_or_else(|e| panic!("payload_id: {e}"));
    let payload_len: usize = need("payload_len")
        .parse()
        .unwrap_or_else(|e| panic!("payload_len: {e}"));
    let ser = need("serializer");
    assert_eq!(op, "put", "commit_one_record: unimplemented op {op}");
    assert_eq!(
        ser, "raw",
        "commit_one_record: unimplemented serializer {ser}"
    );

    let recid = s
        .put(&payload(payload_id, payload_len), &R)
        .unwrap_or_else(|e| panic!("commit_one_record: put failed: {e}"));
    s.commit()
        .unwrap_or_else(|e| panic!("commit_one_record: commit failed: {e}"));
    format!(
        "RESULT action={verb} label={label} recid={recid} payloadId={payload_id} \
         payloadLen={payload_len}"
    )
}

fn nz(recid: u64) -> Recid {
    NonZeroU64::new(recid).expect("manifest recid must be nonzero")
}

/// The reader contract: `verify()`, every named recid in its declared state,
/// and `get_all_recids()` EXACTLY equal to the live+null set. Prealloc and
/// deleted recids are excluded from that set by construction, which the
/// equality (not a containment) is what enforces.
pub fn assert_reader_contract<S: Store>(s: &S, recids: &[&RecidRow], ctx: &str) {
    s.verify()
        .unwrap_or_else(|e| panic!("[{ctx}] verify() failed: {e}"));
    let mut want_all: BTreeSet<Recid> = BTreeSet::new();
    for row in recids {
        let recid = nz(row.recid);
        let label = &row.label;
        match row.state {
            RecidState::Live => {
                let got = s
                    .get(recid, &R)
                    .unwrap_or_else(|e| panic!("[{ctx}] get({label}) failed: {e}"));
                assert_eq!(
                    got,
                    Some(payload(row.payload_id, row.len)),
                    "[{ctx}] {label} (recid {recid}) content mismatch"
                );
                want_all.insert(recid);
            }
            RecidState::Null => {
                assert_eq!(
                    s.get(recid, &R).unwrap(),
                    None,
                    "[{ctx}] {label} (recid {recid}) must read as null"
                );
                want_all.insert(recid);
            }
            RecidState::Prealloc => {
                assert_eq!(
                    s.get(recid, &R).unwrap(),
                    None,
                    "[{ctx}] {label} (recid {recid}) prealloc must read as null"
                );
            }
            RecidState::Deleted => {
                assert!(
                    matches!(s.get(recid, &R), Err(DbError::GetVoid(x)) if x == recid.get()),
                    "[{ctx}] {label} (recid {recid}) must be deleted (GetVoid)"
                );
            }
        }
    }
    let all: BTreeSet<Recid> = s.get_all_recids().unwrap().into_iter().collect();
    assert_eq!(
        all, want_all,
        "[{ctx}] get_all_recids must equal the manifest's live+null set"
    );
}

/// Reads a file a `post` row names, failing with the real error rather than
/// silently treating "unreadable" as "absent".
fn read_named(path: &Path, rel: &str, ctx: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("[{ctx}] cannot read {rel}: {e}"))
}

/// The oracle rows one cell owes, and which of them a handler actually ran.
///
/// Separately unit-tested, because it is the ONE mechanism standing between
/// "executes" and "parses and drops" for three of the four addressed oracle row
/// types — every one except `action`, which has a failure of its own. The
/// whole-file `post` hash subsumes a byte-at-offset assertion, the two-sided
/// unnamed-input rule silently re-verifies a file whose **`unchanged`** row was
/// dropped, and *nothing at all* observes a dropped `reopen`.
///
/// A row is identified by a key AND by its ADDRESS: consuming the right key
/// with a different row is how a handler that grades the wrong object still
/// reports its debt paid.
pub struct Consumption {
    ctx: String,
    owed: Vec<(String, usize)>,
    done: BTreeSet<String>,
}

impl Consumption {
    pub fn new(ctx: &str) -> Consumption {
        Consumption {
            ctx: ctx.to_string(),
            owed: Vec::new(),
            done: BTreeSet::new(),
        }
    }

    pub fn owe<T>(&mut self, key: &str, row: &T) {
        assert!(
            !self.owed.iter().any(|(k, _)| k == key),
            "[{}] two oracle rows share the key {key}",
            self.ctx
        );
        self.owed.push((key.to_string(), row as *const T as usize));
    }

    pub fn consume<T>(&mut self, key: &str, row: &T) {
        let at = self
            .owed
            .iter()
            .find(|(k, _)| k == key)
            .unwrap_or_else(|| panic!("[{}] consumed {key}, which was never owed", self.ctx))
            .1;
        assert_eq!(
            at, row as *const T as usize,
            "[{}] consumed {key} with a different row",
            self.ctx
        );
        assert!(
            self.done.insert(key.to_string()),
            "[{}] consumed {key} twice",
            self.ctx
        );
    }

    pub fn require_all_consumed(&self) {
        let dropped: Vec<&str> = self
            .owed
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| !self.done.contains(*k))
            .collect();
        assert!(
            dropped.is_empty(),
            "[{}] no handler consumed: {dropped:?}. A parsed-and-dropped assertion is a green \
             cell that checked nothing",
            self.ctx
        );
    }
}

/// The two-sided D6 post-state rule, graded against a CAPTURE.
///
/// One side is the obvious one: every file a `post` row names must be in the
/// state that row declares. The other side is the amendment that makes the
/// rule total — **files not named by a `post` row are implicitly `unchanged`**
/// — so an unnamed input must still be there byte for byte, and a file that is
/// neither an input nor named must not exist at all. Without the second side a
/// cell that deleted a segment and wrote three new ones would pass by saying
/// nothing about them.
///
/// It reads the CAPTURE rather than the directory because the cell's `reopen`
/// step is an open: it happens not to rewrite a segment today, and "happens
/// not to" is not a property to hash a corpus against.
///
/// **Presence is decided by the capture's key set, and the capture is built by
/// `symlink_metadata`,** not by `read(..).ok()` — `.ok()` turns a permission
/// error, or the target having been replaced by a DIRECTORY, into "absent", so
/// a `deleted` row would pass on a file that is very much still there in
/// another shape. The C3r review named that one.
pub fn assert_post_state(
    posts: &[&V2Post],
    before: &BTreeMap<String, Vec<u8>>,
    after: &BTreeMap<String, Vec<u8>>,
    ctx: &str,
    owed: &mut Consumption,
) {
    let mut named: BTreeSet<&str> = BTreeSet::new();
    for p in posts {
        let where_ = format!("{ctx} post[{} {}]", p.rel, p.verb);
        assert!(
            named.insert(p.rel.as_str()),
            "{where_}: two post rows for one file"
        );
        let was = before.get(&p.rel);
        let now = after.get(&p.rel);
        match p.verb.as_str() {
            // §2.1: a post row is an explicit OVERRIDE of an input or an
            // explicit NEW file, so each verb says which side it is on.
            "unchanged" => {
                assert!(
                    was.is_some(),
                    "{where_}: names a file that was not an input"
                );
                assert!(now.is_some(), "{where_}: file is gone");
                assert_eq!(was, now, "{where_}: bytes changed");
            }
            "deleted" => {
                assert!(
                    was.is_some(),
                    "{where_}: names a file that was not an input"
                );
                assert!(now.is_none(), "{where_}: file is still there");
            }
            "created" | "truncated" | "modified" => {
                if p.verb == "created" {
                    assert!(
                        was.is_none(),
                        "{where_}: names a file that already existed as an input — an existing \
                         file the cell rewrote is `modified` or `truncated`"
                    );
                } else {
                    assert!(
                        was.is_some(),
                        "{where_}: names a file that was not an input — only `created` may name a \
                         file the cell did not start with"
                    );
                }
                let now = now.unwrap_or_else(|| panic!("{where_}: file is missing"));
                assert_eq!(now.len() as u64, p.len.unwrap(), "{where_}: length");
                assert_eq!(
                    sha256_hex(now),
                    *p.sha.as_ref().unwrap(),
                    "{where_}: SHA-256"
                );
                // …and the VERB's own relation to the input, which the three
                // verbs shared this arm without ever asserting. Round 3 of
                // review measured the consequence: the length and the hash are
                // self-consistent whatever the verb says, so a file that GREW
                // satisfied `truncated` and a file that did not change at all
                // satisfied `modified`. The verb was decoration.
                //
                // I had recorded this as needing C5t's torn-tail images. That
                // was wrong and the reviewer falsified it in six lines of the
                // existing synthetic battery — the rule is about the relation
                // between two byte strings and needs no engine to produce them.
                match p.verb.as_str() {
                    // "the active segment is `truncated:<len>:<sha>` back to
                    // its last valid section end" (contract §10.1): a prefix,
                    // and strictly shorter. Both halves, or a rewrite that
                    // happens to be shorter would pass as a truncation.
                    "truncated" => {
                        // ONE statement: a truncation is a PROPER PREFIX of the
                        // input. Written as two — "it shrank" and "it is a
                        // prefix" — the shrink half has no red of its own,
                        // because a file that grew fails the prefix comparison
                        // too. The campaign measured that: the two-assertion
                        // form left `posttrunc_len` a survivor. Same shape as
                        // `assert_family`'s S2 arm and the read-only probe, and
                        // the same reason.
                        //
                        // **And the collapse is not free**, which round 4
                        // measured: deleting one statement proves that SOME
                        // part of the conjunction matters, never which. `<`
                        // regressed to `<=` with the whole gate and the whole
                        // campaign green, because no input had equal bytes.
                        // Collapsing removes the masking BETWEEN the halves; it
                        // does not give either of them a red. The inputs have
                        // to, and there is now one per half.
                        let was = was.expect("checked above");
                        assert!(
                            now.len() < was.len() && was[..now.len()] == now[..],
                            "{where_}: `truncated` must name a PROPER PREFIX of the input — it \
                             was {} bytes and is now {}, and the contract's truncation is the \
                             active segment cut back to its last valid section end",
                            was.len(),
                            now.len()
                        );
                    }
                    // `modified` exists because Q8's segment GREW, which no
                    // other disposition can describe. It must therefore mean
                    // the file changed — and must not be usable for the shape
                    // that has its own verb, or the two are interchangeable and
                    // neither is a claim.
                    "modified" => {
                        let was = was.expect("checked above");
                        assert_ne!(
                            &was[..],
                            &now[..],
                            "{where_}: `modified` names a file whose bytes did not change"
                        );
                        assert!(
                            !(now.len() < was.len() && was[..now.len()] == now[..]),
                            "{where_}: `modified` names a pure truncation, which is `truncated`"
                        );
                    }
                    _ => {}
                }
            }
            other => panic!("{where_}: unknown disposition verb {other}"),
        }
        // AFTER the disposition was asserted, never before: a row consumed on
        // entry would be accounted for by a handler that had not yet graded it.
        owed.consume(&format!("post {}", p.rel), *p);
    }
    for (rel, was) in before {
        if named.contains(rel.as_str()) {
            continue;
        }
        let now = after
            .get(rel)
            .unwrap_or_else(|| panic!("{ctx}: input {rel} is gone and no post row says so"));
        assert_eq!(
            was, now,
            "{ctx}: input {rel} changed and no post row says so"
        );
    }
    for name in after.keys() {
        assert!(
            before.contains_key(name) || named.contains(name.as_str()),
            "{ctx}: unexpected new file {name}"
        );
    }
}

/// How a cell chooses its opener.
///
/// [`Dispatch::ByManifest`] is the only value production uses;
/// [`Dispatch::AlwaysWal3`] exists so the C5 plan §3.11 mutant can be RUN
/// rather than described — a deletion that merely restored an
/// `opener == "wal3"` refusal would prove parser branching and nothing about
/// this engine.
///
/// **§3.11's mechanism is the one this engine actually has, and C5r measured
/// both halves of it** (java's flip found both false for java, so neither was
/// inherited). `StoreDirect::open_file` refuses the bare segment with
/// `not a MapDB StoreDirect file (bad magic)` and leaves the directory holding
/// `{x}`; `StoreWAL::open` on the same path refuses it as D1 — a regular file
/// at the WAL base path — but takes `<base>.lock` BEFORE the check and leaves
/// `{x, x.lock}`. Both openers reject, so the verdict discriminates nothing;
/// the stray lock does, against the two-sided file-set rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dispatch {
    ByManifest,
    AlwaysWal3,
}

/// Runs schema-v2 cells against this engine — the single executor for BOTH v2
/// roots.
///
/// **Why one type and not two.** `tests/xfixtures-v2/` is the static `v2-core`
/// sample and `tests/xfixtures-v2-corpus/` is the `v2-oracle` preflight root;
/// they differ in which rows they carry, not in what a cell means. Two
/// executors would be two implementations of the post-state rule, the opener
/// dispatch and the reader contract, and this workstream has already shipped
/// the consequence twice — a fix applied to one of two copies is a fix that did
/// not happen (C2j's B-finding). What legitimately differs between the roots is
/// the CARDINALITY rule, and that lives in the callers: the sample owes one
/// cell per fixture per mode, the corpus owes exactly its `applies` rows.
pub struct Cells<'a> {
    sample: &'a SampleV2,
    /// The `fixtureId/mode` of every `ro` accept cell whose read-only handle
    /// was actually probed with a write.
    ///
    /// This exists so the probe is not a LEAF. Deleting
    /// `if ro { assert_write_refused(..) }` leaves nothing to observe: the
    /// standalone discriminating test opens the read-only handle itself, so it
    /// cannot see the executor skipping the call. A set the caller compares
    /// against the cells it ran turns the deletion into an empty set and a red
    /// gate. It is bookkeeping, and it is the cheapest honest answer to lesson
    /// (i) — a rule can be correct, directly tested, and never called.
    pub ro_probed: BTreeSet<String>,
}

impl<'a> Cells<'a> {
    pub fn new(sample: &'a SampleV2) -> Cells<'a> {
        Cells {
            sample,
            ro_probed: BTreeSet::new(),
        }
    }

    /// Every `action`/`bytes`/`reopen`/`family`/`post` row addressed to this
    /// engine in `mode` must name a cell the engine actually runs.
    ///
    /// **Per-cell consumption cannot see this**, and both C5j reviewers proved
    /// it independently: the accountant is built from the rows addressed to the
    /// cell BEING RUN, so a row addressed to a `(fixture, mode)` with no
    /// `expect` row is owed by nobody, consumed by nobody and graded by nobody.
    /// Contract §2.3 says an addressed row no handler consumed is a failure,
    /// and this is the half of that sentence per-cell accounting cannot reach.
    ///
    /// It is scoped to ONE mode because this engine's two halves run in two
    /// binaries — `rw` cells through the public opener in `tests/`, `ro` cells
    /// through the crate-internal one (decision C-D3). Each half grades the
    /// rows addressed to its own mode; `MODES` has exactly two members and the
    /// parser refuses any third, so between them the two halves cover every
    /// row addressed to this engine.
    pub fn require_every_oracle_row_addresses_a_run_cell(
        &self,
        mode: &str,
        ran: &BTreeSet<String>,
    ) {
        let m = &self.sample.manifest;
        let mut orphans: BTreeSet<String> = BTreeSet::new();
        let addressed = |engine: &str, m2: &str, fixture: &str| -> bool {
            engine == ENGINE && m2 == mode && !ran.contains(&format!("{fixture}/{m2}"))
        };
        for a in &m.actions {
            if addressed(&a.engine, &a.mode, &a.fixture) {
                orphans.insert(format!("action {}/{} {}", a.fixture, a.mode, a.verb));
            }
        }
        for b in &m.bytes {
            if addressed(&b.engine, &b.mode, &b.fixture) {
                orphans.insert(format!("bytes {}/{} {}", b.fixture, b.mode, b.rel));
            }
        }
        for r in &m.reopens {
            if addressed(&r.engine, &r.mode, &r.fixture) {
                orphans.insert(format!("reopen {}/{} {}", r.fixture, r.mode, r.family));
            }
        }
        // C8f f1: `family` is the fifth addressed oracle row type — same
        // suite-wide orphan rule as reopen (plan §4.2 item 2).
        for r in &m.families {
            if addressed(&r.engine, &r.mode, &r.fixture) {
                orphans.insert(format!("family {}/{} {}", r.fixture, r.mode, r.family));
            }
        }
        // `post` is an addressed row type. C5j's round 2 found that
        // nothing on either side of the fence caught one addressed to a cell no
        // engine runs; §2.3 names it now, and it has a per-cell debt as well.
        for p in &m.posts {
            if addressed(&p.engine, &p.mode, &p.fixture) {
                orphans.insert(format!("post {}/{} {}", p.fixture, p.mode, p.rel));
            }
        }
        assert!(
            orphans.is_empty(),
            "oracle rows addressed to {ENGINE} whose cell this engine never ran, so no accountant \
             could ever owe them: {orphans:?}"
        );
    }

    /// Runs ONE cell: stage the inputs, open through the opener the `expect`
    /// row names, grade every oracle row addressed here, and account for all of
    /// them.
    ///
    /// The `open` closure is the one thing the two halves do not share: the
    /// integration test opens read-write through the public `StoreWAL::open`,
    /// and the in-crate module opens read-only through `open_cfg`, which is
    /// `pub(crate)` and stays that way (C-D3).
    pub fn run_cell(
        &mut self,
        e: &V2Expect,
        cell: &Path,
        open: &dyn Fn(&Path) -> Result<StoreWAL>,
        dispatch: Dispatch,
    ) {
        let m = &self.sample.manifest;
        let ctx = format!(
            "v2 cell[{} {ENGINE} {} {} {}]",
            e.fixture, e.mode, e.verdict, e.opener
        );

        // Every oracle row addressed to this cell, and nothing else. Rows are
        // struck off as they are consumed; what is left at the end is a claim
        // the executor was handed and dropped.
        let mut owed = Consumption::new(&ctx);
        for a in m.actions_of(&e.fixture, ENGINE, &e.mode) {
            owed.owe(&format!("action {}", a.verb), a);
        }
        for b in m.bytes_of(&e.fixture, ENGINE, &e.mode) {
            owed.owe(&format!("bytes {}@{}", b.rel, b.offset), b);
        }
        for r in m.reopens_of(&e.fixture, ENGINE, &e.mode) {
            owed.owe(&format!("reopen {}", r.family), r);
        }
        for r in m.families_of(&e.fixture, ENGINE, &e.mode) {
            owed.owe(&format!("family {}", r.family), r);
        }
        for p in m.posts_of(&e.fixture, ENGINE, &e.mode) {
            owed.owe(&format!("post {}", p.rel), p);
        }

        let files = m.files_of(&e.fixture);
        assert!(!files.is_empty(), "[{ctx}] fixture has no file rows");
        let mut before: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for f in &files {
            let bytes = self.sample.bytes_of(f).to_vec();
            std::fs::write(cell.join(&f.rel), &bytes).unwrap();
            before.insert(f.rel.clone(), bytes);
        }

        let opener = match dispatch {
            Dispatch::AlwaysWal3 => "wal3",
            Dispatch::ByManifest => e.opener.as_str(),
        };
        let target = cell.join(&e.open_arg);
        // The cell's OWN refusal, on a reject cell. `None` on an accept cell,
        // where there is none and a `reopen` row (Q8's) is the only grade.
        let mut first_refusal: Option<DbError> = None;
        match e.verdict.as_str() {
            "accept" => self.run_accept(&ctx, e, opener, &target, open, &mut owed),
            // A `reject` cell asserts that the open FAILED. Since C5t it can
            // also assert WHICH failure, and that is plan §3.12's point.
            //
            // C8f f1: first-open family comes from the `family` row (plan §4.2),
            // not only via the `reopen` path. `assert_first_open_family` looks
            // up exactly one key for `(fixture, rust, mode)`, grades the held
            // refusal, and consumes once. `assert_reopen` is then stability /
            // second-open only. Mutating R6-audit/rw carries `family` alone.
            //
            // The refusal is HELD rather than graded here. Family grading runs
            // after the capture and the post-state rules, because §3.11's
            // mutant — the direct cell dispatched to the wal3 opener — trips
            // both this family check and the post-row rule it was written to
            // prove, and lesson (h) says such an input measures whichever fires
            // first. Grading here made §3.11's mutant report the family, and
            // §3.11's own rule went unmeasured.
            "reject" => {
                first_refusal = Some(
                    refusal_of(&ctx, opener, &e.mode, &target, open).unwrap_or_else(|| {
                        panic!("[{ctx}] expected a refusal, but the store opened")
                    }),
                );
            }
            v => panic!("[{ctx}] unsupported verdict {v}"),
        }

        // THE CAPTURE, taken before the reopen — see `assert_post_state`.
        let after = capture(cell, &ctx);
        assert_bytes_rows(m, e, &after, &ctx, &mut owed);
        // The post-cardinality guard runs BEFORE the rule it guards, not after:
        // a cell whose post rows were all removed also loses the row naming the
        // lock it creates, so the two-sided file-set rule fires first and this
        // guard would report a red it did not produce (lesson h — an input that
        // trips several checks measures the first).
        //
        // It keys on the MANIFEST's opener rather than the dispatched one.
        // Plan §5.3 item 5's second relaxation
        // is what this engine needs and java did not: `StoreDirect` here takes
        // no `<base>.lock`, so the direct cell legitimately leaves the
        // directory as it found it and carries no post row. The two-sided
        // unnamed-input rule above still runs and is what carries the check —
        // it is exactly what the §3.11 mutant trips. Keying on the DISPATCHED
        // opener would make this guard fire first under `AlwaysWal3` and the
        // mutant would report red for the wrong rule (lesson h).
        assert!(
            !m.posts_of(&e.fixture, ENGINE, &e.mode).is_empty() || e.opener != "wal3",
            "[{ctx}] a wal3 cell with no post rows asserts nothing about the directory it just \
             opened, which is not a check"
        );
        assert_post_state(
            &m.posts_of(&e.fixture, ENGINE, &e.mode),
            &before,
            &after,
            &ctx,
            &mut owed,
        );
        // First-open family grade (C8f f1) BEFORE the stability reopen, so a
        // wrong `family` row reds at `family[..]` and a wrong `reopen` reds at
        // `reopen[..]` — distinct sites, no double-consume (different keys).
        assert_first_open_family(m, e, &ctx, &mut owed, first_refusal.as_ref());
        assert_reopen(m, e, opener, &target, &ctx, &mut owed);
        owed.require_all_consumed();
    }

    fn run_accept(
        &mut self,
        ctx: &str,
        e: &V2Expect,
        opener: &str,
        target: &Path,
        open: &dyn Fn(&Path) -> Result<StoreWAL>,
        owed: &mut Consumption,
    ) {
        assert_eq!(
            opener, "wal3",
            "[{ctx}] an accept cell through a non-wal3 opener is a shape no corpus has and no \
             executor here implements"
        );
        let m = &self.sample.manifest;
        let s =
            open(target).unwrap_or_else(|err| panic!("[{ctx}] accept cell failed to open: {err}"));
        for a in m.actions_of(&e.fixture, ENGINE, &e.mode) {
            // Deliberately not caught: a store that opened and then failed its
            // action is a different fact from one that refused to open, and
            // collapsing the two lets a broken action be read as the verdict.
            run_action(&s, &a.verb, &a.arg_spec);
            owed.consume(&format!("action {}", a.verb), a);
        }
        let recids = m.recids_of(&e.fixture);
        self.require_some_oracle(ctx, e, !recids.is_empty());
        if !recids.is_empty() {
            assert_reader_contract(&s, &recids, ctx);
            assert_every_logged_recid_is_classified(&s, self.sample, &e.fixture, &recids, ctx);
        }
        if e.mode == "ro" {
            self.assert_write_refused(ctx, e, &s);
        }
        s.close().unwrap();
    }

    /// An accept cell must assert SOMETHING about the store it just opened —
    /// the C3j guard, as the disjunction plan §5.3 item 5 asked for.
    ///
    /// C5j's first draft deleted this guard for the sealed root and offered the
    /// distribution seal as its replacement. Both reviewers refused, and proving
    /// them right took one doctored manifest: strip a fixture's recid rows and
    /// its accept cell passes on nothing but the universal `x.lock` post row.
    /// **The seal proves copy fidelity and the guard proves assertion
    /// adequacy**; artifact identity cannot buy a semantic property.
    ///
    /// The disjunction admits every cell either root has: recid rows, an
    /// `action` row whose result a post oracle grades, a `reopen` row whose
    /// claim is the store's permanent unopenability, `mode == ro`, where the
    /// read-only write refusal below is the executable claim, or a `post` row
    /// that says the open CHANGED the tree.
    ///
    /// **The staged run found that last arm**, and no preflight root could
    /// have: `mut-wal3-torn-tail` carries no recid row, no action and no reopen,
    /// and what it asserts is a post state — the tail truncated to the last
    /// valid section end. A byte-exact statement of what recovery left behind is
    /// an assertion about the store, not an absence of one.
    ///
    /// `created` and `unchanged` are deliberately NOT in it. Every wal3 cell
    /// carries the universal `x.lock created` row, so admitting `created` would
    /// make the guard vacuous — it would admit the very cell the doctored proof
    /// above uses. `unchanged` is the two-sided rule's default statement and
    /// asserts that the open did nothing.
    fn require_some_oracle(&self, ctx: &str, e: &V2Expect, has_recids: bool) {
        let m = &self.sample.manifest;
        let mutation_claimed = m
            .posts_of(&e.fixture, ENGINE, &e.mode)
            .iter()
            .any(|p| matches!(p.verb.as_str(), "modified" | "truncated" | "deleted"));
        // THE SIXTH ARM IS JAVA'S ALONE, AND IT IS DELIBERATELY ABSENT HERE.
        //
        // java's `requireSomeOracle` carries a DIVERGENCE arm: another engine
        // reaches a different verdict on this same fixture and mode, so the
        // cell's claim is the verdict itself. It was written for
        // `div-wal3-entry-recid0`, where java's behaviour is UNDEFINED
        // (`recidToOffset` computes `recid - 1`) and pinning a logical state
        // would freeze an accident.
        //
        // This engine had that arm too, for one day, and round 4 measured it:
        // **it can never fire here.** The corpus holds THREE divergent fixtures
        // and therefore SIX divergent (fixture, mode) groups, since each
        // diverges in both `rw` and `ro`: `div-wal3-lsn-exhausted`,
        // `div-wal3-entry-recid0`, `div-wal3-packlong-overlong`. The preflight
        // root holds the two groups of the first. All eight are java ACCEPT
        // against ports REJECT — round 5 re-enumerated them, because round 4's
        // census said "three groups" and a proof that miscounts its own domain
        // is a proof to re-check. This guard runs on the accept arm only, so for
        // every rust accept cell in either root the arm was `false` outright,
        // not masked by an earlier disjunct: deleting it cannot change any
        // result of any run this engine has ever done, the staged one included.
        // It went, rather than staying as a guard nothing can trip.
        //
        // What that costs, stated rather than discovered: if a future corpus
        // ever holds a cell this engine ACCEPTS and another REJECTS, this guard
        // refuses it and java's does not. That refusal is the right red — it
        // says the arm is now reachable and owes a doctored input of its own
        // before it comes back.
        let any = has_recids
            || !m.actions_of(&e.fixture, ENGINE, &e.mode).is_empty()
            || !m.reopens_of(&e.fixture, ENGINE, &e.mode).is_empty()
            || e.mode == "ro"
            || mutation_claimed;
        assert!(
            any,
            "[{ctx}] an accept cell with no recid rows, no action, no reopen, no post row claiming \
             a change and a writable handle asserts nothing about the store it opened, which is \
             not a check. If this cell is one another engine REJECTS, the divergence arm this \
             engine deliberately does not carry is now reachable and owes a doctored input"
        );
    }

    /// D7's read-only mode is observable, in the direction that matters: a
    /// write through the `ro` handle must be refused.
    ///
    /// C3z's review found the general shape this closes — `mode` was parsed,
    /// vocabulary-checked and used to select an opener, and then NOTHING
    /// observed the difference, so every `ro` cell in java and rust was an
    /// ordinary writable open wearing a label.
    ///
    /// ONE assertion, not two. On a writable handle there is no refusal to
    /// inspect, so "it was refused" and "the refusal names the mode" as two
    /// statements are a pair that can only ever be killed by each other. The
    /// claim is a conjunction and it is written as one.
    pub fn assert_write_refused(&mut self, ctx: &str, e: &V2Expect, s: &StoreWAL) {
        let outcome = match s.put(&vec![1u8, 2, 3], &R) {
            Ok(_) => "the write was ACCEPTED".to_string(),
            Err(err) => format!("refused with: {err}"),
        };
        assert!(
            outcome.contains("read-only"),
            "[{ctx}] the probe accepted a writable handle or a refusal that does not name the \
             read-only mode — {outcome}"
        );
        // LAST, and inside this method rather than beside its call: with the
        // recording at the call site, deleting the call and keeping the `add`
        // leaves the gate green — the set then observes that the bookkeeping
        // ran, not that the probe did.
        self.ro_probed.insert(format!("{}/{}", e.fixture, e.mode));
    }
}

/// Opens and returns the refusal, or `None` if the store opened (and closed).
fn refusal_of(
    ctx: &str,
    opener: &str,
    mode: &str,
    target: &Path,
    open: &dyn Fn(&Path) -> Result<StoreWAL>,
) -> Option<DbError> {
    let r: Result<()> = if opener == "direct" {
        assert_eq!(
            mode, "rw",
            "[{ctx}] the direct opener has no read-only mode here"
        );
        StoreDirect::open_file(target).and_then(|s| s.close())
    } else {
        open(target).and_then(|s| s.close())
    };
    r.err()
}

/// Everything in the cell directory, by name, after the cell has run.
///
/// A name that is present but is not a REGULAR file is refused here rather than
/// read as absent: `read(..).ok()` turns a permission error, or a file replaced
/// by a directory of the same name, into "the file is gone", so a `deleted` row
/// would pass on something that is very much still there in another shape. The
/// C3r review named that one.
pub fn capture(cell: &Path, ctx: &str) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for name in dir_entries(cell) {
        let p = cell.join(&name);
        let md = std::fs::symlink_metadata(&p)
            .unwrap_or_else(|e| panic!("[{ctx}] cannot stat {name}: {e}"));
        assert!(md.is_file(), "[{ctx}] {name} is not a regular file");
        out.insert(name.clone(), read_named(&p, &name, ctx));
    }
    out
}

/// Grades every `bytes` row against the CAPTURED post bytes (contract §2.3).
///
/// It is never a pre-open patch: Q8's input segment is 186 bytes and its
/// assertion is at offset 187, so a pre-open reading is not merely wrong, it is
/// out of range. An assertion whose range cannot be reached is a failure, never
/// a skip.
fn assert_bytes_rows(
    m: &V2,
    e: &V2Expect,
    after: &BTreeMap<String, Vec<u8>>,
    ctx: &str,
    owed: &mut Consumption,
) {
    for b in m.bytes_of(&e.fixture, ENGINE, &e.mode) {
        let where_ = format!("{ctx} bytes[{}@{}]", b.rel, b.offset);
        let now = after
            .get(&b.rel)
            .unwrap_or_else(|| panic!("{where_}: names a file the cell directory does not hold"));
        let len = b.hex.len() / 2;
        let end = b.offset as usize + len;
        assert!(
            end <= now.len(),
            "{where_}: the range ends at {end} and the post state is {} bytes",
            now.len()
        );
        let got: String = now[b.offset as usize..end]
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect();
        assert_eq!(got, b.hex, "{where_}: the asserted bytes");
        owed.consume(&format!("bytes {}@{}", b.rel, b.offset), b);
    }
}

/// Grades the cell's OWN (first-open) refusal against the `family` row.
///
/// Plan §4.2 executor invariant (C8f f1):
/// 1. On every rejected open: look up exactly one `family` key for
///    `(fixture, rust, mode)`; absence is failure; `assert_family`; consume
///    once.
/// 2. Accept arms carry no `family` row; a present row is owed and left
///    unconsumed, which fails at suite/cell end — same as orphan reopen.
/// 3. Mutating R6-audit/rw: `family` only (no `reopen`) still grades here.
///
/// Kept AFTER post-state (see `run_cell`) so §3.11's misrouted direct cell
/// reports the lock/file-set red rather than a family red (lesson h).
fn assert_first_open_family(
    m: &V2,
    e: &V2Expect,
    ctx: &str,
    owed: &mut Consumption,
    first: Option<&DbError>,
) {
    let fams = m.families_of(&e.fixture, ENGINE, &e.mode);
    match e.verdict.as_str() {
        "reject" => {
            let f = first.unwrap_or_else(|| {
                panic!("[{ctx}] reject cell has no held first-open refusal to grade")
            });
            // Absence is failure — not a silent skip. Zero rows would leave
            // nothing owed and the first open ungraded; more than one is a
            // parser-level duplicate already, but re-assert for the invariant.
            assert!(
                !fams.is_empty(),
                "[{ctx}] reject arm has no family row for ({}/{}/{}) — first-open family is \
                 required (C8f f1 / plan §4.2)",
                e.fixture,
                ENGINE,
                e.mode
            );
            assert_eq!(
                fams.len(),
                1,
                "[{ctx}] reject arm has {} family rows for ({}/{}/{}); exactly one is required",
                fams.len(),
                e.fixture,
                ENGINE,
                e.mode
            );
            let r = fams[0];
            assert_family(&format!("{ctx} family[{}]", r.family), &r.family, f);
            owed.consume(&format!("family {}", r.family), r);
        }
        "accept" => {
            // A family row on an accept arm is catalogue-illegal; if one is
            // present it stays owed and `require_all_consumed` reds. No consume
            // path here — that is the fail-closed half of the bijection.
            let _ = (fams, first);
        }
        v => panic!("[{ctx}] unsupported verdict {v}"),
    }
}

/// Stability / second-open grade only (C8f f1).
///
/// First-open family grading moved to [`assert_first_open_family`]. On an
/// ACCEPT cell — Q8 — there is no first refusal and the reopen remains the
/// only grade. On a REJECT cell the second open must refuse the same way.
fn assert_reopen(
    m: &V2,
    e: &V2Expect,
    opener: &str,
    target: &Path,
    ctx: &str,
    owed: &mut Consumption,
) {
    for r in m.reopens_of(&e.fixture, ENGINE, &e.mode) {
        let where_ = format!("{ctx} reopen[{}]", r.family);
        // A reopen is a WRITABLE open whatever the cell's own mode was: the
        // claim is that the store is permanently unopenable, and a read-only
        // probe would be a weaker one.
        //
        // Through the cell's OWN opener, not a hard-coded `wal3`. Until C5t only
        // Q8 carried a reopen row and Q8 is a wal3 cell, so the constant was
        // right by accident; `reject-wal3-segment-at-direct` carries one now,
        // and sending it to the WAL opener would grade a `direct-magic` family
        // against a refusal `StoreDirect` never made.
        let refusal = refusal_of(&where_, opener, "rw", target, &|p| StoreWAL::open(p));
        let t = refusal.unwrap_or_else(|| panic!("{where_}: the store opened again"));
        assert_family(&where_, &r.family, &t);
        owed.consume(&format!("reopen {}", r.family), r);
    }
}

/// D1 is the legacy boundary — a v1 artifact sitting where the v3 opener expects
/// a base — and `wal_segments.rs` refuses it with one sentence per row.
///
/// TWO of the three rows, not three. The `.wal` row is family **N6**, which the
/// catalogue names separately and which this predicate must therefore refuse: a
/// D1 arm satisfied by an N6 refusal is a family column that cannot tell the
/// ports' upgrade-safety boundary from Java's own row.
///
/// **The PATH is opaque.** The first draft split on the first `": "` after
/// `" present at "` and called what preceded it the path, which makes a legal
/// Unix path containing `": "` fail the predicate on a genuine refusal (codex
/// round 1 finding 6). The fixed text is stripped from both ends instead and
/// whatever is left is the path, whatever it contains. `" present at "` is part
/// of the exact prefix for the same reason.
fn d1_matches(msg: &str) -> bool {
    const TAIL: &str = ": no migration to v3 — open it with the release that wrote it and copy \
                        the data across, or move it aside";
    let Some(head) = msg.strip_suffix(TAIL) else {
        return false;
    };
    for what in [
        "regular file at the WAL base path (the v3 opener takes a base, not a log file)",
        "v1 checkpoint temp, possibly the only recoverable copy after a v1 crash",
    ] {
        // A non-empty remainder: the row always names a path, and an empty one
        // would mean the message ended where the path should start.
        if let Some(path) = head.strip_prefix(&format!("{what} present at ")) {
            if !path.is_empty() {
                return true;
            }
        }
    }
    false
}

/// S2 is the section-header `lsn <= seg.last_lsn` rule, wrapped by the WAL
/// segment prefix the recovery scan puts on every per-segment refusal.
///
/// Matched WHOLE, with `is_match` over an anchored pattern built by hand: the
/// refusal this grades is one line of the engine and its whole form is
/// knowable, so matching the whole form is what the check should say. An
/// unanchored substring test would accept the S2 wording embedded in unrelated
/// text.
fn s2_matches(msg: &str) -> bool {
    // `WAL segment <name>: section LSN <n> at offset <n> does not follow <n>`
    //
    // The NAME is opaque, for the reason `d1_matches` says at length: a legal
    // segment filename may contain `": "`, and a predicate that reads the name
    // as "everything up to the first `: `" refuses a genuine refusal about one.
    // The first draft split there AND forbade a colon outright, which is the
    // same mistake stated twice; codex round 2 found it after round 1 found it
    // in the D1 predicate. The rest of the sentence is fixed, so the name is
    // whatever lies between the fixed prefix and the next fixed marker — found
    // from the RIGHT, so only the last such marker can end the name.
    let Some(rest) = msg.strip_prefix("WAL segment ") else {
        return false;
    };
    let Some((name, rest)) = rest.rsplit_once(": section LSN ") else {
        return false;
    };
    if name.is_empty() {
        return false;
    }
    let Some((lsn, rest)) = rest.split_once(" at offset ") else {
        return false;
    };
    let Some((off, prev)) = rest.split_once(" does not follow ") else {
        return false;
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    // The two LSNs are SIGNED and the offset is NOT, and the asymmetry is the
    // engine's, not a convenience: an LSN is `i64` on disk, so a CRC-valid
    // section carrying a negative one is a real input this rule can be shown,
    // while the offset is a `u64` and no refusal this engine renders can put a
    // minus sign there. Round 3 found one sign predicate covering all three,
    // which accepted `at offset -2` as S2 — a message no engine produces, so a
    // refusal wearing it is something else entirely and must not be graded here.
    let signed = |s: &str| digits(s.strip_prefix('-').unwrap_or(s));
    signed(lsn) && digits(off) && signed(prev)
}

/// Strip the `WAL segment <name>: ` prefix with an opaque name (may contain
/// `": "` or newlines). Returns the remainder after the last matching marker.
fn after_wal_segment<'a>(msg: &'a str, marker: &str) -> Option<&'a str> {
    let rest = msg.strip_prefix("WAL segment ")?;
    let (_name, rest) = rest.rsplit_once(marker)?;
    if _name.is_empty() {
        return None;
    }
    Some(rest)
}

fn digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit())
}

fn signed_digits(s: &str) -> bool {
    digits(s.strip_prefix('-').unwrap_or(s))
}

fn n6_matches(msg: &str) -> bool {
    // Rust/zig use "to v3"; java still says "to v2". This engine's wording:
    // `v1 single-file WAL present at <path>: no migration to v3 — …`
    const TAIL: &str = ": no migration to v3 — open it with the release that wrote it and copy \
                        the data across, or move it aside";
    let Some(head) = msg.strip_suffix(TAIL) else {
        return false;
    };
    let Some(path) = head.strip_prefix("v1 single-file WAL present at ") else {
        return false;
    };
    !path.is_empty()
}

fn h5_matches(msg: &str) -> bool {
    let Some(rest) = after_wal_segment(msg, ": unsupported WAL format version ") else {
        return false;
    };
    signed_digits(rest)
}

fn h6_matches(msg: &str) -> bool {
    let Some(rest) = after_wal_segment(msg, ": unknown segment flags ") else {
        return false;
    };
    signed_digits(rest)
}

fn h7_matches(msg: &str) -> bool {
    let Some(rest) = after_wal_segment(msg, ": header sequence ") else {
        return false;
    };
    let Some((seq, tail)) = rest.split_once(" does not match its name") else {
        return false;
    };
    signed_digits(seq) && tail.is_empty()
}

fn h9_matches(msg: &str) -> bool {
    let Some(rest) = after_wal_segment(msg, ": header firstLsn ") else {
        return false;
    };
    let Some((lsn, tail)) = rest.split_once(" is not a valid LSN") else {
        return false;
    };
    signed_digits(lsn) && tail.is_empty()
}

fn k4_matches(msg: &str) -> bool {
    let Some(rest) = after_wal_segment(msg, ": clean mark in segment ") else {
        return false;
    };
    let Some((seg, rest)) = rest.split_once(" authorizes removing segment ") else {
        return false;
    };
    let Some((through, tail)) = rest.split_once(", including itself") else {
        return false;
    };
    signed_digits(seg) && signed_digits(through) && tail.is_empty()
}

fn s8_matches(msg: &str) -> bool {
    // Three disjuncts on the 'K' body. K4 is a neighbour on the same mark.
    if let Some(rest) = after_wal_segment(msg, ": clean mark body is ") {
        if let Some((n, tail)) = rest.split_once(" bytes, not 16") {
            return signed_digits(n) && tail.is_empty();
        }
        return false;
    }
    if let Some(rest) = after_wal_segment(msg, ": clean mark attests cleanedThroughSeq ") {
        return signed_digits(rest);
    }
    if let Some(rest) = after_wal_segment(msg, ": clean mark attests logStartLsn ") {
        let Some((start, rest)) =
            rest.split_once(", which is not an LSN at or below the mark's own ")
        else {
            return false;
        };
        return signed_digits(start) && signed_digits(rest);
    }
    false
}

fn s9_matches(msg: &str) -> bool {
    let Some(rest) = after_wal_segment(msg, ": section LSNs must be consecutive: ") else {
        return false;
    };
    let Some((lsn, rest)) = rest.split_once(" at offset ") else {
        return false;
    };
    let Some((off, prev)) = rest.split_once(" after ") else {
        return false;
    };
    signed_digits(lsn) && digits(off) && signed_digits(prev)
}

fn s4_matches(msg: &str) -> bool {
    // Active mid-log FIRST: its wording embeds ": section body CRC mismatch at
    // offset ", which is also the non-final marker — checking non-final first
    // would take the mid-log message, fail the non-final tail, and refuse a
    // genuine mid-log refusal.
    if let Some(rest) = after_wal_segment(
        msg,
        ": mid-log corruption: section body CRC mismatch at offset ",
    ) {
        let Some((off, tail)) = rest.split_once(" but valid sections follow") else {
            return false;
        };
        return digits(off) && tail.is_empty();
    }
    if let Some(rest) = after_wal_segment(msg, ": section body CRC mismatch at offset ") {
        let Some((off, tail)) = rest.split_once(" in a non-final segment") else {
            return false;
        };
        return digits(off) && tail.is_empty();
    }
    false
}

fn r4_floor_matches(msg: &str) -> bool {
    // `WAL retained log begins at LSN <n> in <name> but <why>: sections below it are gone`
    let Some(rest) = msg.strip_prefix("WAL retained log begins at LSN ") else {
        return false;
    };
    let Some((lsn, rest)) = rest.split_once(" in ") else {
        return false;
    };
    let Some((_name, rest)) = rest.rsplit_once(" but ") else {
        return false;
    };
    let Some((_why, tail)) = rest.rsplit_once(": sections below it are gone") else {
        return false;
    };
    signed_digits(lsn) && tail.is_empty() && !_name.is_empty() && !_why.is_empty()
}

fn r4_chain_matches(msg: &str) -> bool {
    // `WAL segment <name> states it begins at LSN <n> but <prev> accounts for LSNs up to <n>: sections between them are gone`
    let Some(rest) = msg.strip_prefix("WAL segment ") else {
        return false;
    };
    let Some((_name, rest)) = rest.rsplit_once(" states it begins at LSN ") else {
        return false;
    };
    if _name.is_empty() {
        return false;
    }
    let Some((stated, rest)) = rest.split_once(" but ") else {
        return false;
    };
    let Some((_prev, rest)) = rest.rsplit_once(" accounts for LSNs up to ") else {
        return false;
    };
    let Some((upto, tail)) = rest.split_once(": sections between them are gone") else {
        return false;
    };
    signed_digits(stated) && signed_digits(upto) && tail.is_empty() && !_prev.is_empty()
}

fn r4_self_matches(msg: &str) -> bool {
    // `WAL segment <name> states it begins at LSN <n> but its first section is <n>: its leading sections are gone`
    let Some(rest) = msg.strip_prefix("WAL segment ") else {
        return false;
    };
    let Some((_name, rest)) = rest.rsplit_once(" states it begins at LSN ") else {
        return false;
    };
    if _name.is_empty() {
        return false;
    }
    let Some((stated, rest)) = rest.split_once(" but its first section is ") else {
        return false;
    };
    let Some((first, tail)) = rest.split_once(": its leading sections are gone") else {
        return false;
    };
    signed_digits(stated) && signed_digits(first) && tail.is_empty()
}

fn r6_audit_matches(msg: &str) -> bool {
    // Exact engine form, numbers opaque.
    let Some(rest) = msg.strip_prefix("WAL replay skipped ") else {
        return false;
    };
    let Some((n, rest)) = rest.split_once(
        " append(s) whose base image is absent and which no later entry superseded (recid ",
    ) else {
        return false;
    };
    let Some((recid, tail)) = rest.split_once("): the log is missing sections it depends on")
    else {
        return false;
    };
    digits(n) && digits(recid) && tail.is_empty()
}

/// True when a corruption payload belongs to any refined family this engine grades.
/// Used by the generic `DataCorruption` arm as an exclusion set (C8f f0: all thirteen
/// L15 families plus the C5t set).
fn refined_family_matches(msg: &str) -> bool {
    d1_matches(msg)
        || s2_matches(msg)
        || n6_matches(msg)
        || h5_matches(msg)
        || h6_matches(msg)
        || h7_matches(msg)
        || h9_matches(msg)
        || k4_matches(msg)
        || s8_matches(msg)
        || s9_matches(msg)
        || s4_matches(msg)
        || r4_floor_matches(msg)
        || r4_chain_matches(msg)
        || r4_self_matches(msg)
        || r6_audit_matches(msg)
        || msg == "not a MapDB StoreDirect file (bad magic)"
}

/// Asserts a refusal belongs to the named contract family.
///
/// The family is read from the manifest row, never hard-coded, so editing
/// `catalogue.reopen` stops the run instead of being graded against a constant
/// this file happens to agree with. A family this engine has no predicate for
/// is a **failure**: the alternative is a green cell whose reopen was checked
/// by nothing.
///
/// **The three families C5t brought here are separated by their MESSAGE**, which
/// is the mirror of how zig separates them. That port's `DbError` carries no
/// payload, so it reads a typed `Diag` and tells `D1` from `DataCorruption` by
/// whether recovery ran at all — a structural test. Here the payload IS the
/// diagnosis and there is no channel saying who produced it, so `DataCorruption`
/// has to be stated as "a corruption verdict that is none of the refined
/// families this engine names". Same partition, expressed with what each engine
/// has; neither is the other's oversight.
pub fn assert_family(where_: &str, family: &str, t: &DbError) {
    if family == "direct-magic" {
        // `StoreDirect::open_file`'s first refusal, matched whole. This is the
        // only cell in the corpus dispatched through the direct opener, so an
        // engine that sent it to the WAL opener by mistake refuses as `D1`
        // instead — the substitution `direct_cell_through_wal_opener` makes, and
        // the reason this predicate must not settle for "some corruption".
        assert!(
            matches!(t, DbError::DataCorruption(c)
                     if c.to_string() == "not a MapDB StoreDirect file (bad magic)"),
            "{where_}: `direct-magic` is StoreDirect's bad-magic refusal, and this is: {t}"
        );
        return;
    }
    if family == "D1" {
        assert!(
            matches!(t, DbError::DataCorruption(c) if d1_matches(&c.to_string())),
            "{where_}: `D1` is the legacy-boundary refusal, and this is: {t}"
        );
        return;
    }
    if family == "DataCorruption" {
        // The UNREFINED corruption family: what the two divergent entry cells
        // land on, where the catalogue names no rule. Stated as an exclusion
        // because that is what it means — a corruption verdict no refined family
        // this engine GRADES describes.
        //
        // C8f f0: every L15 family has a predicate, so all of them are excluded
        // here (N6 included — the day C5t deferred). A refinement another arm
        // can name is a neighbour, whatever it is in the taxonomy.
        let msg = match t {
            DbError::DataCorruption(c) => c.to_string(),
            _ => String::new(),
        };
        assert!(
            matches!(t, DbError::DataCorruption(_)) && !refined_family_matches(&msg),
            "{where_}: `DataCorruption` is a corruption verdict no TRANSPORTED refined family \
             names, and this is: {t}"
        );
        return;
    }
    if family == "S2" {
        // ONE statement for a claim with two halves: the refusal is a
        // corruption verdict AND its payload is the S2 rule's message. The
        // message is read from the PAYLOAD, not from the rendered error, because
        // `Display for DbError` prefixes `"data corruption: "`.
        let payload = match t {
            DbError::DataCorruption(c) => Some(c.to_string()),
            _ => None,
        };
        assert!(
            payload.as_deref().is_some_and(s2_matches),
            "{where_}: not the S2 rule's refusal — it must be a corruption verdict whose payload \
             is the S2 message, and this is: {t}"
        );
        return;
    }
    if family == "StoreFull" {
        // Q8's family in this engine, and the one the corpus's own reject
        // verdict for `div-wal3-lsn-exhausted` produces: a WAL segment
        // namespace with no sequence number left is a capacity ceiling with
        // nothing damaged, so the port refuses to call an intact store corrupt.
        assert!(
            matches!(t, DbError::StoreFull),
            "{where_}: StoreFull is a capacity verdict, got {t}"
        );
        return;
    }
    // ---- C8f f0: L15 remainder (thirteen families) ----
    let payload = match t {
        DbError::DataCorruption(c) => Some(c.to_string()),
        _ => None,
    };
    let ok = match family {
        "N6" => payload.as_deref().is_some_and(n6_matches),
        "H5" => payload.as_deref().is_some_and(h5_matches),
        "H6" => payload.as_deref().is_some_and(h6_matches),
        "H7" => payload.as_deref().is_some_and(h7_matches),
        "H9" => payload.as_deref().is_some_and(h9_matches),
        "K4" => payload.as_deref().is_some_and(k4_matches),
        "S8/K-bounds" => payload.as_deref().is_some_and(s8_matches),
        "S9" => payload.as_deref().is_some_and(s9_matches),
        "S4/mid-log" => payload.as_deref().is_some_and(s4_matches),
        "R4-floor" => payload.as_deref().is_some_and(r4_floor_matches),
        "R4-chain" => payload.as_deref().is_some_and(r4_chain_matches),
        "R4-self" => payload.as_deref().is_some_and(r4_self_matches),
        "R6-audit" => payload.as_deref().is_some_and(r6_audit_matches),
        _ => {
            panic!(
                "{where_}: error family {family} has no predicate in this engine. Refusing rather \
                 than accepting any refusal at all — an unimplemented family graded as 'it threw \
                 something' is the check not running"
            );
        }
    };
    assert!(
        ok,
        "{where_}: not the {family} rule's refusal — it must be a corruption verdict whose \
         payload is that family's message, and this is: {t}"
    );
}

/// Runs every schema-v2 cell addressed to this engine in `mode`, and asserts
/// the set that ran is **exactly** the set the `fixture` rows call for.
///
/// This is the STATIC SAMPLE's cardinality rule. It derives what should run
/// from a different row type than the one that says what will: a count, or the
/// set of modes actually seen, is a projection of the already-truncated input,
/// and the C3j review measured the consequence — deleting one `expect` row left
/// the suite green because another fixture still supplied that mode. Every
/// declared fixture owes one cell per mode, so a missing `expect` row now
/// contradicts the `fixture` row still sitting next to it.
///
/// The preflight corpus cannot use this rule — its cell set is legitimately
/// partial — which is what [`run_v2_corpus_cells`] and `applies` are for.
pub fn run_v2_cells(
    sample: &SampleV2,
    mode: &str,
    session: &Path,
    open: &dyn Fn(&Path) -> Result<StoreWAL>,
) {
    let m = &sample.manifest;
    // The sample is `v2-core`, in BOTH directions. A root that grew an oracle
    // row would be running assertions this rule never bought, and since C5 moved
    // the profile split into the grammar that is a refusal, not a widening.
    assert!(
        m.applies.is_empty()
            && m.actions.is_empty()
            && m.bytes.is_empty()
            && m.reopens.is_empty()
            && m.families.is_empty(),
        "the static sample carries an oracle row; it is v2-core through C7"
    );
    let want: BTreeSet<String> = m
        .declared_fixtures()
        .into_iter()
        .map(|f| format!("{f}/{mode}"))
        .collect();
    assert!(!want.is_empty(), "the v2 sample declares no fixtures");
    let mut cells = Cells::new(sample);
    let ran = run_cells(&mut cells, mode, session, open);
    assert_eq!(
        ran, want,
        "the {ENGINE}/{mode} cells that ran are not the ones the fixture rows call for"
    );
}

/// Runs the preflight CORPUS's cells for this engine in `mode`, under the
/// cardinality rule its partial cell set needs, and applies every rule that is
/// about the SET of cells rather than about one of them.
///
/// Two row types emitted from one catalogue is a pair that moves together, so
/// this also requires `applies == expect` per cell, in both directions. That
/// check is deliberately absent from `manifest_v2.py` — there both sets are
/// compared to the catalogue a few lines apart, so a third comparison could
/// only fire after one of those already had. An engine has no catalogue, so for
/// an engine the disagreement is the only detectable inconsistency, and without
/// it a manifest could have this suite run a cell it holds no verdict for.
///
/// Every doctored-manifest case enters HERE rather than calling the rules
/// directly. That distinction is the entire finding both C5j reviewers made: a
/// test that calls the suite-wide check itself proves the METHOD and leaves its
/// CALL unobserved, so deleting the call from the suite stays green.
pub fn run_v2_corpus_cells(
    sample: &SampleV2,
    mode: &str,
    session: &Path,
    open: &dyn Fn(&Path) -> Result<StoreWAL>,
) {
    let m = &sample.manifest;
    let want: BTreeSet<String> = m
        .applies
        .iter()
        .filter(|a| a.engine == ENGINE && a.mode == mode)
        .map(|a| format!("{}/{}", a.fixture, a.mode))
        .collect();
    assert!(
        !want.is_empty(),
        "the corpus declares no {ENGINE} applies rows for mode {mode}"
    );
    let expects: BTreeSet<String> = m
        .expects
        .iter()
        .filter(|e| e.engine == ENGINE && e.mode == mode)
        .map(|e| format!("{}/{}", e.fixture, e.mode))
        .collect();
    assert_eq!(
        want, expects,
        "the {ENGINE}/{mode} `applies` rows and `expect` rows are different sets"
    );

    let mut cells = Cells::new(sample);
    let ran = run_cells(&mut cells, mode, session, open);
    assert_eq!(
        ran, want,
        "the {ENGINE}/{mode} cells that ran are not the ones `applies` calls for"
    );

    // The other half of contract §2.3's consumption rule.
    cells.require_every_oracle_row_addresses_a_run_cell(mode, &ran);

    // …and the ro write probe really ran on every ro accept cell. Deleting the
    // call inside the executor leaves this set empty, which is the red that
    // call did not have.
    // In `rw` the expected set is empty, and comparing it is not decoration:
    // a probe that fired on a writable handle would land here.
    let ro_cells: BTreeSet<String> = m
        .expects
        .iter()
        .filter(|e| e.engine == ENGINE && e.mode == mode && mode == "ro" && e.verdict == "accept")
        .map(|e| format!("{}/{}", e.fixture, e.mode))
        .collect();
    assert!(
        mode != "ro" || !ro_cells.is_empty(),
        "the corpus has no {ENGINE} ro accept cell, so the read-only probe has no input"
    );
    assert_eq!(
        ro_cells, cells.ro_probed,
        "the ro cells whose read-only handle was probed with a write"
    );
}

fn run_cells(
    cells: &mut Cells<'_>,
    mode: &str,
    session: &Path,
    open: &dyn Fn(&Path) -> Result<StoreWAL>,
) -> BTreeSet<String> {
    let expects: Vec<V2Expect> = cells
        .sample
        .manifest
        .expects
        .iter()
        .filter(|e| e.engine == ENGINE && e.mode == mode)
        .cloned()
        .collect();
    let mut ran: BTreeSet<String> = BTreeSet::new();
    for (i, e) in expects.iter().enumerate() {
        let cell = session.join(format!("v2-{mode}-{i}"));
        let _ = std::fs::remove_dir_all(&cell);
        std::fs::create_dir_all(&cell).unwrap();
        cells.run_cell(e, &cell, open, Dispatch::ByManifest);
        assert!(
            ran.insert(format!("{}/{}", e.fixture, e.mode)),
            "two {mode} cells for {}",
            e.fixture
        );
        std::fs::remove_dir_all(&cell).unwrap();
    }
    ran
}

/// The completeness half of the recid oracle: every recid the LOG mentions is
/// either named by the manifest or void according to the engine.
///
/// Without this, deleting a `prealloc` (or `deleted`) recid row from the
/// manifest is invisible — measured, not assumed. `assert_reader_contract`
/// derives its `get_all_recids` set from the manifest's own live+null rows, so
/// dropping a row that is excluded from that set by construction removes an
/// assertion and adds none; and [`check_recids_against_manifest`] is one-way,
/// so a shorter manifest satisfies it more easily. Both directions of the
/// existing pair get WEAKER when a row disappears, which is the shape a
/// completeness rule has to fix from outside.
///
/// **This is not the direction plan §5 forbids.** §5 forbids asserting that a
/// log contains only the recids the manifest names, because §5.2's rolled-back
/// put need only be invisible through the API. That case is exactly what the
/// escape hatch here admits: a rolled-back recid was never committed, so the
/// engine answers `GetVoid` for it, and the row stays legal without being
/// named. What is refused is the other thing — a recid the log mentions that
/// the engine still ANSWERS for, and that the manifest describes nowhere.
fn assert_every_logged_recid_is_classified<S: Store>(
    s: &S,
    sample: &SampleV2,
    fixture: &str,
    named_rows: &[&RecidRow],
    ctx: &str,
) {
    let named: BTreeSet<u64> = named_rows.iter().map(|r| r.recid).collect();
    let mut mentioned: BTreeSet<u64> = BTreeSet::new();
    for f in sample.manifest.files_of(fixture) {
        let where_ = format!("{}/{}", f.fixture, f.rel);
        for sec in decode(sample.bytes_of(f), &where_).sections {
            if sec.tag == TAG_MARK {
                continue;
            }
            for e in entries(&sec, &where_) {
                mentioned.insert(e.recid as u64);
            }
        }
    }
    for recid in mentioned.difference(&named) {
        assert!(
            matches!(s.get(nz(*recid), &R), Err(DbError::GetVoid(x)) if x == *recid),
            "[{ctx}] recid {recid} appears in the log, the store still answers for it, and no \
             manifest recid row says what it is"
        );
    }
}

/// A scratch directory that removes itself, INCLUDING when the test panics.
///
/// The cleanup is a `Drop` and not a line at the end of each test, because a
/// line at the end of each test is exactly what a panic skips — and this suite's
/// mutation campaign makes every case panic on purpose, as does every
/// `assert_refused` in the batteries. Measured the hard way: three campaign runs
/// left **53,141** session directories in a 61 GB tmpfs and filled it, which
/// stopped being a tidiness question and became an outage that took the shell
/// down with it.
///
/// `Drop` runs during unwinding, so the guard cleans up on the panicking path
/// and the normal one alike. A test that wants the directory kept for diagnosis
/// can `std::mem::forget` it, which is a deliberate act and reads as one. The
/// explicit `remove_dir_all` at the end of each test is left where it stands:
/// it is now an early release rather than the only one, and deleting it would
/// be removing a line that is correct.
pub struct Session(PathBuf);

impl Drop for Session {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl std::ops::Deref for Session {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for Session {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

/// A fresh per-process scratch directory for one test, removed when it drops.
pub fn session_dir(tag: &str) -> Session {
    let d = std::env::temp_dir().join(format!("mapdb5_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("create session dir");
    Session(d)
}

/// Asserts `f` REFUSES its input, and that the refusal carries a message.
///
/// Two things here are load-bearing. The panic hook is silenced for the
/// duration, because `catch_unwind` still prints a "thread panicked" line by
/// default and a suite whose passing output is full of them trains the reader
/// to ignore exactly the lines that matter. And the message is required to be
/// non-empty: a refusal is only useful if it says which rule fired, and an
/// `assert!(cond)` with no message would satisfy a bare "it panicked" check
/// while telling a future reader nothing.
pub fn assert_refused(what: &str, f: impl FnOnce() + std::panic::UnwindSafe) {
    match red_of(f) {
        None => panic!("accepted {what}"),
        Some(msg) => assert!(!msg.is_empty(), "the refusal of {what} carried no message"),
    }
}

/// The panic message `f` produced, or `None` if it returned.
///
/// A refusal this harness EXPECTS prints nothing: `catch_unwind` still emits a
/// "thread panicked" line by default, and a suite whose passing output is full
/// of them trains the reader to ignore exactly the lines that matter.
///
/// **The silencing is per THREAD, not per call, and that is a defect this slice
/// found by tripping over it.** The obvious implementation — `take_hook`,
/// install a no-op, run, put the old one back — swaps a hook that is
/// PROCESS-GLOBAL while cargo runs the cases in one binary concurrently. So one
/// test's expected refusal silences another test's genuine failure, and the
/// second reports as a bare `test ... FAILED` with no message at all. C5r's
/// mutation campaign is what surfaced it: twenty mutants were killed for
/// reasons the runner could not read, which is a campaign that cannot check its
/// own claim. One hook, installed once, consulting a thread-local flag and
/// delegating to the default hook otherwise.
pub fn red_of(f: impl FnOnce() + std::panic::UnwindSafe) -> Option<String> {
    install_quiet_hook();
    // SAVE and restore, rather than set true / set false. A `red_of` inside a
    // `red_of` would otherwise clear the outer call's quiet state on its way
    // out, and the outer call's expected panic would print. Nothing nests today
    // and this is what stops the first thing that does from being a puzzle —
    // round 1's reviewer named it as the next thing they would attack.
    let outer = EXPECTING.with(|e| e.replace(true));
    let outcome = std::panic::catch_unwind(f);
    EXPECTING.with(|e| e.set(outer));
    match outcome {
        Ok(()) => None,
        Err(e) => Some(
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<a panic with no message>".to_string()),
        ),
    }
}

thread_local! {
    /// Whether THIS thread is inside a [`red_of`] and expects the panic it is
    /// about to see.
    static EXPECTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn install_quiet_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !EXPECTING.with(|e| e.get()) {
                default(info);
            }
        }));
    });
}

/// [`assert_refused`] specialised to the manifest reader.
pub fn assert_manifest_refused(what: &str, manifest: &str) {
    assert_refused(&format!("the manifest with {what}"), || {
        parse(manifest);
    });
}
