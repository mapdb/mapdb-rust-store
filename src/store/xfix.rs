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
use mapdb_rust_store::store::{Recid, Store, StoreWAL};
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

// --- schema v1 ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct V1File {
    pub fixture: String,
    pub rel: String,
    pub raw_len: u64,
    pub raw_sha: String,
    pub gz_sha: String,
}

/// `expect <fid> <engine> <verdict> <opener> <placeAs> <openArg>` — seven
/// fields, and the SAME arity as a v2 `expect` row with different columns in
/// them. That collision is why the version line is a hard dispatch and not a
/// hint; see [`parse`].
#[derive(Clone, Debug)]
pub struct V1Expect {
    pub fixture: String,
    pub engine: String,
    pub verdict: String,
    pub opener: String,
    pub place_as: String,
    pub open_arg: String,
}

#[derive(Default, Debug)]
pub struct V1 {
    pub fixture_kinds: BTreeMap<String, String>,
    pub files: Vec<V1File>,
    pub expects: Vec<V1Expect>,
    pub recids: Vec<RecidRow>,
}

impl V1 {
    pub fn files_of(&self, fixture: &str) -> Vec<&V1File> {
        self.files.iter().filter(|f| f.fixture == fixture).collect()
    }

    pub fn recids_of(&self, fixture: &str) -> Vec<&RecidRow> {
        self.recids
            .iter()
            .filter(|r| r.fixture == fixture)
            .collect()
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

#[derive(Default, Debug)]
pub struct V2 {
    pub fixture_kinds: BTreeMap<String, String>,
    pub files: Vec<V2File>,
    pub expects: Vec<V2Expect>,
    pub posts: Vec<V2Post>,
    pub recids: Vec<RecidRow>,
}

impl V2 {
    pub fn files_of(&self, fixture: &str) -> Vec<&V2File> {
        self.files.iter().filter(|f| f.fixture == fixture).collect()
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

/// The parsed manifest, tagged by the schema version its first line declared.
#[derive(Debug)]
pub enum Loaded {
    V1(V1),
    V2(V2),
}

impl Loaded {
    pub fn version(&self) -> u32 {
        match self {
            Loaded::V1(_) => 1,
            Loaded::V2(_) => 2,
        }
    }

    pub fn v1(&self) -> &V1 {
        match self {
            Loaded::V1(m) => m,
            Loaded::V2(_) => panic!("manifest is schema v2, not v1"),
        }
    }

    pub fn v2(&self) -> &V2 {
        match self {
            Loaded::V2(m) => m,
            Loaded::V1(_) => panic!("manifest is schema v1, not v2"),
        }
    }
}

/// Dispatches on the version line, then parses with the grammar that line names.
///
/// **The two grammars collide on arity.** A v1 `expect` row and a v2 `expect`
/// row both have seven fields, and v1's third column is a verdict where v2's is
/// a mode. Guessing the schema from a row's shape would therefore read
/// `accept` as a mode and `wal3` as a verdict without any arity check firing,
/// so the version line is authoritative and an unknown version is refused
/// rather than assumed to be the newest.
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
        "1" => Loaded::V1(parse_v1(&rest)),
        "2" => Loaded::V2(parse_v2(&rest)),
        other => panic!(
            "unsupported manifest schema version {other} — this reader speaks 1 and 2, and \
             refuses rather than guessing: the two grammars share row arities, so a newer \
             schema would be misread field by field without a single check firing"
        ),
    }
}

fn parse_v1(lines: &[&str]) -> V1 {
    let mut m = V1::default();
    for line in lines {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "version" => panic!("a second version row: {line}"),
            "fixture" => {
                arity(&t, 5, line);
                check(
                    m.fixture_kinds
                        .insert(t[1].to_string(), t[2].to_string())
                        .is_none(),
                    || format!("duplicate fixture row for {}: {line}", t[1]),
                );
            }
            "file" => {
                arity(&t, 6, line);
                let f = V1File {
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
            "expect" => {
                arity(&t, 7, line);
                let e = V1Expect {
                    fixture: t[1].to_string(),
                    engine: one_of(t[2], &ENGINES, "engine", line),
                    verdict: one_of(t[3], &["accept", "reject"], "verdict", line),
                    opener: t[4].to_string(),
                    place_as: rel_name(t[5], line),
                    open_arg: rel_name(t[6], line),
                };
                // A v1 cell is identified by (fixture, engine, opener, placeAs),
                // NOT by (fixture, engine): the live tree has both a `direct`
                // and a `wal` cell for the same engine on `wal-v1-rust-tail`,
                // which is exactly what the v1 `opener` column is for. A
                // narrower key rejects the real manifest — it did, here, before
                // the live data corrected it.
                for prior in &m.expects {
                    check(
                        !(prior.fixture == e.fixture
                            && prior.engine == e.engine
                            && prior.opener == e.opener
                            && prior.place_as == e.place_as),
                        || {
                            format!(
                                "duplicate expect row for {}/{}/{}: {line}",
                                e.fixture, e.engine, e.opener
                            )
                        },
                    );
                }
                m.expects.push(e);
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
                // provenance for a derived reject image; nothing here executes it.
            }
            other => panic!("unknown v1 manifest row type {other:?}: {line}"),
        }
    }
    check(!m.files.is_empty(), || {
        "a v1 manifest with no file rows".to_string()
    });
    m
}

fn parse_v2(lines: &[&str]) -> V2 {
    let mut m = V2::default();
    for line in lines {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "version" => panic!("a second version row: {line}"),
            "fixture" => {
                arity(&t, 5, line);
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
            "expect" => {
                arity(&t, 7, line);
                let e = V2Expect {
                    fixture: t[1].to_string(),
                    engine: one_of(t[2], &ENGINES, "engine", line),
                    mode: one_of(t[3], &MODES, "mode", line),
                    verdict: one_of(t[4], &["accept", "reject"], "verdict", line),
                    opener: t[5].to_string(),
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
            "bytes" => {
                arity(&t, 7, line);
                panic!(
                    "a v2 `bytes` row, which this reader does not execute yet (C4 introduces \
                     the derived fixtures it describes): {line}"
                );
            }
            other => panic!("unknown v2 manifest row type {other:?}: {line}"),
        }
    }
    check(!m.files.is_empty(), || {
        "a v2 manifest with no file rows".to_string()
    });
    m
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

pub fn v1_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/xfixtures")
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
    let loaded = parse(&read_root_text(root, "MANIFEST.tsv"));
    let manifest = match loaded {
        Loaded::V2(m) => m,
        Loaded::V1(_) => panic!("{} is schema v1, not v2", root.display()),
    };
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
/// `payload(id, len)` is invertible from its first byte, so rebuilding it from
/// the recovered id and comparing is a total check that these really are bytes
/// this corpus issued — which they cannot be if the entry stream was framed
/// wrongly. A decoder that read the packed-long continuation bit the wrong way
/// round lands mid-payload and fails HERE, rather than producing a plausible
/// file that disagrees with java's for reasons nobody can localise.
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

/// The witness that these content bytes really are bytes this corpus issued.
///
/// `payload(id, len)[i] == (i*131 + id) & 0xff` is invertible from its first
/// byte, so rebuilding it from the recovered id and comparing is total. A
/// decoder that read the packed-long continuation bit the wrong way round
/// lands mid-payload and fails HERE, rather than producing a plausible file
/// that disagrees with java's for reasons nobody can localise. Zero-length
/// content carries no id and is vacuously fine.
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
    if cap == 0 {
        return;
    }
    assert!(
        cap >= 4 + len as i64 && cap & 15 == 0,
        "{where_}: cap {cap} is not a valid capacity for {len} content bytes \
         (must be 16-aligned and at least {})",
        4 + len
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

/// The two-sided D6 post-state rule.
///
/// One side is the obvious one: every file a `post` row names must be in the
/// state that row declares. The other side is the amendment that makes the
/// rule total — **files not named by a `post` row are implicitly `unchanged`**
/// — so an unnamed input must still be there byte for byte, and a file that is
/// neither an input nor named must not exist at all. Without the second side a
/// cell that deleted a segment and wrote three new ones would pass by saying
/// nothing about them.
pub fn assert_post_state(
    cell: &Path,
    before: &BTreeMap<String, Vec<u8>>,
    posts: &[&V2Post],
    ctx: &str,
) {
    let named: BTreeSet<&str> = posts.iter().map(|p| p.rel.as_str()).collect();
    for p in posts {
        let path = cell.join(&p.rel);
        let now = std::fs::read(&path).ok();
        match p.verb.as_str() {
            "deleted" => assert!(
                now.is_none(),
                "[{ctx}] {} must not exist after the cell",
                p.rel
            ),
            "unchanged" => {
                let was = before.get(&p.rel).unwrap_or_else(|| {
                    panic!(
                        "[{ctx}] `unchanged` names {}, which was never an input",
                        p.rel
                    )
                });
                assert_eq!(
                    now.as_ref(),
                    Some(was),
                    "[{ctx}] {} must be byte-unchanged",
                    p.rel
                );
            }
            verb => {
                if verb == "truncated" || verb == "modified" {
                    assert!(
                        before.contains_key(&p.rel),
                        "[{ctx}] `{verb}` names {}, which was never an input — only `created` \
                         may name a file the cell did not start with",
                        p.rel
                    );
                }
                let now =
                    now.unwrap_or_else(|| panic!("[{ctx}] {} must exist after the cell", p.rel));
                assert_eq!(
                    now.len() as u64,
                    p.len.unwrap(),
                    "[{ctx}] {} length after the cell",
                    p.rel
                );
                assert_eq!(
                    sha256_hex(&now),
                    *p.sha.as_ref().unwrap(),
                    "[{ctx}] {} content after the cell",
                    p.rel
                );
            }
        }
    }
    for (rel, was) in before {
        if named.contains(rel.as_str()) {
            continue;
        }
        let now = std::fs::read(cell.join(rel)).unwrap_or_else(|e| {
            panic!("[{ctx}] {rel} is named by no post row, so it must still be there: {e}")
        });
        assert_eq!(&now, was, "[{ctx}] unnamed input {rel} changed");
    }
    for name in dir_entries(cell) {
        assert!(
            before.contains_key(&name) || named.contains(name.as_str()),
            "[{ctx}] {name} is neither an input nor named by a post row"
        );
    }
}

/// Runs every schema-v2 cell addressed to this engine in `mode`, and asserts
/// the set that ran is **exactly** the set the `fixture` rows call for.
///
/// `open` receives the cell's base path. The caller supplies it because that is
/// the one thing the two halves do not share: the integration test opens
/// read-write through the public `StoreWAL::open`, and the in-crate module
/// opens read-only through `open_cfg`, which is `pub(crate)` and stays that
/// way (C-D3).
pub fn run_v2_cells(
    sample: &SampleV2,
    mode: &str,
    session: &Path,
    open: &dyn Fn(&Path) -> Result<StoreWAL>,
) {
    let m = &sample.manifest;
    let want: BTreeSet<String> = m.declared_fixtures();
    assert!(!want.is_empty(), "the v2 sample declares no fixtures");

    let mut ran: BTreeSet<String> = BTreeSet::new();
    for (i, e) in m.expects.iter().enumerate() {
        if e.engine != ENGINE || e.mode != mode {
            continue;
        }
        let ctx = format!(
            "v2 cell {i}: fixture={} mode={} verdict={} opener={} openArg={}",
            e.fixture, e.mode, e.verdict, e.opener, e.open_arg
        );
        assert_eq!(
            e.opener, "wal3",
            "[{ctx}] the only v2 opener this reader executes is `wal3`"
        );
        let files = m.files_of(&e.fixture);
        assert!(!files.is_empty(), "[{ctx}] fixture has no file rows");

        let cell = session.join(format!("v2-{mode}-{i}"));
        let _ = std::fs::remove_dir_all(&cell);
        std::fs::create_dir_all(&cell).unwrap();
        let mut before: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for f in &files {
            let bytes = sample.bytes_of(f).to_vec();
            std::fs::write(cell.join(&f.rel), &bytes).unwrap();
            before.insert(f.rel.clone(), bytes);
        }

        let base = cell.join(&e.open_arg);
        match e.verdict.as_str() {
            "accept" => {
                let s = open(&base)
                    .unwrap_or_else(|err| panic!("[{ctx}] accept cell failed to open: {err}"));
                let recids = m.recids_of(&e.fixture);
                assert_reader_contract(&s, &recids, &ctx);
                assert_every_logged_recid_is_classified(&s, sample, &e.fixture, &recids, &ctx);
                s.close().unwrap();
            }
            "reject" => match open(&base) {
                Err(DbError::DataCorruption(_)) => {}
                Err(other) => panic!("[{ctx}] expected DataCorruption, got: {other}"),
                Ok(s) => {
                    let _ = s.close();
                    panic!("[{ctx}] reject cell opened successfully");
                }
            },
            v => panic!("[{ctx}] unsupported verdict {v}"),
        }

        assert_post_state(&cell, &before, &m.posts_of(&e.fixture, ENGINE, mode), &ctx);
        assert!(
            ran.insert(e.fixture.clone()),
            "[{ctx}] two {mode} cells for the same fixture"
        );
        std::fs::remove_dir_all(&cell).unwrap();
    }
    assert_eq!(
        ran, want,
        "the {ENGINE}/{mode} cells that ran are not the ones the fixture rows call for"
    );
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

/// A fresh per-process scratch directory for one test.
pub fn session_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mapdb5_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("create session dir");
    d
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
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    match outcome {
        Ok(()) => panic!("accepted {what}"),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()));
            assert!(
                msg.is_some_and(|m| !m.is_empty()),
                "the refusal of {what} carried no message"
            );
        }
    }
}

/// [`assert_refused`] specialised to the manifest reader.
pub fn assert_manifest_refused(what: &str, manifest: &str) {
    assert_refused(&format!("the manifest with {what}"), || {
        parse(manifest);
    });
}
