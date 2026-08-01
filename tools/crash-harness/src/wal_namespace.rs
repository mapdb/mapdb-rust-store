//! An **independent** reader of the WAL format v3 segment namespace, and the
//! invariants a crash image and a recovered store must satisfy.
//!
//! Independent is the point. The crash tier's contents oracle already holds the
//! store to a model it computes itself; before slice A2 nothing held the
//! store's *files* to anything, so rotate, the forced `'K'` + unlink, create
//! residue and the recovery successor were exercised but never asserted — the
//! harness would have passed a recovery that reused a segment name, resurrected
//! a retired one, or left a residue file behind. This module re-derives the
//! namespace from the format description rather than calling into
//! `mapdb_rust_store`, so a defect in the store's own enumerator cannot make it
//! agree with itself. It deliberately duplicates ~40 lines of format knowledge:
//!
//! ```text
//! name    := <base> ".wal." <16 lowercase hex digits of segmentSeq>
//! header  := magic "MDBS.WAL"(8) | version i32 = 3 | flags i32 = 0
//!          | segmentSeq i64 | firstLsn i64 | headerCrc i32        // 36 bytes
//! ```
//!
//! All integers big-endian; `headerCrc` is zlib CRC-32 over header bytes
//! `[0, 32)`. [`scan`] reads those headers and nothing else. [`scan_with_marks`]
//! additionally walks each segment's valid section prefix for `'K'` marks,
//! because the floor a forced mark obliges recovery to reach cannot be derived
//! from names alone; that walk mirrors the reference's acceptance predicate row
//! for row and is documented at [`highest_mark`]. Everything else about
//! sections — what they contain, whether replay applied them — stays the
//! store's business, and the store's own recovery tests own it.
//!
//! # What the namespace is allowed to do
//!
//! Only three things ever change the file set, which is why so few files can
//! legitimately move between two observations:
//!
//! - **create** (rotate, or R7's post-truncate rotation, or N1's first
//!   segment) adds exactly one name, always strictly ABOVE every name ever
//!   observed — W6 burns sequence numbers and never reuses one, residue
//!   included;
//! - **unlinkThrough** (phase 3 of a cleaning cycle, and recovery's R5 replay
//!   of one) removes a low run: every removed name is below every survivor;
//! - **residue deletion** (R2) removes the HIGHEST name, and only when its
//!   header is unreadable — a create that crashed between `CREATE_NEW` and the
//!   forced header.
//!
//! Everything an implementation could get wrong here — reusing a burnt name,
//! unlinking a segment the mark did not authorize, leaving a residue file for
//! the next open to trip over, failing to finish a partially applied unlink —
//! violates one of those three, and [`Namespace::check_recovered`] is where
//! that shows up.

use std::collections::BTreeSet;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// magic(8) + version(4) + flags(4) + segmentSeq(8) + firstLsn(8) + headerCrc(4).
pub const SEG_HDR_LEN: usize = 36;
/// Header bytes covered by `headerCrc`.
const SEG_HDR_CRC_LEN: usize = 32;
const MAGIC: &[u8; 8] = b"MDBS.WAL";
const FORMAT_VERSION: i32 = 3;

/// tag(1) + lsn(8) + bodyLen(8) + hdrCrc(4) + bodyCrc(4).
const SEC_HDR_LEN: usize = 25;
/// Section-header bytes covered by `hdrCrc`.
const SEC_HDR_CRC_LEN: usize = 17;
const TAG_SECTION: u8 = b'S';
const TAG_IMAGE: u8 = b'C';
const TAG_MARK: u8 = b'K';
/// cleanedThroughSeq(8) + logStartLsn(8).
const MARK_BODY_LEN: u64 = 16;

/// `validTag` (`StoreWAL.java:717-719`). A tag outside this set fails S3
/// together with the header CRC, so the walk stops there exactly as Java does.
fn valid_tag(tag: u8) -> bool {
    matches!(tag, TAG_SECTION | TAG_IMAGE | TAG_MARK)
}

/// Buffer the body walk streams through. Fixed and stack-resident: a section
/// body is `bodyLen` wide and `bodyLen` comes off the disk, so it must never
/// size an allocation.
const BODY_CHUNK: usize = 8192;
/// Conservative ceilings on ONE segment's walk. Neither is a format rule — a
/// stop only ever lowers the derived floor, which weakens the assertion and can
/// never fail a correct store — but without them a corrupt `bodyLen` chain or a
/// legitimately enormous segment turns a crash verdict into a checker timeout,
/// which reads as a product failure. They bound work jointly and neither alone
/// is the whole story: the byte member bounds the admitted on-disk prefix —
/// checked against each section's END before its body is streamed, which is the
/// only check that constrains a single huge `bodyLen` — and the section member
/// bounds the fixed per-section overhead (the CRC domain, a mark's 16-byte
/// reread) that the byte member does not count. The byte member is D8's DEFAULT
/// `segmentBytes`, so a default-configured store is always walked whole.
/// `(bytes, sections)`. The ONE place the production walk's ceilings are
/// written, so a test can pin them: `highest_mark` destructures this and has no
/// literals of its own.
const WALK_LIMITS: (u64, u64) = (64 << 20, 1 << 20);

/// zlib CRC-32, bitwise and incremental. Deliberately not the store's
/// `crc32fast`: an oracle that imported the same checksum implementation as the
/// code under test would agree with it by construction. Incremental because the
/// CRC domains here are `segment header ‖ offset ‖ bytes` and a body is read in
/// bounded chunks — neither may be assembled into a buffer first.
struct Crc32(u32);

impl Crc32 {
    fn new() -> Self {
        Crc32(0xFFFF_FFFF)
    }
    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u32;
            for _ in 0..8 {
                self.0 = if self.0 & 1 != 0 {
                    (self.0 >> 1) ^ 0xEDB8_8320
                } else {
                    self.0 >> 1
                };
            }
        }
    }
    fn finish(self) -> u32 {
        !self.0
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(bytes);
    c.finish()
}

fn be32(b: &[u8], off: usize) -> i32 {
    i32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn be64(b: &[u8], off: usize) -> i64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[off..off + 8]);
    i64::from_be_bytes(v)
}

/// One enumerated segment file: a name that matched the grammar exactly, plus
/// the verdict of the header table applied to its first 36 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentInfo {
    /// Sequence number parsed from the NAME (the enumeration key).
    pub seq: i64,
    /// `firstLsn` from the header — meaningless unless `bad` is `None`.
    pub first_lsn: i64,
    pub len: u64,
    /// Highest `cleanedThroughSeq` authorized by a `'K'` in this segment's
    /// valid section prefix, if any. Only computed for a valid header.
    pub mark: Option<i64>,
    /// `None` when every header row passed; otherwise a stable reason code.
    pub bad: Option<&'static str>,
}

/// The whole namespace at one instant: every name under the base, classified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Namespace {
    pub base: PathBuf,
    /// Grammar-matching segment names, ascending by seq.
    pub segs: Vec<SegmentInfo>,
    /// Names sharing the `<base>.wal.` prefix that are NOT segments (the store
    /// ignores them; the harness reports them, because under its own workload
    /// nothing should ever create one).
    pub foreign: Vec<String>,
    /// A `<base>.wal` entry — v1's single log file (D1/N6).
    pub legacy_wal: bool,
    /// A `<base>.ckpt` entry — v1's rename-checkpoint (D1).
    pub legacy_ckpt: bool,
    /// Whether this scan walked section prefixes for `'K'` marks. Recorded
    /// rather than assumed: [`Namespace::authorized_floor`] answers `None` both
    /// for "no mark" and for "never looked", and the difference is the whole
    /// unfinished-unlink rule. [`Namespace::check_recovered`] SKIPS the floor
    /// rule when its `pre` never looked — asserting a floor nobody derived
    /// would be worse — so the caller that needs the rule is the one that must
    /// insist, and the crash checker does exactly that before it opens the
    /// store.
    pub marks_scanned: bool,
}

impl SegmentInfo {
    pub fn ok(&self) -> bool {
        self.bad.is_none()
    }
}

/// Reads the 36-byte header and applies rows H1-H9 in the reference's order:
/// the semantic rows are reached only after the CRC passes, so editing a field
/// without resealing is a CRC verdict rather than a semantic one.
///
/// `want_marks` additionally walks the segment's valid section prefix for `'K'`
/// marks. That is the only thing here that reads past 36 bytes, so it is opt-in:
/// the workload scans the namespace at every group boundary and needs names
/// only.
fn classify_header(path: &Path, name_seq: i64, want_marks: bool) -> Result<SegmentInfo, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len();
    let mut info = SegmentInfo {
        seq: name_seq,
        first_lsn: 0,
        len,
        mark: None,
        bad: None,
    };
    // 36 bytes, positionally — NEVER the whole file. A segment is a tuning knob
    // wide (`StoreWAL::set_segment_bytes`) and a crash image may hold hundreds
    // of them; reading each one whole to look at its header would let a large
    // or corrupt image exhaust the checker before it can report a verdict.
    if len == 0 {
        info.bad = Some("h1-empty");
        return Ok(info);
    }
    if len < SEG_HDR_LEN as u64 {
        info.bad = Some("h2-short");
        return Ok(info);
    }
    let mut hdr_buf = [0u8; SEG_HDR_LEN];
    file.read_exact_at(&mut hdr_buf, 0)
        .map_err(|e| format!("read header {}: {e}", path.display()))?;
    let hdr = &hdr_buf[..];
    if crc32(&hdr[..SEG_HDR_CRC_LEN]) as i32 != be32(hdr, 32) {
        info.bad = Some("h3-hdr-crc");
        return Ok(info);
    }
    if &hdr[..8] != MAGIC {
        info.bad = Some("h4-magic");
        return Ok(info);
    }
    if be32(hdr, 8) != FORMAT_VERSION {
        info.bad = Some("h5-version");
        return Ok(info);
    }
    if be32(hdr, 12) != 0 {
        info.bad = Some("h6-flags");
        return Ok(info);
    }
    if be64(hdr, 16) != name_seq {
        info.bad = Some("h7-seq");
        return Ok(info);
    }
    info.first_lsn = be64(hdr, 24);
    if info.first_lsn <= 0 {
        info.bad = Some("h9-first-lsn");
        return Ok(info);
    }
    if want_marks {
        info.mark = highest_mark(&file, &hdr_buf, name_seq, len);
    }
    Ok(info)
}

/// Walks this segment's VALID SECTION PREFIX and returns the highest
/// `cleanedThroughSeq` any `'K'` mark in it authorizes.
///
/// This is the one place the harness reads past a header, and it exists because
/// the requirement it serves cannot be met without it: a recovery that simply
/// *ignores* a forced `'K'` — leaving every segment the mark retired on disk —
/// removes nothing, so a rule about the shape of what disappeared sees nothing
/// wrong. The authorized floor has to come from the mark itself.
///
/// It mirrors `scanSegment`'s acceptance predicate row for row, and errs in ONE
/// direction: **any doubt ends the walk**. Reading too little yields a lower
/// floor and therefore a weaker assertion; reading too much would demand a floor
/// recovery correctly did not reach, and fail a green CI on a correct store.
/// Every early return below is that stop, and it is always safe because the rule
/// this feeds is one-sided (`post.lo() > floor`): a floor below Java's still
/// holds against a store that retired more.
///
/// Mirroring the predicate is not optional, because a section Java REFUSES must
/// not raise this floor. The rows, in Java's order
/// (`StoreWAL.java:617-697`), each of which stops the walk here:
///
/// - **S3** header CRC over `segment header ‖ offset ‖ hdr[0,17)`, and
///   `validTag`. Java folds the tag into the same verdict, so a `'X'` section
///   with a valid CRC ends the prefix rather than being stepped over.
/// - **S5** `0 <= bodyLen <= len - bodyStart`.
/// - **S4** the body CRC of **every** section, not just a mark's. Skipping it
///   would let a torn ordinary body be stepped over and a later mark counted,
///   which is a floor Java never derives.
/// - **S2/S9** LSN density. Java restarts density at each segment boundary and
///   only applies it once `lastLsn != 0`, so it accepts a first section whose
///   LSN disagrees with the header and defers that to R4's self check. Demanding
///   the header's `firstLsn` here instead is strictly more conservative, and it
///   is also how the deliberately unmodelled `lsn == 0` leading-run edge stops.
/// - **S8** a mark's body is exactly 16 bytes, `through > 0`, and
///   `0 < logStart <= ` the mark's own LSN.
/// - **K4** `through < seg.seq`: a mark may not authorize removing its own
///   segment. Java HOLDS such a segment (`StoreWAL.java:691-694`) and the floor
///   it carries never reaches R5 — so counting it here is the one construction
///   that makes this oracle fail a correct store, and it is why every row above
///   is checked rather than assumed.
///
/// Sound because `cleaned_through` is exactly `max` over each segment's valid
/// prefix (`wal_recover.rs:1241-1254`), taken before R4's adjudication and used
/// verbatim by R5's `unlink_through` (`:1289`).
fn highest_mark(file: &File, seg_hdr: &[u8; SEG_HDR_LEN], seg_seq: i64, len: u64) -> Option<i64> {
    let (max_bytes, max_sections) = WALK_LIMITS;
    highest_mark_bounded(file, seg_hdr, seg_seq, len, max_bytes, max_sections)
}

/// [`highest_mark`] with the ceilings injected, so a test can reach them without
/// building a 64 MiB fixture.
fn highest_mark_bounded(
    file: &File,
    seg_hdr: &[u8; SEG_HDR_LEN],
    seg_seq: i64,
    len: u64,
    max_bytes: u64,
    max_sections: u64,
) -> Option<i64> {
    let mut best: Option<i64> = None;
    let mut off = SEG_HDR_LEN as u64;
    let mut expect_lsn = be64(seg_hdr, 24);
    let mut sections: u64 = 0;
    let mut buf = [0u8; BODY_CHUNK];
    loop {
        sections += 1;
        if sections > max_sections {
            return best; // conservative ceiling, never a verdict
        }
        if len - off < SEC_HDR_LEN as u64 {
            return best; // no room for another section header
        }
        let mut hdr = [0u8; SEC_HDR_LEN];
        if file.read_exact_at(&mut hdr, off).is_err() {
            return best; // short read: the prefix ends here
        }
        // S3. The header CRC is domain-bound to this segment's identity AND this
        // offset, so a section copied from elsewhere fails here exactly as it
        // does in recovery.
        let tag = hdr[0];
        let mut c = Crc32::new();
        c.update(seg_hdr);
        c.update(&off.to_be_bytes());
        c.update(&hdr[..SEC_HDR_CRC_LEN]);
        if c.finish() as i32 != be32(&hdr, 17) || !valid_tag(tag) {
            return best;
        }
        // S5.
        let body_len = be64(&hdr, 9);
        if body_len < 0 {
            return best;
        }
        let body_len = body_len as u64;
        let body_start = off + SEC_HDR_LEN as u64; // <= len, checked above
        let Some(body_end) = body_start.checked_add(body_len) else {
            return best;
        };
        if body_end > len {
            return best; // the body is not all there
        }
        // The byte ceiling has to cover the bytes this section will make us
        // HASH, not merely where it starts: `bodyLen` is a disk value that Java
        // accepts for any width that fits the file, so one CRC-valid header
        // claiming a gigabyte would otherwise stream a gigabyte before the
        // ceiling was consulted again.
        if body_end - SEG_HDR_LEN as u64 > max_bytes {
            return best;
        }
        // S4, streamed: the body is `bodyLen` wide and `bodyLen` came off the
        // disk, so it is hashed in fixed chunks and never buffered whole.
        let mut c = Crc32::new();
        c.update(seg_hdr);
        c.update(&off.to_be_bytes());
        let mut p = body_start;
        while p < body_end {
            let n = ((body_end - p) as usize).min(buf.len());
            if file.read_exact_at(&mut buf[..n], p).is_err() {
                return best;
            }
            c.update(&buf[..n]);
            p += n as u64;
        }
        if c.finish() as i32 != be32(&hdr, 21) {
            return best;
        }
        // S2/S9.
        let lsn = be64(&hdr, 1);
        if lsn != expect_lsn {
            return best;
        }
        if tag == TAG_MARK {
            // S8, then K4.
            if body_len != MARK_BODY_LEN {
                return best;
            }
            let mut body = [0u8; MARK_BODY_LEN as usize];
            if file.read_exact_at(&mut body, body_start).is_err() {
                return best;
            }
            let (through, log_start) = (be64(&body, 0), be64(&body, 8));
            if through <= 0 || log_start <= 0 || log_start > lsn || through >= seg_seq {
                return best;
            }
            best = Some(best.map_or(through, |b: i64| b.max(through)));
        }
        // The frozen reference opens an LSN-exhausted image by wrapping
        // (`wal_recover.rs:1313-1324` records the reachable construction), so
        // the exhausted edge is a legal image the oracle must not panic on. It
        // is also the end of any prefix this walk can vouch for.
        let Some(next) = expect_lsn.checked_add(1) else {
            return best;
        };
        expect_lsn = next;
        off = body_end;
    }
}

/// Enumerates the namespace of `base` (the path handed to `DB::make_wal`).
///
/// Reads the directory once and stats/reads only the 36-byte headers, so it is
/// cheap enough for the workload to call at every group boundary. The resulting
/// [`Namespace`] therefore carries no marks and cannot answer the floor rule;
/// use [`scan_with_marks`] where that is needed.
pub fn scan(base: &Path) -> Result<Namespace, String> {
    scan_inner(base, false)
}

/// [`scan`], plus a walk of each valid segment's section prefix for the highest
/// `'K'` it authorizes. Reads the segment bodies, so it is for the crash
/// checker's one pre-open image scan, not for a hot loop.
pub fn scan_with_marks(base: &Path) -> Result<Namespace, String> {
    scan_inner(base, true)
}

fn scan_inner(base: &Path, want_marks: bool) -> Result<Namespace, String> {
    let dir = base
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let stem = base
        .file_name()
        .ok_or_else(|| format!("base {} has no file name", base.display()))?
        .to_str()
        .ok_or_else(|| format!("base {} is not utf-8", base.display()))?
        .to_string();
    let seg_prefix = format!("{stem}.wal.");
    let mut ns = Namespace {
        base: base.to_path_buf(),
        segs: Vec::new(),
        foreign: Vec::new(),
        legacy_wal: false,
        legacy_ckpt: false,
        marks_scanned: want_marks,
    };
    for entry in std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name == format!("{stem}.wal") {
            ns.legacy_wal = true;
            continue;
        }
        if name == format!("{stem}.ckpt") {
            ns.legacy_ckpt = true;
            continue;
        }
        let Some(tail) = name.strip_prefix(&seg_prefix) else {
            continue;
        };
        // The grammar: EXACTLY 16 lowercase hex digits parsing to a
        // non-negative i64. Anything else is a name the store ignores.
        let seq = match (
            tail.len() == 16
                && tail
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            i64::from_str_radix(tail, 16),
        ) {
            (true, Ok(seq)) if seq >= 0 => seq,
            _ => {
                ns.foreign.push(name.to_string());
                continue;
            }
        };
        // A directory or symlink at a segment name is not a segment; the store
        // ignores it, and under this workload nothing creates one.
        if !entry
            .file_type()
            .map_err(|e| format!("file_type {name}: {e}"))?
            .is_file()
        {
            ns.foreign.push(name.to_string());
            continue;
        }
        ns.segs
            .push(classify_header(&entry.path(), seq, want_marks)?);
    }
    ns.segs.sort_by_key(|s| s.seq);
    Ok(ns)
}

impl Namespace {
    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }
    pub fn lo(&self) -> i64 {
        self.segs.first().map(|s| s.seq).unwrap_or(0)
    }
    pub fn hi(&self) -> i64 {
        self.segs.last().map(|s| s.seq).unwrap_or(0)
    }
    pub fn count(&self) -> u64 {
        self.segs.len() as u64
    }
    fn seqs(&self) -> BTreeSet<i64> {
        self.segs.iter().map(|s| s.seq).collect()
    }
    /// Missing sequence numbers inside `[lo, hi]`. A gap is legitimate — a
    /// burnt residue name leaves one, and a crash midway through an unlink run
    /// leaves one — so this is coverage evidence, never a verdict on its own.
    pub fn gaps(&self) -> u64 {
        if self.is_empty() {
            return 0;
        }
        (self.hi() - self.lo() + 1) as u64 - self.count()
    }
    pub fn bad(&self) -> Vec<&SegmentInfo> {
        self.segs.iter().filter(|s| !s.ok()).collect()
    }
    /// The highest `cleanedThroughSeq` any valid `'K'` in this set authorizes —
    /// the floor a recovery of this image is obliged to reach. Recovery
    /// computes the same reduction (`max` over each segment's valid prefix,
    /// `wal_recover.rs:1241-1254`) and feeds it to `unlink_through` (`:1289`);
    /// this one is deliberately never higher (see [`highest_mark`]).
    ///
    /// Always `None` for a [`scan`] that did not look — check `marks_scanned`
    /// before reading anything into the answer.
    pub fn authorized_floor(&self) -> Option<i64> {
        self.segs
            .iter()
            .filter(|s| s.ok())
            .filter_map(|s| s.mark)
            .max()
    }

    /// Invariants of a **crash image**, before anything opens it. Everything
    /// here must hold at every possible cut point, so nothing that depends on
    /// an operation having completed belongs in it.
    pub fn check_image(&self) -> Result<(), String> {
        // Non-strict chain: in an image the active segment may end in a torn
        // section, so a segment with bytes past its header need not hold a
        // valid one, and its successor may legitimately state the same LSN.
        self.check_common("crash image", false)?;
        // R2: an unreadable header is create-crash residue, which can only ever
        // be the highest name — a create writes the header before anything
        // below it can be superseded, and no other operation truncates a
        // segment to nothing.
        for b in self.bad() {
            if b.seq != self.hi() {
                return Err(format!(
                    "crash image: segment {:016x} has an unreadable header ({}) but is not the \
                     highest name ({:016x}) — only a crashed create may leave residue",
                    b.seq,
                    b.bad.unwrap_or("?"),
                    self.hi()
                ));
            }
        }
        Ok(())
    }

    /// Invariants of a **recovered store**, checked against the crash image it
    /// was recovered from. This is the half a cut point cannot excuse: recovery
    /// ran to completion, so every partially applied namespace operation must
    /// now be finished.
    pub fn check_recovered(&self, pre: &Namespace) -> Result<(), String> {
        // Strict chain: after recovery every non-final segment is fully valid
        // (a torn tail is only forgiven on the ACTIVE one, and R7 truncates
        // that one to its valid prefix), so a segment still longer than its
        // header holds at least one section and its successor must start above
        // it.
        self.check_common("recovered", true)?;
        // R2 completed: no residue survives a writable open.
        if let Some(b) = self.bad().first() {
            return Err(format!(
                "recovered: segment {:016x} still has an unreadable header ({}) — R2 must delete \
                 create-crash residue",
                b.seq,
                b.bad.unwrap_or("?")
            ));
        }
        // The floor a forced `'K'` in the image OBLIGES recovery to reach. This
        // is the half a shape test cannot see: a recovery that ignores the mark
        // entirely removes nothing, so "everything removed was a low run" holds
        // vacuously and the segments the mark retired sit there forever (K5/K8
        // make an unfinished unlink the NEXT open's job, not a licence to
        // forget it).
        // Only a scan that actually looked can be asked; `authorized_floor`
        // answers `None` for "no mark" and for "never looked" alike, and
        // silently taking the second for the first is how this rule would
        // become dead code without a test noticing.
        if pre.marks_scanned {
            if let Some(through) = pre.authorized_floor() {
                if self.lo() <= through {
                    return Err(format!(
                        "recovered: a valid 'K' authorizes retiring through {through:016x} but \
                         the lowest surviving segment is {:016x} — R5 must replay the unlink the \
                         crash interrupted",
                        self.lo()
                    ));
                }
            }
        }
        let (pre_seqs, post_seqs) = (pre.seqs(), self.seqs());
        // W6: a name is never reused. Anything recovery created is strictly
        // above every name the image held — and a name whose header was
        // unreadable is BURNT by that same rule, so recovery deleting the
        // residue at N and then creating a fresh segment at N is a reuse even
        // though the two sets have the same members. `created` is therefore
        // computed against the names that survived AS THEMSELVES.
        let pre_valid: BTreeSet<i64> = pre.segs.iter().filter(|s| s.ok()).map(|s| s.seq).collect();
        let created: Vec<i64> = post_seqs.difference(&pre_valid).copied().collect();
        if created.len() > 1 {
            return Err(format!(
                "recovered: {} new segments {:?} — an open creates at most one (N1's first \
                 segment, or R7's post-truncate rotation)",
                created.len(),
                created
            ));
        }
        if let Some(&new) = created.first() {
            if new <= pre.hi() {
                return Err(format!(
                    "recovered: new segment {new:016x} is not above the image's highest name \
                     {:016x} — W6 burns sequence numbers and never reuses one",
                    pre.hi()
                ));
            }
        }
        // unlinkThrough removes a LOW RUN: every retired name is below every
        // survivor. A survivor below a retired name would mean recovery
        // unlinked something the mark did not authorize, or replayed an unlink
        // out of order.
        let retired: Vec<i64> = pre_seqs.difference(&post_seqs).copied().collect();
        if let (Some(&max_retired), Some(&min_kept)) =
            (retired.iter().max(), post_seqs.iter().next())
        {
            if max_retired > min_kept {
                return Err(format!(
                    "recovered: segment {max_retired:016x} was removed while {min_kept:016x} \
                     survives — unlinkThrough removes a low run, never a hole"
                ));
            }
        }
        Ok(())
    }

    /// Hermeticity of the harness's own working directory. **This is not a
    /// namespace rule and is deliberately not reachable from one** — it is
    /// called by the crash checker as an environment precondition, before any
    /// verdict about the store.
    ///
    /// The frozen reference IGNORES everything that is not an exact segment
    /// name: uppercase-hex siblings, wrong-width names, and directories or
    /// symlinks at segment-shaped names all open perfectly well
    /// (`WalSegmentSet.java:279-310`; Appendix A.1 "everything else IGNORED").
    /// A namespace checker that refused them would be making a false claim
    /// about the FORMAT, and would fail CI on a legal store. `.ckpt` is the
    /// same: D1's sentinel is a PORTS rule, and Java's constructor tests only a
    /// regular `<base>.wal` (`WalSegmentSet.java:210-219`).
    ///
    /// What it is instead is a statement about this test. The smoke runner
    /// recreates its work root per campaign and the privileged runner gets a
    /// fresh filesystem per round, so nothing but the workload writes there and
    /// the workload creates none of these. One appearing means the round's
    /// environment is not what the oracle assumes, and every assertion
    /// downstream is about the wrong thing — a harness failure, reported as
    /// one.
    pub fn check_harness_environment(&self) -> Result<(), String> {
        if self.legacy_wal || self.legacy_ckpt {
            return Err(format!(
                "harness environment: the round's own directory holds a v1 artifact \
                 (<base>.wal={}, <base>.ckpt={}) — nothing in this test creates one, and D1 would \
                 refuse the next open",
                self.legacy_wal, self.legacy_ckpt
            ));
        }
        if !self.foreign.is_empty() {
            return Err(format!(
                "harness environment: the round's own directory holds names under the segment \
                 prefix that are not segments: {:?}. The store correctly IGNORES these; this test \
                 creates none, so their presence means something else wrote into the directory",
                self.foreign
            ));
        }
        Ok(())
    }

    /// The rows that hold at every instant, open or closed.
    fn check_common(&self, what: &str, strict_chain: bool) -> Result<(), String> {
        if self.is_empty() {
            return Err(format!(
                "{what}: no segments at all — an open always leaves at least one (N1 creates the \
                 first, K4 keeps the mark's own segment retained)"
            ));
        }
        // R1: seq 0 is reserved for "no clean mark" and is a corruption verdict
        // as a NAME, so a conforming writer never produces one.
        if self.lo() < 1 {
            return Err(format!(
                "{what}: segment name {:016x} — seq 0 is reserved",
                self.lo()
            ));
        }
        // The LSN chain, name order against header order. `firstLsn` is the LSN
        // rotation RESERVED for the first section of that segment, so it never
        // decreases — but two adjacent segments may legitimately state the SAME
        // LSN when the lower one never received it. That is not a curiosity: it
        // is what a cut between a rollover's create and the section it rolled
        // for leaves, and R7 then rotates past the empty segment to a successor
        // holding the identical `nextLsn`. H8 makes a header-only segment legal
        // at any position for exactly this reason. Equality is therefore only
        // refused where the lower segment demonstrably HOLDS something.
        let mut prev: Option<&SegmentInfo> = None;
        for s in self.segs.iter().filter(|s| s.ok()) {
            if let Some(p) = prev {
                let must_increase = strict_chain && p.len > SEG_HDR_LEN as u64;
                if s.first_lsn < p.first_lsn || (must_increase && s.first_lsn == p.first_lsn) {
                    return Err(format!(
                        "{what}: firstLsn {} with seq: {:016x} ({} bytes) states {} but {:016x} \
                         states {}",
                        if must_increase {
                            "does not increase"
                        } else {
                            "decreases"
                        },
                        p.seq,
                        p.len,
                        p.first_lsn,
                        s.seq,
                        s.first_lsn
                    ));
                }
            }
            prev = Some(s);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mapdb-ns-oracle-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    fn header(seq: i64, first_lsn: i64) -> Vec<u8> {
        let mut h = vec![0u8; SEG_HDR_LEN];
        h[..8].copy_from_slice(MAGIC);
        h[8..12].copy_from_slice(&FORMAT_VERSION.to_be_bytes());
        h[12..16].copy_from_slice(&0i32.to_be_bytes());
        h[16..24].copy_from_slice(&seq.to_be_bytes());
        h[24..32].copy_from_slice(&first_lsn.to_be_bytes());
        let crc = crc32(&h[..SEG_HDR_CRC_LEN]) as i32;
        h[32..36].copy_from_slice(&crc.to_be_bytes());
        h
    }

    /// A section: `tag | lsn i64 | bodyLen i64 | hdrCrc i32 | bodyCrc i32`, both
    /// CRCs domain-bound to the segment header and this section's offset.
    fn section(seg_hdr: &[u8], off: u64, tag: u8, lsn: i64, body: &[u8]) -> Vec<u8> {
        let dom = |extra: &[u8]| {
            let mut d = seg_hdr.to_vec();
            d.extend_from_slice(&off.to_be_bytes());
            d.extend_from_slice(extra);
            crc32(&d) as i32
        };
        let mut h = vec![0u8; SEC_HDR_LEN];
        h[0] = tag;
        h[1..9].copy_from_slice(&lsn.to_be_bytes());
        h[9..17].copy_from_slice(&(body.len() as i64).to_be_bytes());
        let hdr_crc = dom(&h[..SEC_HDR_CRC_LEN]);
        h[17..21].copy_from_slice(&hdr_crc.to_be_bytes());
        h[21..25].copy_from_slice(&dom(body).to_be_bytes());
        h.extend_from_slice(body);
        h
    }

    fn mark_body(through: i64, log_start: i64) -> Vec<u8> {
        let mut b = through.to_be_bytes().to_vec();
        b.extend_from_slice(&log_start.to_be_bytes());
        b
    }

    /// Writes a segment whose sections are the given `(tag, body)` pairs, at
    /// consecutive LSNs from `first_lsn`.
    fn write_seg_with(base: &Path, seq: i64, first_lsn: i64, secs: &[(u8, Vec<u8>)]) {
        let hdr = header(seq, first_lsn);
        let mut bytes = hdr.clone();
        for (i, (tag, body)) in secs.iter().enumerate() {
            let off = bytes.len() as u64;
            let s = section(&hdr, off, *tag, first_lsn + i as i64, body);
            bytes.extend_from_slice(&s);
        }
        let p = base.with_file_name(format!(
            "{}.wal.{seq:016x}",
            base.file_name().unwrap().to_str().unwrap()
        ));
        std::fs::write(p, bytes).expect("write segment");
    }

    fn write_seg(base: &Path, seq: i64, first_lsn: i64) {
        let p = base.with_file_name(format!(
            "{}.wal.{seq:016x}",
            base.file_name().unwrap().to_str().unwrap()
        ));
        std::fs::write(p, header(seq, first_lsn)).expect("write segment");
    }

    fn base_in(dir: &Path) -> PathBuf {
        dir.join("store.db")
    }

    /// The CRC is pinned against a known zlib CRC-32 value, so a broken
    /// reimplementation cannot quietly agree with itself.
    #[test]
    fn crc32_matches_the_zlib_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn scan_reads_a_healthy_set_and_accepts_it() {
        let dir = scratch("healthy");
        let base = base_in(&dir);
        write_seg(&base, 3, 10);
        write_seg(&base, 4, 25);
        let ns = scan(&base).expect("scan");
        assert_eq!(ns.count(), 2);
        assert_eq!((ns.lo(), ns.hi(), ns.gaps()), (3, 4, 0));
        ns.check_image().expect("healthy image");
        ns.check_recovered(&ns).expect("healthy recovered");
    }

    #[test]
    fn a_gap_is_reported_but_is_not_itself_a_violation() {
        let dir = scratch("gap");
        let base = base_in(&dir);
        write_seg(&base, 2, 5);
        write_seg(&base, 5, 9);
        let ns = scan(&base).expect("scan");
        assert_eq!(ns.gaps(), 2, "3 and 4 are missing");
        ns.check_image()
            .expect("a burnt or half-unlinked name is legitimate");
    }

    #[test]
    fn every_header_row_gets_its_own_reason_code() {
        let dir = scratch("rows");
        let base = base_in(&dir);
        let name = |seq: i64| {
            base.with_file_name(format!(
                "{}.wal.{seq:016x}",
                base.file_name().unwrap().to_str().unwrap()
            ))
        };
        let reseal = |h: &mut Vec<u8>| {
            let crc = crc32(&h[..SEG_HDR_CRC_LEN]) as i32;
            h[32..36].copy_from_slice(&crc.to_be_bytes());
        };
        std::fs::write(name(1), []).unwrap();
        std::fs::write(name(2), [0u8; 12]).unwrap();
        let mut h = header(3, 1);
        h[20] ^= 0xff; // no reseal
        std::fs::write(name(3), &h).unwrap();
        let mut h = header(4, 1);
        h[0] = b'X';
        reseal(&mut h);
        std::fs::write(name(4), &h).unwrap();
        let mut h = header(5, 1);
        h[8..12].copy_from_slice(&2i32.to_be_bytes());
        reseal(&mut h);
        std::fs::write(name(5), &h).unwrap();
        let mut h = header(6, 1);
        h[12..16].copy_from_slice(&1i32.to_be_bytes());
        reseal(&mut h);
        std::fs::write(name(6), &h).unwrap();
        std::fs::write(name(7), header(70, 1)).unwrap(); // seq != name
        std::fs::write(name(8), header(8, 0)).unwrap(); // firstLsn <= 0
        let ns = scan(&base).expect("scan");
        let codes: Vec<Option<&str>> = ns.segs.iter().map(|s| s.bad).collect();
        assert_eq!(
            codes,
            vec![
                Some("h1-empty"),
                Some("h2-short"),
                Some("h3-hdr-crc"),
                Some("h4-magic"),
                Some("h5-version"),
                Some("h6-flags"),
                Some("h7-seq"),
                Some("h9-first-lsn"),
            ]
        );
    }

    #[test]
    fn residue_is_accepted_on_the_highest_name_and_refused_below_it() {
        let dir = scratch("residue");
        let base = base_in(&dir);
        write_seg(&base, 1, 1);
        let top = base.with_file_name(format!(
            "{}.wal.{:016x}",
            base.file_name().unwrap().to_str().unwrap(),
            2
        ));
        std::fs::write(&top, []).unwrap();
        let ns = scan(&base).expect("scan");
        ns.check_image()
            .expect("residue on the highest name is a crashed create");
        assert!(ns.check_recovered(&ns).is_err(), "recovery must delete it");

        // The same unreadable header one name lower is not residue.
        write_seg(&base, 3, 7);
        let ns = scan(&base).expect("scan");
        let err = ns
            .check_image()
            .expect_err("residue below the highest name");
        assert!(err.contains("not the highest name"), "{err}");
    }

    #[test]
    fn recovery_may_not_reuse_a_burnt_name_or_create_two() {
        let dir = scratch("burn");
        let base = base_in(&dir);
        write_seg(&base, 4, 9);
        let pre = scan(&base).expect("scan");

        // Reusing a name below the image's highest (here: filling the hole at 2
        // after 4 already existed) is a W6 violation even though the resulting
        // set looks tidy.
        let dir2 = scratch("burn2");
        let b2 = base_in(&dir2);
        write_seg(&b2, 2, 3);
        write_seg(&b2, 4, 9);
        let post = scan(&b2).expect("scan");
        let err = post.check_recovered(&pre).expect_err("name reuse");
        assert!(err.contains("never reuses one"), "{err}");

        let dir3 = scratch("burn3");
        let b3 = base_in(&dir3);
        write_seg(&b3, 4, 9);
        write_seg(&b3, 5, 11);
        write_seg(&b3, 6, 12);
        let post = scan(&b3).expect("scan");
        let err = post.check_recovered(&pre).expect_err("two creates");
        assert!(err.contains("creates at most one"), "{err}");
    }

    #[test]
    fn recovery_may_not_unlink_a_hole() {
        let dir = scratch("hole-pre");
        let base = base_in(&dir);
        write_seg(&base, 1, 1);
        write_seg(&base, 2, 4);
        write_seg(&base, 3, 8);
        let pre = scan(&base).expect("scan");

        let dir2 = scratch("hole-post");
        let b2 = base_in(&dir2);
        write_seg(&b2, 1, 1);
        write_seg(&b2, 3, 8);
        let post = scan(&b2).expect("scan");
        let err = post.check_recovered(&pre).expect_err("removed the middle");
        assert!(err.contains("never a hole"), "{err}");

        // Removing the low run is exactly what a completed unlinkThrough does.
        let dir3 = scratch("hole-ok");
        let b3 = base_in(&dir3);
        write_seg(&b3, 3, 8);
        let post = scan(&b3).expect("scan");
        post.check_recovered(&pre).expect("a low run is legitimate");
    }

    #[test]
    fn the_lsn_chain_never_decreases_and_may_repeat_only_across_an_empty_segment() {
        let dir = scratch("chain");
        let base = base_in(&dir);
        write_seg(&base, 1, 40);
        write_seg(&base, 2, 39);
        let ns = scan(&base).expect("scan");
        assert!(
            ns.check_image().unwrap_err().contains("decreases"),
            "a lower LSN under a higher name is corruption at any time"
        );

        // Two header-only segments stating the same LSN: a cut between a
        // rollover's create and the section it rolled for, then R7 rotating
        // past the empty segment with the same `nextLsn`. Observed in the very
        // first smoke round this oracle ran on, and legal under H8.
        let dir = scratch("chain-empty");
        let base = base_in(&dir);
        write_seg(&base, 1, 22);
        write_seg(&base, 2, 22);
        let ns = scan(&base).expect("scan");
        ns.check_image().expect("legal in an image");
        ns.check_recovered(&ns).expect("legal after recovery too");

        // But a segment that HOLDS something cannot share its LSN with the next
        // one — that would be two segments claiming the same section.
        let dir = scratch("chain-nonempty");
        let base = base_in(&dir);
        write_seg(&base, 1, 22);
        let p = base.with_file_name(format!(
            "{}.wal.{:016x}",
            base.file_name().unwrap().to_str().unwrap(),
            1
        ));
        let mut body = std::fs::read(&p).unwrap();
        body.extend_from_slice(&[0u8; 25]);
        std::fs::write(&p, body).unwrap();
        write_seg(&base, 2, 22);
        let ns = scan(&base).expect("scan");
        ns.check_image()
            .expect("an image may still hold a torn tail");
        let err = ns.check_recovered(&ns).expect_err("not after recovery");
        assert!(err.contains("does not increase"), "{err}");
    }

    /// Foreign names and `.ckpt` are a HARNESS-environment verdict, never a
    /// namespace one. The reference ignores both, so a namespace checker that
    /// refused them would fail CI on a legal store; the crash checker asks for
    /// them separately, before it says anything about the product.
    #[test]
    fn v1_artifacts_and_foreign_names_are_a_harness_verdict_not_a_namespace_one() {
        let dir = scratch("legacy");
        let base = base_in(&dir);
        write_seg(&base, 1, 1);
        std::fs::write(dir.join("store.db.ckpt"), b"v1").unwrap();
        let ns = scan(&base).expect("scan");
        assert!(ns.legacy_ckpt);
        ns.check_image()
            .expect("a `.ckpt` sibling does not make the namespace illegal");
        ns.check_recovered(&ns).expect("nor after recovery");
        assert!(ns
            .check_harness_environment()
            .unwrap_err()
            .contains("v1 artifact"));

        std::fs::remove_file(dir.join("store.db.ckpt")).unwrap();
        // Every shape the store ignores: wrong width, uppercase hex, a
        // directory at a well-formed segment name.
        std::fs::write(dir.join("store.db.wal.000000000000000Z"), b"x").unwrap();
        std::fs::write(dir.join("store.db.wal.000000000000000A"), b"x").unwrap();
        std::fs::create_dir(dir.join("store.db.wal.00000000000000ff")).unwrap();
        let ns = scan(&base).expect("scan");
        assert_eq!(ns.count(), 1, "none of them is a segment");
        assert_eq!(ns.foreign.len(), 3);
        ns.check_image()
            .expect("the store IGNORES these, so the namespace is legal");
        ns.check_recovered(&ns).expect("still legal after recovery");
        assert!(ns
            .check_harness_environment()
            .unwrap_err()
            .contains("something else wrote into the directory"));
    }

    /// The defect a shape test cannot see: recovery that ignores a forced `'K'`
    /// removes NOTHING, so "everything removed was a low run" holds vacuously.
    #[test]
    fn a_forced_mark_obliges_recovery_to_reach_its_floor() {
        let dir = scratch("mark");
        let base = base_in(&dir);
        write_seg(&base, 1, 1);
        write_seg(&base, 2, 2);
        // Segment 3 carries a mark retiring everything through 2.
        write_seg_with(&base, 3, 3, &[(TAG_MARK, mark_body(2, 3))]);
        let pre = scan_with_marks(&base).expect("scan");
        assert_eq!(pre.authorized_floor(), Some(2), "the mark was read");

        // A recovery that unlinked nothing: same set, no hole, no reuse — every
        // other rule in this module passes it.
        let err = pre.check_recovered(&pre).expect_err("the mark was ignored");
        assert!(err.contains("authorizes retiring through"), "{err}");

        // Having replayed the unlink, it passes.
        let dir2 = scratch("mark-done");
        let b2 = base_in(&dir2);
        write_seg_with(&b2, 3, 3, &[(TAG_MARK, mark_body(2, 3))]);
        let post = scan(&b2).expect("scan");
        post.check_recovered(&pre).expect("unlink replayed");
    }

    /// `unlinkThrough(t)` removes every segment with `seq <= t`, so the surviving
    /// low name must be strictly ABOVE the floor. A checker written with `<`
    /// instead of `<=` passes the test above and fails only here.
    #[test]
    fn the_floor_is_inclusive_so_the_authorized_segment_itself_must_be_gone() {
        let dir = scratch("mark-eq");
        let base = base_in(&dir);
        write_seg(&base, 2, 2);
        write_seg_with(&base, 3, 3, &[(TAG_MARK, mark_body(2, 3))]);
        let pre = scan_with_marks(&base).expect("scan");
        assert_eq!(pre.authorized_floor(), Some(2));
        // Segment 2 IS the segment the mark authorized; leaving it is the
        // unfinished unlink.
        let err = pre
            .check_recovered(&pre)
            .expect_err("lo == through is not good enough");
        assert!(err.contains("authorizes retiring through"), "{err}");
    }

    /// **The construction that made the first version of this oracle unsound.**
    /// A CRC-valid mark that authorizes removing its OWN segment fails K4, so
    /// Java holds that segment and never derives its floor
    /// (`StoreWAL.java:691-694`). An oracle that read the number anyway would
    /// demand a floor of 100 and fail a recovery that correctly retired
    /// through 2.
    #[test]
    fn a_k4_invalid_mark_is_not_authority_even_with_a_valid_crc() {
        let dir = scratch("mark-k4");
        let base = base_in(&dir);
        // Segment 2's mark authorizes removing segment 100 — including itself.
        write_seg_with(&base, 2, 2, &[(TAG_MARK, mark_body(100, 1))]);
        // Segment 3's mark is the real one.
        write_seg_with(&base, 3, 3, &[(TAG_MARK, mark_body(2, 3))]);
        let pre = scan_with_marks(&base).expect("scan");
        assert_eq!(
            pre.authorized_floor(),
            Some(2),
            "K4 refuses the mark in segment 2; only segment 3's counts"
        );

        // What a CORRECT recovery leaves: everything through 2 retired.
        let dir2 = scratch("mark-k4-post");
        let b2 = base_in(&dir2);
        write_seg_with(&b2, 3, 3, &[(TAG_MARK, mark_body(2, 3))]);
        let post = scan(&b2).expect("scan");
        post.check_recovered(&pre)
            .expect("a correct recovery must not be failed by a mark Java held");
    }

    /// The walk stops at the first thing it cannot vouch for, so a mark behind
    /// damage never raises the floor. Reading too little only weakens the
    /// assertion; reading too much would fail a correct store.
    #[test]
    fn a_mark_is_only_counted_inside_the_valid_section_prefix() {
        let dir = scratch("mark-prefix");
        let base = base_in(&dir);
        let name = |seq: i64| {
            base.with_file_name(format!(
                "{}.wal.{seq:016x}",
                base.file_name().unwrap().to_str().unwrap()
            ))
        };
        let floor = || scan_with_marks(&base).expect("scan").authorized_floor();

        // A torn section ahead of the mark: the walk stops at the tear.
        write_seg_with(&base, 9, 1, &[(TAG_MARK, mark_body(7, 1))]);
        let mut bytes = std::fs::read(name(9)).unwrap();
        bytes[SEG_HDR_LEN + 20] ^= 0xff; // corrupt the section header CRC
        std::fs::write(name(9), &bytes).unwrap();
        assert_eq!(floor(), None);

        // A torn mark BODY is not authority either.
        write_seg_with(&base, 9, 1, &[(TAG_MARK, mark_body(7, 1))]);
        let mut bytes = std::fs::read(name(9)).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0xff; // corrupt the body
        std::fs::write(name(9), &bytes).unwrap();
        assert_eq!(floor(), None);

        // A mark whose body is not all there yet.
        write_seg_with(&base, 9, 1, &[(TAG_MARK, mark_body(7, 1))]);
        let bytes = std::fs::read(name(9)).unwrap();
        std::fs::write(name(9), &bytes[..bytes.len() - 4]).unwrap();
        assert_eq!(floor(), None);

        // A mark AFTER an ordinary section is reached; the highest wins.
        write_seg_with(
            &base,
            9,
            1,
            &[
                (b'S', vec![9u8; 40]),
                (TAG_MARK, mark_body(3, 2)),
                (TAG_MARK, mark_body(5, 3)),
            ],
        );
        assert_eq!(floor(), Some(5));

        // An ordinary section whose BODY is corrupt (its own CRC no longer
        // matches) ends the prefix — a walk that only checksummed marks would
        // step over it and count the 5 behind it.
        let hdr = header(9, 1);
        let mut bytes = hdr.clone();
        let off = bytes.len() as u64;
        bytes.extend_from_slice(&section(&hdr, off, b'S', 1, &[9u8; 40]));
        let corrupt_at = bytes.len() - 1;
        bytes[corrupt_at] ^= 0xff;
        let off = bytes.len() as u64;
        bytes.extend_from_slice(&section(&hdr, off, TAG_MARK, 2, &mark_body(5, 1)));
        std::fs::write(name(9), &bytes).unwrap();
        assert_eq!(floor(), None, "a torn ordinary body ends the prefix");

        // A tag outside {S, C, K} fails S3 with the header CRC, so it ends the
        // prefix rather than being stepped over.
        let mut bytes = hdr.clone();
        let off = bytes.len() as u64;
        bytes.extend_from_slice(&section(&hdr, off, b'X', 1, &[1u8; 8]));
        let off = bytes.len() as u64;
        bytes.extend_from_slice(&section(&hdr, off, TAG_MARK, 2, &mark_body(5, 1)));
        std::fs::write(name(9), &bytes).unwrap();
        assert_eq!(floor(), None, "an unknown tag ends the prefix");

        // S8's logStart rows: it must be positive and at or below the mark's
        // own LSN.
        write_seg_with(&base, 9, 1, &[(TAG_MARK, mark_body(7, 5))]);
        assert_eq!(floor(), None, "logStart 5 is above the mark's LSN 1");
        write_seg_with(&base, 9, 1, &[(TAG_MARK, mark_body(7, 0))]);
        assert_eq!(floor(), None, "logStart 0 is not an LSN");

        // A section whose LSN breaks the run ends the walk before the mark.
        let mut bytes = hdr.clone();
        let off = bytes.len() as u64;
        bytes.extend_from_slice(&section(&hdr, off, b'S', 99, &[1u8; 8]));
        let off = bytes.len() as u64;
        bytes.extend_from_slice(&section(&hdr, off, TAG_MARK, 100, &mark_body(4, 1)));
        std::fs::write(name(9), &bytes).unwrap();
        assert_eq!(floor(), None);
    }

    /// The LSN-exhausted image is one the frozen reference OPENS (it wraps to
    /// `i64::MIN`), so the walk must end at it rather than overflow. Before this
    /// was checked the increment panicked in a debug build of the checker.
    #[test]
    fn an_lsn_exhausted_segment_ends_the_walk_instead_of_overflowing() {
        let dir = scratch("mark-lsn-max");
        let base = base_in(&dir);
        write_seg_with(&base, 9, i64::MAX, &[(TAG_MARK, mark_body(4, 1))]);
        let ns = scan_with_marks(&base).expect("scan");
        assert_eq!(
            ns.authorized_floor(),
            Some(4),
            "the mark at i64::MAX is vouched for; the walk simply cannot continue past it"
        );
    }

    /// The ceilings are a stop, never a verdict: reaching one lowers the floor,
    /// which weakens the assertion and can never fail a correct store.
    #[test]
    fn the_walk_ceilings_stop_conservatively() {
        let dir = scratch("mark-bound");
        let base = base_in(&dir);
        write_seg_with(
            &base,
            9,
            1,
            &[
                (TAG_MARK, mark_body(1, 1)),
                (TAG_MARK, mark_body(2, 2)),
                (TAG_MARK, mark_body(3, 3)),
            ],
        );
        let path = base.with_file_name("store.db.wal.0000000000000009");
        let file = File::open(&path).unwrap();
        let len = file.metadata().unwrap().len();
        let mut hdr = [0u8; SEG_HDR_LEN];
        file.read_exact_at(&mut hdr, 0).unwrap();

        let all = highest_mark_bounded(&file, &hdr, 9, len, u64::MAX, u64::MAX);
        assert_eq!(all, Some(3), "unbounded, every mark is read");
        let capped = highest_mark_bounded(&file, &hdr, 9, len, u64::MAX, 2);
        assert_eq!(capped, Some(2), "the section ceiling stops the walk early");
        // One section here is 25 + 16 = 41 bytes, so a 41-byte budget admits
        // exactly the first and stops before the second.
        let capped = highest_mark_bounded(&file, &hdr, 9, len, 41, u64::MAX);
        assert_eq!(capped, Some(1), "so does the byte ceiling");
        let capped = highest_mark_bounded(&file, &hdr, 9, len, 1, u64::MAX);
        assert_eq!(capped, None, "a budget below one section admits nothing");
    }

    /// The ceiling must bound the bytes HASHED, not the offset a section starts
    /// at: `bodyLen` is a disk value and the reference accepts any width that
    /// fits the file, so a budget consulted only at section boundaries is no
    /// budget at all against a single huge body.
    #[test]
    fn the_byte_ceiling_covers_the_body_a_section_is_about_to_stream() {
        let dir = scratch("mark-bigbody");
        let base = base_in(&dir);
        write_seg_with(
            &base,
            9,
            1,
            &[(b'S', vec![7u8; 100_000]), (TAG_MARK, mark_body(3, 2))],
        );
        let path = base.with_file_name("store.db.wal.0000000000000009");
        let file = File::open(&path).unwrap();
        let len = file.metadata().unwrap().len();
        let mut hdr = [0u8; SEG_HDR_LEN];
        file.read_exact_at(&mut hdr, 0).unwrap();

        assert_eq!(
            highest_mark_bounded(&file, &hdr, 9, len, u64::MAX, u64::MAX),
            Some(3),
            "with no budget the whole segment is walked"
        );
        // The first body alone is 100 KB, so a 50 KB budget must refuse it.
        // This pins the RESULT only: a start-only check would also answer
        // `None` here, having hashed the whole body first. What discriminates
        // start-from-end is the 41-byte case above; what a test in this process
        // cannot yet observe is the reads themselves, so moving this check
        // below the S4 loop survives the suite (round 6, non-blocking).
        assert_eq!(
            highest_mark_bounded(&file, &hdr, 9, len, 50_000, u64::MAX),
            None,
            "the walk stops before streaming a body that crosses the budget"
        );
    }

    /// The ceilings the PRODUCTION walk is wired with. Reaching them
    /// behaviourally would need a 64 MiB fixture, so this pins the values
    /// instead — and `highest_mark` destructures `WALK_LIMITS` rather than
    /// writing any ceiling of its own, so this is the only place either number
    /// can be raised.
    #[test]
    fn the_production_walk_is_actually_bounded() {
        assert_eq!(WALK_LIMITS, (64 << 20, 1 << 20));
    }

    /// S8's width row. A mark body that is not exactly 16 bytes is refused
    /// before its contents mean anything — otherwise a CRC-valid wider body
    /// whose first eight bytes happen to read as a sequence number would
    /// authorize a floor Java refuses outright.
    #[test]
    fn a_mark_body_of_the_wrong_width_is_not_authority() {
        let dir = scratch("mark-width");
        let base = base_in(&dir);
        let mut wide = mark_body(3, 1);
        wide.extend_from_slice(&[0u8; 8]); // 24 bytes, sealed with a valid CRC
        write_seg_with(&base, 9, 1, &[(TAG_MARK, wide)]);
        assert_eq!(
            scan_with_marks(&base).expect("scan").authorized_floor(),
            None
        );
    }

    /// A scan that never looked answers `None` exactly like a scan that found
    /// nothing, so the floor rule is skipped rather than passed vacuously — and
    /// the crash checker asserts `marks_scanned` before it relies on it.
    #[test]
    fn the_floor_rule_is_skipped_by_a_scan_that_did_not_look_for_marks() {
        let dir = scratch("mark-unscanned");
        let base = base_in(&dir);
        write_seg(&base, 1, 1);
        write_seg_with(&base, 3, 3, &[(TAG_MARK, mark_body(2, 3))]);

        let headers_only = scan(&base).expect("scan");
        assert!(!headers_only.marks_scanned);
        assert_eq!(headers_only.authorized_floor(), None);
        headers_only
            .check_recovered(&headers_only)
            .expect("no floor was derived, so no floor is demanded");

        let with_marks = scan_with_marks(&base).expect("scan");
        assert!(with_marks.marks_scanned);
        assert_eq!(with_marks.authorized_floor(), Some(2));
        assert!(with_marks.check_recovered(&with_marks).is_err());
    }

    /// Deleting the residue at N and creating a fresh segment AT N leaves both
    /// sets with the same members, so a membership diff sees no new name at all.
    #[test]
    fn recovery_may_not_create_at_a_residue_name_it_just_deleted() {
        let dir = scratch("residue-reuse");
        let base = base_in(&dir);
        write_seg(&base, 4, 9);
        let top = base.with_file_name(format!(
            "{}.wal.{:016x}",
            base.file_name().unwrap().to_str().unwrap(),
            5
        ));
        std::fs::write(&top, [0u8; 12]).unwrap(); // residue at 5
        let pre = scan(&base).expect("scan");

        let dir2 = scratch("residue-reuse-post");
        let b2 = base_in(&dir2);
        write_seg(&b2, 4, 9);
        write_seg(&b2, 5, 11); // a VALID segment at the burnt name
        let post = scan(&b2).expect("scan");
        let err = post.check_recovered(&pre).expect_err("5 was burnt");
        assert!(err.contains("never reuses one"), "{err}");
    }

    #[test]
    fn an_empty_namespace_is_refused() {
        let dir = scratch("empty");
        let ns = scan(&base_in(&dir)).expect("scan");
        assert!(ns.is_empty());
        assert!(ns.check_image().unwrap_err().contains("no segments at all"));
    }

    #[test]
    fn uppercase_hex_and_wrong_width_are_not_segment_names() {
        let dir = scratch("grammar");
        let base = base_in(&dir);
        std::fs::write(dir.join("store.db.wal.000000000000000A"), header(10, 1)).unwrap();
        std::fs::write(dir.join("store.db.wal.00000001"), header(1, 1)).unwrap();
        std::fs::write(dir.join("store.db.wal.ffffffffffffffff"), header(-1, 1)).unwrap();
        let ns = scan(&base).expect("scan");
        assert!(ns.segs.is_empty(), "{:?}", ns.segs);
        assert_eq!(ns.foreign.len(), 3);
    }
}
