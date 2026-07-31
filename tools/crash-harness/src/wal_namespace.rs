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
//! `[0, 32)`. It reads headers only — sections, LSN chains and the `'K'` mark
//! body stay the store's business, and the store's own recovery tests own them.
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
use std::path::{Path, PathBuf};

/// magic(8) + version(4) + flags(4) + segmentSeq(8) + firstLsn(8) + headerCrc(4).
pub const SEG_HDR_LEN: usize = 36;
/// Header bytes covered by `headerCrc`.
const SEG_HDR_CRC_LEN: usize = 32;
const MAGIC: &[u8; 8] = b"MDBS.WAL";
const FORMAT_VERSION: i32 = 3;

/// zlib CRC-32, bitwise. Deliberately not the store's `crc32fast`: 36 bytes per
/// segment makes the table irrelevant, and an oracle that imported the same
/// checksum implementation as the code under test would agree with it by
/// construction.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
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
}

impl SegmentInfo {
    pub fn ok(&self) -> bool {
        self.bad.is_none()
    }
}

/// Reads the 36-byte header and applies rows H1-H9 in the reference's order:
/// the semantic rows are reached only after the CRC passes, so editing a field
/// without resealing is a CRC verdict rather than a semantic one.
fn classify_header(path: &Path, name_seq: i64) -> Result<SegmentInfo, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    let len = meta.len();
    let mut info = SegmentInfo {
        seq: name_seq,
        first_lsn: 0,
        len,
        bad: None,
    };
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.is_empty() {
        info.bad = Some("h1-empty");
        return Ok(info);
    }
    if bytes.len() < SEG_HDR_LEN {
        info.bad = Some("h2-short");
        return Ok(info);
    }
    let hdr = &bytes[..SEG_HDR_LEN];
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
    }
    Ok(info)
}

/// Enumerates the namespace of `base` (the path handed to `DB::make_wal`).
///
/// Reads the directory once and stats/reads only the 36-byte headers, so it is
/// cheap enough for the workload to call at every group boundary.
pub fn scan(base: &Path) -> Result<Namespace, String> {
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
        ns.segs.push(classify_header(&entry.path(), seq)?);
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
        let (pre_seqs, post_seqs) = (pre.seqs(), self.seqs());
        // W6: a name is never reused. Anything recovery created is strictly
        // above every name the image held, residue included (the residue name
        // is burnt by the same rule).
        let created: Vec<i64> = post_seqs.difference(&pre_seqs).copied().collect();
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

    /// The rows that hold at every instant, open or closed.
    fn check_common(&self, what: &str, strict_chain: bool) -> Result<(), String> {
        if self.legacy_wal || self.legacy_ckpt {
            return Err(format!(
                "{what}: a v1 artifact is present (<base>.wal={}, <base>.ckpt={}) — D1 refuses the \
                 open rather than starting a fresh segment set beside it, so the harness must \
                 never produce one",
                self.legacy_wal, self.legacy_ckpt
            ));
        }
        if !self.foreign.is_empty() {
            return Err(format!(
                "{what}: names under the segment prefix that are not segments: {:?}",
                self.foreign
            ));
        }
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

    #[test]
    fn v1_artifacts_and_foreign_names_are_refused() {
        let dir = scratch("legacy");
        let base = base_in(&dir);
        write_seg(&base, 1, 1);
        std::fs::write(dir.join("store.db.ckpt"), b"v1").unwrap();
        let ns = scan(&base).expect("scan");
        assert!(ns.legacy_ckpt);
        assert!(ns.check_image().unwrap_err().contains("v1 artifact"));

        std::fs::remove_file(dir.join("store.db.ckpt")).unwrap();
        std::fs::write(dir.join("store.db.wal.000000000000000Z"), b"x").unwrap();
        let ns = scan(&base).expect("scan");
        assert_eq!(ns.count(), 1, "the non-hex name is not a segment");
        assert!(ns.check_image().unwrap_err().contains("not segments"));
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
