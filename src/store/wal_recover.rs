//! The WAL's **v3 codec and recovery state machine** — sections, entries, the
//! `'K'` clean mark, and the two-pass replay (R3-R7) that turns a segment set
//! into an in-memory store.
//!
//! Port of the recovery half of Java `StoreWAL` (format v3); the namespace half
//! (N/H/W2/W5/W6, the store lock) is [`wal_segments`](super::wal_segments) and
//! ran already — R0-R2 happen inside [`WalSegmentSet::open`]. This module is
//! everything after that.
//!
//! ```text
//! section := tag u8 ('S' commit | 'C' image | 'K' clean mark)
//!          | lsn i64 | bodyLen i64 | hdrCrc i32 | bodyCrc i32      // 25 bytes
//!          | body
//! mark    := cleanedThroughSeq i64 | logStartLsn i64               // 16 bytes
//! entry   := T_PREALLOC recid
//!          | T_RECORD   recid cap len+1|0 payload?
//!          | T_APPEND   recid (lsn - baseLsn) len payload
//!          | T_DELETE   recid                       // packLong framing
//! ```
//!
//! Both CRCs are **domain-bound**: each is an ordinary CRC-32 fed the segment's
//! 36 header bytes and the section's own offset as a prefix
//! ([`Segment::crc_domain`]). A section byte-copied to another segment, or to
//! another offset in its own, therefore fails its checksums — the property that
//! lets the torn-tail lookahead below trust what it finds.
//!
//! # The shape of recovery, and why it is two passes
//!
//! Pass 1 ([`scan_segment`]) establishes **boundaries only**: how far each
//! segment's valid section prefix runs, its LSN span, and whether anything in it
//! is corrupt — with no per-recid state whatsoever. Pass 2 ([`apply_section`])
//! replays entries in ascending LSN order and is the sole authority on content.
//! A verdict found in pass 1 is **held**, not thrown: the segment carrying it
//! may be below a clean mark and about to be deleted, and refusing to open a
//! store over rot in bytes nobody will ever read is how a recoverable store gets
//! bricked. R4 decides which held verdicts matter.
//!
//! The three questions recovery must answer — where does the retained log
//! legitimately begin, is each missing segment authorized, and does each delta
//! still have the image it extends — are all answered by comparing **numbers a
//! conforming writer recorded** (`firstLsn` in every segment header,
//! `logStartLsn` in every mark, the base LSN in every `T_APPEND`) rather than by
//! inferring intent from LSN density or section tags. That is the whole reason
//! v3 exists.
//!
//! # What is NOT here
//!
//! The section WRITER lives in [`wal_write`](super::wal_write) and the cleaning
//! cycle that emits `'K'` in [`wal`](super::wal); this module reads. There is no
//! public read-only surface either (D7 — the internal read-only mode here is
//! real and tested, and it is all of read-only that this workstream ships).

use super::direct::StoreDirect;
use super::index_val as iv;
use super::wal_segments::{crc_domain_of, Segment, WalSegmentSet, SEG_HDR};
use super::wal_write::{wal_io_event, WalOpKind};
use super::{AppendResult, Recid, StoreDelta};
use crate::error::{DbError, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::num::NonZeroU64;
use std::os::unix::fs::FileExt;

/// tag(1) + lsn(8) + bodyLen(8) + hdrCrc(4) + bodyCrc(4).
pub(crate) const SEC_HDR: usize = 25;
/// Bytes of the section header covered by `hdrCrc` — everything before the two
/// checksums.
pub(crate) const SEC_HDR_CRC_LEN: usize = 17;

/// A committed transaction.
pub(crate) const TAG_SECTION: u8 = b'S';
/// A cleaner-written image: semantically identical to `'S'`, and deliberately so
/// — the retained `'C'` sections are collectively the checkpoint, so there is no
/// "newest image wins" rule to implement.
pub(crate) const TAG_IMAGE: u8 = b'C';
/// A clean mark. Carries no entries and is never handed to the entry decoder.
pub(crate) const TAG_MARK: u8 = b'K';
/// `cleanedThroughSeq i64 | logStartLsn i64`, fixed width — a mark is a fact,
/// not a record, so it is not packLong-framed.
pub(crate) const MARK_BODY_LEN: i64 = 16;

const T_PREALLOC: u8 = 1;
const T_RECORD: u8 = 2;
const T_APPEND: u8 = 3;
const T_DELETE: u8 = 4;

/// `'S'`, `'C'` and `'K'` — **all three**, in the main scan and in the
/// lookahead alike.
///
/// Transcribing v1's two-tag set here is the single easiest way to port this
/// format wrongly: a `'K'` sitting after a rotted section would not be
/// recognised as a valid section, the lookahead would report "nothing valid
/// follows", and deliberate mid-log rot would be silently truncated away as a
/// torn tail.
fn valid_tag(tag: u8) -> bool {
    tag == TAG_SECTION || tag == TAG_IMAGE || tag == TAG_MARK
}

fn be32(b: &[u8], off: usize) -> i32 {
    i32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn be64(b: &[u8], off: usize) -> i64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[off..off + 8]);
    i64::from_be_bytes(v)
}

/// Recid 0 is refused for every entry type by [`entry_recid`], before any of the
/// entry's other fields are read, so this conversion cannot fail by the time an
/// apply reaches it.
fn nz(recid: u64) -> Recid {
    NonZeroU64::new(recid).expect("entry_recid refuses the reserved recid 0")
}

pub(crate) fn parse_sec_hdr(hdr: &[u8; SEC_HDR]) -> (u8, i64, i64, i32, i32) {
    (
        hdr[0],
        be64(hdr, 1),
        be64(hdr, 9),
        be32(hdr, 17),
        be32(hdr, 21),
    )
}

/// The 25 header bytes for a section whose body has already been MEASURED — a
/// length and a body CRC, not the bytes themselves.
///
/// This is the shape the streaming writer needs (`wal_write.rs`): its pass 1
/// produces exactly these two numbers and never materializes the body, so a
/// signature taking `&[u8]` cannot serve it. Split out here rather than
/// duplicated there, so the port keeps ONE encoding of a section header — both
/// A1 reviews found the un-split version and named the same failure mode, a
/// writer and a test kit that drift into two.
pub(crate) fn seal_sec_hdr(
    seg_header: &[u8; SEG_HDR as usize],
    offset: u64,
    tag: u8,
    lsn: i64,
    body_len: u64,
    body_crc: i32,
) -> [u8; SEC_HDR] {
    let mut hdr = [0u8; SEC_HDR];
    hdr[0] = tag;
    hdr[1..9].copy_from_slice(&lsn.to_be_bytes());
    hdr[9..17].copy_from_slice(&(body_len as i64).to_be_bytes());
    let mut h = crc32fast::Hasher::new();
    crc_domain_of(&mut h, seg_header, offset);
    h.update(&hdr[..SEC_HDR_CRC_LEN]);
    hdr[17..21].copy_from_slice(&(h.finalize() as i32).to_be_bytes());
    hdr[21..25].copy_from_slice(&body_crc.to_be_bytes());
    hdr
}

/// [`seal_sec_hdr`] for a caller that HOLDS the body: the byte-level test kit.
/// The production writer never materializes a body, so it always seals from a
/// measured length and CRC instead.
#[cfg(test)]
pub(crate) fn build_sec_hdr(
    seg_header: &[u8; SEG_HDR as usize],
    offset: u64,
    tag: u8,
    lsn: i64,
    body: &[u8],
) -> [u8; SEC_HDR] {
    let mut b = crc32fast::Hasher::new();
    crc_domain_of(&mut b, seg_header, offset);
    b.update(body);
    seal_sec_hdr(
        seg_header,
        offset,
        tag,
        lsn,
        body.len() as u64,
        b.finalize() as i32,
    )
}

/// The 16-byte `'K'` body. Written by the cleaner (A3) and by the test kit.
pub(crate) fn build_mark_body(cleaned_through_seq: i64, log_start_lsn: i64) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&cleaned_through_seq.to_be_bytes());
    b[8..].copy_from_slice(&log_start_lsn.to_be_bytes());
    b
}

/// CRC-32 of a section header in its domain — compare against `hdrCrc`.
fn hdr_crc(seg: &Segment, offset: u64, hdr: &[u8; SEC_HDR]) -> i32 {
    let mut h = crc32fast::Hasher::new();
    seg.crc_domain(&mut h, offset);
    h.update(&hdr[..SEC_HDR_CRC_LEN]);
    h.finalize() as i32
}

/// The segment's file handle. Every read helper here needs one, and every caller
/// opens it once per pass before entering them.
///
/// Panics rather than returning an error, deliberately: a released handle is a
/// sequencing bug in THIS module, and the shape it used to have —
/// `DataCorruption("segment handle released mid-pass")` — told the user their
/// store was damaged when the port had merely mis-ordered its own
/// `ensure_open`/`release` calls. A2 grows considerably more of that
/// choreography, so the lie would get easier to trigger, not harder.
fn handle(seg: &Segment) -> &File {
    seg.file()
        .expect("WAL segment handle released mid-pass: caller must ensure_open")
}

/// `Ok(None)` when the file is shorter than the read demands — the segment
/// shrank under us, which recovery treats exactly as a torn tail.
fn read_at_opt(file: &File, buf: &mut [u8], pos: u64) -> Result<Option<()>> {
    match file.read_exact_at(buf, pos) {
        Ok(()) => Ok(Some(())),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(DbError::Io(e)),
    }
}

fn read_sec_hdr(seg: &Segment, pos: u64) -> Result<Option<[u8; SEC_HDR]>> {
    let mut hdr = [0u8; SEC_HDR];
    Ok(read_at_opt(handle(seg), &mut hdr, pos)?.map(|()| hdr))
}

/// CRC-32 over a section body in its domain, streamed through a bounded window
/// so a body larger than memory still verifies.
fn body_crc(
    seg: &Segment,
    section_offset: u64,
    start: u64,
    end: u64,
    replay_buf: usize,
) -> Result<Option<i32>> {
    let file = handle(seg);
    let mut crc = crc32fast::Hasher::new();
    seg.crc_domain(&mut crc, section_offset);
    if start < end {
        // Narrowed only AFTER the minimum, never before: on a 32-bit target a
        // body of exactly 4 GiB casts to a `usize` 0, which would size the
        // buffer at zero and then loop forever without advancing `p`. The format
        // supports bodies past 2 GiB by design, so this is not a theoretical
        // width.
        let cap = (end - start).min(replay_buf.max(16) as u64) as usize;
        let mut buf = vec![0u8; cap];
        let mut p = start;
        while p < end {
            let n = (end - p).min(buf.len() as u64) as usize;
            if read_at_opt(file, &mut buf[..n], p)?.is_none() {
                return Ok(None);
            }
            crc.update(&buf[..n]);
            p += n as u64;
        }
    }
    Ok(Some(crc.finalize() as i32))
}

// ---------- streaming entry decoder ----------

/// A fixed-size window over one section body, with `u64` file positions.
///
/// Never materializes a body, so a commit larger than memory replays: Java
/// streams both writing and reading for exactly this reason, and a port that
/// reads whole bodies regresses every large transaction into an allocation of
/// its full size.
///
/// It folds no CRC into its reads, and does not need to: a v3 section's CRCs are
/// verified in pass 1, before a single entry is decoded ("garbage never
/// allocates"). The v1 reader this replaced computed one incrementally, because
/// its legacy trailing-seal format could only be checked at the end.
pub(crate) struct SecIn<'a> {
    file: &'a File,
    /// SOFT end: the section being decoded. Reading past it is corruption.
    limit: u64,
    /// HARD end: how far the window may read AHEAD of the soft limit. Equal to
    /// the soft limit for replay, which decodes one section at a time; the
    /// cleaner's scan sets it to the segment's validated end so one window can
    /// span a section boundary.
    ///
    /// That split is what makes the scan cost one syscall per WINDOW instead of
    /// per section. Java measured the difference: a log written by single-op
    /// commits is nearly all section headers, and reading each one with its own
    /// positional read (plus the window drop that followed) issued ~148k reads
    /// to walk 34 MB.
    hard_limit: u64,
    win: Vec<u8>,
    win_start: u64,
    win_pos: usize,
    win_len: usize,
    /// Reads issued, for the scan-cost test. Never reset; a test takes a
    /// difference.
    reads: u64,
}

impl<'a> SecIn<'a> {
    pub(crate) fn new(file: &'a File, bufsize: usize) -> SecIn<'a> {
        SecIn {
            file,
            limit: 0,
            hard_limit: 0,
            win: vec![0u8; bufsize.max(16)],
            win_start: 0,
            win_pos: 0,
            win_len: 0,
            reads: 0,
        }
    }

    /// Positions the reader over `[start, end)` and DROPS the window. Both
    /// bounds become `end`.
    pub(crate) fn reset(&mut self, start: u64, end: u64) {
        self.win_start = start;
        self.limit = end;
        self.hard_limit = end;
        self.win_pos = 0;
        self.win_len = 0;
    }

    /// Positions the reader over `[start, end)` and KEEPS the window when it
    /// already covers `start`. The hard limit is untouched, so this narrows the
    /// soft bound to one section without paying for a re-read.
    pub(crate) fn rebound(&mut self, start: u64, end: u64) {
        self.limit = end;
        if start >= self.win_start && start < self.win_start + self.win_len as u64 {
            self.win_pos = (start - self.win_start) as usize;
        } else {
            self.win_start = start;
            self.win_pos = 0;
            self.win_len = 0;
        }
    }

    /// Sets the hard bound the window may read to, dropping it. Used once per
    /// segment by the cleaner's scan.
    pub(crate) fn reset_hard(&mut self, start: u64, hard_end: u64) {
        self.reset(start, hard_end);
    }

    /// Moves to `pos` within the current bounds, keeping the window when it
    /// covers the target — the payload seek that makes the scan's cost
    /// proportional to entries rather than to the bytes they carry.
    pub(crate) fn seek(&mut self, pos: u64) {
        let limit = self.limit;
        self.rebound(pos, limit);
    }

    pub(crate) fn pos(&self) -> u64 {
        self.win_start + self.win_pos as u64
    }

    pub(crate) fn reads(&self) -> u64 {
        self.reads
    }

    fn remaining(&self) -> u64 {
        self.limit - self.pos()
    }

    /// A read past the section's end is **corruption**, not a torn tail: pass 1
    /// already proved this section whole and CRC-valid, so an entry that runs
    /// off its end means the body's framing disagrees with its own length.
    fn refill(&mut self) -> Result<()> {
        self.win_start = self.pos();
        self.win_pos = 0;
        if self.win_start >= self.limit {
            return Err(DbError::corrupt_msg(format!(
                "WAL entry overran its section body at {}",
                self.limit
            )));
        }
        // Minimum in u64, THEN narrow — see `body_crc`. A 4 GiB remainder that
        // cast to 0 first would leave `win_len` at zero and hand the caller a
        // byte it never read. Filled to the HARD limit, so one window can serve
        // several sections; the soft limit above is what bounds the caller.
        let n =
            (self.hard_limit.max(self.limit) - self.win_start).min(self.win.len() as u64) as usize;
        self.file
            .read_exact_at(&mut self.win[..n], self.win_start)
            .map_err(DbError::Io)?;
        self.win_len = n;
        self.reads += 1;
        Ok(())
    }

    pub(crate) fn read_byte(&mut self) -> Result<u8> {
        if self.pos() >= self.limit || self.win_pos >= self.win_len {
            self.refill()?;
        }
        let b = self.win[self.win_pos];
        self.win_pos += 1;
        Ok(b)
    }

    /// packLong: MSB-first 7-bit groups, high bit terminates.
    ///
    /// Capped at 10 bytes, where Java's decoder loops to the terminator. That
    /// difference is deliberate and pre-existing (v1 caps it too): the canonical
    /// encodings agree, so the cap only changes which *malformed* input is
    /// refused and how quickly. It is recorded as a per-engine expectation in
    /// the corruption fixtures rather than smoothed over.
    pub(crate) fn unpack_long(&mut self) -> Result<u64> {
        let mut ret: u64 = 0;
        for _ in 0..10 {
            let v = self.read_byte()?;
            ret = (ret << 7) | (v & 0x7F) as u64;
            if v & 0x80 != 0 {
                return Ok(ret);
            }
        }
        Err(DbError::corrupt("WAL packed long too long"))
    }

    pub(crate) fn read_fully(&mut self, dst: &mut [u8]) -> Result<()> {
        let mut off = 0;
        while off < dst.len() {
            if self.pos() >= self.limit || self.win_pos >= self.win_len {
                self.refill()?;
            }
            let n = (self.win_len - self.win_pos).min(dst.len() - off);
            dst[off..off + n].copy_from_slice(&self.win[self.win_pos..self.win_pos + n]);
            self.win_pos += n;
            off += n;
        }
        Ok(())
    }
}

/// Capacity as the writer encodes it: 0 for null content, else 16-aligned, big
/// enough for header+content and within the plain-record limit — EXCEPT oversize
/// (linked) records, which the writer encodes with capacity 0. Anything else
/// never came from this writer.
fn cap_valid(cap: u64, data: Option<&[u8]>) -> bool {
    match data {
        None => cap == 0,
        Some(d) => {
            let max = iv::MAX_CAPACITY as u64;
            if cap == 0 {
                4 + d.len() as u64 > max
            } else {
                cap >= 4 + d.len() as u64 && cap <= max && (cap & 15) == 0
            }
        }
    }
}

// ---------- the two per-recid identities (§4.2) ----------

/// The two per-recid identities replay maintains, plus the deferred skip audit.
///
/// **Not a replay floor under another name.** The v2 floor was derived by
/// looking *ahead* for each recid's newest self-contained entry and then
/// deciding what to apply, so a wrong floor silently recovered different data.
/// These are derived purely from what has already been applied and decide
/// nothing on their own: the only thing they can do when they are wrong is
/// refuse the open (the audit), which is why they replaced it.
///
/// The maps survive recovery — A2's commit classifier stamps every `T_APPEND`
/// with `content_base_lsn[recid]`, and fabricating that stamp instead of reading
/// it is a silent-loss channel.
#[derive(Default)]
pub(crate) struct Identities {
    /// LSN of the content image currently applied for a recid. Set by a
    /// content-bearing `T_RECORD`; CLEARED by `T_DELETE`, by a null-content
    /// `T_RECORD` and by `T_PREALLOC`.
    pub(crate) content_base_lsn: HashMap<u64, i64>,
    /// LSN at which a recid's state was last made self-contained. Set by EVERY
    /// self-contained non-void entry; cleared by `T_DELETE`. Consumed by the
    /// cleaner (A3).
    pub(crate) state_lsn: HashMap<u64, i64>,
    /// Recids whose stranded `T_APPEND` replay skipped, minus those a later
    /// self-contained entry has since superseded.
    skipped_appends: HashSet<u64>,
}

impl Identities {
    /// A content-bearing image: both identities move to this section's LSN.
    pub(crate) fn content(&mut self, recid: u64, lsn: i64) {
        self.content_base_lsn.insert(recid, lsn);
        self.state_lsn.insert(recid, lsn);
        self.skipped_appends.remove(&recid);
    }

    /// A self-contained entry that leaves the record with NO content image — a
    /// null `T_RECORD` or a `T_PREALLOC`. Merely declining to set a new content
    /// base is not enough: a recid that was content-live and became null would
    /// keep a stale base, and a later writer could then stamp an append from a
    /// state in which append is not valid.
    pub(crate) fn state_only(&mut self, recid: u64, lsn: i64) {
        self.content_base_lsn.remove(&recid);
        self.state_lsn.insert(recid, lsn);
        self.skipped_appends.remove(&recid);
    }

    /// The record is gone: both identities cleared, any pending skip discharged.
    pub(crate) fn void(&mut self, recid: u64) {
        self.content_base_lsn.remove(&recid);
        self.state_lsn.remove(&recid);
        self.skipped_appends.remove(&recid);
    }

    /// End of replay: every skipped append must have been superseded. A recid
    /// still here means the retained log holds a delta whose base is gone and
    /// nothing later re-established it — the store cannot be reconstructed, so
    /// the open refuses rather than return a record missing acknowledged bytes.
    fn audit(&mut self) -> Result<()> {
        if self.skipped_appends.is_empty() {
            return Ok(());
        }
        let n = self.skipped_appends.len();
        let recid = *self.skipped_appends.iter().next().expect("non-empty");
        self.skipped_appends.clear();
        Err(DbError::corrupt_msg(format!(
            "WAL replay skipped {n} append(s) whose base image is absent and which no later entry \
             superseded (recid {recid}): the log is missing sections it depends on"
        )))
    }
}

// ---------- R3: pass 1 ----------

/// Records a verdict against a segment; the caller stops scanning it and R4
/// decides whether it matters. First message wins.
fn hold(seg: &mut Segment, message: String) {
    if seg.held.is_none() {
        seg.held = Some(message);
    }
}

/// R3, one segment (table S). Leaves `valid_end`, `first_lsn`, `last_lsn` and at
/// most one held verdict on `seg`; returns the highest `cleanedThroughSeq`
/// attested by a valid `'K'` **inside this segment**.
///
/// `look_last` enters as the previous segment's last accepted LSN and is used
/// ONLY as the suspect-header lookahead's anchor. The density checks (S2/S9)
/// deliberately restart at every segment boundary — the cross-boundary link is
/// R4's job, over the retained set alone. Checking it here would refuse a
/// perfectly legitimate crash image (segment 1 present, segment 2 already
/// unlinked, segment 3 carrying the mark that authorized it), and the verdict
/// would originate in a retained segment while being caused by a superseded one
/// — the one shape R4's "discard verdicts from below the mark" cannot rescue.
///
/// `is_active` marks the highest segment, the only one a torn tail can reach:
/// W3 seals every other segment at a section boundary with `force(true)`, so a
/// tear anywhere below the highest name is corruption by construction.
fn scan_segment(
    seg: &mut Segment,
    mut look_last: i64,
    is_active: bool,
    replay_buf: usize,
    mark_log_start: &mut i64,
) -> Result<i64> {
    let mut seg_through: i64 = 0;
    let len = seg.file_len;
    let mut pos = SEG_HDR;
    seg.valid_end = pos;
    while pos + SEC_HDR as u64 <= len {
        let hdr = match read_sec_hdr(seg, pos)? {
            Some(h) => h,
            None => {
                if !is_active {
                    hold(
                        seg,
                        "non-final segment is shorter than its own sections claim".into(),
                    );
                }
                return Ok(seg_through);
            }
        };
        let (tag, lsn, body_len, stored_hdr_crc, stored_body_crc) = parse_sec_hdr(&hdr);
        let body_start = pos + SEC_HDR as u64;

        if hdr_crc(seg, pos, &hdr) != stored_hdr_crc || !valid_tag(tag) {
            // S3. The declared bodyLen is UNTRUSTED — it lives in the bytes that
            // just failed their own checksum — so proving corruption needs a
            // section at exactly the declared end carrying exactly the LSN the
            // damaged one would have been followed by.
            if !is_active {
                hold(
                    seg,
                    format!("section header damaged at offset {pos} in a non-final segment"),
                );
                return Ok(seg_through);
            }
            if body_len >= 0
                && body_len as u64 <= len - body_start
                && any_valid_section_from(
                    seg,
                    body_start + body_len as u64,
                    len,
                    look_last,
                    true,
                    replay_buf,
                )?
            {
                hold(
                    seg,
                    format!(
                        "mid-log corruption: section header damaged at offset {pos} but valid \
                         sections follow (not a torn tail)"
                    ),
                );
            }
            return Ok(seg_through); // torn tail
        }
        if body_len < 0 || body_len as u64 > len - body_start {
            // S5: a verified header whose body runs past the end of the file is
            // a torn tail by construction — there is nothing to look ahead at.
            if !is_active {
                hold(
                    seg,
                    format!(
                        "section body extends past the end of a non-final segment at offset {pos}"
                    ),
                );
            }
            return Ok(seg_through);
        }
        let body_end = body_start + body_len as u64;
        match body_crc(seg, pos, body_start, body_end, replay_buf)? {
            None => {
                if !is_active {
                    hold(
                        seg,
                        "non-final segment is shorter than its own sections claim".into(),
                    );
                }
                return Ok(seg_through);
            }
            Some(c) if c != stored_body_crc => {
                // S4. Here `body_end` IS trusted — the header sealed it — so the
                // lookahead starts at a real section boundary and any strictly
                // future LSN proves durable sections follow.
                if !is_active {
                    hold(
                        seg,
                        format!("section body CRC mismatch at offset {pos} in a non-final segment"),
                    );
                    return Ok(seg_through);
                }
                if any_valid_section_from(seg, body_end, len, look_last, false, replay_buf)? {
                    hold(
                        seg,
                        format!(
                            "mid-log corruption: section body CRC mismatch at offset {pos} but \
                             valid sections follow"
                        ),
                    );
                }
                return Ok(seg_through);
            }
            Some(_) => {}
        }

        // The section is whole. Everything from here is a WRITER-defect class:
        // CRC-valid means these bytes were produced deliberately, so the verdict
        // is corruption rather than a torn tail — but it is still HELD, because
        // this segment may be superseded.
        if seg.last_lsn != 0 {
            // Both density checks live under this guard, and that is frozen
            // reference behaviour, not an oversight: 0 doubles as the "no
            // section seen" sentinel, so an unguarded S2 would refuse the first
            // section of every segment on `0 <= 0`. The visible consequence is
            // that a whole LEADING RUN of crafted lsn==0 sections is accepted
            // and replayed while staying invisible to first_lsn/last_lsn.
            if lsn <= seg.last_lsn {
                hold(
                    seg,
                    format!(
                        "section LSN {lsn} at offset {pos} does not follow {}",
                        seg.last_lsn
                    ),
                );
                return Ok(seg_through);
            }
            if lsn != seg.last_lsn + 1 {
                // S9. LSNs are DENSE by construction — one per section, the
                // reservation never burns one, rollback never mints one — so
                // recovery demands them consecutive rather than merely
                // increasing. A gap is what detects a clean whose 'C' sections
                // vanished WHOLLY, leaving the predecessor ending at a clean
                // section boundary so nothing else looks wrong; without it the
                // mark silently authorizes deleting the only surviving copy.
                hold(
                    seg,
                    format!(
                        "section LSNs must be consecutive: {lsn} at offset {pos} after {}",
                        seg.last_lsn
                    ),
                );
                return Ok(seg_through);
            }
        }
        if tag == TAG_MARK {
            match read_mark(seg, body_start, body_len, lsn, seg.seq)? {
                Err(fault) => {
                    hold(seg, fault);
                    return Ok(seg_through);
                }
                Ok((through, log_start)) => {
                    // The reduction is per-SEGMENT-scan and strict: an equal
                    // through later in the same segment does not displace the
                    // first one's logStartLsn. `mark_log_start` is whatever the
                    // last segment-local maximum set it to and is never
                    // re-derived from the global maximum below — frozen
                    // behaviour, pinned by the Java edge tests.
                    if through > seg_through {
                        seg_through = through;
                        *mark_log_start = log_start;
                    }
                }
            }
        }
        if seg.first_lsn == 0 {
            seg.first_lsn = lsn;
        }
        seg.last_lsn = lsn;
        look_last = lsn;
        pos = body_end;
        seg.valid_end = pos;
    }
    if pos < len && !is_active {
        // S6: W3 leaves no trailing bytes below the highest name.
        hold(
            seg,
            format!(
                "non-final segment has {} trailing bytes past its last section",
                len - pos
            ),
        );
    }
    Ok(seg_through)
}

/// Reads and validates one `'K'` body. `Ok(Err(msg))` is a held fault; the outer
/// `Err` is I/O.
#[allow(clippy::type_complexity)]
fn read_mark(
    seg: &Segment,
    body_start: u64,
    body_len: i64,
    lsn: i64,
    seg_seq: i64,
) -> Result<std::result::Result<(i64, i64), String>> {
    if body_len != MARK_BODY_LEN {
        return Ok(Err(format!(
            "clean mark body is {body_len} bytes, not {MARK_BODY_LEN}"
        )));
    }
    let mut body = [0u8; MARK_BODY_LEN as usize];
    if read_at_opt(handle(seg), &mut body, body_start)?.is_none() {
        return Ok(Err("clean mark body is truncated".into()));
    }
    let through = be64(&body, 0);
    let log_start = be64(&body, 8);
    if through <= 0 {
        return Ok(Err(format!(
            "clean mark attests cleanedThroughSeq {through}"
        )));
    }
    if log_start <= 0 || log_start > lsn {
        return Ok(Err(format!(
            "clean mark attests logStartLsn {log_start}, which is not an LSN at or below the \
             mark's own {lsn}"
        )));
    }
    if through >= seg_seq {
        // K4: a mark may never authorize removing its own segment, which is what
        // makes the retained set non-empty by construction.
        return Ok(Err(format!(
            "clean mark in segment {seg_seq} authorizes removing segment {through}, including itself"
        )));
    }
    Ok(Ok((through, log_start)))
}

/// True when `[from, limit)` holds at least one fully valid section, proving
/// that durable committed sections follow a bad one — corruption, not a torn
/// tail.
///
/// A **framed candidate walk**, not a bytewise search and not a full validation:
/// a candidate is framing-valid on its header CRC, tag and body length alone.
/// `'K'` body constraints, entry decoding and the one-entry-per-recid rule are
/// deliberately NOT checked here — a port that calls its complete section
/// validator classifies torn tails differently from the reference.
///
/// With `exact_next` (untrusted anchor: the damaged section's own bodyLen) the
/// candidate must carry EXACTLY `last_lsn + 2`, the damaged section having been
/// `last_lsn + 1`; otherwise (trusted anchor) any strictly future LSN counts.
/// Both reject "embedded fake" patterns from user data holding copies of earlier
/// sections: stale copies carry old LSNs, and under the CRC domain a copied
/// section fails its checksums at any other offset anyway.
///
/// Never crosses a segment boundary — `limit` is this segment's length.
fn any_valid_section_from(
    seg: &Segment,
    from: u64,
    limit: u64,
    last_lsn: i64,
    exact_next: bool,
    replay_buf: usize,
) -> Result<bool> {
    let mut pos = from;
    while pos + SEC_HDR as u64 <= limit {
        let hdr = match read_sec_hdr(seg, pos)? {
            Some(h) => h,
            None => return Ok(false),
        };
        let (tag, lsn, body_len, stored_hdr_crc, stored_body_crc) = parse_sec_hdr(&hdr);
        let body_start = pos + SEC_HDR as u64;
        if hdr_crc(seg, pos, &hdr) != stored_hdr_crc
            || !valid_tag(tag)
            || body_len < 0
            || body_len as u64 > limit - body_start
        {
            return Ok(false);
        }
        let body_end = body_start + body_len as u64;
        // Wrapping, like the reference: `last_lsn` is a number read off a disk
        // that may hold anything, and a candidate LSN that matches the wrapped
        // value is a legitimate (if unreachable) answer, where a panic is not.
        let lsn_ok = if exact_next {
            lsn == last_lsn.wrapping_add(2)
        } else {
            lsn > last_lsn.wrapping_add(1)
        };
        if lsn_ok {
            match body_crc(seg, pos, body_start, body_end, replay_buf)? {
                Some(c) if c == stored_body_crc => return Ok(true),
                None => return Ok(false),
                Some(_) => {}
            }
        }
        pos = body_end;
    }
    Ok(false)
}

// ---------- R4: adjudicate ----------

/// R4. Returns the index at which the **retained** suffix begins — the segments
/// above `cleaned_through`. Verdicts and LSN discontinuities originating below
/// it are DISCARDED: those segments are superseded and about to be deleted, so
/// rot inside them is irrelevant and throwing on it would brick a store over
/// bytes nobody will read.
///
/// Every check here is an **equality between two recorded numbers**. The lowest
/// retained segment's stated start must equal the newest mark's `logStartLsn`,
/// or 1 when there is no mark. Each subsequent segment's stated start must equal
/// where its present predecessor ended — or, when that predecessor holds no
/// section, where IT said it would start, which is exactly what separates W7's
/// legitimately empty rotate target from a segment whose sections vanished. And
/// a segment must hold what its own header promised.
///
/// A missing sequence number needs **no rule at all**: if it held sections its
/// successor's stated start will not match its predecessor's end; if it held
/// none, nothing is missing. That is why the sequence numbers W6 burns on
/// create-crash residue are simply invisible here.
fn adjudicate(segments: &[Segment], cleaned_through: i64, mark_log_start: i64) -> Result<usize> {
    let start = segments
        .iter()
        .position(|s| s.seq > cleaned_through)
        .unwrap_or(segments.len());
    if start == segments.len() {
        // Unreachable: K4 makes a mark's own segment outrank everything it
        // authorizes removing, so the segment holding the newest mark is always
        // retained. Checked rather than assumed, because everything below
        // depends on it.
        return Err(DbError::corrupt_msg(format!(
            "WAL clean mark {cleaned_through} retires the whole segment set"
        )));
    }
    let retained = &segments[start..];
    for s in retained {
        if let Some(held) = &s.held {
            return Err(DbError::corrupt_msg(format!(
                "WAL segment {}: {held}",
                seg_name(s)
            )));
        }
    }
    // The floor runs ALWAYS, not only when there is no anchor. The two witness
    // different things — the chain witnesses LSN continuity, the floor witnesses
    // the mark-image contract — and making them alternatives leaves a hole: a
    // mark with no image behind it, whose superseded segments are still present,
    // satisfies the chain (their data is below the mark, so no LSN is missing)
    // and violates the floor. The open would then succeed, pass 2 would replay
    // only the retained set, and R5 would unlink the segments holding the only
    // copy of the data.
    let expected_start = if mark_log_start > 0 {
        mark_log_start
    } else {
        1
    };
    let mut prev: Option<&Segment> = None;
    for s in retained {
        let stated = s.header_first_lsn();
        match prev {
            None => {
                if stated != expected_start {
                    return Err(DbError::corrupt_msg(format!(
                        "WAL retained log begins at LSN {stated} in {} but {}: sections below it \
                         are gone",
                        seg_name(s),
                        if mark_log_start > 0 {
                            format!("the clean mark attests it begins at {mark_log_start}")
                        } else {
                            "an unmarked log must begin at LSN 1".to_string()
                        }
                    )));
                }
            }
            Some(p) => {
                // Wrapping: `last_lsn` is a disk field, and a crafted section
                // carrying i64::MAX reaches here through the sentinel guard (the
                // first section of a segment is accepted whatever its LSN). The
                // reference wraps; rust would panic.
                let after = if p.last_lsn != 0 {
                    p.last_lsn.wrapping_add(1)
                } else {
                    p.header_first_lsn()
                };
                if stated != after {
                    return Err(DbError::corrupt_msg(format!(
                        "WAL segment {} states it begins at LSN {stated} but {} accounts for LSNs \
                         up to {}: sections between them are gone",
                        seg_name(s),
                        seg_name(p),
                        after.wrapping_sub(1)
                    )));
                }
            }
        }
        // A segment must also hold what its own header promised, or its prefix
        // was lost. The gate is `first_lsn != 0`, NOT "the segment is nonempty":
        // the two differ exactly on a segment holding only crafted lsn==0
        // sections, which is nonempty with first_lsn 0 and whose self-check the
        // reference therefore SKIPS.
        if s.first_lsn != 0 && s.first_lsn != stated {
            return Err(DbError::corrupt_msg(format!(
                "WAL segment {} states it begins at LSN {stated} but its first section is {}: its \
                 leading sections are gone",
                seg_name(s),
                s.first_lsn
            )));
        }
        prev = Some(s);
    }
    Ok(start)
}

fn seg_name(seg: &Segment) -> String {
    super::wal_segments::file_name(&seg.path)
}

// ---------- R6: pass 2 ----------

/// R6, one segment. Pass 1 is the sole authority on section boundaries: this
/// walk re-reads headers it already validated and never re-derives them, so a
/// disagreement between the two passes is impossible by construction.
fn pass2(
    seg: &Segment,
    inner: &StoreDirect,
    ids: &mut Identities,
    replay_buf: usize,
) -> Result<()> {
    let file = handle(seg);
    let mut input = SecIn::new(file, replay_buf);
    let mut pos = SEG_HDR;
    while pos < seg.valid_end {
        let mut hdr = [0u8; SEC_HDR];
        file.read_exact_at(&mut hdr, pos).map_err(DbError::Io)?;
        let (tag, lsn, body_len, _, _) = parse_sec_hdr(&hdr);
        let body_start = pos + SEC_HDR as u64;
        // Unreachable by construction — pass 1 validated exactly these bytes and
        // `valid_end` is where it stopped — and checked anyway, because "cannot
        // happen" plus an `as u64` cast is a panic rather than a wrong answer if
        // the file changes underneath us. `body_start > valid_end` is tested
        // FIRST: a guard whose own subtraction underflows in the scenario it
        // exists for would be worse than no guard at all.
        if body_start > seg.valid_end
            || body_len < 0
            || body_len as u64 > seg.valid_end - body_start
        {
            return Err(DbError::corrupt_msg(format!(
                "WAL segment {} changed between recovery passes at offset {pos}",
                seg_name(seg)
            )));
        }
        // A 'K' body carries no entries and is NEVER passed to the entry
        // decoder; 'C' is semantically identical to 'S' and gets no special
        // handling.
        if tag != TAG_MARK {
            apply_section(
                inner,
                &mut input,
                body_start,
                body_start + body_len as u64,
                lsn,
                ids,
            )?;
        }
        pos = body_start + body_len as u64;
    }
    Ok(())
}

/// Decodes and applies one CRC-verified section body as the §4.2 **state
/// transition table**; a malformed entry is a writer bug or corruption, never a
/// torn tail. `lsn` is the enclosing section's.
///
/// Every row states what happens to BOTH identities, because getting that wrong
/// is how the in-memory tables desynchronize from the store:
///
/// ```text
/// entry             precondition                      action        contentBase  state    skip
/// T_RECORD content  —                                 wal_put       = lsn        = lsn    clear
/// T_RECORD null     —                                 wal_put(null) cleared      = lsn    clear
/// T_PREALLOC        R is not content-live             wal_prealloc  cleared      = lsn    clear
/// T_PREALLOC        R IS content-live                 DataCorruption
/// T_DELETE          —                                 wal_delete    cleared      cleared  clear
/// T_APPEND          baseLsn == contentBase[R]         append        unchanged    unch.    unch.
/// T_APPEND          contentBase[R] absent or > base   SKIP          unchanged    unch.    add R
/// T_APPEND          contentBase[R] < baseLsn          DataCorruption
/// ```
///
/// A superseded `T_RECORD` is RE-APPLIED rather than skipped, which is correct
/// because it is idempotent and costs only recovery-time work.
fn apply_section(
    inner: &StoreDirect,
    input: &mut SecIn,
    start: u64,
    end: u64,
    lsn: i64,
    ids: &mut Identities,
) -> Result<()> {
    input.reset(start, end);
    // At most one entry per recid per section, for 'C' sections as well as 'S'.
    // The classifier coalesces every append() call for a recid into one entry, so
    // a second entry would mean the ordered-replay reasoning no longer applies to
    // this section.
    let mut seen: HashSet<u64> = HashSet::new();
    while input.pos() < end {
        let ty = input.read_byte()?;
        match ty {
            T_PREALLOC => {
                let recid = entry_recid(&mut seen, input.unpack_long()?)?;
                // wal_prealloc no-ops on ANY set slot, so applying it to a
                // content-live record would silently leave a record that is
                // still there while the identities describe a preallocated one.
                // The precondition is "not content-live" rather than "void or
                // already preallocated" to be TOTAL over doctored images: a
                // null-content target matches neither of those and must not fall
                // through undefined.
                if inner.rec_state(recid)? == super::direct::STATE_LIVE {
                    return Err(DbError::corrupt_msg(format!(
                        "WAL PREALLOC over a content-live record, recid={recid}"
                    )));
                }
                inner.wal_prealloc(recid)?;
                ids.state_only(recid, lsn);
            }
            T_DELETE => {
                let recid = entry_recid(&mut seen, input.unpack_long()?)?;
                // Tolerant of a void target on purpose: that is the shape a
                // skipped-append history leaves behind.
                inner.wal_delete(recid)?;
                ids.void(recid);
            }
            T_RECORD => {
                let recid = entry_recid(&mut seen, input.unpack_long()?)?;
                let cap = input.unpack_long()?;
                let len_plus = input.unpack_long()?;
                let mut data: Option<Vec<u8>> = None;
                if len_plus != 0 {
                    let len = len_plus - 1;
                    if len > i32::MAX as u64 || len > input.remaining() {
                        return Err(DbError::corrupt_msg(format!("bad WAL record length {len}")));
                    }
                    let mut b = vec![0u8; len as usize];
                    input.read_fully(&mut b)?;
                    data = Some(b);
                }
                if !cap_valid(cap, data.as_deref()) {
                    return Err(DbError::corrupt_msg(format!(
                        "bad WAL record capacity {cap}"
                    )));
                }
                inner.wal_put(recid, cap as usize, data.as_deref())?;
                match data {
                    None => ids.state_only(recid, lsn),
                    Some(_) => ids.content(recid, lsn),
                }
            }
            T_APPEND => {
                let recid = entry_recid(&mut seen, input.unpack_long()?)?;
                let base_lsn = decode_base_lsn(input.unpack_long()?, lsn, recid)?;
                let len = input.unpack_long()?;
                if len > i32::MAX as u64 || len > input.remaining() {
                    return Err(DbError::corrupt_msg(format!("bad WAL append length {len}")));
                }
                let base = ids.content_base_lsn.get(&recid).copied();
                if let Some(b) = base {
                    if b < base_lsn {
                        // Unreachable in a conforming set (retirement is a prefix
                        // in LSN order, so a base below the current one cannot be
                        // the missing part); defence in depth over S9.
                        return Err(DbError::corrupt_msg(format!(
                            "WAL append cites base LSN {base_lsn} above the applied base {b}, \
                             recid={recid}: sections are missing"
                        )));
                    }
                }
                let mut b = vec![0u8; len as usize];
                // Consumed either way: the frame is still framed.
                input.read_fully(&mut b)?;
                match base {
                    Some(base) if base == base_lsn => {
                        if inner.append(nz(recid), &b)? == AppendResult::Refused {
                            return Err(DbError::corrupt_msg(format!(
                                "WAL append refused, recid={recid}"
                            )));
                        }
                    }
                    // The base this delta extends is gone (cleaned, or superseded
                    // by a newer image that already contains these bytes): skip
                    // and remember.
                    _ => {
                        ids.skipped_appends.insert(recid);
                    }
                }
            }
            other => {
                return Err(DbError::corrupt_msg(format!("bad WAL entry tag {other}")));
            }
        }
    }
    Ok(())
}

/// The recid an entry names, checked once for both rules that apply to it
/// before any of the entry's other fields are read.
///
/// - **one entry per recid per section**, `'C'` sections included. The
///   classifier coalesces every `append()` call for a recid into one entry, so a
///   second entry would mean the ordered-replay reasoning no longer applies.
/// - **recid 0 is reserved** and never allocated, so no conforming writer emits
///   it. Refusing it here is a deliberate port strictness — the reference
///   decoder does not check, and hands 0 to the inner store — of the same class
///   as the port's 10-byte packLong cap: it changes which MALFORMED images are
///   refused, never which conforming ones are accepted, and the corruption
///   fixtures record it per engine rather than assuming uniform strictness.
fn entry_recid(seen: &mut HashSet<u64>, recid: u64) -> Result<u64> {
    if recid == 0 {
        return Err(DbError::corrupt("WAL entry references reserved recid 0"));
    }
    if !seen.insert(recid) {
        return Err(DbError::corrupt_msg(format!(
            "two WAL entries for recid {recid} in one section"
        )));
    }
    Ok(recid)
}

/// Turns the encoded `packLong(lsn - baseLsn)` back into an absolute base LSN,
/// BEFORE any mutation. The delta must be >= 1 — so an append can never cite a
/// base in its own section, and `baseLsn < lsn` always — and must leave a base
/// LSN >= 1, since LSNs start at 1. Both bounds are what make the table's
/// comparison meaningful instead of an accidental "skip" on a garbage value.
fn decode_base_lsn(delta: u64, lsn: i64, recid: u64) -> Result<i64> {
    // Compared as i64 with the same bits the reference sees, and wrapping where
    // it wraps. `lsn` is a number read off a disk that may hold anything — a
    // crafted section is accepted with ANY lsn while the segment is still empty
    // (the sentinel guard) — so `lsn - 1` on i64::MIN is a reachable input, and
    // in rust it is a panic where in Java it is a wrap.
    let delta = delta as i64;
    if delta < 1 || delta > lsn.wrapping_sub(1) {
        return Err(DbError::corrupt_msg(format!(
            "bad WAL append base delta {delta} in section LSN {lsn}, recid={recid}"
        )));
    }
    Ok(lsn.wrapping_sub(delta))
}

// ---------- the ordered algorithm ----------

/// What recovery hands back to the store: where the log continues, and the
/// identities replay rebuilt.
pub(crate) struct Recovered {
    pub(crate) next_lsn: i64,
    pub(crate) identities: Identities,
}

/// The ordered recovery algorithm, R3-R7. R0-R2 (enumerate, classify, remove
/// create-crash residue) already ran inside [`WalSegmentSet::open`].
///
/// 1. **R3** pass 1 — namespace only, no per-recid state: per segment the valid
///    section prefix, any HELD verdict, LSN continuity; globally the newest
///    `'K'`.
/// 2. **R4** adjudicate — discard every verdict from a superseded segment, throw
///    the rest, and check the floor/chain/self equalities over the retained set.
/// 3. **R5** unlink the superseded segments, then fsync the directory.
/// 4. **R6** pass 2 — apply the §4.2 table in ascending (segment, offset) order,
///    then the skip audit.
/// 5. **R7** finish — `nextLsn`, and IFF the active segment's valid prefix is
///    shorter than its length: truncate, force, rotate (W7), fsync.
///
/// **R5 runs before R6**, so an open that refuses can do so with the namespace
/// already pruned. That is deliberate and fixed, not incidental: "a failed open
/// leaves the files untouched" is not a v3 invariant, and the fixture oracles
/// assert an exact post-open file set per row instead.
///
/// A read-only recovery takes every decision identically and performs none of
/// the mutations: no create, no residue delete, no unlink, no truncate, no
/// rotate, no directory fsync. It still computes `next_lsn` and the identities.
pub(crate) fn recover(
    set: &mut WalSegmentSet,
    inner: &StoreDirect,
    replay_buf: usize,
) -> Result<Recovered> {
    let read_only = set.read_only();
    if set.segments().is_empty() {
        // N1: a fresh store. The writable open creates segment 1 (or the
        // successor of whatever sequence numbers residue burned) with
        // firstLsn = 1; the read-only open creates nothing and simply reports
        // where the log would begin.
        if !read_only {
            set.create_segment(1)?;
        }
        inner.rebuild_free_recids()?;
        return Ok(Recovered {
            next_lsn: 1,
            identities: Identities::default(),
        });
    }

    // ---- R3 ----
    let n = set.segments().len();
    let active_idx = n - 1;
    let mut cleaned_through = 0i64;
    let mut mark_log_start = 0i64;
    let mut carry = 0i64;
    for i in 0..n {
        let seg = &mut set.segments_mut()[i];
        seg.ensure_open()?;
        let scanned = scan_segment(seg, carry, i == active_idx, replay_buf, &mut mark_log_start);
        // Released the moment this segment's scan is done, whatever the verdict;
        // pass 2 reopens the ones it needs. This is what bounds the descriptor
        // count to O(1) rather than O(segments) — a store is allowed to reach
        // roughly twice the live data size in log, so a large one means thousands
        // of segments against a default `ulimit -n` of 1024.
        seg.release();
        cleaned_through = cleaned_through.max(scanned?);
        // A segment ending in an accepted lsn==0 section must NOT erase the
        // preceding segment's anchor: `last_lsn` is still 0 there, and the carry
        // stands. Frozen sentinel behaviour again.
        if seg.last_lsn != 0 {
            carry = seg.last_lsn;
        }
    }

    // ---- R4 ----
    let retained_from = adjudicate(set.segments(), cleaned_through, mark_log_start)?;

    // R7's answer is computed over the RETAINED set, deliberately, rather than
    // over every valid section. The two agree on every conforming image — K4 puts
    // a mark's own segment above everything it authorizes removing, and LSNs
    // ascend with (segment, offset) — and differ only on a doctored image where a
    // superseded segment carries a spuriously high LSN, where the global reading
    // would leave a gap that fails S9 on the NEXT open.
    let mut max_valid_lsn = 0i64;
    for s in &set.segments()[retained_from..] {
        max_valid_lsn = max_valid_lsn.max(s.last_lsn);
    }
    // An all-empty retained set holds no LSN to count from, and "0 + 1" would
    // restart the log at 1 — reissuing LSNs a mark already accounted for. The
    // lowest segment's header says where the log begins; that is the answer, and
    // it is why the field is in the header.
    if max_valid_lsn == 0 {
        // The plain subtraction is safe because H9 refuses a header stating
        // `firstLsn <= 0` before a segment ever joins the set
        // (`wal_segments.rs`, table H). Named here because the dependency is
        // invisible at this call site: relaxing H9 would arm an underflow.
        max_valid_lsn = set.segments()[retained_from].header_first_lsn() - 1;
    }

    // ---- R5 ---- (no-op under read-only)
    set.unlink_through(cleaned_through)?;

    // ---- R6 ----
    let mut ids = Identities::default();
    let n = set.segments().len();
    for i in 0..n {
        if set.segments()[i].seq <= cleaned_through {
            continue; // read-only: superseded segments are still on disk
        }
        let is_active = i == n - 1;
        set.segments_mut()[i].ensure_open()?;
        let applied = pass2(&set.segments()[i], inner, &mut ids, replay_buf);
        if !is_active {
            set.segments_mut()[i].release();
        }
        applied?;
    }
    // The audit runs BEFORE R7's truncate: an open that refuses here has mutated
    // nothing further. The bytes a torn tail would lose were never a valid
    // section, so the ordering is conformance and forensics rather than data —
    // but a port that reordered it would fail the fixtures.
    ids.audit()?;

    // ---- R7 ----
    // A RECORDED divergence, not an oversight. The reference adds without
    // checking and opens the store with `nextLsn == i64::MIN`
    // (`StoreWAL.java:547`), which is reachable on a CRC-valid doctored image: a
    // lone segment stating `firstLsn = i64::MAX` whose single 'K' sits at that
    // LSN satisfies K4, the floor and the self check. The port refuses instead,
    // because the reference's answer opens a store that accepts exactly one more
    // commit — written at a negative LSN — and is then permanently unopenable,
    // since the next scan reads that section as S2. Refusing loses nothing a
    // conforming writer could produce (2^63 transactions), and `StoreFull` says
    // what is true: nothing on disk is damaged, the LSN space is used up. If the
    // owner wants byte-for-byte parity with the reference here, `wrapping_add`
    // is the whole change — it is a one-line reversal, deliberately isolated.
    let next_lsn = max_valid_lsn.checked_add(1).ok_or(DbError::StoreFull)?;
    if !read_only {
        let active = set.active().expect("non-empty");
        let (torn, valid_end) = (active.valid_end < active.file_len, active.valid_end);
        if torn {
            // W7. The truncate is not itself forced, so a crash after
            // truncate-then-shorter-reappend can resurface pre-truncation bytes.
            // Force, then rotate, so later appends never reuse this segment's
            // checksum domain at all. Conditional on an ACTUAL truncation:
            // rotating on every open would burn a sequence number per open and
            // demote a legitimate valid-empty highest segment to non-highest (H8).
            // The force ORDERING here — truncate, then a size-persisting force,
            // then rotate — is a claim about operations that leave no trace in
            // the resulting bytes, and A1 could not test it because the port had
            // no I/O seam at all. A2 built one, so it is observable now.
            let io = set.wal_io().clone();
            let active = set.active_mut().expect("non-empty");
            active.ensure_open()?;
            let seq = active.seq;
            wal_io_event(&io, WalOpKind::Truncate, seq, valid_end, 0, 0)?;
            handle(active).set_len(valid_end)?;
            active.file_len = valid_end;
            // The file's SIZE is the payload here: never sync_data.
            wal_io_event(&io, WalOpKind::ForceFull, seq, valid_end, 0, 0)?;
            handle(active).sync_all()?;
            // A RECORDED divergence, and the port's is the better behaviour.
            // The reference never releases the truncated predecessor
            // (`StoreWAL.java:548-560` releases nothing, and pass 2's `finally`
            // exempts the active segment), so a Java store holds that stale
            // channel after every torn-tail open and TWO once the first commit
            // opens the successor's — observable through its own
            // `openSegmentChannelsForTest`. Nothing reads a segment after
            // recovery, and the O(1)-descriptor rule is a tested invariant here
            // (`wal_segments.rs`: steady state is at most the active handle),
            // so the port releases it. Copying the reference would reintroduce
            // the leak in every engine; pinned by the W7 descriptor test below.
            active.release();
            set.create_segment(next_lsn)?;
        }
    }
    // Replay of delete-then-reuse histories leaves stale free-list entries for
    // revived recids: rebuild the allocator's free list from the final index.
    inner.rebuild_free_recids()?;
    Ok(Recovered {
        next_lsn,
        identities: ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::DataOutput2;
    use crate::store::wal_segments::build_header;
    use crate::store::Store;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    // ---------------------------------------------------------------- test kit
    // The rust half of Java's WalTestKit, extended from A0's namespace-only
    // version to whole segment IMAGES: the byte-level recipe for a section, a
    // mark and an entry lives here once, so a hand-built image cannot drift from
    // what the codec above reads — and, through `build_sec_hdr`, from what A2's
    // writer will emit.

    const SEGH: usize = SEG_HDR as usize;

    fn scratch(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "mapdb5_walrec_{}_{}_{}",
            std::process::id(),
            tag,
            n
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn base_in(dir: &Path) -> PathBuf {
        dir.join("store.db")
    }

    fn seg_path(base: &Path, seq: i64) -> PathBuf {
        let mut s = base.as_os_str().to_os_string();
        s.push(format!(".wal.{seq:016x}"));
        PathBuf::from(s)
    }

    /// One segment file under construction.
    struct SegImage {
        seq: i64,
        header: [u8; SEGH],
        bytes: Vec<u8>,
        /// Offset of every section appended, in order.
        offsets: Vec<u64>,
    }

    impl SegImage {
        fn new(seq: i64, first_lsn: i64) -> SegImage {
            let header = build_header(seq, first_lsn);
            SegImage {
                seq,
                header,
                bytes: header.to_vec(),
                offsets: Vec::new(),
            }
        }

        /// A section of any tag, sealed in its own domain at the offset it
        /// actually lands on.
        fn section(mut self, tag: u8, lsn: i64, body: &[u8]) -> SegImage {
            let off = self.bytes.len() as u64;
            let hdr = build_sec_hdr(&self.header, off, tag, lsn, body);
            self.offsets.push(off);
            self.bytes.extend_from_slice(&hdr);
            self.bytes.extend_from_slice(body);
            self
        }

        fn commit(self, lsn: i64, body: Body) -> SegImage {
            self.section(TAG_SECTION, lsn, &body.finish())
        }

        fn image(self, lsn: i64, body: Body) -> SegImage {
            self.section(TAG_IMAGE, lsn, &body.finish())
        }

        fn mark(self, lsn: i64, through: i64, log_start: i64) -> SegImage {
            self.section(TAG_MARK, lsn, &build_mark_body(through, log_start))
        }

        /// Bytes appended past the last section — a torn tail, or S6's trailing
        /// bytes below the highest name.
        fn raw(mut self, bytes: &[u8]) -> SegImage {
            self.bytes.extend_from_slice(bytes);
            self
        }

        /// Truncates the image to `len` bytes: the shape a crash mid-append
        /// leaves.
        fn cut_to(mut self, len: usize) -> SegImage {
            self.bytes.truncate(len);
            self
        }

        /// Flips a bit at an absolute file offset.
        fn damage(mut self, at: u64) -> SegImage {
            self.bytes[at as usize] ^= 0x40;
            self
        }

        fn off(&self, i: usize) -> u64 {
            self.offsets[i]
        }

        fn len(&self) -> usize {
            self.bytes.len()
        }

        fn write(self, base: &Path) -> SegImage {
            std::fs::write(seg_path(base, self.seq), &self.bytes).expect("write segment");
            self
        }
    }

    /// A section body: entries in the packLong framing, built through the port's
    /// own `DataOutput2` so the test kit cannot encode a long differently from
    /// the writer.
    #[derive(Default)]
    struct Body(DataOutput2);

    impl Body {
        fn new() -> Body {
            Body(DataOutput2::with_capacity(64))
        }

        fn prealloc(mut self, recid: u64) -> Body {
            self.0.write_byte(T_PREALLOC as i32);
            self.0.pack_long(recid);
            self
        }

        fn delete(mut self, recid: u64) -> Body {
            self.0.write_byte(T_DELETE as i32);
            self.0.pack_long(recid);
            self
        }

        /// `content: None` is the null record (`len+1 == 0`, capacity 0).
        fn record(mut self, recid: u64, content: Option<&[u8]>) -> Body {
            self.0.write_byte(T_RECORD as i32);
            self.0.pack_long(recid);
            match content {
                None => {
                    self.0.pack_long(0);
                    self.0.pack_long(0);
                }
                Some(d) => {
                    self.0.pack_long(cap_for(d.len()));
                    self.0.pack_long(d.len() as u64 + 1);
                    self.0.write_all(d);
                }
            }
            self
        }

        /// A record with a hand-chosen capacity, for the `capValid` rows.
        fn record_cap(mut self, recid: u64, cap: u64, content: Option<&[u8]>) -> Body {
            self.0.write_byte(T_RECORD as i32);
            self.0.pack_long(recid);
            self.0.pack_long(cap);
            match content {
                None => self.0.pack_long(0),
                Some(d) => {
                    self.0.pack_long(d.len() as u64 + 1);
                    self.0.write_all(d);
                }
            }
            self
        }

        /// `delta` is `sectionLsn - baseLsn`, exactly as the format stores it.
        fn append(mut self, recid: u64, delta: u64, data: &[u8]) -> Body {
            self.0.write_byte(T_APPEND as i32);
            self.0.pack_long(recid);
            self.0.pack_long(delta);
            self.0.pack_long(data.len() as u64);
            self.0.write_all(data);
            self
        }

        fn raw(mut self, bytes: &[u8]) -> Body {
            self.0.write_all(bytes);
            self
        }

        fn finish(self) -> Vec<u8> {
            self.0.buf
        }
    }

    /// The capacity a conforming writer records for `len` content bytes.
    fn cap_for(len: usize) -> u64 {
        let need = 4 + len as u64;
        need.div_ceil(16) * 16
    }

    struct Recovery {
        set: WalSegmentSet,
        inner: StoreDirect,
        rec: Recovered,
    }

    fn try_recover(base: &Path, read_only: bool, replay_buf: usize) -> Result<Recovery> {
        let mut set = WalSegmentSet::open(base, read_only)?;
        let inner = StoreDirect::new_heap_ts(true)?;
        let rec = recover(&mut set, &inner, replay_buf)?;
        Ok(Recovery { set, inner, rec })
    }

    /// The ordinary path: writable, 1 MiB replay window.
    fn open_rw(base: &Path) -> Result<Recovery> {
        try_recover(base, false, 1 << 20)
    }

    fn open_ro(base: &Path) -> Result<Recovery> {
        try_recover(base, true, 1 << 20)
    }

    fn corrupt_msg<T>(r: Result<T>) -> String {
        match r {
            Err(DbError::DataCorruption(c)) => c.to_string(),
            Err(e) => panic!("expected DataCorruption, got {e}"),
            Ok(_) => panic!("expected DataCorruption, got Ok"),
        }
    }

    fn content(r: &Recovery, recid: u64) -> Option<Vec<u8>> {
        r.inner.raw_get(recid).expect("record is live")
    }

    fn is_void(r: &Recovery, recid: u64) -> bool {
        matches!(r.inner.raw_get(recid), Err(DbError::GetVoid(_)))
    }

    fn seqs(r: &Recovery) -> Vec<i64> {
        r.set.segments().iter().map(|s| s.seq).collect()
    }

    fn on_disk(base: &Path) -> Vec<i64> {
        let mut v: Vec<i64> = Vec::new();
        for e in std::fs::read_dir(base.parent().expect("dir")).expect("read_dir") {
            let name = e.expect("entry").file_name();
            let name = name.to_string_lossy().into_owned();
            if let Some(hex) = name.strip_prefix("store.db.wal.") {
                v.push(i64::from_str_radix(hex, 16).expect("hex"));
            }
        }
        v.sort_unstable();
        v
    }

    fn file_len(base: &Path, seq: i64) -> u64 {
        std::fs::metadata(seg_path(base, seq))
            .expect("segment exists")
            .len()
    }

    // ------------------------------------------------- the cross-engine vector

    /// A segment file **written by the Java implementation**, verbatim: the
    /// `reject-wal-java-v3.walseg` fixture, emitted by
    /// `FixtureWriter.writeWalSegFixture` at `b0ed433-dirty` (the generator
    /// commit the xfixtures manifest records). 36-byte header for seq 1 /
    /// firstLsn 1, followed by one complete `'S'` section at LSN 1 carrying a
    /// single `T_RECORD`: recid 1, capacity 112 — Java's
    /// `(4 + 100 + 15) & ~15` for a first fresh put — and 100 bytes of content.
    ///
    /// Embedded rather than loaded from the fixture file on purpose. These bytes
    /// are the VECTOR; the fixture is scheduled to be retired and re-derived at
    /// Stage C, and a vector that moves with the artifact it was copied from
    /// proves nothing.
    const JAVA_SEGMENT: [u8; 165] = [
        0x4d, 0x44, 0x42, 0x53, 0x2e, 0x57, 0x41, 0x4c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x4a, 0x4d, 0x90, 0x4b, 0x53, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0x9d, 0x82, 0x80, 0xb3, 0xe7, 0xf6, 0xab,
        0xe9, 0x02, 0x81, 0xf0, 0xe5, 0x33, 0xb6, 0x39, 0xbc, 0x3f, 0xc2, 0x45, 0xc8, 0x4b, 0xce,
        0x51, 0xd4, 0x57, 0xda, 0x5d, 0xe0, 0x63, 0xe6, 0x69, 0xec, 0x6f, 0xf2, 0x75, 0xf8, 0x7b,
        0xfe, 0x81, 0x04, 0x87, 0x0a, 0x8d, 0x10, 0x93, 0x16, 0x99, 0x1c, 0x9f, 0x22, 0xa5, 0x28,
        0xab, 0x2e, 0xb1, 0x34, 0xb7, 0x3a, 0xbd, 0x40, 0xc3, 0x46, 0xc9, 0x4c, 0xcf, 0x52, 0xd5,
        0x58, 0xdb, 0x5e, 0xe1, 0x64, 0xe7, 0x6a, 0xed, 0x70, 0xf3, 0x76, 0xf9, 0x7c, 0xff, 0x82,
        0x05, 0x88, 0x0b, 0x8e, 0x11, 0x94, 0x17, 0x9a, 0x1d, 0xa0, 0x23, 0xa6, 0x29, 0xac, 0x2f,
        0xb2, 0x35, 0xb8, 0x3b, 0xbe, 0x41, 0xc4, 0x47, 0xca, 0x4d, 0xd0, 0x53, 0xd6, 0x59, 0xdc,
    ];

    #[test]
    fn a_java_written_section_decodes_to_the_record_java_put_in_it() {
        let dir = scratch("java_vector");
        let base = base_in(&dir);
        std::fs::write(seg_path(&base, 1), JAVA_SEGMENT).expect("write");
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 1), Some(JAVA_SEGMENT[65..].to_vec()));
        assert_eq!(r.rec.next_lsn, 2);
        assert_eq!(r.rec.identities.content_base_lsn.get(&1), Some(&1));
        // What the Java side actually stored, derived from its generator rather
        // than from these bytes: `payload(51, 100)` is `(i * 131 + 51) & 0xff`.
        // Checking the recovered CONTENT against that formula tests the vector's
        // meaning independently of its transcription.
        let recovered = content(&r, 1).expect("live");
        assert_eq!(recovered.len(), 100);
        for (i, b) in recovered.iter().enumerate() {
            assert_eq!(*b as usize, (i * 131 + 51) & 0xff, "payload byte {i}");
        }
    }

    #[test]
    fn this_ports_encoder_reproduces_the_java_bytes_exactly() {
        // The test that catches PAIRED drift, which nothing else here can: every
        // other image in this module is built by the same code that reads it, so
        // moving both sides to a different endianness, CRC polynomial or domain
        // recipe would leave the whole suite green. These 165 bytes came from
        // the other implementation.
        let payload = &JAVA_SEGMENT[65..];
        let rebuilt = SegImage::new(1, 1).commit(1, Body::new().record_cap(1, 112, Some(payload)));
        assert_eq!(rebuilt.bytes, JAVA_SEGMENT.to_vec());
    }

    // ------------------------------------------------------ the CRC domain

    #[test]
    fn a_section_is_bound_to_its_segment() {
        let dir = scratch("dom_seg");
        let base = base_in(&dir);
        // Two segments whose sections sit at identical offsets. Segment 2's
        // section is byte-copied from segment 1's, which is what an operator
        // "repairing" a log by copying a good segment over a bad one does.
        let one = SegImage::new(1, 1).commit(1, Body::new().record(10, Some(b"one")));
        let two = SegImage::new(2, 2).commit(2, Body::new().record(10, Some(b"one")));
        assert_eq!(one.len(), two.len(), "same shape, so same offsets");
        let mut forged = two.bytes.clone();
        forged[SEGH..].copy_from_slice(&one.bytes[SEGH..]);
        one.write(&base);
        std::fs::write(seg_path(&base, 2), &forged).expect("write");

        // The forged section fails its header CRC in segment 2's domain. Segment
        // 2 is the highest name, so that reads as a torn tail: the section is
        // discarded, not replayed.
        let r = open_rw(&base).expect("opens");
        assert_eq!(r.set.segments()[1].valid_end, SEG_HDR);
        assert_eq!(r.rec.next_lsn, 2, "only segment 1's section counted");
    }

    #[test]
    fn a_section_is_bound_to_its_offset() {
        let dir = scratch("dom_off");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"b")));
        // Move the FIRST section's bytes over the second one's: same segment,
        // same length, different offset.
        let (a, b) = (img.off(0) as usize, img.off(1) as usize);
        let n = b - a;
        let mut bytes = img.bytes.clone();
        let first = bytes[a..b].to_vec();
        bytes[b..b + n].copy_from_slice(&first);
        std::fs::write(seg_path(&base, 1), &bytes).expect("write");

        let r = open_rw(&base).expect("opens");
        assert_eq!(r.rec.next_lsn, 2, "the relocated copy is not a section");
        assert_eq!(content(&r, 10), Some(b"a".to_vec()));
        assert!(is_void(&r, 11));
    }

    #[test]
    fn the_domain_covers_the_segments_stated_start() {
        let dir = scratch("dom_first");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1).commit(1, Body::new().record(10, Some(b"a")));
        let mut bytes = img.bytes.clone();
        // Restate firstLsn as 2 and RESEAL, so the header itself stays valid and
        // the edit is only visible through the section CRCs it invalidates.
        bytes[24..32].copy_from_slice(&2i64.to_be_bytes());
        let crc = crc32fast::hash(&bytes[..32]) as i32;
        bytes[32..36].copy_from_slice(&crc.to_be_bytes());
        std::fs::write(seg_path(&base, 1), &bytes).expect("write");

        // The section no longer verifies, so the segment reads as empty — and an
        // empty segment must then satisfy the floor with its restated start,
        // which says 2 where an unmarked log must begin at 1.
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("must begin at LSN 1"), "{msg}");
    }

    #[test]
    fn a_body_larger_than_the_replay_window_verifies_and_replays() {
        let dir = scratch("stream");
        let base = base_in(&dir);
        let big = vec![0xA5u8; 300_000];
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(&big)))
            .write(&base);
        // A window far smaller than the body forces refills in both the CRC pass
        // and the entry decoder.
        let r = try_recover(&base, false, 64).expect("opens");
        assert_eq!(content(&r, 10), Some(big));
    }

    // ------------------------------------------------------ table S

    #[test]
    fn a_torn_tail_in_the_active_segment_truncates_forces_and_rotates() {
        let dir = scratch("torn");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"b")));
        let keep = img.off(1) as usize + 12; // half of the second section's header
        img.cut_to(keep).write(&base);

        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 10), Some(b"a".to_vec()));
        assert!(is_void(&r, 11));
        assert_eq!(r.rec.next_lsn, 2);
        // W7: truncated to the valid prefix, and a successor created so the old
        // CRC domain is never appended to again.
        assert_eq!(on_disk(&base), vec![1, 2]);
        assert_eq!(file_len(&base, 1), SEG_HDR + SEC_HDR as u64 + 5);
        assert_eq!(r.set.active().expect("active").seq, 2);
        assert_eq!(r.set.active().expect("active").header_first_lsn(), 2);
    }

    #[test]
    fn w7_leaves_no_stale_descriptor_and_no_stale_accounting() {
        let dir = scratch("w7_state");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"b")));
        let keep = img.off(1) as usize + 12;
        img.cut_to(keep).write(&base);
        let r = open_rw(&base).expect("opens");

        // The cached length must follow the truncation, or `create_segment`
        // charges the PRE-truncate length into `sealed_bytes` and every later
        // reader of `log_bytes` — the cleaner's trigger, in A3 — works from an
        // inflated log size. Compared against the bytes actually on disk,
        // because the set's own two accessors read the same cached field and
        // would agree with each other while both being wrong.
        let on_device: u64 = on_disk(&base).iter().map(|&s| file_len(&base, s)).sum();
        assert_eq!(r.set.log_bytes(), on_device);
        assert_eq!(r.set.log_bytes(), r.set.log_bytes_exact());

        // And the deliberate divergence: the reference keeps the truncated
        // predecessor's channel open here. See the comment at the release.
        assert_eq!(r.set.open_file_count(), 0, "no handle survives W7");
    }

    #[test]
    fn an_untorn_open_does_not_rotate() {
        let dir = scratch("no_rotate");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        // Rotating on every open would burn a sequence number per open and
        // demote a legitimately empty highest segment to non-highest (H8).
        assert_eq!(on_disk(&base), vec![1]);
        assert_eq!(r.rec.next_lsn, 2);
    }

    #[test]
    fn a_damaged_header_followed_by_the_exact_next_section_is_corruption() {
        let dir = scratch("s3_exact");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"b")))
            .commit(3, Body::new().record(12, Some(b"c")));
        // Rot the second section's TAG. Its declared bodyLen survives, so the
        // walk starts exactly at section 3, which carries lastLsn+2 == 3.
        let at = img.off(1);
        img.damage(at).write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("mid-log corruption"), "{msg}");
        assert!(msg.contains("header damaged"), "{msg}");
    }

    #[test]
    fn a_damaged_header_with_nothing_after_it_is_a_torn_tail() {
        let dir = scratch("s3_tail");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"b")));
        let at = img.off(1);
        img.damage(at).write(&base);
        let r = open_rw(&base).expect("opens: torn tail");
        assert_eq!(r.rec.next_lsn, 2);
        assert!(is_void(&r, 11));
    }

    #[test]
    fn the_lookahead_wants_exactly_the_next_lsn_after_a_damaged_header() {
        let dir = scratch("s3_wrong_lsn");
        let base = base_in(&dir);
        // Sections 1, 2, then a section carrying LSN 9: valid in itself, but not
        // the LSN that would have followed the damaged one. The reference calls
        // that a torn tail — the untrusted anchor makes "something valid is over
        // there" too weak a proof on its own.
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"b")))
            .commit(9, Body::new().record(12, Some(b"c")));
        let at = img.off(1);
        img.damage(at).write(&base);
        let r = open_rw(&base).expect("opens: torn tail");
        assert_eq!(r.rec.next_lsn, 2);
    }

    #[test]
    fn a_body_crc_mismatch_followed_by_any_future_lsn_is_corruption() {
        let dir = scratch("s4_mid");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"bbbb")))
            .commit(3, Body::new().record(12, Some(b"c")));
        // Rot a BODY byte: the header still seals the section's end, so the
        // lookahead anchor is trusted and any strictly future LSN proves rot.
        let at = img.off(1) + SEC_HDR as u64 + 2;
        img.damage(at).write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("body CRC mismatch"), "{msg}");
    }

    #[test]
    fn a_body_crc_mismatch_at_the_end_is_a_torn_tail() {
        let dir = scratch("s4_tail");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"bbbb")));
        let at = img.off(1) + SEC_HDR as u64 + 2;
        img.damage(at).write(&base);
        let r = open_rw(&base).expect("opens: torn tail");
        assert_eq!(r.rec.next_lsn, 2);
        assert!(is_void(&r, 11));
    }

    #[test]
    fn a_clean_mark_counts_as_proof_that_sections_follow() {
        let dir = scratch("k_lookahead");
        let base = base_in(&dir);
        // THE flagged trap: a port whose validTag is v1's two-tag set does not
        // see the 'K' here, reports "nothing valid follows", and silently
        // truncates deliberate mid-log rot away as a torn tail.
        let img = SegImage::new(2, 3)
            .commit(3, Body::new().record(10, Some(b"a")))
            .commit(4, Body::new().record(11, Some(b"b")))
            .mark(5, 1, 3);
        let at = img.off(1) + SEC_HDR as u64;
        img.damage(at).write(&base);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(1, Some(b"x")))
            .commit(2, Body::new().record(2, Some(b"y")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("mid-log corruption"), "{msg}");
    }

    #[test]
    fn a_damaged_section_below_the_highest_name_is_corruption_without_a_lookahead() {
        let dir = scratch("s3_nonfinal");
        let base = base_in(&dir);
        // Nothing follows the damaged section inside segment 1, so in the active
        // segment this shape would be a legal torn tail. Below the highest name
        // W3 rules that out: a sealed segment ends exactly at a section boundary.
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"b")));
        let at = img.off(1);
        img.damage(at).write(&base);
        SegImage::new(2, 3)
            .commit(3, Body::new().record(12, Some(b"c")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("non-final segment"), "{msg}");
    }

    #[test]
    fn trailing_bytes_below_the_highest_name_are_corruption() {
        let dir = scratch("s6");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .raw(&[0u8; 7])
            .write(&base);
        SegImage::new(2, 2)
            .commit(2, Body::new().record(11, Some(b"b")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("trailing bytes"), "{msg}");
    }

    #[test]
    fn trailing_bytes_in_the_active_segment_are_a_torn_tail() {
        let dir = scratch("s6_active");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .raw(&[0u8; 7])
            .write(&base);
        let valid = img.len() - 7;
        let r = open_rw(&base).expect("opens");
        assert_eq!(r.rec.next_lsn, 2);
        assert_eq!(file_len(&base, 1), valid as u64, "truncated to the prefix");
        assert_eq!(on_disk(&base), vec![1, 2], "and rotated");
    }

    #[test]
    fn a_body_running_past_the_end_is_a_torn_tail_in_the_active_segment() {
        let dir = scratch("s5");
        let base = base_in(&dir);
        // A CRC-valid header whose body was never written: the crash shape a
        // header-first writer produces.
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"bbbbbbbbbb")));
        let keep = img.off(1) as usize + SEC_HDR;
        img.cut_to(keep).write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(r.rec.next_lsn, 2);
    }

    #[test]
    fn a_crc_valid_section_under_an_unknown_tag_is_not_a_section() {
        let dir = scratch("s3_tag");
        let base = base_in(&dir);
        // Sealed correctly in its own domain, so only `valid_tag` separates it
        // from a real section. A scanner that accepted it would hand its body to
        // the entry decoder and replay data the reference discards as a damaged
        // active tail.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .section(b'X', 2, &Body::new().record(11, Some(b"b")).finish())
            .write(&base);
        let r = open_rw(&base).expect("opens: torn tail");
        assert_eq!(content(&r, 10), Some(b"a".to_vec()));
        assert!(is_void(&r, 11), "the unknown tag was never replayed");
        assert_eq!(r.rec.next_lsn, 2);
    }

    #[test]
    fn the_trusted_anchor_accepts_any_future_lsn_not_just_the_next() {
        let dir = scratch("s4_distant");
        let base = base_in(&dir);
        // The damaged section's own header sealed where its body ends, so the
        // walk starts at a REAL section boundary and does not need the exact
        // next LSN — this is what separates the trusted anchor from the
        // untrusted one, and LSN 9 satisfies only the relaxed rule.
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"bbbb")))
            .commit(9, Body::new().record(12, Some(b"c")));
        let at = img.off(1) + SEC_HDR as u64 + 2;
        img.damage(at).write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("body CRC mismatch"), "{msg}");
    }

    #[test]
    fn the_lookahead_walks_past_a_framed_candidate_that_does_not_qualify() {
        let dir = scratch("s4_walk");
        let base = base_in(&dir);
        // A framed candidate carrying a stale LSN does not end the search: the
        // walk advances by that candidate's own length and keeps looking. A port
        // that returned false at the first non-qualifying frame would call this
        // a torn tail and truncate committed sections away.
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"bbbb")))
            .commit(2, Body::new().record(12, Some(b"stale")))
            .commit(5, Body::new().record(13, Some(b"c")));
        let at = img.off(1) + SEC_HDR as u64 + 2;
        img.damage(at).write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("mid-log corruption"), "{msg}");
    }

    #[test]
    fn a_lookahead_candidate_must_pass_its_own_body_crc_too() {
        let dir = scratch("s4_cand_body");
        let base = base_in(&dir);
        // The clause that says a candidate proves nothing unless its BODY also
        // verifies. Section 3 is framed correctly and carries a qualifying LSN,
        // so a walk that stopped at the LSN test would call this mid-log
        // corruption; its body is damaged too, so nothing here is a durable
        // section and the reference truncates the pair as a torn tail.
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"bbbb")))
            .commit(3, Body::new().record(12, Some(b"cccc")));
        let (two, three) = (img.off(1), img.off(2));
        let img = img
            .damage(two + SEC_HDR as u64 + 2)
            .damage(three + SEC_HDR as u64 + 2);
        img.write(&base);
        let r = open_rw(&base).expect("opens: torn tail");
        assert_eq!(r.rec.next_lsn, 2);
        assert_eq!(content(&r, 10), Some(b"a".to_vec()));
        assert!(is_void(&r, 12));
    }

    #[test]
    fn a_body_crc_mismatch_below_the_highest_name_is_corruption() {
        let dir = scratch("s4_nonfinal");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"aaaa")))
            .commit(2, Body::new().record(11, Some(b"b")));
        let at = img.off(0) + SEC_HDR as u64 + 2;
        img.damage(at).write(&base);
        SegImage::new(2, 3)
            .commit(3, Body::new().record(12, Some(b"c")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("body CRC mismatch"), "{msg}");
        assert!(msg.contains("non-final segment"), "{msg}");
    }

    #[test]
    fn a_body_past_the_end_below_the_highest_name_is_corruption() {
        let dir = scratch("s5_nonfinal");
        let base = base_in(&dir);
        let img = SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"bbbbbbbbbb")));
        let keep = img.off(1) as usize + SEC_HDR;
        img.cut_to(keep).write(&base);
        SegImage::new(2, 3)
            .commit(3, Body::new().record(12, Some(b"c")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("extends past the end"), "{msg}");
    }

    #[test]
    fn a_repeated_lsn_is_held_even_in_the_active_segment() {
        let dir = scratch("s2");
        let base = base_in(&dir);
        // CRC-valid means deliberate: a writer-defect class, refused rather than
        // truncated away, even at the very end of the highest segment.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(1, Body::new().record(11, Some(b"b")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("does not follow"), "{msg}");
    }

    #[test]
    fn an_lsn_gap_is_held_even_in_the_active_segment() {
        let dir = scratch("s9");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(3, Body::new().record(11, Some(b"b")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("must be consecutive"), "{msg}");
    }

    // ------------------------------------------------------ table K

    /// Every mark-body fault, and the message that names it. All of them are
    /// HELD, so a mark in a superseded segment can still be wrong without
    /// bricking the store — which the next test proves.
    #[test]
    fn every_malformed_mark_is_held() {
        /// name, the mark to append, the phrase that must name the fault.
        type Case = (&'static str, fn(SegImage) -> SegImage, &'static str);
        let cases: Vec<Case> = vec![
            (
                "wrong body length",
                |s: SegImage| s.section(TAG_MARK, 3, &[0u8; 8]),
                "clean mark body is 8 bytes",
            ),
            (
                "through 0",
                |s: SegImage| s.mark(3, 0, 1),
                "attests cleanedThroughSeq 0",
            ),
            (
                "negative through",
                |s: SegImage| s.mark(3, -4, 1),
                "attests cleanedThroughSeq -4",
            ),
            (
                "log start 0",
                |s: SegImage| s.mark(3, 1, 0),
                "attests logStartLsn 0",
            ),
            (
                "log start above the mark's own lsn",
                |s: SegImage| s.mark(3, 1, 4),
                "attests logStartLsn 4",
            ),
            (
                "K4: authorizes its own segment",
                |s: SegImage| s.mark(3, 2, 1),
                "including itself",
            ),
        ];
        for (name, build, expected) in cases {
            let dir = scratch("k_bad");
            let base = base_in(&dir);
            SegImage::new(1, 1)
                .commit(1, Body::new().record(10, Some(b"a")))
                .write(&base);
            build(SegImage::new(2, 2).commit(2, Body::new().record(11, Some(b"b")))).write(&base);
            let msg = corrupt_msg(open_rw(&base));
            assert!(msg.contains(expected), "{name}: {msg}");
        }
    }

    #[test]
    fn a_valid_mark_unlinks_the_segments_below_it_and_fsyncs_once() {
        let dir = scratch("k_unlink");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"gone")))
            .write(&base);
        SegImage::new(2, 2)
            .commit(2, Body::new().record(11, Some(b"gone too")))
            .write(&base);
        // Segment 3 is self-contained: it re-states recid 10 as a 'C' image, and
        // its mark says the log now begins at its own LSN 3.
        SegImage::new(3, 3)
            .image(3, Body::new().record(10, Some(b"kept")))
            .mark(4, 2, 3)
            .write(&base);

        let r = open_rw(&base).expect("opens");
        assert_eq!(on_disk(&base), vec![3], "R5 removed the superseded prefix");
        assert_eq!(seqs(&r), vec![3]);
        assert_eq!(content(&r, 10), Some(b"kept".to_vec()));
        assert!(is_void(&r, 11), "a superseded segment is never replayed");
        assert_eq!(r.rec.next_lsn, 5);
        assert_eq!(r.set.dir_fsyncs(), 1, "one fsync after the batch");
    }

    #[test]
    fn a_mark_below_the_active_segment_still_wins() {
        let dir = scratch("k_below_active");
        let base = base_in(&dir);
        // The conforming shape a clean followed by a rotation leaves: the
        // winning mark is NOT in the highest segment. Worth its own test
        // because every other mark test here puts the mark in the active
        // segment, so "the last scan wins" and "the maximum wins" would agree
        // on all of them.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"gone")))
            .write(&base);
        SegImage::new(2, 2)
            .image(2, Body::new().record(11, Some(b"kept")))
            .mark(3, 1, 2)
            .write(&base);
        SegImage::new(3, 4).write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(on_disk(&base), vec![2, 3]);
        assert_eq!(content(&r, 11), Some(b"kept".to_vec()));
        assert_eq!(r.rec.next_lsn, 4);
    }

    #[test]
    fn within_one_segment_the_greatest_mark_wins_not_the_newest() {
        let dir = scratch("k_greatest");
        let base = base_in(&dir);
        for seq in 1..=2 {
            SegImage::new(seq, seq)
                .commit(seq, Body::new().record(seq as u64, Some(b"x")))
                .write(&base);
        }
        SegImage::new(3, 3)
            .image(3, Body::new().record(9, Some(b"i")))
            .mark(4, 2, 3) // greater
            .mark(5, 1, 1) // newer, but lesser: does not displace it
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(on_disk(&base), vec![3]);
        assert_eq!(r.rec.next_lsn, 6);
    }

    #[test]
    fn a_later_greater_mark_in_the_same_segment_does_displace_the_first() {
        let dir = scratch("k_later_greater");
        let base = base_in(&dir);
        // The mirror of the equal-mark test, and it moves BOTH halves of the
        // reduction: the greater mark raises the removal boundary (segment 2
        // goes as well as segment 1) and re-points the log start (2 → 3, which
        // is what the retained header states).
        SegImage::new(1, 1)
            .commit(1, Body::new().record(1, Some(b"x")))
            .write(&base);
        SegImage::new(2, 2)
            .commit(2, Body::new().record(2, Some(b"y")))
            .write(&base);
        SegImage::new(3, 3)
            .image(3, Body::new().record(9, Some(b"i")))
            .mark(4, 1, 2)
            .mark(5, 2, 3)
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(on_disk(&base), vec![3], "the greater mark set the boundary");
        assert_eq!(r.rec.next_lsn, 6);
    }

    #[test]
    fn an_equal_mark_does_not_displace_the_first_one() {
        let dir = scratch("k_equal");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(1, Some(b"x")))
            .write(&base);
        // Two marks attesting the SAME through. The reduction is strict
        // (`through > local`), so the FIRST one's logStartLsn stands — and here
        // the two disagree, so the floor check is what observes which won.
        SegImage::new(2, 2)
            .image(2, Body::new().record(9, Some(b"i")))
            .mark(3, 1, 2)
            .mark(4, 1, 3)
            .write(&base);
        let r = open_rw(&base).expect("the first mark's logStartLsn 2 matches the header");
        assert_eq!(on_disk(&base), vec![2]);
        assert_eq!(r.rec.next_lsn, 5);
    }

    #[test]
    fn the_log_start_comes_from_the_last_segment_holding_a_mark() {
        let dir = scratch("k_last_seg");
        let base = base_in(&dir);
        // The reduction is per-SEGMENT-SCAN: `segThrough` restarts at 0 in every
        // segment, so a later segment's mark sets markLogStartLsn even when its
        // `through` merely EQUALS the global maximum. Both marks here attest
        // through 1; they disagree about where the log starts, and only the
        // later segment's answer (2) matches the lowest retained header. A port
        // that re-derived the field from the global reduction would take the
        // first mark's 1 and refuse this image.
        SegImage::new(1, 1)
            .image(1, Body::new().record(9, Some(b"i")))
            .write(&base);
        SegImage::new(2, 2).mark(2, 1, 1).write(&base);
        SegImage::new(3, 3).mark(3, 1, 2).write(&base);
        let r = open_rw(&base).expect("the LAST segment's mark names the log start");
        assert_eq!(on_disk(&base), vec![2, 3]);
        assert_eq!(seqs(&r), vec![2, 3]);
        assert_eq!(r.rec.next_lsn, 4);
    }

    #[test]
    fn a_hold_stops_the_segments_scan_where_it_stands() {
        let dir = scratch("k_hold_stops");
        let base = base_in(&dir);
        // White-box, because this rule is invisible from outside `recover`: a
        // mark AFTER a fault in the same segment must not be collected, and the
        // valid prefix must end at the fault. Both facts feed decisions taken
        // elsewhere (which segments R5 removes, where W7 truncates), so the
        // ordering is pinned here rather than left to a fixture that cannot see
        // it. First message wins, too.
        let img = SegImage::new(3, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(1, Body::new().record(11, Some(b"repeat"))) // S2: held
            .mark(2, 1, 1)
            .write(&base);
        let fault_at = img.off(1);

        let mut set = WalSegmentSet::open(&base, false).expect("namespace opens");
        let seg = &mut set.segments_mut()[0];
        seg.ensure_open().expect("open");
        let mut mark_log_start = 0i64;
        let through = scan_segment(seg, 0, true, 1 << 20, &mut mark_log_start).expect("scan");
        assert_eq!(through, 0, "the mark past the fault was never examined");
        assert_eq!(mark_log_start, 0);
        assert_eq!(seg.valid_end, fault_at);
        assert!(seg
            .held
            .as_deref()
            .expect("held")
            .contains("does not follow"));
    }

    // ------------------------------------------------------ R4

    #[test]
    fn a_held_verdict_in_a_superseded_segment_is_discarded() {
        let dir = scratch("r4_discard");
        let base = base_in(&dir);
        // Segment 1 is rotten in a way that is corruption on its own (an LSN gap
        // in a CRC-valid section). It is below the mark, so refusing here would
        // brick a store over bytes about to be deleted.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(7, Body::new().record(11, Some(b"b")))
            .write(&base);
        SegImage::new(2, 8)
            .image(8, Body::new().record(10, Some(b"kept")))
            .mark(9, 1, 8)
            .write(&base);
        let r = open_rw(&base).expect("opens: the verdict is below the mark");
        assert_eq!(on_disk(&base), vec![2]);
        assert_eq!(content(&r, 10), Some(b"kept".to_vec()));
    }

    #[test]
    fn a_held_verdict_in_a_retained_segment_refuses() {
        let dir = scratch("r4_retained");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .image(1, Body::new().record(10, Some(b"i")))
            .mark(2, 0, 0) // malformed: held, and this segment is retained
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("cleanedThroughSeq 0"), "{msg}");
    }

    #[test]
    fn an_unmarked_log_must_begin_at_lsn_1() {
        let dir = scratch("r4_floor");
        let base = base_in(&dir);
        SegImage::new(1, 5)
            .commit(5, Body::new().record(10, Some(b"a")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("must begin at LSN 1"), "{msg}");
        assert!(msg.contains("sections below it are gone"), "{msg}");
    }

    #[test]
    fn the_floor_refuses_a_retained_log_that_starts_below_the_mark() {
        let dir = scratch("r4_floor_mark");
        let base = base_in(&dir);
        // The mark says the log begins at 3, but the lowest retained segment
        // states 2: the image the mark was issued against is not there. Nothing
        // else notices — the chain is satisfied, because segment 1's data is
        // below the mark and no LSN is missing.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, 2)
            .commit(2, Body::new().record(11, Some(b"b")))
            .mark(3, 1, 3)
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("clean mark attests it begins at 3"), "{msg}");
    }

    #[test]
    fn the_chain_refuses_a_segment_whose_sections_are_gone() {
        let dir = scratch("r4_chain");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        // Segment 2 states it begins at 4: LSNs 2 and 3 are accounted for by
        // nobody. A missing sequence NUMBER needs no rule — this is how the loss
        // surfaces.
        SegImage::new(2, 4)
            .commit(4, Body::new().record(11, Some(b"b")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("sections between them are gone"), "{msg}");
    }

    #[test]
    fn an_empty_segment_chains_by_its_stated_start() {
        let dir = scratch("r4_empty");
        let base = base_in(&dir);
        // W7's rotate target: created, never appended to. It must chain by what
        // it SAID it would start at, which is what separates "always empty" from
        // "its sections vanished" (H8).
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, 2).write(&base);
        SegImage::new(3, 2)
            .commit(2, Body::new().record(11, Some(b"b")))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(r.rec.next_lsn, 3);
        assert_eq!(content(&r, 11), Some(b"b".to_vec()));
    }

    #[test]
    fn the_self_check_refuses_a_segment_missing_its_leading_sections() {
        let dir = scratch("r4_self");
        let base = base_in(&dir);
        // Header says the segment starts at 1; its first section is 2. Under the
        // chain alone this passes at the head of the log, so the self check is
        // what catches it.
        SegImage::new(1, 1)
            .commit(2, Body::new().record(10, Some(b"a")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("its leading sections are gone"), "{msg}");
    }

    #[test]
    fn the_self_check_runs_on_every_retained_segment_not_just_the_first() {
        let dir = scratch("r4_self_later");
        let base = base_in(&dir);
        // Segment 2's stated start chains correctly off segment 1, so the chain
        // is satisfied and only the self check sees that its own first section
        // is not the one it promised.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, 2)
            .commit(3, Body::new().record(11, Some(b"b")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("its leading sections are gone"), "{msg}");
        assert!(msg.contains("0000000000000002"), "{msg}");
    }

    #[test]
    fn the_next_lsn_comes_from_the_retained_set_not_from_every_valid_section() {
        let dir = scratch("r7_retained_max");
        let base = base_in(&dir);
        // A superseded segment carrying a spuriously high LSN. Reading the
        // maximum globally would set nextLsn to 51 and leave a gap that fails
        // S9 on the NEXT open; the reference takes it over the retained set,
        // and records the deviation from its own prose for exactly this image.
        SegImage::new(1, 50)
            .commit(50, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, 3)
            .image(3, Body::new().record(11, Some(b"i")))
            .mark(4, 1, 3)
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(on_disk(&base), vec![2]);
        assert_eq!(r.rec.next_lsn, 5, "4 + 1, not 50 + 1");
    }

    #[test]
    fn an_exhausted_lsn_space_refuses_instead_of_wrapping() {
        let dir = scratch("r7_exhausted");
        let base = base_in(&dir);
        // A RECORDED divergence from the frozen reference, and the image that
        // reaches it: one segment stating firstLsn = i64::MAX whose single 'K'
        // sits at that LSN passes K4, the floor and the self check. The
        // reference opens with nextLsn = i64::MIN — a store that takes one more
        // commit, at a negative LSN, and is then unopenable because the next
        // scan reads that section as S2. `StoreFull` is the honest answer: the
        // bytes are intact, the LSN space is used up. Nothing a conforming
        // writer can produce reaches here (2^63 transactions).
        SegImage::new(2, i64::MAX)
            .mark(i64::MAX, 1, i64::MAX)
            .write(&base);
        match open_rw(&base) {
            Err(DbError::StoreFull) => {}
            other => panic!("expected StoreFull, got {:?}", other.map(|_| "Ok")),
        }
    }

    #[test]
    fn an_lsn_at_the_top_of_the_range_does_not_panic_the_chain() {
        let dir = scratch("r4_maxlsn");
        let base = base_in(&dir);
        // Reachable, not theoretical: the density checks do not apply to a
        // segment's FIRST section, so one crafted section can carry i64::MAX,
        // and a mark attesting `logStartLsn = i64::MAX` makes the floor accept
        // the header that states it. The chain then computes `last_lsn + 1` on
        // i64::MAX — a wrap in the reference and a PANIC in a debug rust build,
        // which is the difference between refusing an image and taking the
        // process down with it.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, i64::MAX)
            .mark(i64::MAX, 1, i64::MAX)
            .write(&base);
        SegImage::new(3, 1).write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("sections between them are gone"), "{msg}");
    }

    #[test]
    fn an_append_base_delta_cannot_overflow_the_section_lsn() {
        // The decoder's bounds are compared in i64 with the reference's bits and
        // wrap where it wraps. R4's self check happens to gate every negative
        // `lsn` out of pass 2 today, so this exercises the arithmetic directly
        // rather than through an image — the guard is there because that gating
        // is a property of a DIFFERENT rule, not of this function.
        // The reference's own answer on the extreme: `lsn - 1` wraps to
        // i64::MAX, so the delta passes its bound and the base wraps too. Pinned
        // as-is — a port that "fixed" it by refusing would accept a different
        // set of images than the reference on a doctored one.
        assert_eq!(decode_base_lsn(1, i64::MIN, 10).expect("wraps"), i64::MAX);
        // A delta above i64::MAX is negative in the reference's arithmetic, and
        // that is what rejects it — not an unsigned comparison.
        assert!(decode_base_lsn(u64::MAX, i64::MIN, 10).is_err());
        assert!(decode_base_lsn(0, 5, 10).is_err());
        assert!(decode_base_lsn(5, 5, 10).is_err());
        assert!(decode_base_lsn(u64::MAX, 5, 10).is_err());
        assert_eq!(decode_base_lsn(4, 5, 10).expect("valid"), 1);
        assert_eq!(
            decode_base_lsn(1, i64::MAX, 10).expect("valid"),
            i64::MAX - 1
        );
    }

    // ------------------------- the frozen lsn==0 sentinel edge (J0 pins these)

    #[test]
    fn a_leading_run_of_lsn_zero_sections_is_accepted_and_replayed() {
        let dir = scratch("z_run");
        let base = base_in(&dir);
        // Both density checks live under `last_lsn != 0`, so a whole LEADING RUN
        // of crafted lsn==0 sections is accepted — and replayed. Ports must
        // reproduce this, not fix it.
        SegImage::new(1, 1)
            .commit(0, Body::new().record(10, Some(b"zero")))
            .commit(0, Body::new().record(11, Some(b"also zero")))
            .commit(1, Body::new().record(12, Some(b"real")))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 10), Some(b"zero".to_vec()));
        assert_eq!(content(&r, 11), Some(b"also zero".to_vec()));
        assert_eq!(content(&r, 12), Some(b"real".to_vec()));
        assert_eq!(
            r.rec.next_lsn, 2,
            "the zeros stayed invisible to the maximum"
        );
    }

    #[test]
    fn an_lsn_zero_section_after_a_real_one_is_corruption() {
        let dir = scratch("z_after");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(0, Body::new().record(11, Some(b"b")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("does not follow"), "{msg}");
    }

    #[test]
    fn an_lsn_zero_only_segment_chains_by_its_stated_start_and_does_not_advance_it() {
        let dir = scratch("z_chain");
        let base = base_in(&dir);
        // Segment 2 holds sections but no LSN the chain can see: it is "empty"
        // for chaining purposes, so segment 3 must state what SEGMENT 2 said it
        // would start at — 2, not 3.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, 2)
            .commit(0, Body::new().record(11, Some(b"z")))
            .write(&base);
        SegImage::new(3, 2)
            .commit(2, Body::new().record(12, Some(b"c")))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 11), Some(b"z".to_vec()), "and it IS replayed");
        assert_eq!(r.rec.next_lsn, 3);
    }

    #[test]
    fn the_self_check_is_skipped_for_an_lsn_zero_only_segment() {
        let dir = scratch("z_self");
        let base = base_in(&dir);
        // first_lsn stays 0, and the gate is `first_lsn != 0` rather than "the
        // segment is nonempty" — a port transcribing "nonempty" refuses an image
        // the reference accepts.
        SegImage::new(1, 1)
            .commit(0, Body::new().record(10, Some(b"z")))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 10), Some(b"z".to_vec()));
        // All-empty retained set: nextLsn counts from the header, not from 0+1.
        assert_eq!(r.rec.next_lsn, 1);
    }

    #[test]
    fn an_accepted_lsn_zero_does_move_the_scan_local_anchor() {
        let dir = scratch("z_local_anchor");
        let base = base_in(&dir);
        // The other half of the two-level anchor, and it points the opposite way
        // from the carry test: WITHIN a segment scan, an accepted zero updates
        // the lookahead anchor to 0. Segment 1 ends at LSN 5, so the anchor
        // enters segment 2 at 5; the accepted zero drops it to 0, and the
        // damaged header that follows is then proven corrupt by a section
        // carrying 0 + 2. Had the anchor stayed at 5 the walk would have wanted
        // 7, found nothing, and truncated the tail away instead.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(1, Some(b"a")))
            .commit(2, Body::new().record(2, Some(b"b")))
            .commit(3, Body::new().record(3, Some(b"c")))
            .commit(4, Body::new().record(4, Some(b"d")))
            .commit(5, Body::new().record(5, Some(b"e")))
            .write(&base);
        let img = SegImage::new(2, 6)
            .commit(0, Body::new().record(10, Some(b"z")))
            .commit(1, Body::new().record(11, Some(b"damaged")))
            .commit(2, Body::new().record(12, Some(b"proof")));
        let at = img.off(1);
        img.damage(at).write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("mid-log corruption"), "{msg}");
    }

    #[test]
    fn an_append_inside_an_lsn_zero_section_refuses_on_its_delta() {
        let dir = scratch("z_append");
        let base = base_in(&dir);
        // There IS an image-level route into the decoder with `lsn == 0`, which
        // a round-2 review corrected me on: leading zero-LSN sections replay,
        // and one may carry a `T_APPEND`. Both bounds of the delta rule are
        // total at zero — `delta >= 1` and `delta <= lsn - 1 == -1` cannot both
        // hold — so the reference refuses, and so does this port.
        SegImage::new(1, 1)
            .commit(0, Body::new().record(10, Some(b"a")))
            .commit(0, Body::new().append(10, 1, b"b"))
            .commit(1, Body::new().record(11, Some(b"c")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("bad WAL append base delta 1"), "{msg}");
        assert!(msg.contains("section LSN 0"), "{msg}");
    }

    #[test]
    fn a_leading_zero_does_not_excuse_the_first_nonzero_from_the_header() {
        let dir = scratch("z_first_nonzero");
        let base = base_in(&dir);
        // The zeros are invisible to `first_lsn`, so the first NONZERO section
        // becomes both recorded endpoints and must answer to the header like any
        // other. Here it does not.
        SegImage::new(1, 1)
            .commit(0, Body::new().record(10, Some(b"z")))
            .commit(2, Body::new().record(11, Some(b"b")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("its leading sections are gone"), "{msg}");
    }

    #[test]
    fn an_lsn_zero_only_segment_does_not_erase_the_cross_segment_anchor() {
        let dir = scratch("z_anchor");
        let base = base_in(&dir);
        // Segment 2 ends with an accepted lsn==0 section, so the carry stays at
        // segment 1's last LSN. The anchor is only observable through the
        // lookahead, so segment 3 carries a damaged header followed by exactly
        // carry+2 == 3: corruption if the anchor survived, torn tail if not.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, 2)
            .commit(0, Body::new().record(11, Some(b"z")))
            .write(&base);
        let img = SegImage::new(3, 2)
            .commit(2, Body::new().record(12, Some(b"c")))
            .commit(3, Body::new().record(13, Some(b"d")));
        let at = img.off(0);
        img.damage(at).write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("mid-log corruption"), "{msg}");
    }

    // ------------------------------------------------------ R6, the §4.2 table

    #[test]
    fn a_content_record_sets_both_identities() {
        let dir = scratch("id_content");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(r.rec.identities.content_base_lsn.get(&10), Some(&1));
        assert_eq!(r.rec.identities.state_lsn.get(&10), Some(&1));
    }

    #[test]
    fn a_null_record_clears_the_content_base_but_keeps_the_state() {
        let dir = scratch("id_null");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(10, None))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 10), None, "null content, not void");
        assert_eq!(r.rec.identities.content_base_lsn.get(&10), None);
        assert_eq!(r.rec.identities.state_lsn.get(&10), Some(&2));
    }

    #[test]
    fn prealloc_over_a_content_live_record_refuses() {
        let dir = scratch("id_prealloc_live");
        let base = base_in(&dir);
        // wal_prealloc no-ops on a set slot, so applying it here would leave a
        // live record while the identities describe a preallocated one.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().prealloc(10))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("PREALLOC over a content-live record"), "{msg}");
    }

    #[test]
    fn prealloc_over_a_null_record_is_allowed_and_state_only() {
        let dir = scratch("id_prealloc_null");
        let base = base_in(&dir);
        // The precondition is "not content-live", stated to be TOTAL: a
        // null-content target is neither void nor already preallocated, and a
        // port phrasing it "void or P" diverges here.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, None))
            .commit(2, Body::new().prealloc(10))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(r.rec.identities.content_base_lsn.get(&10), None);
        assert_eq!(r.rec.identities.state_lsn.get(&10), Some(&2));
    }

    #[test]
    fn delete_clears_both_identities() {
        let dir = scratch("id_delete");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().delete(10))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert!(is_void(&r, 10));
        assert_eq!(r.rec.identities.content_base_lsn.get(&10), None);
        assert_eq!(r.rec.identities.state_lsn.get(&10), None);
    }

    #[test]
    fn deleting_a_record_that_was_never_established_is_a_no_op() {
        let dir = scratch("id_delete_void");
        let base = base_in(&dir);
        // The shape a cleaned log leaves: the section that created recid 10 is
        // gone, and this delete is the only surviving mention of it. Refusing
        // here would turn a correctly cleaned log into an unopenable store.
        SegImage::new(1, 1)
            .commit(1, Body::new().delete(10).record(11, Some(b"b")))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert!(is_void(&r, 10));
        assert_eq!(content(&r, 11), Some(b"b".to_vec()));
    }

    #[test]
    fn an_append_on_its_stated_base_applies() {
        let dir = scratch("id_append");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().append(10, 1, b"bc"))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 10), Some(b"abc".to_vec()));
        // Neither identity moves: an append is not a self-contained state.
        assert_eq!(r.rec.identities.content_base_lsn.get(&10), Some(&1));
        assert_eq!(r.rec.identities.state_lsn.get(&10), Some(&1));
    }

    #[test]
    fn an_append_whose_base_is_gone_is_skipped_and_then_audited() {
        let dir = scratch("id_skip");
        let base = base_in(&dir);
        // The mark retires the segment holding recid 10's image, and nothing
        // re-establishes it: the store cannot be reconstructed, so the open
        // refuses rather than return a record missing acknowledged bytes.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, 2)
            .image(2, Body::new().record(9, Some(b"i")))
            .mark(3, 1, 2)
            .commit(4, Body::new().append(10, 3, b"bc"))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("skipped 1 append"), "{msg}");
        assert!(msg.contains("recid 10"), "{msg}");
        // R5 ran BEFORE R6, so this refusal observes a namespace already pruned.
        // "A failed open leaves the files untouched" is NOT a v3 invariant, and
        // pretending otherwise is how a port ends up asserting it somewhere.
        assert_eq!(on_disk(&base), vec![2]);
    }

    #[test]
    fn the_audit_refuses_before_w7_touches_the_torn_tail() {
        let dir = scratch("audit_before_w7");
        let base = base_in(&dir);
        // The ordering rule needs BOTH faults in one image to be visible: a
        // stranded append (which the audit refuses) and a torn active tail
        // (which W7 would truncate). Running W7 first would leave the same file
        // SET, so only the active segment's LENGTH can witness the order.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        let img = SegImage::new(2, 2)
            .image(2, Body::new().record(9, Some(b"i")))
            .mark(3, 1, 2)
            .commit(4, Body::new().append(10, 3, b"stranded"))
            .commit(5, Body::new().record(12, Some(b"tail")));
        let torn = img.off(3) as usize + 9;
        let img = img.cut_to(torn).write(&base);
        let before = img.len() as u64;

        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("skipped 1 append"), "{msg}");
        assert_eq!(on_disk(&base), vec![2], "R5 pruned before the refusal");
        assert_eq!(
            file_len(&base, 2),
            before,
            "the audit refused BEFORE W7 truncated the tail"
        );
    }

    #[test]
    fn a_mark_before_a_torn_tail_still_authorizes_its_unlinks() {
        let dir = scratch("k_torn_mark");
        let base = base_in(&dir);
        // K3: a mark living in the torn-tail-prone active segment may itself be
        // truncated, and that only under-collects. The converse needs pinning
        // too — a mark that survived AHEAD of the tear must still drive R5. An
        // implementation that discarded the segment-local maximum on the
        // torn-tail return would silently stop cleaning at recovery, and no
        // other assertion here would notice.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"gone")))
            .write(&base);
        let img = SegImage::new(2, 2)
            .image(2, Body::new().record(11, Some(b"kept")))
            .mark(3, 1, 2)
            .commit(4, Body::new().record(12, Some(b"tail")));
        let torn = img.off(2) as usize + 9;
        img.cut_to(torn).write(&base);

        let r = open_rw(&base).expect("opens");
        assert_eq!(on_disk(&base), vec![2, 3], "unlinked, then rotated");
        assert_eq!(content(&r, 11), Some(b"kept".to_vec()));
        assert!(is_void(&r, 12), "the torn section is not replayed");
        assert_eq!(r.rec.next_lsn, 4);
    }

    #[test]
    fn a_skipped_append_is_discharged_by_a_later_self_contained_entry() {
        let dir = scratch("id_skip_ok");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, 2)
            .image(2, Body::new().record(9, Some(b"i")))
            .mark(3, 1, 2)
            .commit(4, Body::new().append(10, 3, b"bc"))
            .commit(5, Body::new().record(10, Some(b"whole")))
            .write(&base);
        let r = open_rw(&base).expect("opens: the skip was superseded");
        assert_eq!(content(&r, 10), Some(b"whole".to_vec()));
        assert_eq!(r.rec.next_lsn, 6);
    }

    #[test]
    fn a_skipped_appends_payload_is_still_consumed() {
        let dir = scratch("id_skip_frame");
        let base = base_in(&dir);
        // The entry after the skipped append must decode, which it only can if
        // the skip consumed its payload: the frame is still framed.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, 2)
            .image(2, Body::new().record(9, Some(b"i")))
            .mark(3, 1, 2)
            .commit(
                4,
                Body::new()
                    .append(10, 3, b"payload-that-must-be-skipped")
                    .record(12, Some(b"after")),
            )
            .commit(5, Body::new().record(10, Some(b"whole")))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 12), Some(b"after".to_vec()));
    }

    #[test]
    fn prealloc_over_a_void_record_establishes_it() {
        let dir = scratch("id_prealloc_void");
        let base = base_in(&dir);
        // The row that proves `wal_prealloc` actually ACTS; the other two
        // prealloc tests only prove what it refuses and what it tolerates.
        SegImage::new(1, 1)
            .commit(1, Body::new().prealloc(10))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 10), None, "preallocated: present, no content");
        assert!(!is_void(&r, 10));
        assert_eq!(r.rec.identities.state_lsn.get(&10), Some(&1));
        assert_eq!(r.rec.identities.content_base_lsn.get(&10), None);
    }

    #[test]
    fn every_self_contained_entry_discharges_a_pending_skip() {
        let dir = scratch("id_skip_discharge");
        let base = base_in(&dir);
        // Three stranded appends, discharged three different ways — a null
        // record, a prealloc and a delete. Only the content-record path was
        // covered before, and the audit is a refusal channel: a row that failed
        // to clear the set would turn a recoverable log into a refused open.
        SegImage::new(1, 1)
            .commit(
                1,
                Body::new()
                    .record(10, Some(b"a"))
                    .record(11, Some(b"b"))
                    .record(12, Some(b"c")),
            )
            .write(&base);
        SegImage::new(2, 2)
            .image(2, Body::new().record(9, Some(b"i")))
            .mark(3, 1, 2)
            .commit(
                4,
                Body::new()
                    .append(10, 3, b"x")
                    .append(11, 3, b"y")
                    .append(12, 3, b"z"),
            )
            .commit(5, Body::new().record(10, None).prealloc(11).delete(12))
            .write(&base);
        let r = open_rw(&base).expect("every skip was discharged");
        assert_eq!(content(&r, 10), None);
        assert_eq!(content(&r, 11), None);
        assert!(is_void(&r, 12));
    }

    #[test]
    fn an_append_citing_a_base_below_the_applied_one_refuses() {
        let dir = scratch("id_append_low");
        let base = base_in(&dir);
        // recid 10's applied image is LSN 1; the append at LSN 3 cites base 2.
        // Unreachable in a conforming set — retirement is a prefix in LSN order,
        // so a base ABOVE the applied one cannot be the missing part — and
        // defence in depth over the density rule.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"filler")))
            .commit(3, Body::new().append(10, 1, b"c"))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("above the applied base"), "{msg}");
    }

    #[test]
    fn an_append_delta_outside_its_bounds_refuses() {
        for (delta, lsn) in [(0u64, 2i64), (2, 2)] {
            let dir = scratch("id_delta");
            let base = base_in(&dir);
            SegImage::new(1, 1)
                .commit(1, Body::new().record(10, Some(b"a")))
                .commit(lsn, Body::new().append(10, delta, b"c"))
                .write(&base);
            let msg = corrupt_msg(open_rw(&base));
            assert!(
                msg.contains("bad WAL append base delta"),
                "delta {delta}: {msg}"
            );
        }
    }

    #[test]
    fn a_superseded_record_is_reapplied() {
        let dir = scratch("id_resupply");
        let base = base_in(&dir);
        // Idempotent, so replay does not try to be clever about which image
        // "wins" — it applies them all in LSN order and the last one stands.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"first")))
            .commit(2, Body::new().record(10, Some(b"second")))
            .commit(3, Body::new().record(10, Some(b"third")))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 10), Some(b"third".to_vec()));
        assert_eq!(r.rec.identities.content_base_lsn.get(&10), Some(&3));
    }

    #[test]
    fn two_entries_for_one_recid_in_one_section_refuse() {
        let dir = scratch("id_twice");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")).record(10, Some(b"b")))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("two WAL entries for recid 10"), "{msg}");
    }

    #[test]
    fn the_one_entry_rule_covers_image_sections_too() {
        let dir = scratch("id_twice_c");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .image(1, Body::new().record(10, Some(b"a")).delete(10))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("two WAL entries for recid 10"), "{msg}");
    }

    #[test]
    fn an_unknown_entry_tag_refuses() {
        let dir = scratch("id_tag");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().raw(&[9u8, 0x81]))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("bad WAL entry tag 9"), "{msg}");
    }

    #[test]
    fn an_entry_overrunning_its_section_body_refuses() {
        let dir = scratch("id_overrun");
        let base = base_in(&dir);
        // A record claiming 40 bytes of payload inside a body that holds none:
        // CRC-valid, so it is a writer defect, not a torn tail.
        let mut out = DataOutput2::with_capacity(16);
        out.write_byte(T_RECORD as i32);
        out.pack_long(10);
        out.pack_long(48);
        out.pack_long(41);
        SegImage::new(1, 1)
            .section(TAG_SECTION, 1, &out.buf)
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("bad WAL record length"), "{msg}");
    }

    #[test]
    fn a_capacity_no_conforming_writer_would_record_refuses() {
        // capValid in full. The rule is not "any plausible number": the writer
        // records 0 for null content AND for an oversize (linked) record, whose
        // chunk chain has no plain capacity, and a 16-aligned capacity big
        // enough for the 4-byte header plus content otherwise.
        let over = iv::MAX_CAPACITY as u64;
        let rows: Vec<(u64, Option<&[u8]>, bool)> = vec![
            (0, None, true),        // the null record
            (16, None, false),      // null content never carries one
            (16, Some(b"a"), true), // 16 >= 4 + 1, aligned
            // EXACT fit, and the row a conforming writer actually emits for
            // 12 bytes: `cap_for(12) == 16 == 4 + 12`. Without it, tightening
            // the bound to `cap > 4 + len` refuses images Java accepts and the
            // whole matrix stays green.
            (16, Some(&[0u8; 12]), true),
            (0, Some(b"a"), false),         // 0 is reserved for oversize
            (4, Some(b"a"), false),         // big enough, not 16-aligned
            (16, Some(&[0u8; 13]), false),  // aligned, too small for 4 + 13
            (over + 16, Some(b"a"), false), // past the plain-record limit
            (16, Some(b""), true),          // zero-length content is content
        ];
        for (cap, content, ok) in rows {
            let dir = scratch("id_cap");
            let base = base_in(&dir);
            SegImage::new(1, 1)
                .commit(1, Body::new().record_cap(10, cap, content))
                .write(&base);
            let got = open_rw(&base);
            assert_eq!(
                got.is_ok(),
                ok,
                "cap {cap} with {:?}: {:?}",
                content.map(|c| c.len()),
                got.err().map(|e| e.to_string())
            );
        }
    }

    #[test]
    fn an_oversize_record_replays_through_the_linked_path() {
        let dir = scratch("id_oversize");
        let base = base_in(&dir);
        // The other arm of `cap_valid`: a record too large for the plain
        // capacity model is written with capacity 0, because a chunk chain has
        // no plain capacity and the layout is re-chosen on replay. Nothing else
        // in this module exercises either that arm or the inner store's linked
        // write, and both are on the ordinary path for any large value.
        let big = vec![0x5Au8; iv::MAX_CAPACITY - 3];
        assert!(4 + big.len() > iv::MAX_CAPACITY, "must be oversize");
        SegImage::new(1, 1)
            .commit(1, Body::new().record_cap(10, 0, Some(&big)))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 10), Some(big));
    }

    #[test]
    fn an_append_the_inner_store_refuses_is_corruption() {
        let dir = scratch("id_refused");
        let base = base_in(&dir);
        // An exact-fit record has no headroom, so the append cannot be applied
        // in place. The log says it was; the log is therefore not one this
        // writer produced.
        SegImage::new(1, 1)
            .commit(1, Body::new().record_cap(10, 16, Some(&[7u8; 12])))
            .commit(2, Body::new().append(10, 1, &[9u8; 40]))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("append refused"), "{msg}");
    }

    #[test]
    fn an_overlong_packed_long_refuses_rather_than_running_on() {
        let dir = scratch("id_packlong");
        let base = base_in(&dir);
        // The port's 10-byte cap, which the reference does not have: it loops to
        // the terminator. A recorded strictness difference, so it gets a test
        // that names it rather than living only in a comment.
        SegImage::new(1, 1)
            .commit(
                1,
                Body::new().raw(&[T_RECORD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            )
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("packed long too long"), "{msg}");
    }

    #[test]
    fn a_revived_recid_is_not_handed_out_again_after_replay() {
        let dir = scratch("r7_freelist");
        let base = base_in(&dir);
        // Delete-then-revive leaves the deleted recid on the allocator's free
        // list while the later section makes it live again. Without R7's
        // `rebuild_free_recids` the next allocation hands out a LIVE recid and
        // overwrites it — a silent corruption channel that no other assertion in
        // this module can see, because it only shows up on the first allocation
        // AFTER recovery.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().delete(10))
            .commit(3, Body::new().record(10, Some(b"revived")))
            .write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(content(&r, 10), Some(b"revived".to_vec()));
        let fresh = r.inner.preallocate().expect("allocate after recovery");
        assert_ne!(fresh.get(), 10, "handed out a live recid");
        assert_eq!(content(&r, 10), Some(b"revived".to_vec()));
    }

    #[test]
    fn a_reserved_recid_zero_refuses() {
        let dir = scratch("id_recid0");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(1, Some(b"a")))
            .commit(2, Body::new().append(0, 1, b"x"))
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        assert!(msg.contains("reserved recid 0"), "{msg}");
    }

    #[test]
    fn a_mark_body_is_never_handed_to_the_entry_decoder() {
        let dir = scratch("k_not_entries");
        let base = base_in(&dir);
        // A mark whose 16 bytes begin with 0x00 — not a valid entry tag. If the
        // decoder ever saw a 'K' body the open would refuse.
        SegImage::new(1, 1)
            .image(1, Body::new().record(10, Some(b"i")))
            .mark(2, 0x0000_0000_0000_0001, 1)
            .write(&base);
        let msg = corrupt_msg(open_rw(&base));
        // K4 holds it (segment 1 cannot authorize removing segment 1), which is
        // a MARK verdict, not an entry-decoder one.
        assert!(msg.contains("including itself"), "{msg}");
    }

    // ------------------------------------------------------ R7

    #[test]
    fn next_lsn_follows_the_highest_retained_section() {
        let dir = scratch("r7_next");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .commit(2, Body::new().record(11, Some(b"b")))
            .write(&base);
        SegImage::new(2, 3)
            .commit(3, Body::new().record(12, Some(b"c")))
            .write(&base);
        assert_eq!(open_rw(&base).expect("opens").rec.next_lsn, 4);
    }

    #[test]
    fn an_all_empty_retained_set_counts_from_the_header() {
        let dir = scratch("r7_empty");
        let base = base_in(&dir);
        // "0 + 1" would restart the log at 1 and reissue LSNs a mark already
        // accounted for, which is why firstLsn is in the header. Note what this
        // test can and cannot show: the fallback is only REACHABLE in an
        // unmarked log — K4 keeps a mark's own segment retained, and that
        // segment is never empty — and the floor then forces the lowest header
        // to state 1, so "count from the header" and "0 + 1" agree on every
        // image that survives R4. The branch is kept because the reference has
        // it, not because a fixture can separate the two readings.
        SegImage::new(1, 1).write(&base);
        SegImage::new(2, 1).write(&base);
        let r = open_rw(&base).expect("opens");
        assert_eq!(r.rec.next_lsn, 1);
        assert_eq!(on_disk(&base), vec![1, 2], "empty is not torn: no rotate");
    }

    #[test]
    fn a_fresh_store_creates_its_first_segment_and_starts_at_lsn_1() {
        let dir = scratch("r7_fresh");
        let base = base_in(&dir);
        let r = open_rw(&base).expect("opens");
        assert_eq!(on_disk(&base), vec![1]);
        assert_eq!(r.rec.next_lsn, 1);
        assert_eq!(r.set.active().expect("active").header_first_lsn(), 1);
    }

    #[test]
    fn a_fresh_store_beside_burnt_residue_uses_the_burnt_successor() {
        let dir = scratch("r7_burnt");
        let base = base_in(&dir);
        // W6: the residue's name is burned even though the file is removed, so a
        // stale directory entry can never alias a segment a later create reuses.
        std::fs::write(seg_path(&base, 7), [0u8; 4]).expect("residue");
        let r = open_rw(&base).expect("opens");
        assert_eq!(on_disk(&base), vec![8]);
        assert_eq!(r.rec.next_lsn, 1);
    }

    // ------------------------------------------------------ read-only mode

    #[test]
    fn a_read_only_recovery_mutates_nothing() {
        let dir = scratch("ro_nothing");
        let base = base_in(&dir);
        // Every mutation R5/R7 could make is armed here: a superseded segment to
        // unlink, create-crash residue to delete, and a torn tail to truncate
        // and rotate.
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"gone")))
            .write(&base);
        let img = SegImage::new(2, 2)
            .image(2, Body::new().record(11, Some(b"kept")))
            .mark(3, 1, 2)
            .commit(4, Body::new().record(12, Some(b"tail")));
        let torn = img.off(2) as usize + 9;
        img.cut_to(torn).write(&base);
        std::fs::write(seg_path(&base, 3), [0u8; 4]).expect("residue");
        let before: Vec<(i64, u64)> = on_disk(&base)
            .into_iter()
            .map(|s| (s, file_len(&base, s)))
            .collect();

        let r = open_ro(&base).expect("opens");
        assert_eq!(content(&r, 11), Some(b"kept".to_vec()));
        assert!(is_void(&r, 10), "superseded segments are not replayed");
        assert_eq!(r.rec.next_lsn, 4);
        let after: Vec<(i64, u64)> = on_disk(&base)
            .into_iter()
            .map(|s| (s, file_len(&base, s)))
            .collect();
        assert_eq!(before, after, "no create, unlink, truncate or rotate");
        assert_eq!(r.set.dir_fsyncs(), 0);
    }

    #[test]
    fn a_read_only_recovery_reaches_the_same_answers_as_a_writable_one() {
        let dir = scratch("ro_same");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(1, Body::new().record(10, Some(b"a")))
            .write(&base);
        SegImage::new(2, 2)
            .image(2, Body::new().record(10, Some(b"kept")))
            .mark(3, 1, 2)
            .write(&base);
        let ro = open_ro(&base).expect("ro opens");
        let (ro_next, ro_content) = (ro.rec.next_lsn, content(&ro, 10));
        drop(ro);
        let rw = open_rw(&base).expect("rw opens");
        assert_eq!(ro_next, rw.rec.next_lsn);
        assert_eq!(ro_content, content(&rw, 10));
    }

    #[test]
    fn a_read_only_fresh_store_creates_no_segment() {
        let dir = scratch("ro_fresh");
        let base = base_in(&dir);
        let r = open_ro(&base).expect("opens");
        assert_eq!(on_disk(&base), Vec::<i64>::new());
        assert_eq!(r.rec.next_lsn, 1);
        assert!(r.set.active().is_none());
    }

    #[test]
    fn a_read_only_recovery_refuses_the_same_images() {
        let dir = scratch("ro_refuse");
        let base = base_in(&dir);
        SegImage::new(1, 1)
            .commit(2, Body::new().record(10, Some(b"a")))
            .write(&base);
        let msg = corrupt_msg(open_ro(&base));
        assert!(msg.contains("its leading sections are gone"), "{msg}");
    }

    // ------------------------------------------------------ descriptors

    #[test]
    fn recovery_leaves_one_descriptor_open_whatever_the_segment_count() {
        let dir = scratch("fds");
        let base = base_in(&dir);
        let n = 40;
        for seq in 1..=n {
            SegImage::new(seq, seq)
                .commit(seq, Body::new().record(seq as u64, Some(b"x")))
                .write(&base);
        }
        let r = open_rw(&base).expect("opens");
        assert_eq!(seqs(&r).len(), n as usize);
        // Both passes release as they go: nothing reads a segment after
        // recovery, and a store is allowed to reach thousands of them.
        assert_eq!(r.set.open_file_count(), 1, "only the active segment");
    }
}
