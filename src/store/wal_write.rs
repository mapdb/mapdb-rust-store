//! The **section writer** — the only place a v3 section reaches the device, and
//! the durability-event seam the writer obligations are stated over.
//!
//! Port of Java `StoreWAL.appendSection` / `BodySink` / `rollover` (slice A2).
//! `wal_segments.rs` owns which FILES exist, `wal_recover.rs` owns how bytes are
//! read back, and this module owns how they get written:
//!
//! - **W1/W4** — a section's force completes before this function returns, so
//!   no section is appended before its predecessor's force finished. Recovery's
//!   mid-log-rot inference ("a valid section follows an invalid one ⇒
//!   corruption") is sound only under that.
//! - **W3** — rollover happens only at a section boundary, after the sealed
//!   segment's last section is forced with a SIZE-persisting force, so a
//!   non-final segment ends exactly at a section boundary with zero trailing
//!   bytes.
//! - **W9** — a failed or partial write/force fails the store CLOSED. Every
//!   error out of [`append_section`] is that failure; the caller must not let a
//!   retry append into a segment that may hold partial bytes.
//!
//! # Two passes, never one buffer
//!
//! The body is emitted TWICE — a measure pass (length + CRC, no I/O) and a write
//! pass — instead of being accumulated into a `Vec`. That is what lets one
//! commit exceed 2 GiB: `bodyLen` is an `i64` in the format and this writer
//! actually uses the range. The header is still written FIRST and final, so the
//! crash shapes are the same ones recovery classifies (a tear mid-body leaves a
//! valid header over a short or CRC-bad body — S3/S4/S5), and the
//! pass-divergence check runs BEFORE the force, so a nondeterministic emitter
//! fails the commit closed rather than acknowledging a section that replay
//! rejects as bit rot.

use super::wal_segments::{crc_domain_of, WalSegmentSet};
use crate::error::{DbError, Result};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// tag(1) + lsn(8) + bodyLen(8) + hdrCrc(4) + bodyCrc(4) — re-exported from the
/// codec so the writer and the reader cannot disagree about the number.
use super::wal_recover::{seal_sec_hdr, SEC_HDR};

/// Java's `BodySink` buffer, byte for byte: entry framing (tens of bytes per
/// entry) is coalesced through it so a large commit is not syscall-bound, while
/// a payload at or past this size bypasses it and is written where it lies.
const SINK_BUF: usize = 64 << 10;

// ---------------------------------------------------------------- the io seam

/// The durability-relevant file operations. `SEC_HEADER` and `SEC_BODY` are
/// separate constants deliberately: a failure between them is a *partial section
/// write*, and that is precisely the state W9 exists to forbid appending after.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WalOpKind {
    Create,
    SegHeader,
    SecHeader,
    SecBody,
    ForceData,
    ForceFull,
    Truncate,
    Unlink,
    DirSync,
}

/// One reported operation. `seq` is the segment's sequence number (0 for
/// `DirSync`), `off` the byte offset it starts at — for a force, the file length
/// it makes durable — `len` the bytes it writes (0 where it writes none), and
/// `tag` the section tag for section events, else 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WalIoEvent {
    pub(crate) kind: WalOpKind,
    pub(crate) seq: i64,
    pub(crate) off: u64,
    pub(crate) len: u64,
    pub(crate) tag: u8,
}

/// Writer fault-injection and trace seam, called immediately **before** each
/// operation; returning an error makes that operation fail exactly as the
/// platform would.
///
/// Why it exists at all: W1-W5 and W7 are ORDERING claims about operations that
/// leave no trace in the resulting bytes. Until this seam they were held by
/// structural argument alone — the calls appear in the right order in the source
/// and nothing checks that they still do. A1's review deferred W7's force
/// ordering for exactly this reason (the port had no I/O seam at all); A2 needs
/// one for W2/W3/W9 regardless, so W7 is observable here too.
///
/// What it does NOT model: this is an I/O-FAILURE seam, not a power-loss one.
/// Throwing here makes a syscall fail; it does not make written bytes vanish.
/// Power-loss images — torn tails at every offset, non-prefix unlink subsets,
/// create-crash residue — are the crash harness's and the recovery suite's
/// subject and stay there.
///
/// **Per store, not per process.** Java installs its `WalIo` in a static and
/// serializes the tests that use it; a rust test binary runs its tests on
/// parallel threads in one process, where a global seam would leak fault
/// injection from one test into another. The seam is therefore a field on the
/// store and on the segment set, handed in at open. This is a test-harness
/// difference, not a format one.
pub(crate) trait WalIo: Send + Sync {
    fn before(&self, e: &WalIoEvent) -> Result<()>;
}

/// Reports one operation, if a seam is installed.
pub(crate) fn wal_io_event(
    io: &Option<Arc<dyn WalIo>>,
    kind: WalOpKind,
    seq: i64,
    off: u64,
    len: u64,
    tag: u8,
) -> Result<()> {
    match io {
        None => Ok(()),
        Some(io) => io.before(&WalIoEvent {
            kind,
            seq,
            off,
            len,
            tag,
        }),
    }
}

// --------------------------------------------------------------- the body sink

/// One pass over a section body. The measure pass (`file == None`) accumulates
/// length and CRC only; the write pass also writes the bytes at increasing
/// offsets. Offsets and the running length are `u64`: a body may exceed 2 GiB and
/// no whole-body allocation exists in either pass.
pub(crate) struct BodySink<'a> {
    file: Option<&'a File>,
    crc: crc32fast::Hasher,
    pos: u64,
    count: u64,
    buf: Vec<u8>,
}

impl<'a> BodySink<'a> {
    fn measure(crc: crc32fast::Hasher) -> BodySink<'a> {
        BodySink {
            file: None,
            crc,
            pos: 0,
            count: 0,
            buf: Vec::new(),
        }
    }

    fn writer(file: &'a File, body_start: u64, crc: crc32fast::Hasher) -> BodySink<'a> {
        BodySink {
            file: Some(file),
            crc,
            pos: body_start,
            count: 0,
            buf: Vec::with_capacity(SINK_BUF),
        }
    }

    /// Emits `b` into this pass. The CRC and the length advance in BOTH passes;
    /// only the write pass touches the device.
    pub(crate) fn write(&mut self, b: &[u8]) -> Result<()> {
        self.crc.update(b);
        self.count += b.len() as u64;
        let Some(file) = self.file else {
            return Ok(());
        };
        if b.len() >= SINK_BUF {
            self.flush()?;
            file.write_all_at(b, self.pos)?;
            self.pos += b.len() as u64;
            return Ok(());
        }
        if self.buf.len() + b.len() > SINK_BUF {
            self.flush()?;
        }
        self.buf.extend_from_slice(b);
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let file = self.file.expect("measure pass never buffers");
        file.write_all_at(&self.buf, self.pos)?;
        self.pos += self.buf.len() as u64;
        self.buf.clear();
        Ok(())
    }
}

// ------------------------------------------------------------ the fail-closed
// guard

/// Marks the store closed if the writer leaves through a PANIC after I/O began.
///
/// Java distinguishes `IOException`/`RuntimeException` (always fail closed) from
/// `Error` (escapes with the store open if it arrived before any I/O, closes the
/// handle if after). Rust's equivalents of the first two are the `Err` returns
/// this module produces, and its equivalent of the third is an unwinding panic —
/// which cannot be turned into a return value without `catch_unwind`. So the
/// guard reproduces the part that matters: once a rollover or a section write has
/// begun, the physical file may extend past `file_len`, and a retry with a
/// shorter body would leave a stale tail that sealing the segment later turns
/// into a torn NON-FINAL segment recovery must refuse. Arming the guard makes the
/// store refuse everything afterwards.
///
/// It does not close the inner store or release the namespace, as Java's
/// `closeAfterWalFailure` does: those need `&mut` state the unwinding frame no
/// longer safely owns. Dropping the `StoreWAL` releases both.
struct IoGuard<'a> {
    closed: &'a AtomicBool,
    armed: bool,
}

impl<'a> IoGuard<'a> {
    fn new(closed: &'a AtomicBool) -> IoGuard<'a> {
        IoGuard {
            closed,
            armed: false,
        }
    }

    fn arm(&mut self) {
        self.armed = true;
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for IoGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.closed.store(true, Ordering::Release);
        }
    }
}

// ------------------------------------------------------------ append a section

/// Appends one complete section to the active segment and forces it, rolling
/// over first when the segment is full.
///
/// `emit` runs TWICE and **must produce identical bytes both times**; if it does
/// not, this refuses to acknowledge the section (before the force) rather than
/// leave a stored `bodyCrc` that replay reads as bit rot.
///
/// Every `Err` return is a W9 failure: the caller closes the store. On success
/// the active segment's `file_len` and `valid_end` have moved to the section's
/// end, and `set.log_bytes()` accounts for it.
pub(crate) fn append_section<F>(
    set: &mut WalSegmentSet,
    segment_bytes: u64,
    io: &Option<Arc<dyn WalIo>>,
    closed: &AtomicBool,
    tag: u8,
    lsn: i64,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(&mut BodySink) -> Result<()>,
{
    let mut guard = IoGuard::new(closed);

    // W3: the rollover condition is checked ONLY here, at a section boundary,
    // and only when the active segment is nonempty — so one section may exceed
    // `segment_bytes` and an oversize section gets a segment to itself, rather
    // than a segment being sealed with nothing in it.
    let roll = {
        let active = active(set)?;
        active.file_len >= segment_bytes && !active.empty()
    };
    if roll {
        guard.arm();
        {
            let active = set.active_mut().expect("checked above");
            active.ensure_open()?;
            let (seq, len) = (active.seq, active.file_len);
            wal_io_event(io, WalOpKind::ForceFull, seq, len, 0, 0)?;
            // force(true), never a data-only sync: this seals the segment and
            // its SIZE is the payload. D5 — the distinction is spec, and W3's
            // whole load collapses if a port's data sync loses a sealed
            // segment's tail extent, because recovery would then see a torn
            // NON-FINAL segment and refuse a legitimate image.
            active.file().expect("just opened").sync_all()?;
            // The sealed segment will never be read or written again by this
            // store (nothing reads a segment after recovery). Releasing here is
            // the same recorded divergence as W7's: the reference keeps the
            // stale handle, and copying it would put an O(segments) descriptor
            // leak in every engine.
            active.release();
        }
        set.create_segment(lsn)?;
    }

    let (seg_header, off, seq) = {
        let active = active(set)?;
        (active.header, active.file_len, active.seq)
    };

    // ---- pass 1: measure. No I/O, no allocation proportional to the body.
    let mut bcrc = crc32fast::Hasher::new();
    crc_domain_of(&mut bcrc, &seg_header, off);
    let mut measure = BodySink::measure(bcrc);
    emit(&mut measure)?;
    let body_len = measure.count;
    let body_crc = measure.crc.finalize() as i32;
    let hdr = seal_sec_hdr(&seg_header, off, tag, lsn, body_len, body_crc);

    let active = set.active_mut().expect("checked above");
    active.ensure_open()?;
    guard.arm();
    let file = active.file().expect("just opened");

    wal_io_event(io, WalOpKind::SecHeader, seq, off, SEC_HDR as u64, tag)?;
    file.write_all_at(&hdr, off)?;
    let body_start = off + SEC_HDR as u64;
    wal_io_event(io, WalOpKind::SecBody, seq, body_start, body_len, tag)?;

    // ---- pass 2: write. Must reproduce pass 1's bytes exactly.
    let mut wcrc = crc32fast::Hasher::new();
    crc_domain_of(&mut wcrc, &seg_header, off);
    let mut writer = BodySink::writer(file, body_start, wcrc);
    emit(&mut writer)?;
    writer.flush()?;
    if writer.count != body_len || writer.crc.finalize() as i32 != body_crc {
        return Err(DbError::corrupt_msg(format!(
            "WAL section body diverged between the CRC pass and the write pass ({} vs {body_len} \
             bytes); refusing to acknowledge",
            writer.count
        )));
    }

    // force(false) — a DATA sync. This relies on the POSIX guarantee that
    // fdatasync persists "the metadata required to retrieve the data", which for
    // an append means the new file size. Where the SIZE itself is the payload —
    // creating a segment (W2), sealing one at rollover (W3), the post-truncate
    // force (W7) — a full sync is used instead, and the distinction is spec.
    let end = body_start + body_len;
    wal_io_event(io, WalOpKind::ForceData, seq, end, 0, tag)?;
    file.sync_data()?;

    active.file_len = end;
    active.valid_end = end;
    guard.disarm();
    Ok(())
}

/// The active segment, or the one error this module cannot recover from: a
/// writable store always has one after recovery (R7 creates it), so its absence
/// is a sequencing bug in the caller rather than anything about the store.
fn active(set: &WalSegmentSet) -> Result<&super::wal_segments::Segment> {
    set.active()
        .ok_or_else(|| DbError::wrong_config("WAL store has no active segment".to_string()))
}
