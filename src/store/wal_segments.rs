//! The WAL's **multi-file namespace** — `<base>.wal.<16 lowercase hex digits>`,
//! the store lock, and every operation that changes which files exist: create,
//! seal, unlink, directory fsync.
//!
//! Port of Java `WalSegmentSet` (format v3). It owns the **namespace** (N) and
//! **segment header** (H) decision tables and the writer obligations that are
//! about files rather than bytes (W2, W5, W6, and the force-flavour rule);
//! sections, entries and the recovery state machine (S/K/R) stay in `wal.rs`.
//! The split is the Java one and is deliberate: the expensive part of the
//! format to port is this state machine, not the codec.
//!
//! ```text
//! name    := <base> ".wal." <16 lowercase hex digits of segmentSeq>
//! header  := magic "MDBS.WAL"(8) | version i32 = 3 | flags i32 = 0
//!          | segmentSeq i64 | firstLsn i64 | headerCrc i32        // 36 bytes
//! ```
//!
//! All integers big-endian; `headerCrc` is zlib CRC-32 over header bytes
//! `[0, 32)`. The first segment of a store has `segmentSeq = 1`, so `0` is free
//! to mean "no clean mark". Sixteen fixed hex digits make lexicographic order
//! equal numeric order in every port's directory listing; the *name* is the
//! enumeration key and the *header* is the authority, which is what catches a
//! copied or renamed segment (N5/H7).
//!
//! # Status: slice A0, not yet reachable from a public open
//!
//! This is the first slice of the v3 adoption (`todo/store-wal3/`). The public
//! [`StoreWAL`](super::wal::StoreWAL) still speaks v1 and does not consult this
//! module — deliberately, because a segmented namespace holding v1-domain
//! sections is neither format and has no normative reader. The cutover to v3 is
//! one atomic change (slice A2). Two rows are therefore NOT here yet and land
//! with that cutover: D1's ports-only legacy boundary (a regular file at the
//! bare base path, or a `<base>.ckpt` left by v1's rename-checkpoint, must
//! refuse rather than be ignored) and the D4 platform gate. N6 — the v1
//! single-file log at `<base>.wal` — is Java's own row and is implemented here,
//! where Java has it.

// A0 ships this layer unhooked (see above); its consumers arrive in A1/A2.
#![allow(dead_code)]

use crate::error::{DbError, Result};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

/// magic(8) + version(4) + flags(4) + segmentSeq(8) + firstLsn(8) + headerCrc(4).
pub(crate) const SEG_HDR: u64 = 36;
/// Bytes of the segment header covered by `headerCrc`.
pub(crate) const SEG_HDR_CRC_LEN: usize = 32;
pub(crate) const MAGIC: [u8; 8] = *b"MDBS.WAL";
/// v3 adds `firstLsn` to the header and a second `i64` to the `'K'` body.
///
/// Those two fields exist to **delete inference**. v2 asked recovery to work
/// out whether a missing segment was authorized, and where the retained log
/// legitimately began, from circumstantial evidence — LSN density, the position
/// of the mark, the tag of the first retained section. Six defects lived in
/// that reasoning across four revisions, two of them permanent bricks and two
/// silent data loss. Recording the two facts directly turns every one of those
/// questions into an equality between two numbers a conforming writer wrote
/// down.
pub(crate) const FORMAT_VERSION: i32 = 3;
/// Sequence number of a store's first segment; 0 is reserved for "no clean mark".
pub(crate) const FIRST_SEQ: i64 = 1;

fn crc32(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}

fn be32(b: &[u8], off: usize) -> i32 {
    i32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn be64(b: &[u8], off: usize) -> i64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[off..off + 8]);
    i64::from_be_bytes(v)
}

/// One segment file. `valid_end` and the LSN fields are pass-1 results filled in
/// by the recovery scanner; everything else is namespace state owned by this
/// module.
pub(crate) struct Segment {
    pub(crate) seq: i64,
    pub(crate) path: PathBuf,
    read_only: bool,
    /// Opened ON DEMAND and released as soon as a recovery pass is done with
    /// this segment.
    ///
    /// Holding one handle per segment for the store's lifetime is what a
    /// straightforward implementation does, and it does not scale: nothing
    /// reads a segment after recovery — the record map lives in the
    /// memory-backed inner store and only the ACTIVE segment is ever appended
    /// to — while the log is allowed to reach roughly twice the live data size,
    /// so a large store means thousands of open descriptors against a default
    /// `ulimit -n` of 1024. A legitimate store would fail to open with
    /// `EMFILE`, and an attacker-supplied directory of valid header-only
    /// segments could force it deliberately.
    file: Option<File>,
    /// The 36 header bytes, verbatim — used as an identity string in the
    /// section CRC domain.
    pub(crate) header: [u8; SEG_HDR as usize],
    pub(crate) file_len: u64,
    /// End offset of the valid section prefix (pass 1). Never below [`SEG_HDR`].
    pub(crate) valid_end: u64,
    /// LSNs of the first and last accepted sections, or 0 when the segment
    /// holds none. **0 doubles as "none seen"**, and that ambiguity is frozen
    /// reference behaviour, not an oversight — see the Java pin tests
    /// (`StoreWALFrozenEdgeTest`).
    pub(crate) first_lsn: i64,
    pub(crate) last_lsn: i64,
    /// A corruption verdict found in this segment, HELD until R4 decides it is
    /// relevant.
    pub(crate) held: Option<String>,
}

impl Segment {
    fn new(
        seq: i64,
        path: PathBuf,
        read_only: bool,
        header: [u8; SEG_HDR as usize],
        file_len: u64,
    ) -> Segment {
        Segment {
            seq,
            path,
            read_only,
            file: None,
            header,
            file_len,
            valid_end: SEG_HDR,
            first_lsn: 0,
            last_lsn: 0,
            held: None,
        }
    }

    /// The file handle, opening it if this segment does not currently hold one.
    pub(crate) fn file(&mut self) -> Result<&File> {
        if self.file.is_none() {
            let f = if self.read_only {
                OpenOptions::new().read(true).open(&self.path)?
            } else {
                OpenOptions::new().read(true).write(true).open(&self.path)?
            };
            self.file = Some(f);
        }
        Ok(self.file.as_ref().expect("just opened"))
    }

    /// Closes the handle if one is held; the segment stays usable and reopens on
    /// demand. Called as soon as a recovery pass finishes with a segment, which
    /// is what bounds the descriptor count to O(1) instead of O(segments).
    /// Nothing is written through these handles without a preceding force, so a
    /// lost close never loses data.
    pub(crate) fn release(&mut self) {
        self.file = None;
    }

    pub(crate) fn holds_file(&self) -> bool {
        self.file.is_some()
    }

    /// Feeds `crc` this section's **domain separator**: the 36 header bytes
    /// verbatim followed by the big-endian section offset. An ordinary CRC-32
    /// over a prefix — NOT a preloaded register, which would force every port to
    /// reimplement a private convention.
    ///
    /// Binding the segment identity rejects a section byte-copied between
    /// segments; binding the offset rejects one copied to a different offset in
    /// the same segment. The domain intentionally includes the header's own
    /// `headerCrc` field: the 36 bytes are an identity string, not a re-parsed
    /// structure. It therefore also covers `firstLsn`, so a segment whose
    /// stated start is edited invalidates every section CRC in it.
    ///
    /// (Java's javadoc says `[0..28)`; the Java CODE, this port and the
    /// byte-level test kit all use all 36 bytes. 36 is authoritative.)
    pub(crate) fn crc_domain(&self, crc: &mut crc32fast::Hasher, section_offset: u64) {
        crc.update(&self.header);
        crc.update(&section_offset.to_be_bytes());
    }

    /// **The LSN this segment's first section holds** — `nextLsn` at the moment
    /// the writer created it, recorded in the header so recovery never has to
    /// infer it. A segment that holds no section still states where its first
    /// one would have gone, which is exactly what separates "this segment was
    /// always empty" from "its sections vanished".
    pub(crate) fn header_first_lsn(&self) -> i64 {
        be64(&self.header, 24)
    }

    /// True while this segment holds no accepted section (H8).
    pub(crate) fn empty(&self) -> bool {
        self.valid_end == SEG_HDR
    }
}

/// One enumerated header's verdict. Java encodes the same three-way answer as
/// `null` / plain string / `"!"`-prefixed string; an enum says it in the type.
enum HeaderVerdict {
    /// Valid v3 header (H8 included: a header-only segment is legitimate).
    Ok,
    /// H1-H4: the *torn-create* shapes. Residue when this is the highest name,
    /// corruption anywhere below it.
    Torn(String),
    /// H5-H7/H9: a CRC-valid header carrying wrong content — a writer defect or
    /// a copied file, never a torn create. Corruption wherever it appears.
    Corrupt(String),
}

pub(crate) struct WalSegmentSet {
    base: PathBuf,
    dir: PathBuf,
    prefix: String,
    read_only: bool,
    /// Ascending by sequence number.
    segments: Vec<Segment>,
    /// W6: one above the highest sequence number seen in ANY enumerated name,
    /// orphans included.
    next_seq: i64,
    /// Total `file_len` of every segment EXCEPT the highest, which is the only
    /// one that grows. Maintained at the two points that change which segments
    /// exist, so [`log_bytes`](Self::log_bytes) is O(1): it is consulted on
    /// every commit (the cleaning trigger), under the WAL write lock, and
    /// summing the list there is proportional to the number of committed
    /// sections at the minimum segment size.
    sealed_bytes: u64,
    /// The store lock. Held for as long as this handle is open — dropping the
    /// `File` closes the descriptor, which releases the `flock`. `None` only in
    /// the read-only-medium case (see [`take_store_lock`](Self::take_store_lock)).
    lock: Option<File>,
}

impl WalSegmentSet {
    /// Opens the namespace: takes the store lock, enumerates and classifies
    /// (R0/R1), and removes create-crash residue (R2). Leaves the surviving
    /// segments in the set, ascending, with no file handles held;
    /// section-level recovery is the caller's job.
    ///
    /// `base` is the store path as opened, absolutized and then used verbatim —
    /// never canonicalized and never reduced to a basename, or two opens by
    /// different paths would disagree on the namespace. This mirrors Java's
    /// `getAbsoluteFile()`.
    pub(crate) fn open(base: &Path, read_only: bool) -> Result<WalSegmentSet> {
        let abs = if base.is_absolute() {
            base.to_path_buf()
        } else {
            std::env::current_dir()?.join(base)
        };
        let dir = abs
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let name = abs.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
            DbError::wrong_config(format!("WAL base path has no file name: {}", abs.display()))
        })?;
        let prefix = format!("{name}.wal.");

        let mut set = WalSegmentSet {
            base: abs,
            dir,
            prefix,
            read_only,
            segments: Vec::new(),
            next_seq: FIRST_SEQ,
            sealed_bytes: 0,
            lock: None,
        };
        // Every early return from here drops `set`, which drops the lock handle
        // and so releases the store lock — Java's `finally { closeQuietly() }`.
        set.take_store_lock()?;
        // N6: the v1 single-file log. There is no migration, and silently
        // starting a fresh segment set beside it would strand every committed
        // transaction in it — the one outcome the format break exists to
        // prevent. Regular files only, the same discipline N4 applies: a
        // DIRECTORY at that name is not a v1 log.
        let v1 = with_suffix(&set.base, ".wal");
        if is_regular_file(&v1) {
            return Err(DbError::corrupt_msg(format!(
                "v1 single-file WAL present at {}: no migration to v3",
                v1.display()
            )));
        }
        let found = set.enumerate();
        set.classify(&found)?;
        Ok(set)
    }

    /// §3.1: exactly one process may run open, recovery or writing at a time.
    /// Recovery unlinks, truncates and rotates, and two concurrent opens would
    /// also pick the same next sequence number. v1 took no lock; this is new.
    ///
    /// The primitive is `flock`, not POSIX record locks: record locks are owned
    /// by the *process*, so a second open in the same process would silently
    /// succeed by upgrading the first one's lock, while Java refuses that case
    /// (`OverlappingFileLockException`). `flock` is owned by the open file
    /// description, so two handles in one process exclude each other exactly as
    /// two processes do — the behaviour Java's refusal describes.
    fn take_store_lock(&mut self) -> Result<()> {
        let lock_path = with_suffix(&self.base, ".lock");
        let handle = if self.read_only {
            // §3.1 is TWO-SIDED: a reader must be rejected while a writer holds
            // the exclusive lock, AND a writer must be rejected while a reader
            // holds a shared one. So CREATE the lock file when the directory
            // allows it, even though this open will not modify the store.
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lock_path)
            {
                Ok(f) => f,
                Err(cannot_create) => {
                    // Going lockless is the one outcome that reintroduces the
                    // race, so it needs a POSITIVE reason — not merely "the
                    // create failed". An error here can be a transient I/O
                    // fault, a quota, or an ACL on this one pathname, none of
                    // which imply that no writer can create the file and lock
                    // it exclusively.
                    if lock_path.exists() {
                        // Ambiguity resolved: the file is there, so a shared
                        // lock is still attainable on a read-only handle. This
                        // is not a fallback to lockless at all.
                        File::open(&lock_path)?
                    } else if !is_writable_dir(&self.dir) {
                        // Positively a read-only medium: no writer can create
                        // the lock file or a segment, so there is nothing to be
                        // excluded by and nothing to exclude.
                        return Ok(());
                    } else {
                        return Err(DbError::Locked(format!(
                            "cannot take a shared store lock on {} and the directory is writable, \
                             so a writer may be running ({cannot_create})",
                            lock_path.display()
                        )));
                    }
                }
            }
        } else {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lock_path)?
        };
        if !try_flock(&handle, !self.read_only)? {
            return Err(DbError::Locked(format!(
                "WAL store {} is already open{}",
                self.base.display(),
                if self.read_only { " for writing" } else { "" }
            )));
        }
        self.lock = Some(handle);
        Ok(())
    }

    // ---------- R0: enumerate ----------

    /// R0/N4. Collects every **regular file** whose name is exactly the prefix
    /// followed by 16 lowercase hex digits with a non-negative `i64` value.
    /// Directories, symlinks, uppercase hex, wrong lengths and the `.lock` file
    /// are not segments and are ignored — ignored, not rejected, because a
    /// store directory is allowed to contain other things. Sequence GAPS are
    /// legal: integrity comes from the recorded LSNs, not from contiguity.
    fn enumerate(&self) -> Vec<i64> {
        let mut found: Vec<i64> = Vec::new();
        // Java's `dir.list()` answers null for an unreadable/absent directory
        // and the constructor treats that as "no segments"; a fresh store then
        // creates its first segment, which is where a genuinely broken
        // directory surfaces as the I/O error it is.
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(_) => return found,
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = match name.to_str() {
                Some(n) => n,
                None => continue,
            };
            let hex = match name.strip_prefix(&self.prefix) {
                Some(h) => h,
                None => continue,
            };
            if hex.len() != 16 {
                continue;
            }
            // Uppercase hex does not match, and enumeration is case-SENSITIVE
            // even on a case-insensitive filesystem; a value >= 2^63 (negative
            // as i64) is not a segment.
            if !hex
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            {
                continue;
            }
            let seq = match u64::from_str_radix(hex, 16) {
                Ok(v) if v <= i64::MAX as u64 => v as i64,
                _ => continue,
            };
            // `file_type()` from a directory entry does not follow symlinks, so
            // a symlink to a valid segment is not a segment.
            match entry.file_type() {
                Ok(ft) if ft.is_file() => {}
                _ => continue,
            }
            found.push(seq);
        }
        found.sort_unstable();
        found
    }

    pub(crate) fn segment_file(&self, seq: i64) -> PathBuf {
        self.dir.join(format!("{}{:016x}", self.prefix, seq))
    }

    // ---------- R1/R2: classify, remove residue ----------

    /// R1 then R2. Applies table H to every enumerated name, unlinks
    /// create-crash residue, and records the maximum sequence over ALL names —
    /// **including the residue it is about to remove** (W6), so a stale
    /// directory entry can never alias a segment a later create reuses.
    ///
    /// The asymmetry in table H is the whole point: a torn create produces an
    /// invalid `headerCrc` with overwhelming probability, so an invalid header
    /// on the **highest** name is an ordinary crash artifact, while the same
    /// bytes anywhere else are corruption — something above it exists, so its
    /// creation completed once.
    fn classify(&mut self, found: &[i64]) -> Result<()> {
        let max_observed = found.iter().copied().max().unwrap_or(0);
        self.next_seq = max_observed
            .checked_add(1)
            .ok_or_else(|| DbError::corrupt_msg("WAL segment sequence overflow"))?;

        let highest = found.last().copied();
        let mut residue: Vec<i64> = Vec::new();
        for &seq in found {
            // Sequence 0 is RESERVED for "no clean mark", so no conforming
            // writer can ever create it: FIRST_SEQ is 1 and next_seq only
            // increases. Rejected here, at R1, rather than left to fall through
            // — R4 would refuse it today only as a side effect of its retained
            // set coming out empty, which is an accident, not a rule.
            if seq == 0 {
                return Err(DbError::corrupt_msg(format!(
                    "WAL segment {}: sequence 0 is reserved for \"no clean mark\" and is never a segment",
                    file_name(&self.segment_file(0))
                )));
            }
            let path = self.segment_file(seq);
            let file = if self.read_only {
                OpenOptions::new().read(true).open(&path)?
            } else {
                OpenOptions::new().read(true).write(true).open(&path)?
            };
            let len = file.metadata()?.len();
            let mut hdr = [0u8; SEG_HDR as usize];
            match read_header(&file, len, &mut hdr, seq)? {
                // The handle is NOT retained: a recovery pass reopens it on
                // demand and releases it again, so the descriptor count stays
                // O(1) in the segment count.
                HeaderVerdict::Ok => {
                    self.segments
                        .push(Segment::new(seq, path, self.read_only, hdr, len))
                }
                HeaderVerdict::Corrupt(fault) => {
                    return Err(DbError::corrupt_msg(format!(
                        "WAL segment {}: {fault}",
                        file_name(&path)
                    )))
                }
                HeaderVerdict::Torn(fault) => {
                    if Some(seq) == highest {
                        // H1-H4 on the highest name: the create crashed. A
                        // read-only open excludes it from the set but keeps the
                        // file — the next writable open removes it.
                        residue.push(seq);
                    } else {
                        return Err(DbError::corrupt_msg(format!(
                            "WAL segment {}: {fault} (not the highest segment, so its create completed)",
                            file_name(&path)
                        )));
                    }
                }
            }
        }
        if !self.read_only && !residue.is_empty() {
            for seq in residue {
                remove_if_exists(&self.segment_file(seq))?;
            }
            self.fsync_dir()?;
        }
        self.recompute_sealed_bytes();
        Ok(())
    }

    // ---------- the namespace mutations ----------

    /// W2: `create → write header → force(true) → fsync the directory`, and
    /// only then may a section be appended. Without the directory fsync the
    /// whole segment can vanish on a crash, taking acknowledged commits with
    /// it; without the size-persisting force the header itself can be lost.
    /// Returns the new active segment, appended to the set.
    pub(crate) fn create_segment(&mut self, first_lsn: i64) -> Result<&mut Segment> {
        if self.read_only {
            return Err(DbError::ReadOnly);
        }
        if first_lsn <= 0 {
            return Err(DbError::corrupt_msg(format!(
                "segment firstLsn must be positive: {first_lsn}"
            )));
        }
        let seq = self.next_seq;
        self.next_seq = seq
            .checked_add(1)
            .ok_or_else(|| DbError::corrupt_msg("WAL segment sequence overflow"))?;
        let path = self.segment_file(seq);
        let hdr = build_header(seq, first_lsn);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        let created = (|| -> Result<()> {
            file.write_all_at(&hdr, 0)?;
            // The file's SIZE is part of the payload here: never sync_data.
            file.sync_all()?;
            self.fsync_dir()
        })();
        if let Err(e) = created {
            drop(file);
            let _ = std::fs::remove_file(&path);
            return Err(e);
        }
        // The Segment reopens on demand, like every other one.
        drop(file);

        // The segment this one displaces stops growing here, so its length
        // joins the sealed total.
        if let Some(prev) = self.segments.last() {
            self.sealed_bytes += prev.file_len;
        }
        self.segments
            .push(Segment::new(seq, path, self.read_only, hdr, SEG_HDR));
        Ok(self.segments.last_mut().expect("just pushed"))
    }

    /// W5: unlink every segment at or below `through_seq`, then fsync the
    /// directory. Called only after the `'K'` authorizing it is forced. A failed
    /// unlink is a leak that the next open retries (K5/K8) — never permission to
    /// advance an unproven mark.
    ///
    /// Ascending order is used for tidiness only. It does **not** make the
    /// crash-visible set a prefix: syscall order says nothing about the order
    /// removals persist before the fsync, so an interior gap is a legitimate
    /// crash image (N3/K9).
    pub(crate) fn unlink_through(&mut self, through_seq: i64) -> Result<()> {
        if self.read_only || through_seq <= 0 {
            return Ok(());
        }
        let n = self
            .segments
            .iter()
            .take_while(|s| s.seq <= through_seq)
            .count();
        if n == 0 {
            // No delete and NO directory fsync: this is the shape a recovered
            // mark takes when an earlier attempt already removed every file it
            // authorizes.
            return Ok(());
        }
        // Drop them from the live set BEFORE any delete can fail: their handles
        // are released here, so a failed unlink must not leave a
        // closed-but-listed segment behind for some later reader to use. The
        // files are then just a leak the next open retries.
        let retiring: Vec<Segment> = self.segments.drain(..n).collect();
        // RECOMPUTED, not decremented: subtracting each removed length is
        // correct only while the highest segment is never in the prefix — true
        // today (K4 plus R4's refusal of a mark that retires the whole set),
        // but that is a property of two other rules rather than of this method,
        // and getting it wrong drifts the counter silently in the direction
        // that stops cleaning.
        self.recompute_sealed_bytes();
        for s in retiring {
            let path = s.path.clone();
            drop(s);
            remove_if_exists(&path)?;
        }
        self.fsync_dir()
    }

    pub(crate) fn fsync_dir(&self) -> Result<()> {
        File::open(&self.dir)?.sync_all()?;
        Ok(())
    }

    fn recompute_sealed_bytes(&mut self) {
        self.sealed_bytes = 0;
        for i in 0..self.segments.len().saturating_sub(1) {
            self.sealed_bytes += self.segments[i].file_len;
        }
    }

    // ---------- accessors ----------

    pub(crate) fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub(crate) fn segments_mut(&mut self) -> &mut [Segment] {
        &mut self.segments
    }

    /// The highest-sequence segment with a valid header, or `None` for a fresh
    /// store.
    pub(crate) fn active(&self) -> Option<&Segment> {
        self.segments.last()
    }

    pub(crate) fn active_mut(&mut self) -> Option<&mut Segment> {
        self.segments.last_mut()
    }

    pub(crate) fn next_seq(&self) -> i64 {
        self.next_seq
    }

    pub(crate) fn read_only(&self) -> bool {
        self.read_only
    }

    pub(crate) fn base(&self) -> &Path {
        &self.base
    }

    /// How many segments currently hold an open file handle. Steady state after
    /// recovery is at most one — the active segment — and that bound is the
    /// point, so it is observable rather than merely intended.
    pub(crate) fn open_file_count(&self) -> usize {
        self.segments.iter().filter(|s| s.holds_file()).count()
    }

    /// Sum of the segment files' current lengths: what the log actually costs on
    /// the device. O(1).
    pub(crate) fn log_bytes(&self) -> u64 {
        match self.segments.last() {
            None => 0,
            Some(last) => self.sealed_bytes + last.file_len,
        }
    }

    /// The same number the slow way. Test-only, to pin `sealed_bytes` against
    /// drift.
    pub(crate) fn log_bytes_exact(&self) -> u64 {
        self.segments.iter().map(|s| s.file_len).sum()
    }

    /// Releases every segment handle and then the store lock, in that order.
    /// `Drop` does the same; this exists for the call sites that must observe
    /// the release point (D2's lock-owning namespace cleanup, slice A2).
    pub(crate) fn close(&mut self) {
        self.segments.clear();
        self.lock = None;
    }
}

impl Drop for WalSegmentSet {
    fn drop(&mut self) {
        self.close();
    }
}

/// Reads and validates one segment header (table H).
fn read_header(
    file: &File,
    len: u64,
    into: &mut [u8; SEG_HDR as usize],
    name_seq: i64,
) -> Result<HeaderVerdict> {
    if len == 0 {
        return Ok(HeaderVerdict::Torn("empty segment file".into())); // H1
    }
    if len < SEG_HDR {
        return Ok(HeaderVerdict::Torn(format!(
            "segment header truncated at {len} bytes" // H2
        )));
    }
    let mut read = 0usize;
    while read < into.len() {
        match file.read_at(&mut into[read..], read as u64)? {
            0 => return Ok(HeaderVerdict::Torn("segment header short read".into())),
            n => read += n,
        }
    }
    if crc32(&into[..SEG_HDR_CRC_LEN]) as i32 != be32(into, SEG_HDR_CRC_LEN) {
        return Ok(HeaderVerdict::Torn("segment header CRC mismatch".into())); // H3
    }
    if into[..8] != MAGIC {
        return Ok(HeaderVerdict::Torn("not a mapdb WAL segment".into())); // H4
    }
    let version = be32(into, 8);
    if version != FORMAT_VERSION {
        return Ok(HeaderVerdict::Corrupt(format!(
            "unsupported WAL format version {version}" // H5
        )));
    }
    let flags = be32(into, 12);
    if flags != 0 {
        return Ok(HeaderVerdict::Corrupt(format!(
            "unknown segment flags {flags}"
        ))); // H6
    }
    let seq = be64(into, 16);
    if seq != name_seq {
        return Ok(HeaderVerdict::Corrupt(format!(
            "header sequence {seq} does not match its name" // H7
        )));
    }
    let first_lsn = be64(into, 24);
    if first_lsn <= 0 {
        return Ok(HeaderVerdict::Corrupt(format!(
            "header firstLsn {first_lsn} is not a valid LSN" // H9
        )));
    }
    Ok(HeaderVerdict::Ok)
}

/// The 36 header bytes a conforming writer produces for `(seq, first_lsn)`.
pub(crate) fn build_header(seq: i64, first_lsn: i64) -> [u8; SEG_HDR as usize] {
    let mut hdr = [0u8; SEG_HDR as usize];
    hdr[..8].copy_from_slice(&MAGIC);
    hdr[8..12].copy_from_slice(&FORMAT_VERSION.to_be_bytes());
    hdr[12..16].copy_from_slice(&0i32.to_be_bytes());
    hdr[16..24].copy_from_slice(&seq.to_be_bytes());
    hdr[24..32].copy_from_slice(&first_lsn.to_be_bytes());
    let crc = crc32(&hdr[..SEG_HDR_CRC_LEN]);
    hdr[SEG_HDR_CRC_LEN..].copy_from_slice(&(crc as i32).to_be_bytes());
    hdr
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    PathBuf::from(s)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string()
}

/// Regular file, symlinks NOT followed — the N4/N6 discipline.
fn is_regular_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|m| m.is_file())
        .unwrap_or(false)
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(DbError::Io(e)),
    }
}

/// `flock(LOCK_EX|LOCK_NB)` / `flock(LOCK_SH|LOCK_NB)`. `Ok(false)` means the
/// lock is held by someone else; a real error propagates.
fn try_flock(file: &File, exclusive: bool) -> Result<bool> {
    let op = if exclusive {
        libc::LOCK_EX
    } else {
        libc::LOCK_SH
    } | libc::LOCK_NB;
    // SAFETY: `fd` is a live descriptor owned by `file` for the duration of the
    // call, and `flock` touches no memory.
    let rc = unsafe { libc::flock(file.as_raw_fd(), op) };
    if rc == 0 {
        return Ok(true);
    }
    let err = std::io::Error::last_os_error();
    match err.raw_os_error() {
        Some(e) if e == libc::EWOULDBLOCK || e == libc::EAGAIN || e == libc::EINTR => Ok(false),
        _ => Err(DbError::Io(err)),
    }
}

/// `access(dir, W_OK)` — effective writability, which `std` cannot express
/// (`Permissions::readonly` reports the owner mode bit, not whether THIS process
/// may create a file). Only used to establish the positive "read-only medium"
/// proof of the read-only lock path.
fn is_writable_dir(dir: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let mut bytes = dir.as_os_str().as_bytes().to_vec();
    bytes.push(0);
    // SAFETY: `bytes` is NUL-terminated and outlives the call.
    unsafe { libc::access(bytes.as_ptr() as *const libc::c_char, libc::W_OK) == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    // ---------------------------------------------------------------- test kit
    // The rust half of Java's WalTestKit: the byte-level recipe lives in one
    // place so a hand-built image cannot drift from the writer's.

    fn scratch(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "mapdb5_walseg_{}_{}_{}",
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

    fn write_segment(base: &Path, seq: i64, bytes: &[u8]) {
        std::fs::write(seg_path(base, seq), bytes).expect("write segment");
    }

    /// A valid segment image: header only (H8), which is all A0 can produce —
    /// sections arrive with the codec in A1.
    fn header_image(seq: i64, first_lsn: i64) -> Vec<u8> {
        build_header(seq, first_lsn).to_vec()
    }

    /// Recomputes `headerCrc` in place, so a doctored header stays CRC-valid
    /// and reaches the semantic rows H5-H7/H9 instead of H3.
    fn reseal(hdr: &mut [u8]) {
        let crc = crc32(&hdr[..SEG_HDR_CRC_LEN]) as i32;
        hdr[SEG_HDR_CRC_LEN..SEG_HDR as usize].copy_from_slice(&crc.to_be_bytes());
    }

    fn seq_list(set: &WalSegmentSet) -> Vec<i64> {
        set.segments().iter().map(|s| s.seq).collect()
    }

    fn is_corrupt<T>(r: Result<T>) -> bool {
        matches!(r, Err(DbError::DataCorruption(_)))
    }

    fn corrupt_msg<T>(r: Result<T>) -> String {
        match r {
            Err(DbError::DataCorruption(c)) => c.to_string(),
            Err(e) => panic!("expected DataCorruption, got {e}"),
            Ok(_) => panic!("expected DataCorruption, got Ok"),
        }
    }

    // ---------------------------------------------------------------- the header

    /// The byte-level cross-check against the reference implementation: these
    /// 36 bytes are the header of `reject-wal-java-v3.walseg`, a segment
    /// produced by the Java writer (xfixtures, Stage 2). If this port's builder
    /// and Java's disagree by one byte, every section CRC in the file differs
    /// too, because the header IS the CRC domain.
    #[test]
    fn header_bytes_match_the_java_writer() {
        let java: [u8; 36] = [
            0x4d, 0x44, 0x42, 0x53, 0x2e, 0x57, 0x41, 0x4c, // "MDBS.WAL"
            0x00, 0x00, 0x00, 0x03, // version 3
            0x00, 0x00, 0x00, 0x00, // flags 0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, // seq 1
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, // firstLsn 1
            0x4a, 0x4d, 0x90, 0x4b, // headerCrc
        ];
        assert_eq!(java, build_header(1, 1));
    }

    /// The CRC domain, cross-checked against the same Java artifact: its first
    /// section sits at offset 36 and carries tag 'S', lsn 1, bodyLen 104, and
    /// Java sealed its header CRC as 0x9d8280b3 over
    /// `segmentHeader[0,36) || be64(36) || sectionHeader[0,17)`. A port that
    /// seeds a register instead of feeding a prefix, or that uses the 28 bytes
    /// the Java javadoc claims, fails here.
    #[test]
    fn crc_domain_matches_the_java_writer() {
        let seg = Segment::new(
            1,
            PathBuf::from("unused"),
            true,
            build_header(1, 1),
            SEG_HDR,
        );
        let mut sec_hdr = [0u8; 17];
        sec_hdr[0] = b'S';
        sec_hdr[1..9].copy_from_slice(&1i64.to_be_bytes());
        sec_hdr[9..17].copy_from_slice(&104i64.to_be_bytes());
        let mut crc = crc32fast::Hasher::new();
        seg.crc_domain(&mut crc, SEG_HDR);
        crc.update(&sec_hdr);
        assert_eq!(0x9d82_80b3_u32, crc.finalize());
    }

    /// The domain binds a section to its segment AND to its offset: the same 17
    /// header bytes seal differently at another offset and in another segment.
    #[test]
    fn crc_domain_binds_segment_and_offset() {
        let a = Segment::new(1, PathBuf::from("a"), true, build_header(1, 1), SEG_HDR);
        let b = Segment::new(2, PathBuf::from("b"), true, build_header(2, 1), SEG_HDR);
        let at = |s: &Segment, off: u64| {
            let mut c = crc32fast::Hasher::new();
            s.crc_domain(&mut c, off);
            c.update(b"the same seventeen");
            c.finalize()
        };
        assert_ne!(at(&a, 36), at(&a, 61), "offset must be bound");
        assert_ne!(at(&a, 36), at(&b, 36), "segment identity must be bound");
    }

    // ---------------------------------------------------------------- N: enumeration

    #[test]
    fn enumeration_ignores_everything_that_is_not_a_segment_name() {
        let dir = scratch("enum");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        write_segment(&base, 0x2a, &header_image(0x2a, 1));
        // near misses, all ignored rather than rejected
        std::fs::write(dir.join("store.db.wal.000000000000000"), b"short").unwrap();
        std::fs::write(dir.join("store.db.wal.00000000000000001"), b"long").unwrap();
        std::fs::write(dir.join("store.db.wal.00000000000000AB"), b"upper").unwrap();
        std::fs::write(dir.join("store.db.wal.zzzzzzzzzzzzzzzz"), b"nonhex").unwrap();
        std::fs::write(dir.join("store.db.wal.ffffffffffffffff"), b"negative i64").unwrap();
        std::fs::write(dir.join("other.db.wal.0000000000000001"), b"another store").unwrap();
        std::fs::write(dir.join("notes.txt"), b"unrelated").unwrap();
        std::fs::create_dir(dir.join("store.db.wal.0000000000000009")).unwrap();

        let set = WalSegmentSet::open(&base, false).expect("open");
        assert_eq!(
            vec![1, 0x2a],
            seq_list(&set),
            "gaps are legal, order ascending"
        );
        // W6: nextSeq is one above the highest NAME, not the count.
        assert_eq!(0x2b, set.next_seq());
    }

    /// R1. Sequence 0 is reserved for "no clean mark" and is never a segment —
    /// a file at that name is corruption, not residue.
    #[test]
    fn sequence_zero_is_refused() {
        let dir = scratch("seq0");
        let base = base_in(&dir);
        write_segment(&base, 0, &header_image(0, 1));
        let msg = corrupt_msg(WalSegmentSet::open(&base, false));
        assert!(msg.contains("sequence 0 is reserved"), "{msg}");
    }

    /// N6. The v1 single-file log refuses the open before anything is
    /// enumerated, created or deleted.
    #[test]
    fn v1_single_file_log_is_refused_not_migrated() {
        let dir = scratch("n6");
        let base = base_in(&dir);
        let v1 = with_suffix(&base, ".wal");
        std::fs::write(&v1, b"MDBS.WAL\0\0\0\x01\0\0\0\0").unwrap();
        let msg = corrupt_msg(WalSegmentSet::open(&base, false));
        assert!(msg.contains("no migration"), "{msg}");
        assert!(v1.exists(), "the refusal must not delete the evidence");
    }

    // ---------------------------------------------------------------- H: header table

    /// H1-H4 on the highest name are create-crash residue: a writable open
    /// unlinks them, and W6 has already burnt the sequence number so the fresh
    /// segment cannot reuse it.
    #[test]
    fn torn_create_residue_on_the_highest_name_is_removed() {
        for (tag, bytes) in [
            ("h1", Vec::new()),
            ("h2", vec![0u8; 16]),
            ("h3", {
                let mut h = header_image(2, 1);
                h[24] ^= 0x01; // firstLsn edited without resealing
                h
            }),
            ("h4", {
                let mut h = header_image(2, 1);
                h[0] = b'X';
                reseal(&mut h);
                h
            }),
        ] {
            let dir = scratch(tag);
            let base = base_in(&dir);
            write_segment(&base, 1, &header_image(1, 1));
            write_segment(&base, 2, &bytes);

            let set = WalSegmentSet::open(&base, false).expect(tag);
            assert_eq!(vec![1], seq_list(&set), "{tag}: residue is not in the set");
            assert!(!seg_path(&base, 2).exists(), "{tag}: residue is unlinked");
            assert_eq!(3, set.next_seq(), "{tag}: W6 burnt the residue's number");
        }
    }

    /// The same shapes anywhere below the highest name are corruption: a
    /// segment exists above them, so their create completed once.
    #[test]
    fn torn_create_shapes_below_the_highest_name_are_corruption() {
        let dir = scratch("h-low");
        let base = base_in(&dir);
        write_segment(&base, 1, &[0u8; 16]);
        write_segment(&base, 2, &header_image(2, 1));
        let msg = corrupt_msg(WalSegmentSet::open(&base, false));
        assert!(msg.contains("not the highest segment"), "{msg}");
        assert!(seg_path(&base, 1).exists(), "corruption deletes nothing");
    }

    /// H5-H7/H9: a CRC-valid header carrying wrong content is a writer defect or
    /// a copied file — corruption wherever it appears, INCLUDING the highest
    /// name, where a torn create would have been forgiven.
    #[test]
    fn resealed_semantic_faults_are_corruption_even_on_the_highest_name() {
        // (tag, the edit that makes the header semantically wrong, expected message)
        type Case = (&'static str, fn(&mut Vec<u8>), &'static str);
        let cases: [Case; 4] = [
            (
                "h5",
                |h| h[8..12].copy_from_slice(&2i32.to_be_bytes()),
                "unsupported WAL format version 2",
            ),
            (
                "h6",
                |h| h[12..16].copy_from_slice(&1i32.to_be_bytes()),
                "unknown segment flags 1",
            ),
            (
                "h7",
                |h| h[16..24].copy_from_slice(&9i64.to_be_bytes()),
                "does not match its name",
            ),
            (
                "h9",
                |h| h[24..32].copy_from_slice(&0i64.to_be_bytes()),
                "is not a valid LSN",
            ),
        ];
        for (tag, edit, expect) in cases {
            let dir = scratch(tag);
            let base = base_in(&dir);
            let mut h = header_image(1, 1);
            edit(&mut h);
            reseal(&mut h);
            write_segment(&base, 1, &h);
            let msg = corrupt_msg(WalSegmentSet::open(&base, false));
            assert!(msg.contains(expect), "{tag}: {msg}");
            assert!(
                seg_path(&base, 1).exists(),
                "{tag}: corruption deletes nothing"
            );
        }
    }

    /// H8. A valid header-only segment is legitimate AT ANY POSITION — W7's
    /// post-truncation rotate produces one, and sweeping it would create an
    /// interior gap.
    #[test]
    fn a_valid_header_only_segment_is_legitimate_at_any_position() {
        let dir = scratch("h8");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        write_segment(&base, 2, &header_image(2, 1));
        write_segment(&base, 3, &header_image(3, 1));
        let set = WalSegmentSet::open(&base, false).expect("open");
        assert_eq!(vec![1, 2, 3], seq_list(&set));
        assert!(set.segments().iter().all(Segment::empty));
        assert_eq!(3 * SEG_HDR, set.log_bytes());
        assert_eq!(set.log_bytes_exact(), set.log_bytes());
    }

    /// A read-only open reaches the same verdicts but performs no mutation: the
    /// residue is excluded from the set and LEFT ON DISK for the next writable
    /// open to remove.
    #[test]
    fn a_read_only_open_excludes_residue_but_keeps_it() {
        let dir = scratch("ro-residue");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        write_segment(&base, 2, &[0u8; 16]);
        let set = WalSegmentSet::open(&base, true).expect("open");
        assert_eq!(vec![1], seq_list(&set));
        assert!(seg_path(&base, 2).exists(), "read-only mutates nothing");
        assert_eq!(3, set.next_seq());
    }

    // ---------------------------------------------------------------- W2/W5/W6

    /// W2 + W6. A create writes the whole header, forces it with the size, and
    /// takes the next unburnt sequence number.
    #[test]
    fn create_segment_writes_a_valid_header_at_the_burnt_successor() {
        let dir = scratch("w2");
        let base = base_in(&dir);
        // Residue only: its name is burnt, so the fresh segment is 8, not 1.
        write_segment(&base, 7, &[0u8; 4]);
        let mut set = WalSegmentSet::open(&base, false).expect("open");
        assert!(set.segments().is_empty());

        let seq = set.create_segment(1).expect("create").seq;
        assert_eq!(8, seq);
        assert_eq!(9, set.next_seq());
        assert_eq!(
            build_header(8, 1).to_vec(),
            std::fs::read(seg_path(&base, 8)).unwrap()
        );
        assert_eq!(SEG_HDR, set.log_bytes());

        // A second create seals the first: its length joins the sealed total.
        set.create_segment(2).expect("create");
        assert_eq!(2 * SEG_HDR, set.log_bytes());
        assert_eq!(set.log_bytes_exact(), set.log_bytes());
        assert_eq!(vec![8, 9], seq_list(&set));
    }

    /// A name is never reused, even after every file that carried it is gone.
    #[test]
    fn a_sequence_number_is_never_reused() {
        let dir = scratch("w6");
        let base = base_in(&dir);
        write_segment(&base, 5, &header_image(5, 1));
        let mut set = WalSegmentSet::open(&base, false).expect("open");
        set.unlink_through(5).expect("unlink");
        assert!(set.segments().is_empty());
        assert!(!seg_path(&base, 5).exists());
        assert_eq!(6, set.create_segment(1).expect("create").seq);
    }

    /// W5. The whole prefix leaves the live set before the first delete, and the
    /// files follow.
    #[test]
    fn unlink_through_removes_the_prefix_and_nothing_else() {
        let dir = scratch("w5");
        let base = base_in(&dir);
        for seq in 1..=4 {
            write_segment(&base, seq, &header_image(seq, 1));
        }
        let mut set = WalSegmentSet::open(&base, false).expect("open");
        set.unlink_through(2).expect("unlink");
        assert_eq!(vec![3, 4], seq_list(&set));
        assert!(!seg_path(&base, 1).exists());
        assert!(!seg_path(&base, 2).exists());
        assert!(seg_path(&base, 3).exists());
        assert_eq!(2 * SEG_HDR, set.log_bytes());
        assert_eq!(set.log_bytes_exact(), set.log_bytes());

        // A mark whose unlink already completed: nothing to do, no error.
        set.unlink_through(2).expect("idempotent");
        assert_eq!(vec![3, 4], seq_list(&set));
    }

    /// A read-only set never unlinks and never creates.
    #[test]
    fn a_read_only_set_refuses_to_mutate_the_namespace() {
        let dir = scratch("ro-mutate");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        let mut set = WalSegmentSet::open(&base, true).expect("open");
        assert!(matches!(set.create_segment(1), Err(DbError::ReadOnly)));
        set.unlink_through(1).expect("no-op");
        assert!(seg_path(&base, 1).exists());
        assert_eq!(vec![1], seq_list(&set));
    }

    // ---------------------------------------------------------------- the store lock

    /// Two writable opens of the same namespace cannot coexist — including in
    /// ONE process, which is where POSIX record locks would have silently
    /// admitted the second.
    #[test]
    fn a_second_writable_open_is_refused() {
        let dir = scratch("lock-rw");
        let base = base_in(&dir);
        let first = WalSegmentSet::open(&base, false).expect("first");
        assert!(matches!(
            WalSegmentSet::open(&base, false),
            Err(DbError::Locked(_))
        ));
        assert!(
            with_suffix(&base, ".lock").exists(),
            "the lock file is created"
        );
        drop(first);
        // The lock is released with the handle, so the next open succeeds.
        WalSegmentSet::open(&base, false).expect("after release");
    }

    /// Shared readers coexist; a writer is excluded while one is live, and a
    /// reader is excluded while the writer is.
    #[test]
    fn read_only_opens_share_and_exclude_a_writer() {
        let dir = scratch("lock-ro");
        let base = base_in(&dir);
        let r1 = WalSegmentSet::open(&base, true).expect("first reader");
        let r2 = WalSegmentSet::open(&base, true).expect("second reader");
        assert!(matches!(
            WalSegmentSet::open(&base, false),
            Err(DbError::Locked(_))
        ));
        drop(r1);
        drop(r2);
        let w = WalSegmentSet::open(&base, false).expect("writer after readers");
        assert!(matches!(
            WalSegmentSet::open(&base, true),
            Err(DbError::Locked(_))
        ));
        drop(w);
    }

    /// A read-only open CREATES the lock file when the directory allows it —
    /// "read-only" is not "no filesystem write".
    #[test]
    fn a_read_only_open_creates_the_lock_file() {
        let dir = scratch("lock-ro-create");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        let lock = with_suffix(&base, ".lock");
        assert!(!lock.exists());
        let set = WalSegmentSet::open(&base, true).expect("open");
        assert!(lock.exists());
        drop(set);
    }
}
