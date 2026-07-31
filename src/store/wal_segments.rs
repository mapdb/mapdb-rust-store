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
//! # The legacy boundary (D1)
//!
//! The ports' v1 opener took the WAL FILE path, so after the v3 cutover the same
//! call site hands what is now a BASE. Three pre-existing artifacts therefore
//! refuse the open rather than being ignored, and none of them is ever deleted:
//! a regular file at `<base>.wal` (Java's own N6 row), a regular file at
//! `<base>` itself, and a `<base>.ckpt` left by v1's rename-checkpoint — which
//! after a v1 crash may be the only recoverable copy. Silently starting a fresh
//! segment set beside any of them is the one outcome the format break exists to
//! prevent.

use super::wal_write::{wal_io_event, WalIo, WalOpKind};
use crate::error::{DbError, Result};
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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

    /// Opens this segment's file if it does not already hold one. Idempotent.
    ///
    /// Deliberately split from [`file`](Self::file): a single
    /// `file(&mut self) -> Result<&File>` keeps an EXCLUSIVE borrow of the
    /// segment alive for as long as the caller holds the handle, so it cannot
    /// read a section and then feed [`crc_domain`](Self::crc_domain) or update
    /// `valid_end`/`last_lsn` from the same segment — which is precisely what a
    /// recovery pass does, per section, for every section. `ensure_open` ends
    /// its borrow at the semicolon and `file` then borrows shared.
    pub(crate) fn ensure_open(&mut self) -> Result<()> {
        if self.file.is_none() {
            let f = if self.read_only {
                OpenOptions::new().read(true).open(&self.path)?
            } else {
                OpenOptions::new().read(true).write(true).open(&self.path)?
            };
            self.file = Some(f);
        }
        Ok(())
    }

    /// The handle, or `None` when [`ensure_open`](Self::ensure_open) has not run
    /// since the last [`release`](Self::release).
    pub(crate) fn file(&self) -> Option<&File> {
        self.file.as_ref()
    }

    /// Closes the handle if one is held; the segment stays usable and reopens on
    /// demand. Called as soon as a recovery pass finishes with a segment, which
    /// is what bounds the descriptor count to O(1) instead of O(segments).
    /// Nothing is written through these handles without a preceding force, so a
    /// lost close never loses data.
    pub(crate) fn release(&mut self) {
        self.file = None;
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
        crc_domain_of(crc, &self.header, section_offset);
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
    /// `<base file name>.wal.` as **native bytes**. Never a `String`: a Unix
    /// path is a byte string, and requiring UTF-8 here would make a perfectly
    /// legal namespace unopenable in this port alone (Java derives the prefix
    /// from `File.getName()` with no such requirement, and defines acceptance by
    /// an ASCII suffix and file type — `WalSegmentSet.java:199-207, 279-311`).
    prefix: OsString,
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
    /// `File` closes the descriptor, which releases the OFD lock taken on it.
    /// `None` only in the read-only-medium case
    /// (see [`take_store_lock`](Self::take_store_lock)).
    lock: Option<File>,
    /// The in-process half of the same lock; released after `lock`.
    process_claim: Option<ProcessClaim>,
    /// True once [`close`](Self::close) has run: the namespace mutations must
    /// not run without the lock this handle no longer holds.
    closed: bool,
    /// The durability seam (A2). Per set rather than per process — see
    /// [`WalIo`](super::wal_write::WalIo).
    io: Option<Arc<dyn WalIo>>,
    /// Test-only durability observation, per set. Java exposes the same points
    /// through its event seam; a byte comparison of a SUCCESSFUL create cannot
    /// tell a missing fsync from a present one, and the no-op `unlink_through`
    /// must be shown not to fsync at all.
    #[cfg(test)]
    dir_fsyncs: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    segment_syncs: std::sync::atomic::AtomicU64,
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
    #[cfg(test)]
    pub(crate) fn open(base: &Path, read_only: bool) -> Result<WalSegmentSet> {
        Self::open_with_io(base, read_only, None)
    }

    /// [`open`](Self::open) with a durability seam installed for the whole
    /// lifetime of the set, including the create and unlink this open itself
    /// performs (R2's residue removal).
    pub(crate) fn open_with_io(
        base: &Path,
        read_only: bool,
        io: Option<Arc<dyn WalIo>>,
    ) -> Result<WalSegmentSet> {
        let abs = if base.is_absolute() {
            base.to_path_buf()
        } else {
            std::env::current_dir()?.join(base)
        };
        let dir = abs
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let name = abs.file_name().ok_or_else(|| {
            DbError::wrong_config(format!("WAL base path has no file name: {}", abs.display()))
        })?;
        let mut prefix = name.to_os_string();
        prefix.push(".wal.");

        let mut set = WalSegmentSet {
            base: abs,
            dir,
            prefix,
            read_only,
            segments: Vec::new(),
            next_seq: FIRST_SEQ,
            sealed_bytes: 0,
            lock: None,
            process_claim: None,
            closed: false,
            io,
            #[cfg(test)]
            dir_fsyncs: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            segment_syncs: std::sync::atomic::AtomicU64::new(0),
        };
        // Every early return from here drops `set`, which drops the lock handle
        // and so releases the store lock — Java's `finally { closeQuietly() }`.
        set.take_store_lock()?;
        // D1: the legacy boundary. All three rows REFUSE, delete nothing, and
        // fire before any v3 segment is created. Regular files only, the same
        // discipline N4 applies: a DIRECTORY at one of these names is not a
        // legacy artifact.
        //
        // N6 is Java's own row; the other two are the ports' upgrade-safety
        // boundary and have no Java counterpart, because Java's base has never
        // named a file. A v1 caller passed the WAL FILE path, so the same call
        // site now passes a BASE — and N6 alone would look at `<arg>.wal`, miss
        // the old log sitting at `<arg>`, and open a fresh empty store beside
        // the user's only durable copy.
        for (path, what) in [
            (with_suffix(&set.base, ".wal"), "v1 single-file WAL"),
            (
                set.base.clone(),
                "regular file at the WAL base path (the v3 opener takes a base, not a log file)",
            ),
            (
                with_suffix(&set.base, ".ckpt"),
                "v1 checkpoint temp, possibly the only recoverable copy after a v1 crash",
            ),
        ] {
            if is_regular_file(&path) {
                return Err(DbError::corrupt_msg(format!(
                    "{what} present at {}: no migration to v3 — open it with the release that \
                     wrote it and copy the data across, or move it aside",
                    path.display()
                )));
            }
        }
        let found = set.enumerate();
        set.classify(&found)?;
        Ok(set)
    }

    /// §3.1: exactly one process may run open, recovery or writing at a time.
    /// Recovery unlinks, truncates and rotates, and two concurrent opens would
    /// also pick the same next sequence number. v1 took no lock; this is new.
    ///
    /// The lock has **two halves**, because Java's has two halves and neither
    /// one alone reproduces it:
    ///
    /// 1. An **OFD record lock** (`fcntl(F_OFD_SETLK)`) on `<base>.lock`.
    ///    Not `flock`: BSD locks and POSIX record locks are independent lock
    ///    classes on Linux, so a `flock` here would not exclude — at all — a
    ///    Java process holding the same store through `FileChannel.tryLock`,
    ///    which is a record lock (`WalSegmentSet.java:267-274`). Uniformity
    ///    across implementations is the ruling this port exists to serve, and a
    ///    lock that only excludes its own language is not one. Measured against
    ///    a live JVM holder rather than assumed: `flock` acquires straight
    ///    through Java's lock, `F_OFD_SETLK` is refused by it and acquires as
    ///    soon as Java releases. OFD (rather than plain `F_SETLK`) because
    ///    ownership is the open file description, not the process: a second open
    ///    in one process is refused instead of silently upgrading the first
    ///    one's lock, and closing any other descriptor on the file cannot drop
    ///    this lock.
    /// 2. A **process-local claim** keyed by the lock file's `(device, inode)`.
    ///    Java holds its locks in a JVM-wide table and refuses ANY overlapping
    ///    second lock in the same JVM — `OverlappingFileLockException` does not
    ///    consider lock MODE, so even two read-only opens of one store are
    ///    refused (verified against the JVM, not merely read off the javadoc).
    ///    No kernel lock can express that: two OFD read locks are compatible by
    ///    construction, which is the entire point of a read lock. So the port
    ///    keeps the table Java keeps.
    ///
    /// The two halves are released in the reverse order they are taken (the
    /// file, then the claim), so no window exists in which this process has
    /// forgotten the store while the kernel still holds its lock.
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
                        // Java's read-only-medium branch, and Java's exact
                        // heuristic: `access(W_OK)` answers for THIS process's
                        // credentials. It is evidence that no writer can create
                        // the lock file, not proof — another uid, or root, still
                        // can. The behaviour is frozen by the reference (see
                        // `WalSegmentSet.java:248-255`, whose comment claims
                        // more than the check delivers); tightening it to a real
                        // `ST_RDONLY` mount test would change the set of stores
                        // that open, so it is an owner decision, not a port one.
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
        // Identity of the LOCK FILE, not of the path used to reach it: two opens
        // naming the same store through different paths (a symlinked directory,
        // a bind mount, `./db` vs `db`) must collide, and Java's lock table is
        // likewise keyed by file identity rather than by pathname.
        let md = handle.metadata()?;
        // Dropped on every error path below, which is what releases the claim.
        let claim = ProcessClaim::take((md.dev(), md.ino())).ok_or_else(|| {
            DbError::Locked(format!(
                "WAL store {} is already open in this process",
                self.base.display()
            ))
        })?;
        if !try_ofd_lock(&handle, !self.read_only)? {
            return Err(DbError::Locked(format!(
                "WAL store {} is locked by another process",
                self.base.display()
            )));
        }
        self.lock = Some(handle);
        self.process_claim = Some(claim);
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
            // Matched as BYTES: a name is not required to be UTF-8 to be a
            // segment, and neither is the base path it hangs off.
            let name = entry.file_name();
            let hex = match name.as_bytes().strip_prefix(self.prefix.as_bytes()) {
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
                .iter()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
            {
                continue;
            }
            // Sixteen ASCII hex digits: UTF-8 by construction.
            let hex = std::str::from_utf8(hex).expect("ascii hex");
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
        let mut name = self.prefix.clone();
        name.push(format!("{seq:016x}"));
        self.dir.join(name)
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
        // A namespace that has run out of sequence numbers is exhausted, not
        // damaged: every byte on disk is intact and readable, there is simply no
        // name left to create. `StoreFull` is the capacity ceiling; Java throws
        // a plain `DBException` here (`WalSegmentSet.java:329-334`).
        self.next_seq = max_observed.checked_add(1).ok_or(DbError::StoreFull)?;

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
        if self.closed {
            return Err(DbError::StoreClosed);
        }
        // Java throws a plain `DBException` for all three of these; the port
        // picks the nearest non-corruption variants, because NOTHING on disk is
        // damaged in any of them and a caller that treats the store as corrupt
        // would be reacting to its own bug. (`WalSegmentSet.java:435-441`.)
        if self.read_only {
            return Err(DbError::ReadOnly);
        }
        if first_lsn <= 0 {
            return Err(DbError::wrong_config(format!(
                "segment firstLsn must be positive: {first_lsn}"
            )));
        }
        let seq = self.next_seq;
        // The name is burned here, BEFORE any I/O (W6) — a failed create must
        // never hand its sequence number to the next one. On overflow Java wraps
        // `nextSeq` negative and only then throws, leaving a store that would
        // answer the next create with a negative name; the checked assignment
        // leaves the counter where it was, so the refusal simply repeats.
        self.next_seq = seq.checked_add(1).ok_or(DbError::StoreFull)?;
        let path = self.segment_file(seq);
        let hdr = build_header(seq, first_lsn);

        wal_io_event(&self.io, WalOpKind::Create, seq, 0, 0, 0)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        let created = (|| -> Result<()> {
            wal_io_event(&self.io, WalOpKind::SegHeader, seq, 0, SEG_HDR, 0)?;
            file.write_all_at(&hdr, 0)?;
            wal_io_event(&self.io, WalOpKind::ForceFull, seq, SEG_HDR, 0, 0)?;
            // The file's SIZE is part of the payload here: never sync_data.
            file.sync_all()?;
            #[cfg(test)]
            self.segment_syncs
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        if self.closed {
            return Err(DbError::StoreClosed);
        }
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
            let (path, seq) = (s.path.clone(), s.seq);
            drop(s);
            wal_io_event(&self.io, WalOpKind::Unlink, seq, 0, 0, 0)?;
            remove_if_exists(&path)?;
        }
        self.fsync_dir()
    }

    pub(crate) fn fsync_dir(&self) -> Result<()> {
        wal_io_event(&self.io, WalOpKind::DirSync, 0, 0, 0, 0)?;
        File::open(&self.dir)?.sync_all()?;
        #[cfg(test)]
        self.dir_fsyncs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Directory fsyncs performed by this set so far.
    #[cfg(test)]
    pub(crate) fn dir_fsyncs(&self) -> u64 {
        self.dir_fsyncs.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Full segment-file syncs (`sync_all`, never `sync_data`) performed by
    /// this set so far.
    #[cfg(test)]
    pub(crate) fn segment_syncs(&self) -> u64 {
        self.segment_syncs
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn recompute_sealed_bytes(&mut self) {
        self.sealed_bytes = 0;
        for i in 0..self.segments.len().saturating_sub(1) {
            self.sealed_bytes += self.segments[i].file_len;
        }
    }

    // ---------- accessors ----------

    /// The durability seam installed at open, for the recovery paths that
    /// perform namespace-visible I/O of their own (R7's truncate and force).
    pub(crate) fn wal_io(&self) -> &Option<Arc<dyn WalIo>> {
        &self.io
    }

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

    #[cfg(test)]
    pub(crate) fn next_seq(&self) -> i64 {
        self.next_seq
    }

    pub(crate) fn read_only(&self) -> bool {
        self.read_only
    }

    /// How many segments currently hold an open file handle. Steady state after
    /// recovery is at most one — the active segment — and that bound is the
    /// point, so it is observable rather than merely intended.
    pub(crate) fn open_file_count(&self) -> usize {
        self.segments.iter().filter(|s| s.file().is_some()).count()
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
    #[cfg(test)]
    pub(crate) fn log_bytes_exact(&self) -> u64 {
        self.segments.iter().map(|s| s.file_len).sum()
    }

    /// **D2's lock-owning namespace cleanup**: delete every file this base owns
    /// — the segments, by the same enumeration rule the open used, plus
    /// `<base>.lock` — then fsync the directory once and release the lock.
    ///
    /// It runs WHILE THE LOCK IS STILL HELD, and that ordering is the whole
    /// point. Close-then-delete is racy in two ways: once close releases the
    /// lock a second opener can acquire the namespace and have its live segments
    /// deleted underneath it, and unlinking the lock PATHNAME while another
    /// instance may exist lets a third opener create a fresh lock inode and
    /// "acquire" a namespace someone else is already using. The lock file goes
    /// last, under the lock, as the owning instance's final act.
    ///
    /// Names that are not this base's segments are preserved: enumeration
    /// ignores them (N4), and a delete-after-close must not sweep a directory it
    /// was merely given a path into. Errors propagate — a best-effort delete
    /// would report a clean removal of files that are still there.
    pub(crate) fn delete_namespace(&mut self) -> Result<()> {
        if self.closed {
            return Err(DbError::StoreClosed);
        }
        if self.read_only {
            return Err(DbError::ReadOnly);
        }
        // RE-enumerated rather than taken from `segments`: the live list holds
        // what recovery retained, and the directory may also hold names this
        // open legitimately left behind (a read-only-style residue kept by an
        // earlier writer, an interior gap). All of them are this base's.
        for seq in self.enumerate() {
            let path = self.segment_file(seq);
            wal_io_event(&self.io, WalOpKind::Unlink, seq, 0, 0, 0)?;
            remove_if_exists(&path)?;
        }
        self.segments.clear();
        self.sealed_bytes = 0;
        // The lock file is unlinked while its lock is still held, so no opener
        // can be between "created the inode" and "locked it" for THIS pathname.
        let lock_path = with_suffix(&self.base, ".lock");
        remove_if_exists(&lock_path)?;
        self.fsync_dir()?;
        self.close();
        Ok(())
    }

    /// Releases every segment handle, then the store lock file, then this
    /// process's claim on the namespace — in that order. `Drop` does the same;
    /// this exists for the call sites that must observe the release point (D2's
    /// lock-owning namespace cleanup, slice A2).
    ///
    /// The set stays alive but is **closed**: the namespace mutations refuse
    /// from here on, because they would otherwise run without the lock that
    /// makes them safe. D2's cleanup must therefore delete while the set is
    /// still open and call this last.
    pub(crate) fn close(&mut self) {
        self.segments.clear();
        self.lock = None;
        self.process_claim = None;
        self.closed = true;
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

/// [`Segment::crc_domain`] for a caller holding header BYTES rather than a
/// segment — the section writer, which seals a section before the segment it
/// extends has been re-read, and the byte-level test kit.
pub(crate) fn crc_domain_of(
    crc: &mut crc32fast::Hasher,
    header: &[u8; SEG_HDR as usize],
    section_offset: u64,
) {
    crc.update(header);
    crc.update(&section_offset.to_be_bytes());
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

/// The file name for a DIAGNOSTIC. Lossy on purpose: a non-UTF-8 name must
/// still appear in the message that names it, and a message is never a key.
pub(crate) fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
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

/// The `(device, inode)` of every store lock this process currently holds —
/// the port's copy of the JVM-wide lock table Java's `tryLock` consults (see
/// [`take_store_lock`](WalSegmentSet::take_store_lock) for why a kernel lock
/// cannot stand in for it).
static OPEN_STORES: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Membership in [`OPEN_STORES`], released on drop — including on an unwind,
/// which is one thing Java's table cannot promise about a partially-constructed
/// `WalSegmentSet`.
struct ProcessClaim {
    key: (u64, u64),
}

impl ProcessClaim {
    /// `None` when this process already holds that store, whatever the mode of
    /// either open.
    fn take(key: (u64, u64)) -> Option<ProcessClaim> {
        // A poisoned mutex here would mean a panic inside these few lines; the
        // table's invariant does not depend on the panicking thread, so recover
        // rather than turn every later open into a panic.
        let mut held = OPEN_STORES.lock().unwrap_or_else(|e| e.into_inner());
        if held.contains(&key) {
            return None;
        }
        held.push(key);
        Some(ProcessClaim { key })
    }
}

impl Drop for ProcessClaim {
    fn drop(&mut self) {
        let mut held = OPEN_STORES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = held.iter().position(|k| *k == self.key) {
            held.swap_remove(i);
        }
    }
}

/// A non-blocking whole-file OFD record lock, `F_WRLCK` or `F_RDLCK`.
/// `Ok(false)` means another owner holds a conflicting lock; a real error
/// propagates.
fn try_ofd_lock(file: &File, exclusive: bool) -> Result<bool> {
    // SAFETY: `flock` is a plain C struct of integers; an all-zero value is a
    // valid one, and every field this call reads is set below.
    let mut fl: libc::flock = unsafe { std::mem::zeroed() };
    fl.l_type = if exclusive {
        libc::F_WRLCK
    } else {
        libc::F_RDLCK
    } as libc::c_short;
    fl.l_whence = libc::SEEK_SET as libc::c_short;
    fl.l_start = 0;
    // 0 means "to end of file, however the file grows" — Java's
    // `tryLock(0, Long.MAX_VALUE, shared)` covers the same whole-file range.
    fl.l_len = 0;
    loop {
        // SAFETY: `fd` is a live descriptor owned by `file` for the duration of
        // the call, and `fl` is a fully-initialised `flock` that outlives it.
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_OFD_SETLK, &fl) };
        if rc == 0 {
            return Ok(true);
        }
        let err = std::io::Error::last_os_error();
        match err.raw_os_error() {
            // The documented "another owner holds it" answers. (EAGAIN and
            // EWOULDBLOCK are the same value on Linux; both are named because
            // POSIX permits either.)
            Some(e) if e == libc::EACCES || e == libc::EAGAIN || e == libc::EWOULDBLOCK => {
                return Ok(false)
            }
            // An interrupted syscall says NOTHING about another owner. Reporting
            // it as contention would refuse an open that a signal happened to
            // land on; retry, which is what a non-blocking acquisition can
            // always safely do.
            Some(e) if e == libc::EINTR => continue,
            _ => return Err(DbError::Io(err)),
        }
    }
}

/// `access(dir, W_OK)` — the same probe Java's `Files.isWritable` makes on
/// Unix, and `std` cannot express it (`Permissions::readonly` reports the owner
/// mode bit, not whether a file may be created here).
///
/// Note what it does NOT prove: it answers for this process's real credentials
/// only, so a `false` is evidence of a read-only medium rather than proof that
/// no writer can appear. See the caller.
fn is_writable_dir(dir: &Path) -> bool {
    // An interior NUL cannot name a real directory, and truncating at it would
    // silently probe a DIFFERENT path. Answer "writable", the conservative side:
    // it leads to the fail-closed refusal rather than to a lockless open.
    let Ok(c) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else {
        return true;
    };
    // SAFETY: `c` is NUL-terminated and outlives the call.
    unsafe { libc::access(c.as_ptr(), libc::W_OK) == 0 }
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

    /// Every shape of table H's torn-create class, for a segment named `seq`:
    /// H1 empty, H2 short, H3 CRC mismatch, H4 CRC-valid wrong magic. Shared by
    /// the highest-name (residue) and below-the-highest (corruption) tests so
    /// the two cannot drift apart on which shapes they cover.
    fn torn_shapes(seq: i64) -> Vec<(String, Vec<u8>)> {
        vec![
            ("h1".to_string(), Vec::new()),
            ("h2".to_string(), vec![0u8; 16]),
            ("h3".to_string(), {
                let mut h = header_image(seq, 1);
                h[24] ^= 0x01; // firstLsn edited without resealing
                h
            }),
            ("h4".to_string(), {
                let mut h = header_image(seq, 1);
                h[0] = b'X';
                reseal(&mut h);
                h
            }),
        ]
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

    fn set_mode(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).expect("chmod");
    }

    /// The permission-dependent rungs of the lock ladder prove nothing when the
    /// test runs as root, which ignores the mode bits they turn on.
    fn running_as_root() -> bool {
        // SAFETY: `geteuid` takes no arguments and cannot fail.
        unsafe { libc::geteuid() == 0 }
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

        // A SYMLINK at an exact segment name, pointing at a valid segment: the
        // name matches, the file type does not. `DirEntry::file_type` answers
        // for the link itself, which is the whole reason it is used.
        std::os::unix::fs::symlink(
            seg_path(&base, 1),
            dir.join("store.db.wal.0000000000000003"),
        )
        .unwrap();

        let set = WalSegmentSet::open(&base, false).expect("open");
        assert_eq!(
            vec![1, 0x2a],
            seq_list(&set),
            "gaps are legal, order ascending"
        );
        // W6: nextSeq is one above the highest NAME, not the count.
        assert_eq!(0x2b, set.next_seq());
        assert!(
            dir.join("store.db.wal.0000000000000003").exists(),
            "an ignored entry is ignored, not removed"
        );
    }

    /// W6 has no successor to burn: the refusal is explicit, not a wrap into a
    /// negative name. Java pins the same case
    /// (`a_sequence_number_at_the_maximum_is_refused_rather_than_wrapping`).
    #[test]
    fn a_sequence_number_at_the_maximum_is_refused_rather_than_wrapping() {
        let dir = scratch("w6-max");
        let base = base_in(&dir);
        write_segment(&base, i64::MAX, &header_image(i64::MAX, 1));
        assert!(matches!(
            WalSegmentSet::open(&base, false),
            Err(DbError::StoreFull)
        ));
    }

    /// A base path is a byte string, not text: a namespace under a name that is
    /// not valid UTF-8 is still a namespace.
    #[test]
    fn a_base_path_that_is_not_utf8_still_enumerates() {
        use std::os::unix::ffi::OsStringExt;
        let dir = scratch("nonutf8");
        let base = dir.join(OsString::from_vec(vec![b'd', b'b', 0xff]));
        write_segment(&base, 1, &header_image(1, 1));
        write_segment(&base, 2, &header_image(2, 1));
        let set = WalSegmentSet::open(&base, false).expect("open");
        assert_eq!(vec![1, 2], seq_list(&set));
        assert_eq!(3, set.next_seq());
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

    /// N6's accepting side, which is the half a "does `<base>.wal` exist?"
    /// implementation gets wrong: only a REGULAR file is a v1 log. A directory
    /// at that name is not one, and neither is a symlink — the same NOFOLLOW
    /// discipline N4 applies to segment names.
    #[test]
    fn only_a_regular_file_at_the_v1_name_refuses_the_open() {
        let dir = scratch("n6-nonfile");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        let v1 = with_suffix(&base, ".wal");

        std::fs::create_dir(&v1).unwrap();
        let set = WalSegmentSet::open(&base, false).expect("a directory is not a v1 log");
        assert_eq!(vec![1], seq_list(&set));
        drop(set);
        std::fs::remove_dir(&v1).unwrap();

        std::os::unix::fs::symlink(seg_path(&base, 1), &v1).unwrap();
        let set = WalSegmentSet::open(&base, false).expect("a symlink is not a v1 log");
        assert_eq!(vec![1], seq_list(&set));
        drop(set);
        assert!(
            std::fs::symlink_metadata(&v1).is_ok(),
            "neither is removed by an open that accepted it"
        );
    }

    // ---------------------------------------------------------------- H: header table

    /// H1-H4 on the highest name are create-crash residue: a writable open
    /// unlinks them, and W6 has already burnt the sequence number so the fresh
    /// segment cannot reuse it.
    #[test]
    fn torn_create_residue_on_the_highest_name_is_removed() {
        for (tag, bytes) in torn_shapes(2) {
            let dir = scratch(&tag);
            let base = base_in(&dir);
            write_segment(&base, 1, &header_image(1, 1));
            write_segment(&base, 2, &bytes);

            let set = WalSegmentSet::open(&base, false).expect(&tag);
            assert_eq!(vec![1], seq_list(&set), "{tag}: residue is not in the set");
            assert!(!seg_path(&base, 2).exists(), "{tag}: residue is unlinked");
            assert_eq!(3, set.next_seq(), "{tag}: W6 burnt the residue's number");
        }
    }

    /// The same shapes anywhere below the highest name are corruption: a
    /// segment exists above them, so their create completed once. Every shape is
    /// tried, because the highest-only forgiveness is a property of the
    /// POSITION, and an implementation that special-cased one shape rather than
    /// the position would survive a single-shape test.
    #[test]
    fn torn_create_shapes_below_the_highest_name_are_corruption() {
        for (tag, bytes) in torn_shapes(1) {
            let dir = scratch(&format!("h-low-{tag}"));
            let base = base_in(&dir);
            write_segment(&base, 1, &bytes);
            write_segment(&base, 2, &header_image(2, 1));
            let msg = corrupt_msg(WalSegmentSet::open(&base, false));
            assert!(msg.contains("not the highest segment"), "{tag}: {msg}");
            assert!(
                seg_path(&base, 1).exists(),
                "{tag}: corruption deletes nothing"
            );
        }
    }

    /// The header is validated CRC FIRST, semantics second, and the order is
    /// load-bearing: an unsealed edit to a semantic field is a torn create (the
    /// bytes never became a header), while the SAME edit resealed is corruption.
    /// One image differing only in whether the CRC was recomputed.
    #[test]
    fn an_unsealed_semantic_edit_is_torn_and_the_resealed_one_is_corruption() {
        let edit = |h: &mut Vec<u8>| h[8..12].copy_from_slice(&2i32.to_be_bytes());

        let dir = scratch("h-order-torn");
        let base = base_in(&dir);
        let mut h = header_image(1, 1);
        edit(&mut h); // NOT resealed: headerCrc no longer matches
        write_segment(&base, 1, &h);
        let set = WalSegmentSet::open(&base, false).expect("torn create on the highest name");
        assert!(set.segments().is_empty());
        assert!(!seg_path(&base, 1).exists(), "residue is unlinked");

        let dir = scratch("h-order-sealed");
        let base = base_in(&dir);
        let mut h = header_image(1, 1);
        edit(&mut h);
        reseal(&mut h);
        write_segment(&base, 1, &h);
        let msg = corrupt_msg(WalSegmentSet::open(&base, false));
        assert!(msg.contains("unsupported WAL format version 2"), "{msg}");
        assert!(seg_path(&base, 1).exists(), "corruption deletes nothing");
    }

    /// H5-H7/H9: a CRC-valid header carrying wrong content is a writer defect or
    /// a copied file — corruption wherever it appears, INCLUDING the highest
    /// name, where a torn create would have been forgiven.
    #[test]
    fn resealed_semantic_faults_are_corruption_even_on_the_highest_name() {
        // (tag, the edit that makes the header semantically wrong, expected message)
        type Case = (&'static str, fn(&mut Vec<u8>), &'static str);
        let cases: [Case; 5] = [
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
                "h9-zero",
                |h| h[24..32].copy_from_slice(&0i64.to_be_bytes()),
                "is not a valid LSN",
            ),
            // A NEGATIVE start, which an `== 0` check would wave through.
            (
                "h9-negative",
                |h| h[24..32].copy_from_slice(&(-1i64).to_be_bytes()),
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
        for (tag, bytes) in torn_shapes(2) {
            let dir = scratch(&format!("ro-residue-{tag}"));
            let base = base_in(&dir);
            write_segment(&base, 1, &header_image(1, 1));
            write_segment(&base, 2, &bytes);
            let set = WalSegmentSet::open(&base, true).expect(&tag);
            assert_eq!(vec![1], seq_list(&set), "{tag}");
            assert!(
                seg_path(&base, 2).exists(),
                "{tag}: read-only mutates nothing"
            );
            assert_eq!(3, set.next_seq(), "{tag}: W6 still burns the name");
        }
    }

    /// The corrupt verdicts are read-only's too: a read-only open is not a
    /// lenient one, it merely declines to repair.
    #[test]
    fn a_read_only_open_reaches_the_same_corrupt_verdicts() {
        let dir = scratch("ro-corrupt");
        let base = base_in(&dir);
        let mut h = header_image(1, 1);
        h[24..32].copy_from_slice(&0i64.to_be_bytes());
        reseal(&mut h);
        write_segment(&base, 1, &h);
        assert!(is_corrupt(WalSegmentSet::open(&base, true)));
        assert!(seg_path(&base, 1).exists());

        // ...and the below-the-highest torn shape, which read-only must not
        // forgive merely because it cannot delete it.
        let dir = scratch("ro-corrupt-low");
        let base = base_in(&dir);
        write_segment(&base, 1, &[0u8; 16]);
        write_segment(&base, 2, &header_image(2, 1));
        assert!(is_corrupt(WalSegmentSet::open(&base, true)));
        assert!(seg_path(&base, 1).exists());
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

    /// A mark can name a sequence that no file ever carried (gaps are legal), and
    /// it retires everything at or below it — not "up to the next name".
    #[test]
    fn unlink_through_a_gap_retires_everything_below_it() {
        let dir = scratch("w5-gap");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        write_segment(&base, 3, &header_image(3, 1));
        let mut set = WalSegmentSet::open(&base, false).expect("open");
        set.unlink_through(2).expect("unlink");
        assert_eq!(vec![3], seq_list(&set));
        assert!(!seg_path(&base, 1).exists());
        assert!(seg_path(&base, 3).exists());
    }

    /// W5 with NOTHING to do performs no directory fsync — the shape a recovered
    /// mark takes when an earlier attempt already removed every file it
    /// authorizes. A "fsync anyway, it is harmless" implementation pays for it on
    /// every open of every cleaned store.
    #[test]
    fn unlink_through_below_the_lowest_name_does_not_fsync() {
        let dir = scratch("w5-nofsync");
        let base = base_in(&dir);
        write_segment(&base, 5, &header_image(5, 1));
        let mut set = WalSegmentSet::open(&base, false).expect("open");
        let before = set.dir_fsyncs();
        set.unlink_through(4).expect("no match");
        assert_eq!(before, set.dir_fsyncs(), "no match, no fsync");
        set.unlink_through(5).expect("match");
        assert_eq!(before + 1, set.dir_fsyncs(), "one fsync after the batch");
    }

    /// `sealed_bytes` is every segment BUT the growing one, so segments of
    /// unequal length are needed to tell it from any other subset — with equal
    /// lengths, "all but the highest" and "all but the lowest" agree.
    #[test]
    fn the_log_size_accounts_for_segments_of_unequal_length() {
        let dir = scratch("bytes");
        let base = base_in(&dir);
        // Valid headers with trailing bytes: at A0 a segment's tail is not yet
        // parsed, so the file length is whatever it is.
        for (seq, extra) in [(1i64, 0usize), (2, 100), (3, 7)] {
            let mut img = header_image(seq, 1);
            img.extend(std::iter::repeat_n(0xab, extra));
            write_segment(&base, seq, &img);
        }
        let mut set = WalSegmentSet::open(&base, false).expect("open");
        assert_eq!(3 * SEG_HDR + 107, set.log_bytes());
        assert_eq!(set.log_bytes_exact(), set.log_bytes());
        set.unlink_through(1).expect("unlink");
        assert_eq!(2 * SEG_HDR + 107, set.log_bytes());
        assert_eq!(set.log_bytes_exact(), set.log_bytes());
        set.unlink_through(3).expect("unlink the rest");
        assert_eq!(0, set.log_bytes());
        assert_eq!(0, set.log_bytes_exact());
    }

    /// W2's durability points are not observable in the resulting bytes: a
    /// create that skipped both syncs writes the same 36 bytes. Count them.
    #[test]
    fn a_create_forces_the_segment_and_then_the_directory() {
        let dir = scratch("w2-fsync");
        let base = base_in(&dir);
        let mut set = WalSegmentSet::open(&base, false).expect("open");
        assert_eq!(0, set.segment_syncs());
        let dir_before = set.dir_fsyncs();
        set.create_segment(1).expect("create");
        assert_eq!(
            1,
            set.segment_syncs(),
            "the segment is forced WITH its size"
        );
        assert_eq!(
            dir_before + 1,
            set.dir_fsyncs(),
            "and the directory entry after it"
        );
    }

    /// A fresh namespace hands the caller an empty set and the first name; N1's
    /// create is the caller's to make (`StoreWAL`, slice A2).
    #[test]
    fn a_fresh_namespace_starts_empty_at_sequence_one() {
        let dir = scratch("n1");
        let base = base_in(&dir);
        let mut set = WalSegmentSet::open(&base, false).expect("open");
        assert!(set.segments().is_empty());
        assert_eq!(FIRST_SEQ, set.next_seq());
        assert_eq!(0, set.log_bytes());
        let seg = set.create_segment(1).expect("create");
        assert_eq!(FIRST_SEQ, seg.seq);
        assert_eq!(1, seg.header_first_lsn());
    }

    /// A create needs a real LSN; both refusals are caller errors, not damage.
    #[test]
    fn a_segment_cannot_start_at_a_non_positive_lsn() {
        let dir = scratch("w2-lsn");
        let base = base_in(&dir);
        let mut set = WalSegmentSet::open(&base, false).expect("open");
        for bad in [0i64, -1, i64::MIN] {
            assert!(
                matches!(set.create_segment(bad), Err(DbError::WrongConfiguration(_))),
                "firstLsn {bad} must be refused"
            );
        }
        assert_eq!(FIRST_SEQ, set.next_seq(), "a refused create burns no name");
    }

    // ---------------------------------------------------------------- descriptors

    /// The descriptor discipline A1's two-pass scanner depends on: classification
    /// retains no handle, a pass opens one on demand and gives it back, and the
    /// handle honours the set's read-only mode.
    #[test]
    fn segments_open_and_release_their_handles_on_demand() {
        if running_as_root() {
            return; // root can open a 0444 file read-write
        }
        let dir = scratch("fds");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        write_segment(&base, 2, &header_image(2, 1));
        set_mode(&seg_path(&base, 1), 0o444);

        let mut set = WalSegmentSet::open(&base, true).expect("open");
        assert_eq!(0, set.open_file_count(), "classification retains nothing");

        let seg = &mut set.segments_mut()[0];
        assert!(seg.file().is_none());
        // A read-only set opens read-only, which is the only way this succeeds.
        seg.ensure_open().expect("open the handle");
        assert!(seg.file().is_some());
        seg.ensure_open().expect("idempotent");
        assert_eq!(1, set.open_file_count());

        set.segments_mut()[0].release();
        assert_eq!(0, set.open_file_count());
        assert!(set.segments()[0].file().is_none());
        // Reopens after a release, as a second recovery pass does.
        set.segments_mut()[0].ensure_open().expect("reopen");
        assert_eq!(1, set.open_file_count());
        drop(set);
        set_mode(&seg_path(&base, 1), 0o644);
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

    /// A second open in THIS process is refused whatever the two modes are —
    /// including read-only against read-only, which no kernel lock refuses and
    /// which the process claim exists to catch.
    ///
    /// This is Java's rule, not an invention: `tryLock` consults a JVM-wide
    /// table that does not consider lock mode, so a second `StoreWAL` on one
    /// store raises `OverlappingFileLockException` even when both opens are
    /// read-only (`WalSegmentSet.java:267-274`). Verified directly against the
    /// JVM — two `tryLock(0, MAX, true)` calls on separate channels of one file
    /// refuse the second — rather than inferred from the javadoc.
    #[test]
    fn a_second_open_in_this_process_is_refused_whatever_the_modes() {
        for (tag, first_ro, second_ro) in [
            ("rw-rw", false, false),
            ("rw-ro", false, true),
            ("ro-rw", true, false),
            ("ro-ro", true, true),
        ] {
            let dir = scratch(tag);
            let base = base_in(&dir);
            let first = WalSegmentSet::open(&base, first_ro).expect(tag);
            assert!(
                matches!(
                    WalSegmentSet::open(&base, second_ro),
                    Err(DbError::Locked(_))
                ),
                "{tag}: the second open must be refused"
            );
            drop(first);
            WalSegmentSet::open(&base, second_ro).expect("after release");
        }
    }

    /// The claim is keyed by the lock file's identity, not by the pathname, so
    /// two spellings of one store still collide.
    #[test]
    fn the_same_store_reached_by_two_paths_is_one_store() {
        let dir = scratch("lock-path");
        let base = base_in(&dir);
        let first = WalSegmentSet::open(&base, false).expect("first");
        let same = dir.join(".").join("store.db");
        assert!(matches!(
            WalSegmentSet::open(&same, false),
            Err(DbError::Locked(_))
        ));
        drop(first);
    }

    /// A refused open must not disturb the holder: the classic self-unlock
    /// hazard of POSIX record locks is that the loser's failed attempt (or its
    /// close of the same file) drops the winner's lock. A third open proves the
    /// first is still held.
    #[test]
    fn a_refused_open_does_not_release_the_holders_lock() {
        let dir = scratch("lock-selfunlock");
        let base = base_in(&dir);
        let first = WalSegmentSet::open(&base, false).expect("first");
        assert!(matches!(
            WalSegmentSet::open(&base, false),
            Err(DbError::Locked(_))
        ));
        assert!(
            matches!(WalSegmentSet::open(&base, false), Err(DbError::Locked(_))),
            "the first open still holds the store"
        );
        drop(first);
        WalSegmentSet::open(&base, false).expect("after release");
    }

    /// `close()` releases the namespace — both halves — without waiting for the
    /// value to be dropped, which is the release point D2's cleanup observes.
    #[test]
    fn close_releases_the_lock_and_the_claim() {
        let dir = scratch("lock-close");
        let base = base_in(&dir);
        let mut first = WalSegmentSet::open(&base, false).expect("first");
        first.close();
        let second = WalSegmentSet::open(&base, false).expect("after close");
        drop(second);
        drop(first);
    }

    /// The KERNEL half, exercised directly because the process claim refuses
    /// every in-process pair before the kernel ever sees it. OFD locks are owned
    /// by the open file description: two read locks share, a write lock is
    /// excluded by a read lock, and closing one description does not release
    /// another's lock (the defect that makes plain `F_SETLK` unusable here).
    #[test]
    fn ofd_locks_are_owned_by_the_open_file_description() {
        let dir = scratch("ofd");
        let path = dir.join("l");
        let open = || {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .unwrap()
        };
        let a = open();
        let b = open();
        assert!(try_ofd_lock(&a, false).unwrap(), "first read lock");
        assert!(try_ofd_lock(&b, false).unwrap(), "read locks share");
        let w = open();
        assert!(!try_ofd_lock(&w, true).unwrap(), "a writer is excluded");
        drop(b);
        assert!(
            !try_ofd_lock(&w, true).unwrap(),
            "closing another description must not release a's lock"
        );
        drop(a);
        assert!(
            try_ofd_lock(&w, true).unwrap(),
            "released by the last owner"
        );
        // ...and an exclusive lock excludes a reader in the other direction.
        let r = open();
        assert!(!try_ofd_lock(&r, false).unwrap());
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

    /// Rung 2 of the read-only ladder: the read-write create fails, but the lock
    /// file is THERE, so a shared lock is still attainable on a read-only
    /// handle. This is not a fallback to lockless at all.
    #[test]
    fn a_read_only_open_falls_back_to_a_read_only_lock_handle() {
        if running_as_root() {
            return; // root ignores the mode bits this rung turns on
        }
        let dir = scratch("lock-rung2");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        let lock = with_suffix(&base, ".lock");
        std::fs::write(&lock, b"").unwrap();
        set_mode(&lock, 0o444);
        let set = WalSegmentSet::open(&base, true).expect("read-only handle rung");
        assert_eq!(vec![1], seq_list(&set));
        drop(set);
        set_mode(&lock, 0o644);
    }

    /// Rung 3: no lock file and a directory this process cannot write. Java's
    /// read-only-medium branch — the reader is admitted with no lock, and a
    /// writable open cannot get in at all.
    #[test]
    fn a_read_only_medium_admits_a_lockless_reader_and_no_writer() {
        if running_as_root() {
            return;
        }
        let dir = scratch("lock-rung3");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        set_mode(&dir, 0o555);
        let set = WalSegmentSet::open(&base, true).expect("lockless reader");
        assert_eq!(vec![1], seq_list(&set));
        assert!(
            !with_suffix(&base, ".lock").exists(),
            "no lock file was created"
        );
        assert!(
            WalSegmentSet::open(&base, false).is_err(),
            "a writer cannot take the lock it needs"
        );
        drop(set);
        set_mode(&dir, 0o755);
    }

    /// Rung 4: the create failed, the file does not exist, and the directory IS
    /// writable — so a writer may be running. Inconclusive fails CLOSED.
    #[test]
    fn an_inconclusive_read_only_lock_refuses_rather_than_going_lockless() {
        let dir = scratch("lock-rung4");
        // A name whose `.lock` sibling exceeds NAME_MAX: the create fails, the
        // path cannot exist, and the directory is plainly writable.
        let base = dir.join("x".repeat(252));
        match WalSegmentSet::open(&base, true) {
            Err(DbError::Locked(_)) => {}
            Err(e) => panic!("inconclusive must fail closed, got {e}"),
            Ok(_) => panic!("inconclusive must not open locklessly"),
        }
    }

    // ---------------------------------------------------------------- closed state

    /// After `close()` the set no longer holds the lock, so the operations that
    /// need it refuse rather than mutating an unowned namespace.
    #[test]
    fn a_closed_set_refuses_to_mutate_the_namespace() {
        let dir = scratch("closed");
        let base = base_in(&dir);
        write_segment(&base, 1, &header_image(1, 1));
        let mut set = WalSegmentSet::open(&base, false).expect("open");
        set.close();
        assert!(matches!(set.create_segment(1), Err(DbError::StoreClosed)));
        assert!(matches!(set.unlink_through(1), Err(DbError::StoreClosed)));
        assert!(seg_path(&base, 1).exists(), "nothing was removed");
    }
}
