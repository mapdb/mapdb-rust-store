//! `StoreWAL` — transactional store: an in-memory [`StoreDirect`] volume plus a
//! write-ahead log file (spec 02 §7, Java `StoreWAL`).
//!
//! Uncommitted mutations are staged in memory; [`StoreWAL::commit`] serializes
//! them as one WAL section, fsyncs (the durability point), then applies them to
//! the inner (memory-backed) StoreDirect. Recovery replays all committed
//! sections from the start of the file.
//!
//! # On-disk format v1
//!
//! This is **this implementation's** format v1, as ported. It is not a shared
//! contract: the Java engine has since moved to a segmented format v3, and this
//! port will refuse to open one (`unsupported WAL format version`). See
//! `README.md` — the on-disk format is not stabilised and no cross-engine
//! compatibility is claimed.
//! ```text
//! file       := fileHeader section*
//! fileHeader := magic "MDBS.WAL" (8) | version i32=1 | flags i32=0        (16 B)
//! section    := tag u8 ('S' commit, 'C' checkpoint)
//!             | lsn i64 (strictly increasing)
//!             | bodyLen i64
//!             | hdrCrc i32 = CRC32(tag ++ lsn ++ bodyLen)
//!             | bodyCrc i32 = CRC32(body)
//!             | body: entries T_PREALLOC/T_RECORD/T_APPEND/T_DELETE (packLong framing)
//! ```
//! CRCs are validated BEFORE any entry is decoded (garbage never allocates);
//! replay is entry-by-entry in O(1) memory; a damaged section FOLLOWED by a
//! valid one is distinguishable from a torn tail — mid-log corruption raises
//! `DataCorruption` while a bad section at EOF is truncated (decision D4).
//!
//! Checkpointing rewrites the log as one snapshot section of the inner store's
//! committed state, written to `<file>.ckpt`, fsynced, then atomically renamed
//! over the log (the rename is the commit point).
//!
//! Background maintenance (R6 executor) is not implemented; the synchronous
//! inline auto-checkpoint in [`commit`](StoreWAL::commit) is the correctness
//! fallback (P7: a disabled executor never affects correctness).

use crate::error::{DbError, Result};
use crate::io::{DataOutput2, SliceInput};
use crate::ser::Serializer;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::num::NonZeroU64;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::direct::{STATE_LIVE, STATE_VOID};
use super::index_val as iv;
use super::lease::LeaseTable;
use super::{AppendResult, Recid, Record, RecordRead, Store, StoreDelta, StoreDirect, StoreTx};
use parking_lot::RwLock;
use std::sync::Arc;

const T_PREALLOC: u8 = 1;
const T_RECORD: u8 = 2;
const T_APPEND: u8 = 3;
const T_DELETE: u8 = 4;
/// Legacy (headerless format) trailing seal tag; v1 sections are length-prefixed.
const T_COMMIT: u8 = 8;

const MAGIC: [u8; 8] = *b"MDBS.WAL";
const FORMAT_VERSION: i32 = 1;
/// File header: magic(8) + version(4) + flags(4).
const FILE_HDR: u64 = 16;
/// Section header: tag(1) + lsn(8) + bodyLen(8) + hdrCrc(4) + bodyCrc(4).
const SEC_HDR: usize = 25;
/// Bytes of the section header covered by hdrCrc (tag + lsn + bodyLen).
const SEC_HDR_CRC_LEN: usize = 17;
const TAG_SECTION: u8 = b'S';
const TAG_CKPT: u8 = b'C';

/// Default streaming-replay window (bytes); ctor override forces refill edges in tests.
const DEFAULT_REPLAY_BUF: usize = 1 << 20;
/// Default log size past which `commit()` triggers an automatic checkpoint.
pub const DEFAULT_AUTO_CHECKPOINT_BYTES: i64 = 1 << 30;

#[inline]
fn crc32(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}

#[inline]
fn nz(recid: u64) -> Recid {
    NonZeroU64::new(recid).expect("recid 0 is never allocated")
}

/// Fallible recid conversion for decode paths: a CRC-valid but semantically
/// invalid entry carrying recid 0 (reserved) must return `DataCorruption`, not
/// panic in `nz`.
#[inline]
fn nz_res(recid: u64) -> WalRes<Recid> {
    NonZeroU64::new(recid).ok_or(WalStop::Fatal(DbError::corrupt(
        "WAL entry references reserved recid 0",
    )))
}

/// Per-recid staged mutation set (uncommitted). Content == (base or inner) ++ appends.
#[derive(Default)]
struct Staged {
    created: bool,
    base_set: bool,
    /// `None` with `base_set == true` means explicit null content.
    base: Option<Vec<u8>>,
    headroom: usize,
    deleted: bool,
    appends: Vec<Vec<u8>>,
    appends_len: usize,
}

impl Staged {
    fn new(created: bool) -> Staged {
        Staged {
            created,
            ..Default::default()
        }
    }
}

/// Classified commit operation, computed before any apply (state must not shift mid-apply).
struct WalOp {
    /// One of T_*, or 0 for "created+deleted: apply-only cleanup, not logged".
    op: u8,
    recid: u64,
    cap: usize,
    data: Option<Vec<u8>>,
}

// ---------- streaming replay control flow (mirrors Java's TornTail exception) ----------

/// A replay step either succeeds, hits a torn tail (stop at the last valid
/// commit — availability), or hits a fatal condition (mid-log corruption / IO).
enum WalStop {
    Torn,
    Fatal(DbError),
}
type WalRes<T> = std::result::Result<T, WalStop>;

impl From<DbError> for WalStop {
    fn from(e: DbError) -> Self {
        WalStop::Fatal(e)
    }
}

fn io_stop(e: std::io::Error) -> WalStop {
    if e.kind() == ErrorKind::UnexpectedEof {
        WalStop::Torn
    } else {
        WalStop::Fatal(DbError::Io(e))
    }
}

/// Positioned full read; a short read (file shorter than claimed) is a torn tail.
fn read_at(file: &File, buf: &mut [u8], pos: u64) -> WalRes<()> {
    file.read_exact_at(buf, pos).map_err(io_stop)
}

/// CRC32 over the body range `[start, end)`, streamed through a bounded buffer.
fn body_crc(file: &File, start: u64, end: u64, bufsize: usize) -> WalRes<u32> {
    let mut crc = crc32fast::Hasher::new();
    if start < end {
        let cap = ((end - start) as usize).min(bufsize.max(16));
        let mut buf = vec![0u8; cap];
        let mut p = start;
        while p < end {
            let n = ((end - p) as usize).min(buf.len());
            read_at(file, &mut buf[..n], p)?;
            crc.update(&buf[..n]);
            p += n as u64;
        }
    }
    Ok(crc.finalize())
}

/// Streaming WAL decoder: a fixed-size window over the file with u64 positions,
/// bounded by `[start, limit)`, plus an incremental CRC32 (used by the legacy
/// trailing-seal format only). Never materializes the log, so 2 GiB+ files replay.
struct WalIn<'a> {
    file: &'a File,
    limit: u64,
    win: Vec<u8>,
    win_start: u64,
    win_pos: usize,
    win_len: usize,
    crc: crc32fast::Hasher,
}

impl<'a> WalIn<'a> {
    fn new(file: &'a File, bufsize: usize) -> WalIn<'a> {
        WalIn {
            file,
            limit: 0,
            win: vec![0u8; bufsize.max(16)],
            win_start: 0,
            win_pos: 0,
            win_len: 0,
            crc: crc32fast::Hasher::new(),
        }
    }

    fn reset(&mut self, start: u64, end: u64) {
        self.win_start = start;
        self.limit = end;
        self.win_pos = 0;
        self.win_len = 0;
        self.crc = crc32fast::Hasher::new();
    }

    #[inline]
    fn pos(&self) -> u64 {
        self.win_start + self.win_pos as u64
    }

    #[inline]
    fn remaining(&self) -> u64 {
        self.limit - self.pos()
    }

    fn refill(&mut self) -> WalRes<()> {
        self.win_start = self.pos();
        self.win_pos = 0;
        if self.win_start >= self.limit {
            return Err(WalStop::Torn);
        }
        let n = ((self.limit - self.win_start) as usize).min(self.win.len());
        read_at(self.file, &mut self.win[..n], self.win_start)?;
        self.win_len = n;
        Ok(())
    }

    /// Unsigned byte, NOT folded into the CRC (callers fold via `crc_tag`).
    fn read_byte_raw(&mut self) -> WalRes<u8> {
        if self.win_pos >= self.win_len {
            self.refill()?;
        }
        let b = self.win[self.win_pos];
        self.win_pos += 1;
        Ok(b)
    }

    fn crc_tag(&mut self, tag: u8) {
        self.crc.update(&[tag]);
    }

    /// Packed long, CRC'd, capped at 10 bytes (over-long run = corruption).
    fn unpack_long(&mut self) -> WalRes<u64> {
        let mut ret: u64 = 0;
        for _ in 0..10 {
            let v = self.read_byte_raw()?;
            self.crc.update(&[v]);
            ret = (ret << 7) | (v & 0x7F) as u64;
            if v & 0x80 != 0 {
                return Ok(ret);
            }
        }
        Err(WalStop::Fatal(DbError::corrupt("WAL packed long too long")))
    }

    /// Payload bytes, CRC'd.
    fn read_fully(&mut self, dst: &mut [u8]) -> WalRes<()> {
        let mut off = 0;
        while off < dst.len() {
            if self.win_pos >= self.win_len {
                self.refill()?;
            }
            let n = (self.win_len - self.win_pos).min(dst.len() - off);
            dst[off..off + n].copy_from_slice(&self.win[self.win_pos..self.win_pos + n]);
            self.win_pos += n;
            off += n;
        }
        self.crc.update(dst);
        Ok(())
    }

    /// Big-endian i32, NOT CRC'd (the stored section CRC itself).
    fn read_int_raw(&mut self) -> WalRes<i32> {
        let mut r: i32 = 0;
        for _ in 0..4 {
            r = (r << 8) | self.read_byte_raw()? as i32;
        }
        Ok(r)
    }

    fn crc_value(&self) -> u32 {
        self.crc.clone().finalize()
    }

    fn crc_reset(&mut self) {
        self.crc = crc32fast::Hasher::new();
    }
}

/// Capacity as the writer encodes it: 0 for null content, else 16-aligned, big
/// enough for header+content, within the plain-record limit — EXCEPT oversize
/// (linked) records, which the writer encodes with capacity 0.
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

/// The lock-guarded mutable state (Java's single ReadWriteLock covers all of it).
struct WalState {
    inner: StoreDirect,
    file: File,
    staged: HashMap<u64, Staged>,
    next_lsn: i64,
    checkpoint_basis: u64,
    auto_checkpoint_bytes: i64,
    /// Append position in the log (Java tracks this as `ch.position()`).
    log_pos: u64,
    replay_buf: usize,
    /// Set when a durability-path step (e.g. the post-rename directory fsync)
    /// failed after its visible effect: the store must not report any later
    /// commit/checkpoint durable until reopened. Guarded by the state lock.
    poisoned: bool,
}

/// Transactional store (spec 02 §7).
pub struct StoreWAL {
    st: RwLock<WalState>,
    path: PathBuf,
    lease_table: Arc<LeaseTable>,
    closed: AtomicBool,
    /// Bumped on every `rollback` so open collections know their append-only
    /// structural caches (e.g. the btree left-edge spine) may have been reverted
    /// to a shorter tree and must be rebuilt before the next structural op.
    struct_gen: AtomicU64,
}

impl StoreWAL {
    pub fn open(path: &Path) -> Result<StoreWAL> {
        Self::open_with(path, true, DEFAULT_REPLAY_BUF)
    }

    pub fn open_ts(path: &Path, thread_safe: bool) -> Result<StoreWAL> {
        Self::open_with(path, thread_safe, DEFAULT_REPLAY_BUF)
    }

    /// `replay_buf` is a test hook: a tiny window forces refill edges in streaming replay.
    pub fn open_with(path: &Path, thread_safe: bool, replay_buf: usize) -> Result<StoreWAL> {
        let inner = StoreDirect::new_heap_ts(thread_safe)?;
        let tmp = ckpt_tmp(path);
        let created = !path.exists();

        // crash-during-checkpoint recovery: a complete temp snapshot wins.
        if created && tmp.exists() {
            if let Some(state) = try_recover_from_ckpt_temp(path, &tmp, thread_safe, replay_buf)? {
                return Ok(Self::wrap(path, state));
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        let mut state = WalState {
            inner,
            file,
            staged: HashMap::new(),
            next_lsn: 1,
            checkpoint_basis: 0,
            auto_checkpoint_bytes: DEFAULT_AUTO_CHECKPOINT_BYTES,
            log_pos: FILE_HDR,
            replay_buf,
            poisoned: false,
        };
        state.recover_opened_channel(created, path)?;
        Ok(Self::wrap(path, state))
    }

    fn wrap(path: &Path, state: WalState) -> StoreWAL {
        StoreWAL {
            st: RwLock::new(state),
            path: path.to_path_buf(),
            lease_table: LeaseTable::new(),
            closed: AtomicBool::new(false),
            struct_gen: AtomicU64::new(0),
        }
    }

    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(DbError::StoreClosed)
        } else {
            Ok(())
        }
    }

    /// Take the write guard and re-check `closed` while holding it. `close`
    /// publishes `closed` under this same lock, so every write path that goes
    /// through here is linearized with close — no staged mutation (or durable
    /// append) can slip in after `close()` completed.
    fn write_open(&self) -> Result<parking_lot::RwLockWriteGuard<'_, WalState>> {
        let st = self.st.write();
        if self.closed.load(Ordering::Acquire) {
            return Err(DbError::StoreClosed);
        }
        Ok(st)
    }

    /// Force a log-compacting checkpoint (also exposed as `compact`).
    pub fn checkpoint(&self) -> Result<()> {
        let mut st = self.write_open()?;
        st.checkpoint_locked(&self.path)
    }

    pub fn set_auto_checkpoint_bytes(&self, bytes: i64) -> Result<()> {
        self.check_closed()?;
        self.st.write().auto_checkpoint_bytes = bytes;
        Ok(())
    }
}

fn ckpt_tmp(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".ckpt");
    PathBuf::from(s)
}

fn write_file_header(file: &File) -> Result<()> {
    let mut h = [0u8; FILE_HDR as usize];
    h[..8].copy_from_slice(&MAGIC);
    h[8..12].copy_from_slice(&FORMAT_VERSION.to_be_bytes());
    h[12..16].copy_from_slice(&0i32.to_be_bytes());
    file.write_all_at(&h, 0)?;
    Ok(())
}

/// True when the file carries the v1 magic; rejects unknown future versions
/// and nonzero header flags.
fn is_v1(file: &File, size: u64) -> Result<bool> {
    if size < FILE_HDR {
        return Ok(false);
    }
    let mut h = [0u8; FILE_HDR as usize];
    file.read_exact_at(&mut h, 0)?;
    if h[..8] != MAGIC {
        return Ok(false);
    }
    let version = i32::from_be_bytes([h[8], h[9], h[10], h[11]]);
    if version != FORMAT_VERSION {
        return Err(DbError::corrupt_msg(format!(
            "unsupported WAL format version {version}"
        )));
    }
    // v1 declares flags == 0 (bytes 12..16); a nonzero word marks an
    // incompatible variant this reader does not understand. This must be an
    // EXPLICIT corruption error raised before any replay/truncation/legacy
    // fallthrough — returning `false` here would route a current-magic file
    // into the framed-MDB guard with a misleading error path. Strictness
    // parity with Java v3, which rejects nonzero segment flags.
    let flags = i32::from_be_bytes([h[12], h[13], h[14], h[15]]);
    if flags != 0 {
        return Err(DbError::corrupt_msg(format!(
            "unsupported WAL header flags {flags:#x}"
        )));
    }
    Ok(true)
}

/// A framed MapDB-family header must never be reinterpreted as the legacy
/// headerless WAL. In particular, this makes a hard magic swap reject old v1
/// files instead of treating their first byte as a torn legacy instruction and
/// destructively migrating an empty prefix.
fn has_framed_magic_prefix(file: &File, size: u64) -> Result<bool> {
    if size < 3 {
        return Ok(false);
    }
    let mut prefix = [0u8; 3];
    file.read_exact_at(&mut prefix, 0)?;
    Ok(prefix == *b"MDB")
}

/// fsync the directory so a create/rename of `path` is itself durable. This is
/// on the durability path (initial WAL creation, checkpoint promotion), so its
/// failure MUST propagate — a swallowed error would report a commit/checkpoint
/// durable while its namespace change is not.
fn fsync_dir(path: &Path) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let dir = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    File::open(dir)?.sync_all()?;
    Ok(())
}

/// Recover a store directly from a complete `<file>.ckpt` snapshot (crash after
/// the snapshot was fsynced but before the atomic rename). Returns `None` if the
/// temp is absent; `Err` if present but not a complete v1 snapshot.
fn try_recover_from_ckpt_temp(
    path: &Path,
    tmp: &Path,
    thread_safe: bool,
    replay_buf: usize,
) -> Result<Option<WalState>> {
    if !tmp.exists() {
        return Ok(None);
    }
    let inner = StoreDirect::new_heap_ts(thread_safe)?;
    let lsn;
    let size;
    {
        let tmp_file = OpenOptions::new().read(true).write(true).open(tmp)?;
        size = tmp_file.metadata()?.len();
        if !is_v1(&tmp_file, size)? {
            return Err(DbError::corrupt("checkpoint temp is not a v1 WAL snapshot"));
        }
        if size < FILE_HDR + SEC_HDR as u64 {
            return Err(DbError::corrupt(
                "checkpoint temp is missing its snapshot section",
            ));
        }
        let mut hdr = [0u8; SEC_HDR];
        tmp_file.read_exact_at(&mut hdr, FILE_HDR)?;
        let (tag, section_lsn, body_len, stored_hdr_crc, stored_body_crc) = parse_sec_hdr(&hdr);
        lsn = section_lsn;
        let body_start = FILE_HDR + SEC_HDR as u64;
        let hdr_ok = crc32(&hdr[..SEC_HDR_CRC_LEN]) as i32 == stored_hdr_crc;
        if tag != TAG_CKPT
            || !hdr_ok
            // LSN must be positive with an available successor: promoting a
            // nonpositive LSN yields next_lsn <= 0, whose next commit writes a
            // section that normal replay rejects as non-increasing (unreopenable).
            || section_lsn <= 0
            || section_lsn == i64::MAX
            || body_len < 0
            || body_start + body_len as u64 != size
            || body_crc(&tmp_file, body_start, size, replay_buf).map_err(fatal_only)?
                != stored_body_crc as u32
        {
            return Err(DbError::corrupt(
                "checkpoint temp is not a complete snapshot",
            ));
        }
        let mut win = WalIn::new(&tmp_file, replay_buf);
        apply_section(&inner, &mut win, body_start, size).map_err(fatal_only)?;
        inner.rebuild_free_recids()?;
    }
    // promote temp → log (atomic), then reopen the log.
    std::fs::rename(tmp, path)?;
    fsync_dir(path)?;
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    let log_pos = file.metadata()?.len();
    Ok(Some(WalState {
        inner,
        file,
        staged: HashMap::new(),
        next_lsn: lsn
            .checked_add(1)
            .ok_or_else(|| DbError::corrupt("WAL LSN space exhausted"))?,
        checkpoint_basis: size,
        auto_checkpoint_bytes: DEFAULT_AUTO_CHECKPOINT_BYTES,
        log_pos,
        replay_buf,
        poisoned: false,
    }))
}

/// A torn tail during a context that requires completeness is itself corruption.
fn fatal_only(stop: WalStop) -> DbError {
    match stop {
        WalStop::Fatal(e) => e,
        WalStop::Torn => DbError::corrupt("WAL snapshot truncated"),
    }
}

fn parse_sec_hdr(hdr: &[u8; SEC_HDR]) -> (u8, i64, i64, i32, i32) {
    let tag = hdr[0];
    let lsn = i64::from_be_bytes(hdr[1..9].try_into().unwrap());
    let body_len = i64::from_be_bytes(hdr[9..17].try_into().unwrap());
    let hdr_crc = i32::from_be_bytes(hdr[17..21].try_into().unwrap());
    let body_crc = i32::from_be_bytes(hdr[21..25].try_into().unwrap());
    (tag, lsn, body_len, hdr_crc, body_crc)
}

/// Decode+apply one CRC-verified section body into `inner` (O(1) memory).
fn apply_section(inner: &StoreDirect, win: &mut WalIn, start: u64, end: u64) -> WalRes<()> {
    win.reset(start, end);
    while win.pos() < end {
        let ty = win.read_byte_raw()?;
        match ty {
            x if x == T_PREALLOC => {
                inner.wal_prealloc(nz_res(win.unpack_long()?)?.get())?;
            }
            x if x == T_DELETE => {
                inner.delete(nz_res(win.unpack_long()?)?)?;
            }
            x if x == T_RECORD => {
                let recid = nz_res(win.unpack_long()?)?.get();
                let cap = win.unpack_long()?;
                let len_plus = win.unpack_long()?;
                let mut data: Option<Vec<u8>> = None;
                if len_plus != 0 {
                    let len = len_plus - 1;
                    if len > i32::MAX as u64 || len > win.remaining() {
                        return Err(WalStop::Fatal(DbError::corrupt_msg(format!(
                            "bad WAL record length {len}"
                        ))));
                    }
                    let mut b = vec![0u8; len as usize];
                    win.read_fully(&mut b)?;
                    data = Some(b);
                }
                if !cap_valid(cap, data.as_deref()) {
                    return Err(WalStop::Fatal(DbError::corrupt_msg(format!(
                        "bad WAL record capacity {cap}"
                    ))));
                }
                inner.wal_put(recid, cap as usize, data.as_deref())?;
            }
            x if x == T_APPEND => {
                let recid = nz_res(win.unpack_long()?)?;
                let len = win.unpack_long()?;
                if len > i32::MAX as u64 || len > win.remaining() {
                    return Err(WalStop::Fatal(DbError::corrupt_msg(format!(
                        "bad WAL append length {len}"
                    ))));
                }
                let mut b = vec![0u8; len as usize];
                win.read_fully(&mut b)?;
                if inner.append(recid, &b)? == AppendResult::Refused {
                    return Err(WalStop::Fatal(DbError::corrupt_msg(format!(
                        "WAL append refused, recid={recid}"
                    ))));
                }
            }
            other => {
                return Err(WalStop::Fatal(DbError::corrupt_msg(format!(
                    "bad WAL entry tag {other}"
                ))));
            }
        }
    }
    Ok(())
}

impl WalState {
    fn recover_opened_channel(&mut self, created: bool, path: &Path) -> Result<()> {
        let size = self.file.metadata()?.len();
        let mut legacy = false;
        let valid_end = if size == 0 {
            write_file_header(&self.file)?;
            self.file.sync_all()?;
            if created {
                fsync_dir(path)?;
            }
            FILE_HDR
        } else if is_v1(&self.file, size)? {
            self.replay_v1(size)?
        } else if has_framed_magic_prefix(&self.file, size)? {
            return Err(DbError::corrupt("unsupported WAL magic"));
        } else {
            legacy = true;
            self.replay_legacy(size)?
        };
        self.file.set_len(valid_end)?;
        self.log_pos = valid_end;
        // replay of delete-then-reuse histories leaves stale free-list entries:
        // rebuild the allocator's free list from the final index.
        self.inner.rebuild_free_recids()?;
        if legacy {
            self.checkpoint_locked(path)?; // migrate to v1
        }
        Ok(())
    }

    /// Scans v1 sections; applies each CRC-valid one; returns the end offset of
    /// the last valid section. Torn tail truncates; a CRC-failing section
    /// FOLLOWED by a valid one is mid-log corruption and errors.
    fn replay_v1(&mut self, size: u64) -> Result<u64> {
        let mut win = WalIn::new(&self.file, self.replay_buf);
        let mut pos = FILE_HDR;
        let mut last_lsn: i64 = 0;
        while pos + SEC_HDR as u64 <= size {
            let step = (|| -> WalRes<Option<(i64, u64)>> {
                let mut hdr = [0u8; SEC_HDR];
                read_at(&self.file, &mut hdr, pos)?;
                let (tag, lsn, body_len, stored_hdr_crc, stored_body_crc) = parse_sec_hdr(&hdr);
                let body_start = pos + SEC_HDR as u64;
                let hdr_ok = crc32(&hdr[..SEC_HDR_CRC_LEN]) as i32 == stored_hdr_crc
                    && (tag == TAG_SECTION || tag == TAG_CKPT);
                if !hdr_ok {
                    // header torn/rotted: bodyLen untrusted — `None` signals "suspect".
                    return Ok(None);
                }
                if body_len < 0 || body_len as u64 > size - body_start {
                    // verified header, body past EOF: torn tail by construction.
                    return Err(WalStop::Torn);
                }
                let body_end = body_start + body_len as u64;
                if body_crc(&self.file, body_start, body_end, self.replay_buf)?
                    != stored_body_crc as u32
                {
                    // bodyEnd TRUSTED (hdrCrc valid): anything valid after it = bit rot.
                    if any_valid_section_from(
                        &self.file,
                        body_end,
                        size,
                        last_lsn,
                        false,
                        self.replay_buf,
                    )? {
                        return Err(WalStop::Fatal(DbError::corrupt_msg(format!(
                            "WAL mid-log corruption: section body CRC mismatch at offset {pos} but valid sections follow (not a torn tail)"
                        ))));
                    }
                    return Err(WalStop::Torn);
                }
                Ok(Some((lsn, body_end)))
            })();

            let (lsn, body_end) = match step {
                Ok(Some(v)) => v,
                Ok(None) => {
                    // suspect header: torn tail unless a later valid section proves rot.
                    return self.suspect_section(pos, size, last_lsn);
                }
                Err(WalStop::Torn) => return Ok(pos),
                Err(WalStop::Fatal(e)) => return Err(e),
            };
            if lsn <= last_lsn {
                return Err(DbError::corrupt_msg(format!(
                    "WAL LSN not increasing at offset {pos}: {lsn} after {last_lsn}"
                )));
            }
            apply_section(&self.inner, &mut win, pos + SEC_HDR as u64, body_end)
                .map_err(fatal_only)?;
            last_lsn = lsn;
            self.next_lsn = lsn
                .checked_add(1)
                .ok_or_else(|| DbError::corrupt("WAL LSN space exhausted"))?;
            pos = body_end;
        }
        Ok(pos)
    }

    /// A section whose header fails its own CRC. The declared bodyLen is
    /// untrusted, so calling it corruption needs the section at the declared end
    /// to be fully valid AND carry EXACTLY the next expected LSN (`last_lsn + 2`).
    fn suspect_section(&self, pos: u64, size: u64, last_lsn: i64) -> Result<u64> {
        // reread the untrusted bodyLen from the damaged header.
        let mut hdr = [0u8; SEC_HDR];
        match read_at(&self.file, &mut hdr, pos) {
            Ok(()) => {}
            Err(WalStop::Torn) => return Ok(pos),
            Err(WalStop::Fatal(e)) => return Err(e),
        }
        let (_, _, body_len, _, _) = parse_sec_hdr(&hdr);
        let body_start = pos + SEC_HDR as u64;
        if body_len >= 0 && body_len as u64 <= size - body_start {
            let follows = any_valid_section_from(
                &self.file,
                body_start + body_len as u64,
                size,
                last_lsn,
                true,
                self.replay_buf,
            )
            .map_err(fatal_only)?;
            if follows {
                return Err(DbError::corrupt_msg(format!(
                    "WAL mid-log corruption: section header damaged at offset {pos} but valid sections follow (not a torn tail)"
                )));
            }
        }
        Ok(pos)
    }

    /// Legacy (headerless) log: trailing-COMMIT-seal sections. Returns end offset
    /// of the last valid section.
    fn replay_legacy(&mut self, size: u64) -> Result<u64> {
        if size == 0 {
            return Ok(0);
        }
        let mut win = WalIn::new(&self.file, self.replay_buf);
        win.reset(0, size);
        let mut valid_end = 0u64;
        let mut pending: Vec<WalOp> = Vec::new();
        let res = (|| -> WalRes<()> {
            while win.remaining() > 0 {
                let ty = win.read_byte_raw()?;
                if ty == T_COMMIT {
                    let computed = win.crc_value() as i32;
                    if computed != win.read_int_raw()? {
                        return Ok(()); // torn/corrupt tail
                    }
                    apply_ops(&self.inner, &pending)?;
                    pending.clear();
                    valid_end = win.pos();
                    win.crc_reset();
                    continue;
                }
                win.crc_tag(ty);
                match ty {
                    x if x == T_PREALLOC => {
                        pending.push(WalOp {
                            op: T_PREALLOC,
                            recid: win.unpack_long()?,
                            cap: 0,
                            data: None,
                        });
                    }
                    x if x == T_DELETE => {
                        pending.push(WalOp {
                            op: T_DELETE,
                            recid: win.unpack_long()?,
                            cap: 0,
                            data: None,
                        });
                    }
                    x if x == T_RECORD => {
                        let recid = win.unpack_long()?;
                        let cap = win.unpack_long()?;
                        let len_plus = win.unpack_long()?;
                        let mut data: Option<Vec<u8>> = None;
                        if len_plus != 0 {
                            let len = len_plus - 1;
                            if len > i32::MAX as u64 || len > win.remaining() {
                                return Ok(()); // torn
                            }
                            let mut b = vec![0u8; len as usize];
                            win.read_fully(&mut b)?;
                            data = Some(b);
                        }
                        if !cap_valid(cap, data.as_deref()) {
                            return Ok(()); // garbage capacity: torn tail
                        }
                        pending.push(WalOp {
                            op: T_RECORD,
                            recid,
                            cap: cap as usize,
                            data,
                        });
                    }
                    x if x == T_APPEND => {
                        let recid = win.unpack_long()?;
                        let len = win.unpack_long()?;
                        if len > i32::MAX as u64 || len > win.remaining() {
                            return Ok(()); // torn
                        }
                        let mut b = vec![0u8; len as usize];
                        win.read_fully(&mut b)?;
                        pending.push(WalOp {
                            op: T_APPEND,
                            recid,
                            cap: 0,
                            data: Some(b),
                        });
                    }
                    _ => return Ok(()), // unknown instruction: torn tail
                }
            }
            Ok(())
        })();
        match res {
            Ok(()) | Err(WalStop::Torn) => Ok(valid_end),
            Err(WalStop::Fatal(e)) => Err(e),
        }
    }

    /// Merged content = (staged base or inner content) ++ staged appends; `None` = null.
    fn merged(&self, recid: u64, s: &Staged) -> Result<Option<Vec<u8>>> {
        let base: Option<Vec<u8>> = if s.base_set {
            s.base.clone()
        } else if self.inner.rec_state(recid)? == STATE_LIVE {
            self.inner.raw_get(recid)?
        } else {
            None
        };
        if base.is_none() && s.appends.is_empty() {
            return Ok(None);
        }
        let base_len = base.as_ref().map_or(0, |b| b.len());
        let mut m = Vec::with_capacity(base_len + s.appends_len);
        if let Some(b) = &base {
            m.extend_from_slice(b);
        }
        for a in &s.appends {
            m.extend_from_slice(a);
        }
        Ok(Some(m))
    }

    /// Staged entry for a write; establishes GetVoid on deleted/void recids.
    fn staged_for_write(&mut self, recid: u64) -> Result<&mut Staged> {
        if let Some(s) = self.staged.get(&recid) {
            if s.deleted {
                return Err(DbError::GetVoid(recid));
            }
        } else {
            if self.inner.rec_state(recid)? == STATE_VOID {
                return Err(DbError::GetVoid(recid));
            }
            self.staged.insert(recid, Staged::new(false));
        }
        Ok(self.staged.get_mut(&recid).unwrap())
    }

    fn commit_locked(&mut self, path: &Path) -> Result<()> {
        if self.poisoned {
            return Err(DbError::corrupt(
                "WAL poisoned by an earlier durability failure",
            ));
        }
        // Panic-safety: refuse before writing rather than overflow `next_lsn += 1`
        // below. LSN exhaustion (2^63 sections) is infeasible in practice.
        if self.next_lsn == i64::MAX {
            return Err(DbError::corrupt("WAL LSN space exhausted"));
        }
        if self.staged.is_empty() {
            return Ok(());
        }
        // classify all ops BEFORE applying any (apply shifts inner state).
        let mut recids: Vec<u64> = self.staged.keys().copied().collect();
        recids.sort_unstable();
        let mut ops: Vec<WalOp> = Vec::with_capacity(recids.len());
        for recid in recids {
            let s = self.staged.get(&recid).unwrap();
            if s.deleted {
                if !s.created {
                    ops.push(WalOp {
                        op: T_DELETE,
                        recid,
                        cap: 0,
                        data: None,
                    });
                } else {
                    ops.push(WalOp {
                        op: 0,
                        recid,
                        cap: 0,
                        data: None,
                    }); // created+deleted: cleanup only
                }
            } else if !s.base_set && s.appends.is_empty() {
                ops.push(WalOp {
                    op: T_PREALLOC,
                    recid,
                    cap: 0,
                    data: None,
                });
            } else if s.base_set || self.inner.rec_state(recid)? != STATE_LIVE {
                let m = self.merged(recid, s)?;
                // `cap == 0` in a T_RECORD is valid ONLY for null content or a
                // genuinely oversize (linked) record. A plain record whose
                // content+headroom rounds past MAX_CAPACITY must NOT collapse to
                // cap 0 (that produces a WAL section neither Rust nor Java can
                // reopen); reject it as RecordTooLarge instead.
                let cap_l: u64 = match &m {
                    None => 0,
                    Some(b) => {
                        let base = 4 + b.len() as u64;
                        if base > iv::MAX_CAPACITY as u64 {
                            0 // genuinely oversize → linked on apply/replay
                        } else {
                            plain_cap(b.len(), s.headroom)?
                        }
                    }
                };
                ops.push(WalOp {
                    op: T_RECORD,
                    recid,
                    cap: cap_l as usize,
                    data: m,
                });
            } else {
                // live base in inner: log only the appended tail.
                let s = self.staged.get(&recid).unwrap();
                let mut m = Vec::with_capacity(s.appends_len);
                for a in &s.appends {
                    m.extend_from_slice(a);
                }
                ops.push(WalOp {
                    op: T_APPEND,
                    recid,
                    cap: 0,
                    data: Some(m),
                });
            }
        }

        // build the section body.
        let mut body = DataOutput2::with_capacity(1024);
        for op in &ops {
            match op.op {
                x if x == T_PREALLOC || x == T_DELETE => {
                    body.write_byte(op.op as i32);
                    body.pack_long(op.recid);
                }
                x if x == T_RECORD => {
                    body.write_byte(T_RECORD as i32);
                    body.pack_long(op.recid);
                    body.pack_long(op.cap as u64);
                    match &op.data {
                        None => body.pack_long(0),
                        Some(d) => {
                            body.pack_long(d.len() as u64 + 1);
                            body.write_all(d);
                        }
                    }
                }
                x if x == T_APPEND => {
                    body.write_byte(T_APPEND as i32);
                    body.pack_long(op.recid);
                    let d = op.data.as_ref().unwrap();
                    body.pack_long(d.len() as u64);
                    body.write_all(d);
                }
                _ => {} // op 0: not logged
            }
        }

        // section header (tag, lsn, bodyLen) + CRCs; fsync = durability point (D1).
        let body_len = body.buf.len() as i64;
        let mut hdr = [0u8; SEC_HDR];
        hdr[0] = TAG_SECTION;
        hdr[1..9].copy_from_slice(&self.next_lsn.to_be_bytes());
        hdr[9..17].copy_from_slice(&body_len.to_be_bytes());
        let hcrc = crc32(&hdr[..SEC_HDR_CRC_LEN]) as i32;
        let bcrc = crc32(&body.buf) as i32;
        hdr[17..21].copy_from_slice(&hcrc.to_be_bytes());
        hdr[21..25].copy_from_slice(&bcrc.to_be_bytes());
        self.file.write_all_at(&hdr, self.log_pos)?;
        self.file
            .write_all_at(&body.buf, self.log_pos + SEC_HDR as u64)?;
        self.file.sync_data()?;
        self.log_pos += SEC_HDR as u64 + body.buf.len() as u64;
        self.next_lsn += 1;

        // apply to the inner volume.
        for op in &ops {
            match op.op {
                0 => {
                    self.inner.delete(nz(op.recid))?; // created+deleted: free the P recid
                }
                x if x == T_PREALLOC => { /* already P in inner since op time */ }
                x if x == T_RECORD => {
                    self.inner.wal_put(op.recid, op.cap, op.data.as_deref())?;
                }
                x if x == T_APPEND => {
                    let d = op.data.as_ref().unwrap();
                    if self.inner.append(nz(op.recid), d)? == AppendResult::Refused {
                        return Err(DbError::corrupt_msg(format!(
                            "commit append refused, recid={}",
                            op.recid
                        )));
                    }
                }
                x if x == T_DELETE => {
                    self.inner.delete(nz(op.recid))?;
                }
                _ => {}
            }
        }
        self.staged.clear();
        self.maybe_auto_checkpoint_locked(path)?;
        Ok(())
    }

    fn maybe_auto_checkpoint_locked(&mut self, path: &Path) -> Result<()> {
        let limit = self.auto_checkpoint_bytes;
        if limit <= 0 {
            return Ok(());
        }
        let doubled = self.checkpoint_basis.saturating_mul(2);
        if self.log_pos >= (limit as u64).max(doubled) {
            self.checkpoint_locked(path)?;
        }
        Ok(())
    }

    /// Rewrite the log as one snapshot section of the inner store's committed
    /// state, atomically replacing the log. The rename is the commit point.
    fn checkpoint_locked(&mut self, path: &Path) -> Result<()> {
        if self.poisoned {
            return Err(DbError::corrupt(
                "WAL poisoned by an earlier durability failure",
            ));
        }
        if self.next_lsn == i64::MAX {
            return Err(DbError::corrupt("WAL LSN space exhausted"));
        }
        let tmp = ckpt_tmp(path);
        let _ = std::fs::remove_file(&tmp);

        // 1) stream the snapshot section to the temp file, make it durable.
        // Keep `out` open PAST the rename so we can install it directly as the
        // new log handle: reopening `path` after the rename and failing would
        // strand the store on the now-unlinked pre-checkpoint inode.
        let out = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&tmp)?;
        write_file_header(&out)?;
        // placeholder section header, patched below.
        out.write_all_at(&[0u8; SEC_HDR], FILE_HDR)?;

        let (hdr, body_size) = {
            let mut w = WalSnapshotWriter::new(&out, FILE_HDR + SEC_HDR as u64);
            self.inner
                .wal_snapshot(|recid, prealloc, cap_bytes, content| {
                    if prealloc {
                        w.prealloc(recid)
                    } else {
                        w.record(recid, cap_bytes, content.as_deref())
                    }
                })?;
            w.flush()?;
            let body_crc = w.crc.clone().finalize() as i32;
            let mut hdr = [0u8; SEC_HDR];
            hdr[0] = TAG_CKPT;
            hdr[1..9].copy_from_slice(&self.next_lsn.to_be_bytes());
            hdr[9..17].copy_from_slice(&(w.body_len as i64).to_be_bytes());
            let hcrc = crc32(&hdr[..SEC_HDR_CRC_LEN]) as i32;
            hdr[17..21].copy_from_slice(&hcrc.to_be_bytes());
            hdr[21..25].copy_from_slice(&body_crc.to_be_bytes());
            (hdr, w.body_len)
        };
        out.write_all_at(&hdr, FILE_HDR)?;
        out.sync_all()?; // snapshot fully durable before it may replace the log
        let size = FILE_HDR + SEC_HDR as u64 + body_size;

        // 2) atomic swap: the rename is the checkpoint's commit point. Install
        // the retained handle and advance in-memory state BEFORE the (fallible)
        // directory fsync, so the store is always consistent with the promoted
        // file even if that final durability step returns an error.
        std::fs::rename(&tmp, path)?;
        self.file = out;
        self.log_pos = size;
        self.checkpoint_basis = size;
        self.next_lsn += 1; // the snapshot section consumed one LSN
                            // The rename is visible but its directory-entry durability is unconfirmed
                            // if this fsync fails; POSIX does not guarantee a later file `sync_data`
                            // makes the rename durable. Poison so no subsequent commit/checkpoint can
                            // report false durability until the store is reopened.
        if let Err(e) = fsync_dir(path) {
            self.poisoned = true;
            return Err(e);
        }
        Ok(())
    }
}

fn apply_ops(inner: &StoreDirect, ops: &[WalOp]) -> WalRes<()> {
    for op in ops {
        let recid = nz_res(op.recid)?;
        match op.op {
            x if x == T_PREALLOC => inner.wal_prealloc(recid.get())?,
            x if x == T_RECORD => inner.wal_put(recid.get(), op.cap, op.data.as_deref())?,
            x if x == T_APPEND => {
                let d = op.data.as_ref().unwrap();
                if inner.append(recid, d)? == AppendResult::Refused {
                    return Err(WalStop::Fatal(DbError::corrupt_msg(format!(
                        "WAL append refused, recid={}",
                        op.recid
                    ))));
                }
            }
            x if x == T_DELETE => inner.delete(recid)?,
            other => {
                return Err(WalStop::Fatal(DbError::corrupt_msg(format!(
                    "bad WAL op {other}"
                ))))
            }
        }
    }
    Ok(())
}

/// True when `[from, size)` holds ≥1 fully valid section proving durable
/// committed sections follow a bad one. `exact_next`: untrusted anchor requires
/// exactly `last_lsn + 2`; else any strictly-future LSN (`> last_lsn + 1`).
fn any_valid_section_from(
    file: &File,
    from: u64,
    size: u64,
    last_lsn: i64,
    exact_next: bool,
    bufsize: usize,
) -> WalRes<bool> {
    let mut pos = from;
    while pos + SEC_HDR as u64 <= size {
        let mut hdr = [0u8; SEC_HDR];
        match read_at(file, &mut hdr, pos) {
            Ok(()) => {}
            Err(WalStop::Torn) => return Ok(false),
            Err(e) => return Err(e),
        }
        let (tag, lsn, body_len, stored_hdr_crc, stored_body_crc) = parse_sec_hdr(&hdr);
        let body_start = pos + SEC_HDR as u64;
        if crc32(&hdr[..SEC_HDR_CRC_LEN]) as i32 != stored_hdr_crc
            || (tag != TAG_SECTION && tag != TAG_CKPT)
            || body_len < 0
            || body_len as u64 > size - body_start
        {
            return Ok(false);
        }
        // checked: overflow at the LSN ceiling means no such successor exists, so
        // there is no match (a saturating add would falsely accept i64::MAX as
        // "two later" when last_lsn == i64::MAX-1).
        let lsn_ok = if exact_next {
            matches!(last_lsn.checked_add(2), Some(x) if lsn == x)
        } else {
            matches!(last_lsn.checked_add(1), Some(x) if lsn > x)
        };
        if lsn_ok
            && body_crc(file, body_start, body_start + body_len as u64, bufsize)?
                == stored_body_crc as u32
        {
            return Ok(true);
        }
        pos = body_start + body_len as u64;
    }
    Ok(false)
}

/// Streaming snapshot body writer: buffers ~1 MiB chunks, tracks total body
/// length (u64 — bodies may exceed 2 GiB) and a rolling CRC32.
struct WalSnapshotWriter<'a> {
    file: &'a File,
    at: u64,
    out: DataOutput2,
    crc: crc32fast::Hasher,
    body_len: u64,
}

impl<'a> WalSnapshotWriter<'a> {
    const FLUSH_AT: usize = 1 << 20;

    fn new(file: &'a File, at: u64) -> WalSnapshotWriter<'a> {
        WalSnapshotWriter {
            file,
            at,
            out: DataOutput2::with_capacity(64 * 1024),
            crc: crc32fast::Hasher::new(),
            body_len: 0,
        }
    }

    fn prealloc(&mut self, recid: u64) -> Result<()> {
        self.out.write_byte(T_PREALLOC as i32);
        self.out.pack_long(recid);
        self.maybe_flush()
    }

    fn record(&mut self, recid: u64, cap_bytes: usize, content: Option<&[u8]>) -> Result<()> {
        self.out.write_byte(T_RECORD as i32);
        self.out.pack_long(recid);
        self.out.pack_long(cap_bytes as u64);
        match content {
            None => self.out.pack_long(0),
            Some(d) => {
                self.out.pack_long(d.len() as u64 + 1);
                self.out.write_all(d);
            }
        }
        self.maybe_flush()
    }

    fn maybe_flush(&mut self) -> Result<()> {
        if self.out.buf.len() >= Self::FLUSH_AT {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.out.buf.is_empty() {
            return Ok(());
        }
        self.crc.update(&self.out.buf);
        self.file.write_all_at(&self.out.buf, self.at)?;
        self.at += self.out.buf.len() as u64;
        self.body_len += self.out.buf.len() as u64;
        self.out.buf.clear();
        Ok(())
    }
}

// ---------- Store / StoreDelta / StoreTx ----------

impl Store for StoreWAL {
    fn preallocate(&self) -> Result<Recid> {
        let mut st = self.write_open()?;
        let recid = st.inner.preallocate()?;
        st.staged.insert(recid.get(), Staged::new(true));
        Ok(recid)
    }

    fn put<R: Record>(&self, value: &R, ser: &(impl Serializer<R> + Sync)) -> Result<Recid> {
        let bytes = serialize(value, ser);
        let mut st = self.write_open()?;
        let recid = st.inner.preallocate()?;
        let mut s = Staged::new(true);
        s.base_set = true;
        s.base = Some(bytes);
        st.staged.insert(recid.get(), s);
        Ok(recid)
    }

    fn get<R: Record>(&self, recid: Recid, ser: &(impl Serializer<R> + Sync)) -> Result<Option<R>> {
        self.check_closed()?;
        let st = self.st.read();
        match st.staged.get(&recid.get()) {
            None => st.inner.get(recid, ser),
            Some(s) => {
                if s.deleted {
                    return Err(DbError::GetVoid(recid.get()));
                }
                match st.merged(recid.get(), s)? {
                    None => Ok(None),
                    Some(m) => {
                        let mut inp = SliceInput::new(&m);
                        Ok(Some(ser.deserialize(&mut inp, Some(m.len()))?))
                    }
                }
            }
        }
    }

    fn read(&self, recid: Recid, action: &mut dyn RecordRead) -> Result<i64> {
        self.check_closed()?;
        let st = self.st.read();
        match st.staged.get(&recid.get()) {
            None => st.inner.read(recid, action),
            Some(s) => {
                if s.deleted {
                    return Err(DbError::GetVoid(recid.get()));
                }
                match st.merged(recid.get(), s)? {
                    None => action.on_null(),
                    Some(m) => {
                        let mut inp = SliceInput::new(&m);
                        action.on_bytes(&mut inp, m.len())
                    }
                }
            }
        }
    }

    fn update<R: Record>(
        &self,
        recid: Recid,
        value: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
    ) -> Result<()> {
        self.update_with_headroom_opt(recid, value, ser, 0)
    }

    fn compare_and_swap<R: Record>(
        &self,
        recid: Recid,
        expect: Option<&R>,
        new: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
    ) -> Result<bool> {
        let mut st = self.write_open()?;
        // resolve current logical value.
        let current: Option<R> = match st.staged.get(&recid.get()) {
            None => {
                if st.inner.rec_state(recid.get())? == STATE_VOID {
                    return Err(DbError::GetVoid(recid.get()));
                }
                st.inner.get(recid, ser)?
            }
            Some(s) => {
                if s.deleted {
                    return Err(DbError::GetVoid(recid.get()));
                }
                match st.merged(recid.get(), s)? {
                    None => None,
                    Some(m) => {
                        let mut inp = SliceInput::new(&m);
                        Some(ser.deserialize(&mut inp, Some(m.len()))?)
                    }
                }
            }
        };
        let eq = match (&current, expect) {
            (None, None) => true,
            (Some(c), Some(e)) => ser.equals(c, e),
            _ => false,
        };
        if !eq {
            return Ok(false);
        }
        let new_bytes = new.map(|v| serialize(v, ser));
        let s = st.staged_for_write(recid.get())?;
        s.base_set = true;
        s.base = new_bytes;
        s.headroom = 0;
        s.appends.clear();
        s.appends_len = 0;
        Ok(true)
    }

    fn delete(&self, recid: Recid) -> Result<()> {
        let mut st = self.write_open()?;
        let s = st.staged_for_write(recid.get())?;
        s.deleted = true;
        s.base_set = false;
        s.base = None;
        s.appends.clear();
        s.appends_len = 0;
        Ok(())
    }

    fn commit(&self) -> Result<()> {
        // write_open re-checks `closed` under the lock: otherwise a commit of a
        // pure staged preallocation could append+fsync a section after `close`
        // completed, since applying it does not touch the inner store.
        let mut st = self.write_open()?;
        st.commit_locked(&self.path)
    }

    fn compact(&self) -> Result<()> {
        self.checkpoint()
    }

    fn close(&self) -> Result<()> {
        // Acquire the write lock BEFORE publishing `closed`, so an in-flight
        // commit/checkpoint that rechecks `closed` under the lock observes the
        // close atomically (no append after close). Any op still runs to
        // completion first (it holds the lock); we then win it and shut down.
        let mut st = self.st.write();
        if self.closed.swap(true, Ordering::AcqRel) {
            // Already closed — but if the first close's directory-fsync retry
            // failed, the checkpoint rename's durability is STILL unconfirmed
            // (`st.poisoned` stays set). Returning Ok here would recreate the
            // false success the first close correctly reported as an error, so
            // re-enter and retry the fsync until it succeeds (mirrors
            // StoreDirect::close re-entering while poisoned). Resources were
            // already released best-effort by the first close.
            if !st.poisoned {
                return Ok(());
            }
            return match fsync_dir(&self.path) {
                Ok(()) => {
                    st.poisoned = false;
                    Ok(())
                }
                Err(e) => Err(e),
            };
        }
        // If a prior checkpoint left the rename's directory durability
        // unconfirmed (poisoned), retry the directory fsync now. A file
        // `sync_data` does NOT make the earlier rename durable, so we must not
        // report a clean close until the directory fsync succeeds.
        let poison_err = if st.poisoned {
            match fsync_dir(&self.path) {
                Ok(()) => {
                    st.poisoned = false;
                    None
                }
                Err(e) => Some(e),
            }
        } else {
            None
        };
        // Always release resources, then surface the durability error (if any).
        let sync = st.file.sync_data();
        let inner = st.inner.close();
        if let Some(e) = poison_err {
            return Err(e);
        }
        sync?;
        inner?;
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn verify(&self) -> Result<()> {
        self.check_closed()?;
        self.st.read().inner.verify()
    }

    fn get_all_recids(&self) -> Result<Vec<Recid>> {
        self.check_closed()?;
        let st = self.st.read();
        let mut set: std::collections::BTreeSet<u64> = st
            .inner
            .get_all_recids()?
            .into_iter()
            .map(|r| r.get())
            .collect();
        for (recid, s) in &st.staged {
            if s.deleted {
                set.remove(recid);
            } else if s.base_set || !s.appends.is_empty() {
                set.insert(*recid);
            } else {
                set.remove(recid); // pure prealloc
            }
        }
        Ok(set.into_iter().map(nz).collect())
    }

    fn get_current_size(&self) -> u64 {
        let st = self.st.read();
        st.inner.get_current_size()
    }

    fn is_tx(&self) -> bool {
        true
    }

    fn structural_generation(&self) -> u64 {
        self.struct_gen.load(Ordering::Acquire)
    }
}

impl StoreWAL {
    fn update_with_headroom_opt<R: Record>(
        &self,
        recid: Recid,
        value: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
        headroom: usize,
    ) -> Result<()> {
        self.check_closed()?;
        let bytes = value.map(|v| serialize(v, ser));
        // Fail fast (as Java `StoreDirect.update_with_headroom` does) when a
        // plain-sized content plus its headroom would exceed MAX_CAPACITY, rather
        // than staging it and only failing at commit.
        if let Some(b) = &bytes {
            if 4 + b.len() as u64 <= iv::MAX_CAPACITY as u64 {
                plain_cap(b.len(), headroom)?;
            }
        }
        let mut st = self.write_open()?;
        let s = st.staged_for_write(recid.get())?;
        s.base_set = true;
        s.base = bytes;
        s.headroom = headroom;
        s.appends.clear();
        s.appends_len = 0;
        Ok(())
    }
}

impl StoreDelta for StoreWAL {
    fn append(&self, recid: Recid, data: &[u8]) -> Result<AppendResult> {
        let mut st = self.write_open()?;
        // capacity refusal is enforced now against the inner base if already live.
        {
            let s = st.staged_for_write(recid.get())?;
            let _ = s;
        }
        let rid = recid.get();
        let base_live =
            !st.staged.get(&rid).unwrap().base_set && st.inner.rec_state(rid)? == STATE_LIVE;
        if base_live {
            let cap_rem = st.inner.capacity_remaining(recid)?;
            let appends_len = st.staged.get(&rid).unwrap().appends_len;
            if appends_len + data.len() > cap_rem {
                return Ok(AppendResult::Refused);
            }
        }
        let base_len = if base_live {
            st.inner.raw_get(rid)?.map_or(0, |b| b.len())
        } else {
            let s = st.staged.get(&rid).unwrap();
            s.base.as_ref().map_or(0, |b| b.len())
        };
        let s = st.staged.get_mut(&rid).unwrap();
        s.appends.push(data.to_vec());
        s.appends_len += data.len();
        Ok(AppendResult::NewSize(base_len + s.appends_len))
    }

    fn capacity_remaining(&self, recid: Recid) -> Result<usize> {
        self.check_closed()?;
        let st = self.st.read();
        match st.staged.get(&recid.get()) {
            None => st.inner.capacity_remaining(recid),
            Some(s) => {
                if s.deleted {
                    return Err(DbError::GetVoid(recid.get()));
                }
                if s.base_set {
                    return Ok(usize::MAX); // capacity established at commit
                }
                if st.inner.rec_state(recid.get())? == STATE_LIVE {
                    return Ok(st
                        .inner
                        .capacity_remaining(recid)?
                        .saturating_sub(s.appends_len));
                }
                Ok(usize::MAX)
            }
        }
    }

    fn update_with_headroom<R: Record>(
        &self,
        recid: Recid,
        value: &R,
        ser: &(impl Serializer<R> + Sync),
        headroom: usize,
    ) -> Result<()> {
        self.update_with_headroom_opt(recid, Some(value), ser, headroom)
    }
}

impl StoreTx for StoreWAL {
    fn rollback(&self) -> Result<()> {
        let mut st = self.write_open()?;
        let created: Vec<u64> = st
            .staged
            .iter()
            .filter(|(_, s)| s.created)
            .map(|(r, _)| *r)
            .collect();
        for recid in created {
            st.inner.delete(nz(recid))?; // free the P recid
        }
        st.staged.clear();
        drop(st);
        // Signal open collections that their append-only structural caches may
        // now describe a taller-than-real tree (a reverted uncommitted grow).
        self.struct_gen.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

impl super::StoreLease for StoreWAL {
    fn lease_table(&self) -> &Arc<LeaseTable> {
        &self.lease_table
    }
}

// WAL reads resolve staged bytes then delegate to the inner locked read (D4).
impl super::StoreTornRead for StoreWAL {}

/// Rounded plain-record capacity for `content_len` + `headroom` bytes, or
/// `RecordTooLarge` on overflow / exceeding MAX_CAPACITY. Caller guarantees the
/// content is not itself oversize (`4 + content_len <= MAX_CAPACITY`).
fn plain_cap(content_len: usize, headroom: usize) -> Result<u64> {
    let need = (4u64 + content_len as u64)
        .checked_add(headroom as u64)
        .ok_or(DbError::RecordTooLarge)?;
    let rounded = need.checked_add(15).ok_or(DbError::RecordTooLarge)? & !15;
    if rounded > iv::MAX_CAPACITY as u64 {
        return Err(DbError::RecordTooLarge);
    }
    Ok(rounded)
}

fn serialize<R>(value: &R, ser: &(impl Serializer<R> + Sync)) -> Vec<u8> {
    let mut out = DataOutput2::with_capacity(ser.size_hint() + 4);
    ser.serialize(&mut out, value);
    out.into_vec()
}
