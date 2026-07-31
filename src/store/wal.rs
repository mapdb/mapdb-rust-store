//! `StoreWAL` — transactional store: an in-memory [`StoreDirect`] volume plus a
//! write-ahead log (spec 02 §7, Java `StoreWAL`).
//!
//! Uncommitted mutations are staged in memory; [`StoreWAL::commit`] emits them
//! as one WAL section, forces it (the durability point), then applies them to
//! the inner memory-backed store. Recovery replays the log's retained sections.
//!
//! # On-disk format v3
//!
//! The log is a **segment set**, not a file: `<base>.wal.<16 hex digits>`, one
//! store lock at `<base>.lock`, and a cleaning cycle that retires whole segments
//! instead of rewriting the log. The opener takes the store's BASE path.
//!
//! ```text
//! segment := header section*
//! header  := magic "MDBS.WAL"(8) | version i32=3 | flags i32=0
//!          | segmentSeq i64 | firstLsn i64 | headerCrc i32           (36 B)
//! section := tag u8 ('S' commit, 'C' image, 'K' clean mark)
//!          | lsn i64 | bodyLen i64 | hdrCrc i32 | bodyCrc i32        (25 B)
//!          | body
//! ```
//!
//! Both section CRCs are computed over a **domain prefix** — the segment's 36
//! header bytes followed by the section's big-endian offset — so a section
//! byte-copied to another segment or another offset fails its own checksum.
//! The three modules that make up the format:
//!
//! - [`wal_segments`](super::wal_segments) — the namespace, the lock, and every
//!   operation that changes which files exist (tables N and H).
//! - [`wal_recover`](super::wal_recover) — the section/entry codec and the
//!   two-pass recovery state machine (tables S, K and R).
//! - [`wal_write`](super::wal_write) — the streaming two-pass section writer and
//!   the durability event seam (table W).
//!
//! # Compatibility
//!
//! There is **no migration** from this port's v1 single-file log, and none from
//! any other format: a v1 log, a bare file at the base path, or a v1 `.ckpt`
//! temp REFUSES the open with guidance and is never deleted (D1). The v3 format
//! is shared with the Java and zig engines byte for byte; it is an
//! implementation fact rather than a stability promise — see `README.md`.

use crate::error::{DbError, Result};
use crate::io::{DataOutput2, SliceInput};
use crate::ser::Serializer;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::direct::{STATE_LIVE, STATE_VOID};
use super::index_val as iv;
use super::lease::LeaseTable;
use super::wal_recover::{
    build_mark_body, recover, Identities, Recovered, MARK_BODY_LEN, TAG_MARK, TAG_SECTION,
};
use super::wal_segments::WalSegmentSet;
use super::wal_write::{append_section, wal_io_event, BodySink, WalIo, WalOpKind};
use super::{AppendResult, Recid, Record, RecordRead, Store, StoreDelta, StoreDirect, StoreTx};
use parking_lot::RwLock;
use std::sync::Arc;

const T_PREALLOC: u8 = 1;
const T_RECORD: u8 = 2;
const T_APPEND: u8 = 3;
const T_DELETE: u8 = 4;
/// Not an entry type: "created and deleted inside one transaction", which is
/// applied (the preallocated recid is freed) and never logged.
const T_TRANSIENT: u8 = 0;

/// Default streaming-replay window (bytes); the ctor override forces refill
/// edges in tests.
const DEFAULT_REPLAY_BUF: usize = 1 << 20;

/// Default segment size. The writer seals and rolls PAST this, at a section
/// boundary, so one section may exceed it and an oversize section gets a segment
/// to itself.
pub const DEFAULT_SEGMENT_BYTES: u64 = 64 << 20;

/// Smallest legal segment size: a header plus one section header. Anything
/// below it cannot hold a single section.
pub const MIN_SEGMENT_BYTES: u64 = super::wal_segments::SEG_HDR + 25;

/// Floor under the cleaning trigger: a log smaller than this is never cleaned,
/// however small the live data is. Without a floor a store holding a few hundred
/// bytes would clean on every commit.
pub const DEFAULT_MIN_LOG_BYTES: u64 = 1 << 30;

/// Default space-amplification target: clean once the log exceeds this multiple
/// of the live data. It bounds SPACE, not write amplification.
pub const DEFAULT_SPACE_AMPLIFICATION: u32 = 2;

#[inline]
fn nz(recid: u64) -> Recid {
    NonZeroU64::new(recid).expect("recid 0 is never allocated")
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

/// Classified commit operation, computed before any apply (state must not shift
/// mid-apply).
struct WalOp {
    /// One of `T_*`, or [`T_TRANSIENT`].
    op: u8,
    recid: u64,
    cap: usize,
    data: Option<Vec<u8>>,
    /// `T_APPEND` only: the LSN of the content image this delta extends, read
    /// from the live identities at classify time and written to the log as
    /// `packLong(sectionLsn - base_lsn)`. 0 for every other op, all of which are
    /// self-contained.
    base_lsn: i64,
}

/// Options for [`StoreWAL::open_cfg`]. Crate-internal: it carries the writer
/// event seam, which is a test surface rather than an API.
pub(crate) struct WalOptions {
    pub(crate) thread_safe: bool,
    pub(crate) read_only: bool,
    pub(crate) segment_bytes: u64,
    /// Streaming window for replay and for the cleaner's scan; a tiny value
    /// forces refill edges in tests.
    pub(crate) replay_buf: usize,
    pub(crate) wal_io: Option<Arc<dyn WalIo>>,
}

impl Default for WalOptions {
    fn default() -> WalOptions {
        WalOptions {
            thread_safe: true,
            read_only: false,
            segment_bytes: DEFAULT_SEGMENT_BYTES,
            replay_buf: DEFAULT_REPLAY_BUF,
            wal_io: None,
        }
    }
}

/// The lock-guarded mutable state (Java's single ReadWriteLock covers all of it).
struct WalState {
    inner: StoreDirect,
    segs: WalSegmentSet,
    staged: HashMap<u64, Staged>,
    /// Next section LSN — exactly consecutive within a segment.
    next_lsn: i64,
    segment_bytes: u64,
    min_log_bytes: u64,
    space_amplification: u32,
    /// Streaming window for replay and for A3's cleaner scan.
    #[allow(dead_code)] // consumed by A3's cleaner; A2 and A3 land together
    replay_buf: usize,
    read_only: bool,
    /// The two per-recid identities, maintained atomically with the committed
    /// apply of the entry that sets them — never before, never from staged
    /// state. Replay rebuilds them; the commit classifier reads them.
    ids: Identities,
    /// Committed self-contained entries over this store's lifetime. The
    /// cleaner's futility latch uses it as a staleness clock (A3).
    committed_state_changes: u64,
    wal_io: Option<Arc<dyn WalIo>>,
}

/// Transactional store (spec 02 §7).
pub struct StoreWAL {
    /// Boxed: the WAL state is the largest store state in the family (a segment
    /// namespace, the two identity maps and the staged set), and `StoreWAL` is
    /// one variant of the DB layer's store enum, where an inline copy would size
    /// every other variant with it.
    st: RwLock<Box<WalState>>,
    /// The store's BASE path — `<base>.wal.<hex>` are its segments. Absolutized
    /// by the namespace layer; this is the caller's spelling.
    base: PathBuf,
    lease_table: Arc<LeaseTable>,
    closed: AtomicBool,
    /// Bumped on every `rollback` so open collections know their append-only
    /// structural caches (e.g. the btree left-edge spine) may have been reverted
    /// to a shorter tree and must be rebuilt before the next structural op.
    struct_gen: AtomicU64,
    /// D2: delete the whole namespace inside `close`, while the lock is still
    /// held. Set by the DB layer's delete-after-close mode.
    delete_on_close: AtomicBool,
}

impl StoreWAL {
    /// Opens the store at `base`; its segments are `<base>.wal.<16 hex>`.
    pub fn open(base: &Path) -> Result<StoreWAL> {
        Self::open_cfg(base, WalOptions::default())
    }

    pub fn open_ts(base: &Path, thread_safe: bool) -> Result<StoreWAL> {
        Self::open_cfg(
            base,
            WalOptions {
                thread_safe,
                ..Default::default()
            },
        )
    }

    /// `replay_buf` is a test hook: a tiny window forces refill edges in
    /// streaming replay.
    pub fn open_with(base: &Path, thread_safe: bool, replay_buf: usize) -> Result<StoreWAL> {
        Self::open_cfg(
            base,
            WalOptions {
                thread_safe,
                replay_buf,
                ..Default::default()
            },
        )
    }

    /// Opens with a non-default segment size. Rollover happens at a section
    /// boundary PAST this many bytes, so one section may exceed it.
    pub fn open_segment_bytes(base: &Path, segment_bytes: u64) -> Result<StoreWAL> {
        Self::open_cfg(
            base,
            WalOptions {
                segment_bytes,
                ..Default::default()
            },
        )
    }

    pub(crate) fn open_cfg(base: &Path, opts: WalOptions) -> Result<StoreWAL> {
        if opts.segment_bytes < MIN_SEGMENT_BYTES {
            return Err(DbError::wrong_config(format!(
                "WAL segment size {} is below the {MIN_SEGMENT_BYTES}-byte minimum (a segment \
                 header plus one section header)",
                opts.segment_bytes
            )));
        }
        // D4, the platform gate: a durable writable open REQUIRES a working
        // directory fsync — the acknowledgement rule is "the section is forced
        // AND the directory entry of the segment holding it is durable" — and
        // Windows cannot express one. Refused BY NAME at open, with no override:
        // an escape hatch that skipped the fsync would make acknowledged commits
        // undurable across a crash while appearing to work. The predicate is
        // Java's exact one (an OS test, not a probe). Read-only opens are exempt
        // — they unlink, truncate and rotate nothing, so they make no durability
        // claim a missing directory fsync could break.
        if !opts.read_only && cfg!(target_os = "windows") {
            return Err(DbError::wrong_config(
                "StoreWAL durable mode is unsupported on Windows: it requires an fsync of the \
                 segment directory, which the platform cannot express, and skipping it would make \
                 acknowledged commits undurable across a crash"
                    .to_string(),
            ));
        }
        let inner = StoreDirect::new_heap_ts(opts.thread_safe)?;
        let mut segs = WalSegmentSet::open_with_io(base, opts.read_only, opts.wal_io.clone())?;
        // A failed recovery drops `segs`, which releases the store lock — Java's
        // `finally { closeQuietly() }`.
        let Recovered {
            next_lsn,
            identities,
        } = match recover(&mut segs, &inner, opts.replay_buf) {
            Ok(r) => r,
            Err(e) => {
                segs.close();
                let _ = inner.close();
                return Err(e);
            }
        };
        Ok(StoreWAL {
            st: RwLock::new(Box::new(WalState {
                inner,
                segs,
                staged: HashMap::new(),
                next_lsn,
                segment_bytes: opts.segment_bytes,
                min_log_bytes: DEFAULT_MIN_LOG_BYTES,
                space_amplification: DEFAULT_SPACE_AMPLIFICATION,
                replay_buf: opts.replay_buf,
                read_only: opts.read_only,
                ids: identities,
                committed_state_changes: 0,
                wal_io: opts.wal_io,
            })),
            base: base.to_path_buf(),
            lease_table: LeaseTable::new(),
            closed: AtomicBool::new(false),
            struct_gen: AtomicU64::new(0),
            delete_on_close: AtomicBool::new(false),
        })
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
    fn write_open(&self) -> Result<parking_lot::RwLockWriteGuard<'_, Box<WalState>>> {
        let st = self.st.write();
        if self.closed.load(Ordering::Acquire) {
            return Err(DbError::StoreClosed);
        }
        if st.read_only {
            return Err(DbError::ReadOnly);
        }
        Ok(st)
    }

    /// Bytes the log currently costs on the device.
    pub fn log_bytes(&self) -> Result<u64> {
        self.check_closed()?;
        Ok(self.st.read().segs.log_bytes())
    }

    /// Floor under the cleaning trigger (D8): a log smaller than this is never
    /// cleaned automatically, however small the live data is.
    pub fn set_min_log_bytes(&self, bytes: u64) -> Result<()> {
        self.check_closed()?;
        self.st.write().min_log_bytes = bytes;
        Ok(())
    }

    /// Space-amplification target (D8): clean once the log exceeds this multiple
    /// of the live data.
    pub fn set_space_amplification(&self, factor: u32) -> Result<()> {
        self.check_closed()?;
        if factor == 0 {
            return Err(DbError::wrong_config(
                "space amplification must be at least 1".to_string(),
            ));
        }
        self.st.write().space_amplification = factor;
        Ok(())
    }

    /// D2: delete this base's whole segment namespace inside [`close`], while
    /// the store lock is still held. Used by the DB layer's delete-after-close
    /// and temporary-store modes.
    pub fn set_delete_on_close(&self, delete: bool) {
        self.delete_on_close.store(delete, Ordering::Release);
    }

    /// Cleans the log all the way down: retire every segment below a freshly
    /// rolled one by re-emitting, above them, a self-contained image of every
    /// record they still own, then a forced `'K'` mark authorizing their
    /// removal, then the unlink.
    ///
    /// **Slice A2 implements the roll half only.** Rolling first is what makes
    /// the eventual cycle a whole-log clean — every section-bearing segment is
    /// then strictly below the active one — so this is the first step of the
    /// real operation rather than a stub of a different one. The re-emission
    /// cycle, its mark and its unlink arrive in A3, which is why A2 is not
    /// independently shippable: a store whose only log-bounding operation is
    /// "start a new segment" has no log-bounding operation.
    pub fn checkpoint(&self) -> Result<()> {
        let mut st = self.write_open()?;
        st.roll_active_if_nonempty(&self.closed)
    }

    /// The store's base path: its segments are `<base>.wal.<16 hex>`.
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// How many segment files this store currently holds open. The steady state
    /// is at most ONE — the active segment — and that bound is the point, so it
    /// is observable rather than merely intended.
    pub fn open_segment_files(&self) -> usize {
        self.st.read().segs.open_file_count()
    }

    /// Sequence numbers of the live segments, ascending.
    pub fn segment_seqs(&self) -> Vec<i64> {
        self.st
            .read()
            .segs
            .segments()
            .iter()
            .map(|s| s.seq)
            .collect()
    }

    /// The LSN the next section will carry.
    pub fn next_lsn(&self) -> i64 {
        self.st.read().next_lsn
    }
}

impl WalState {
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

    /// Classifies the staged set into the ops one commit section will carry.
    /// Runs BEFORE any apply, because applying shifts the inner store's state.
    fn classify(&self) -> Result<Vec<WalOp>> {
        let mut recids: Vec<u64> = self.staged.keys().copied().collect();
        recids.sort_unstable();
        let mut ops: Vec<WalOp> = Vec::with_capacity(recids.len());
        for recid in recids {
            let s = self.staged.get(&recid).unwrap();
            if s.deleted {
                ops.push(WalOp {
                    // created+deleted in one transaction: apply-only cleanup,
                    // not logged.
                    op: if s.created { T_TRANSIENT } else { T_DELETE },
                    recid,
                    cap: 0,
                    data: None,
                    base_lsn: 0,
                });
            } else if !s.base_set && s.appends.is_empty() {
                // T_PREALLOC exists to make a NEWLY ALLOCATED recid durable. On
                // a record that was already committed, an empty staged entry
                // means nothing was changed, so nothing is logged — structural
                // defence in depth: no path that leaves an empty entry behind
                // can turn it into a prealloc over a live record, which §4.2
                // rejects on replay.
                if s.created {
                    ops.push(WalOp {
                        op: T_PREALLOC,
                        recid,
                        cap: 0,
                        data: None,
                        base_lsn: 0,
                    });
                }
            } else if s.base_set || self.inner.rec_state(recid)? != STATE_LIVE {
                let m = self.merged(recid, s)?;
                ops.push(WalOp {
                    op: T_RECORD,
                    recid,
                    cap: record_cap(m.as_deref(), s.headroom) as usize,
                    data: m,
                    base_lsn: 0,
                });
            } else {
                // Live plain base in inner: log only the appended tail.
                let mut m = Vec::with_capacity(s.appends_len);
                for a in &s.appends {
                    m.extend_from_slice(a);
                }
                // This branch is reached only when the record is committed,
                // content-bearing and plain, which is exactly the shape that has
                // a content base — so the identity must be there. Its absence is
                // a writer bug, and the design's weakest point is a WRONG stamp,
                // so refuse to invent one: a delta with a fabricated base is a
                // silent-loss channel. (Java raises an `AssertionError` here,
                // which is an `Error`: it escapes with the store open and the
                // transaction intact, exactly as this panic does.)
                let base_lsn =
                    *self.ids.content_base_lsn.get(&recid).unwrap_or_else(|| {
                        panic!("no content base LSN for appended recid {recid}")
                    });
                ops.push(WalOp {
                    op: T_APPEND,
                    recid,
                    cap: 0,
                    data: Some(m),
                    base_lsn,
                });
            }
        }
        Ok(ops)
    }

    fn commit_locked(&mut self, closed: &AtomicBool) -> Result<()> {
        if self.staged.is_empty() {
            return Ok(());
        }
        let ops = self.classify()?;
        let section_lsn = self.next_lsn;
        // Validated BEFORE `append_section` can roll over or write, so a
        // mis-stamped delta fails the commit with the store open and the staged
        // transaction intact.
        for op in &ops {
            if op.op == T_APPEND {
                assert!(
                    section_lsn > op.base_lsn,
                    "append base LSN {} is not below its section LSN {section_lsn}, recid={}",
                    op.base_lsn,
                    op.recid
                );
            }
        }

        // The emitter runs TWICE (measure + write passes) over this immutable
        // ops snapshot, so a commit staging more than 2 GiB emits one genuinely
        // huge section instead of dying in a doubling `Vec`. Deterministic by
        // construction: `ops`, their payloads and `section_lsn` are all fixed
        // before the first pass.
        let write = append_section(
            &mut self.segs,
            self.segment_bytes,
            &self.wal_io,
            closed,
            TAG_SECTION,
            section_lsn,
            |sink| emit_entries(&ops, section_lsn, sink),
        );
        if let Err(e) = write {
            // W9: a failed or partial write/force fails the store CLOSED, so no
            // retry can append a complete section after the partial bytes. (v1
            // returned the error with the store open; the next open then read
            // the retry's acknowledged section as mid-log garbage and discarded
            // it — a latent v1 defect this obligation exists to prevent.)
            self.fail_closed(closed);
            return Err(e);
        }
        self.next_lsn += 1;
        // The cleaner's staleness clock: SELF-CONTAINED entries only. An append
        // extends a record whose image is already the log's youngest, so it
        // obsoletes nothing, while a record, a delete and a prealloc each
        // supersede whatever stood before.
        for op in &ops {
            if matches!(op.op, T_RECORD | T_DELETE | T_PREALLOC) {
                self.committed_state_changes += 1;
            }
        }

        // Apply to the inner volume. PAST THE DURABILITY POINT: the section is
        // on disk and owns an LSN, so if any apply fails, memory and log have
        // diverged and this handle can never be made consistent again — a
        // retried commit would re-emit the same frames under a NEW section LSN
        // and the forced one would be applied twice on reopen. Fail closed; the
        // durable state on disk is intact and reopen replays it correctly.
        if let Err(e) = self.apply_committed(&ops, section_lsn) {
            self.fail_closed(closed);
            return Err(DbError::corrupt_msg(format!(
                "WAL commit failed after the durability point ({e:?}); store closed, reopen to \
                 recover the committed section"
            )));
        }
        self.staged.clear();
        self.auto_clean_locked(closed)
    }

    /// Applies one committed section's ops and moves the identities by the SAME
    /// §4.2 transition row replay would take for the entry just written — that
    /// shared table is what keeps the live maps and a rebuilt-from-log copy
    /// identical.
    fn apply_committed(&mut self, ops: &[WalOp], section_lsn: i64) -> Result<()> {
        for op in ops {
            match op.op {
                T_TRANSIENT => {
                    self.inner.delete(nz(op.recid))?; // created+deleted: free the P recid
                                                      // Nothing was logged, so nothing established an identity for
                                                      // this incarnation; clearing is defensive, not load-bearing.
                    self.ids.void(op.recid);
                }
                T_PREALLOC => {
                    /* already P in inner since op time */
                    self.ids.state_only(op.recid, section_lsn);
                }
                T_RECORD => {
                    self.inner.wal_put(op.recid, op.cap, op.data.as_deref())?;
                    match &op.data {
                        None => self.ids.state_only(op.recid, section_lsn),
                        Some(_) => self.ids.content(op.recid, section_lsn),
                    }
                }
                T_APPEND => {
                    let d = op.data.as_ref().expect("T_APPEND carries its payload");
                    if self.inner.append(nz(op.recid), d)? == AppendResult::Refused {
                        return Err(DbError::corrupt_msg(format!(
                            "commit append refused, recid={}",
                            op.recid
                        )));
                    }
                    // An append leaves both identities where they are: the base
                    // image it extends is still the one a later append cites.
                }
                T_DELETE => {
                    self.inner.delete(nz(op.recid))?;
                    self.ids.void(op.recid);
                }
                other => unreachable!("unknown classified op {other}"),
            }
        }
        Ok(())
    }

    /// The store cannot be made consistent again: close it rather than let a
    /// caller retry into a segment holding partial bytes. Durable state on disk
    /// is intact and reopen replays it.
    fn fail_closed(&mut self, closed: &AtomicBool) {
        closed.store(true, Ordering::Release);
        self.segs.close();
        let _ = self.inner.close();
    }

    /// Seals the active segment and starts a successor, if the active one holds
    /// any section. W3's force flavour applies: the seal persists SIZE.
    fn roll_active_if_nonempty(&mut self, closed: &AtomicBool) -> Result<()> {
        let roll = match self.segs.active() {
            None => false,
            Some(a) => !a.empty(),
        };
        if !roll {
            return Ok(());
        }
        let r = (|| -> Result<()> {
            let active = self.segs.active_mut().expect("checked above");
            active.ensure_open()?;
            let (seq, len) = (active.seq, active.file_len);
            wal_io_event(&self.wal_io, WalOpKind::ForceFull, seq, len, 0, 0)?;
            active.file().expect("just opened").sync_all()?;
            active.release();
            self.segs.create_segment(self.next_lsn)?;
            Ok(())
        })();
        if let Err(e) = r {
            // A half-created segment is not recoverable in place.
            self.fail_closed(closed);
            return Err(e);
        }
        Ok(())
    }

    /// The automatic cleaning trigger, run after every commit (D8). **A3.**
    fn auto_clean_locked(&mut self, _closed: &AtomicBool) -> Result<()> {
        Ok(())
    }

    /// Writes a `'K'` mark: the fact that everything at or below
    /// `cleaned_through_seq` may be removed, and where the retained log begins.
    /// Forced before any unlink (W5).
    #[allow(dead_code)] // consumed by A3's cleaner; A2 and A3 land together
    fn append_mark(
        &mut self,
        closed: &AtomicBool,
        cleaned_through_seq: i64,
        log_start_lsn: i64,
    ) -> Result<()> {
        let body = build_mark_body(cleaned_through_seq, log_start_lsn);
        debug_assert_eq!(body.len() as i64, MARK_BODY_LEN);
        let lsn = self.next_lsn;
        let r = append_section(
            &mut self.segs,
            self.segment_bytes,
            &self.wal_io,
            closed,
            TAG_MARK,
            lsn,
            |sink| sink.write(&body),
        );
        if let Err(e) = r {
            self.fail_closed(closed);
            return Err(e);
        }
        self.next_lsn += 1;
        Ok(())
    }
}

/// Emits the entry stream of one commit section. Called once per pass; the two
/// calls must produce identical bytes, which they do because `ops` and
/// `section_lsn` are fixed before the first one.
fn emit_entries(ops: &[WalOp], section_lsn: i64, sink: &mut BodySink) -> Result<()> {
    let mut frame = DataOutput2::with_capacity(64);
    for op in ops {
        frame.buf.clear();
        match op.op {
            T_PREALLOC | T_DELETE => {
                frame.write_byte(op.op as i32);
                frame.pack_long(op.recid);
            }
            T_RECORD => {
                frame.write_byte(T_RECORD as i32);
                frame.pack_long(op.recid);
                frame.pack_long(op.cap as u64);
                frame.pack_long(op.data.as_ref().map_or(0, |d| d.len() as u64 + 1));
            }
            T_APPEND => {
                frame.write_byte(T_APPEND as i32);
                frame.pack_long(op.recid);
                // Base identity, as a delta against this section's own LSN
                // (§4.2): >= 1 by construction, because the base was established
                // by a strictly earlier section, and typically one byte because
                // a hot record's base is recent.
                frame.pack_long((section_lsn - op.base_lsn) as u64);
                frame.pack_long(op.data.as_ref().expect("T_APPEND payload").len() as u64);
            }
            T_TRANSIENT => continue, // not logged
            other => unreachable!("unknown classified op {other}"),
        }
        sink.write(&frame.buf)?;
        if matches!(op.op, T_RECORD | T_APPEND) {
            if let Some(d) = &op.data {
                sink.write(d)?;
            }
        }
    }
    Ok(())
}

/// Capacity as the writer encodes it, for merged content `m` plus a `headroom`
/// hint: 0 for null content and for genuinely oversize content (stored linked),
/// else 16-aligned and big enough for header+content.
///
/// Headroom is a HINT; the record is the promise. A staged base reports
/// unlimited capacity, so an append can push the merged content to the plain
/// maximum and the requested headroom then overflows it. Clamping keeps the
/// record plain with an exact capacity, which is what a later `T_APPEND` needs.
/// Falling to capacity 0 there would make the writer acknowledge a commit the
/// decoder rejects as a garbage capacity (`cap_valid` allows 0 only when the
/// CONTENT itself is oversize), i.e. an unopenable log.
fn record_cap(m: Option<&[u8]>, headroom: usize) -> u64 {
    let Some(m) = m else { return 0 };
    // u64 throughout: the sum of a plain-sized record and a large headroom is
    // checked against the ceiling, never wrapped into it.
    let cap = (4u64 + m.len() as u64)
        .saturating_add(headroom as u64)
        .saturating_add(15)
        & !15u64;
    if cap > iv::MAX_CAPACITY as u64 {
        if 4 + m.len() as u64 <= iv::MAX_CAPACITY as u64 {
            iv::MAX_CAPACITY as u64
        } else {
            0 // genuinely oversize content: stored as a linked chain
        }
    } else {
        cap
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
        // pure staged preallocation could append+force a section after `close`
        // completed, since applying it does not touch the inner store.
        let mut st = self.write_open()?;
        st.commit_locked(&self.closed)
    }

    fn compact(&self) -> Result<()> {
        self.checkpoint()
    }

    fn close(&self) -> Result<()> {
        // Acquire the write lock BEFORE publishing `closed`, so an in-flight
        // commit that rechecks `closed` under the lock observes the close
        // atomically (no append after close). Any op still runs to completion
        // first (it holds the lock); we then win it and shut down.
        let mut st = self.st.write();
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        // D2: the namespace deletion runs while the store lock is STILL HELD —
        // close-then-delete would let a second opener acquire the namespace and
        // have its live segments deleted underneath it.
        let deleted = if self.delete_on_close.load(Ordering::Acquire) && !st.read_only {
            st.segs.delete_namespace()
        } else {
            Ok(())
        };
        st.segs.close();
        let inner = st.inner.close();
        deleted?;
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
        // Fail at update time, not commit time. Oversize CONTENT is fine (stored
        // linked at commit) — but content that fits a plain record must also fit
        // with its headroom, because linked records take no appends and silently
        // going linked would break the guarantee the headroom was asked for.
        if let Some(b) = &bytes {
            let plain = 4u64 + b.len() as u64;
            if plain <= iv::MAX_CAPACITY as u64
                && (plain.saturating_add(headroom as u64).saturating_add(15) & !15u64)
                    > iv::MAX_CAPACITY as u64
            {
                return Err(DbError::RecordTooLarge);
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
        let rid = recid.get();
        let was_staged = st.staged.contains_key(&rid);
        st.staged_for_write(rid)?;
        let base_live =
            !st.staged.get(&rid).unwrap().base_set && st.inner.rec_state(rid)? == STATE_LIVE;
        if base_live {
            let cap_rem = st.inner.capacity_remaining(recid)?;
            let appends_len = st.staged.get(&rid).unwrap().appends_len;
            if appends_len + data.len() > cap_rem {
                // REFUSED is a no-op, so it must stage NOTHING. An empty
                // `Staged` left behind here used to be classified as T_PREALLOC
                // at commit: it burnt an LSN, and a prealloc naming a
                // content-live record is exactly what §4.2 rejects on replay.
                if !was_staged {
                    st.staged.remove(&rid);
                }
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
        // NEITHER IDENTITY MOVES ON ROLLBACK, and that is the whole rule: both
        // are set only by a committed apply, so a transaction that never
        // committed established nothing to undo. The `created` recids deleted
        // just above were preallocated in inner at op time but never logged, so
        // they hold no identity either.
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

fn serialize<R>(value: &R, ser: &(impl Serializer<R> + Sync)) -> Vec<u8> {
    let mut out = DataOutput2::with_capacity(ser.size_hint() + 4);
    ser.serialize(&mut out, value);
    out.into_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ser::serializers::LongSer;
    use crate::store::wal_write::{WalIoEvent, WalOpKind};
    use std::sync::atomic::AtomicU64;
    use std::sync::Mutex;

    const L: LongSer = LongSer;

    fn scratch(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "mapdb5_walwrite_{}_{}_{}",
            std::process::id(),
            tag,
            n
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// Records every durability operation and, optionally, fails one of them —
    /// the two halves of Java's `WalIo`, which is where the ORDERING claims of
    /// W1-W7 and W9 stop being structural arguments about the source.
    /// Predicate deciding which operation the trace makes fail.
    type FailAt = Box<dyn Fn(&WalIoEvent, usize) -> bool + Send + Sync>;

    struct Trace {
        events: Mutex<Vec<WalIoEvent>>,
        fail: Mutex<Option<FailAt>>,
    }

    impl Trace {
        fn new() -> Arc<Trace> {
            Arc::new(Trace {
                events: Mutex::new(Vec::new()),
                fail: Mutex::new(None),
            })
        }

        fn fail_when(
            self: &Arc<Self>,
            f: impl Fn(&WalIoEvent, usize) -> bool + Send + Sync + 'static,
        ) {
            *self.fail.lock().unwrap() = Some(Box::new(f));
        }

        fn kinds(&self) -> Vec<WalOpKind> {
            self.events.lock().unwrap().iter().map(|e| e.kind).collect()
        }

        fn take(&self) -> Vec<WalIoEvent> {
            std::mem::take(&mut *self.events.lock().unwrap())
        }
    }

    impl WalIo for Trace {
        fn before(&self, e: &WalIoEvent) -> Result<()> {
            let mut ev = self.events.lock().unwrap();
            ev.push(*e);
            let n = ev.len() - 1;
            drop(ev);
            if let Some(f) = self.fail.lock().unwrap().as_ref() {
                if f(e, n) {
                    return Err(DbError::Io(std::io::Error::other(format!(
                        "injected failure at {:?}",
                        e.kind
                    ))));
                }
            }
            Ok(())
        }
    }

    fn open_traced(base: &Path, segment_bytes: u64, trace: &Arc<Trace>) -> Result<StoreWAL> {
        StoreWAL::open_cfg(
            base,
            WalOptions {
                segment_bytes,
                wal_io: Some(trace.clone() as Arc<dyn WalIo>),
                ..Default::default()
            },
        )
    }

    // ---------------------------------------------------------------- W1/W4

    #[test]
    fn every_section_is_forced_before_the_next_one_starts() {
        let dir = scratch("w1");
        let base = dir.join("s.db");
        let trace = Trace::new();
        let s = open_traced(&base, DEFAULT_SEGMENT_BYTES, &trace).unwrap();
        trace.take(); // discard the open's create events
        for i in 0..3i64 {
            s.put(&i, &L).unwrap();
            s.commit().unwrap();
        }
        assert_eq!(
            trace.kinds(),
            [
                WalOpKind::SecHeader,
                WalOpKind::SecBody,
                WalOpKind::ForceData,
                WalOpKind::SecHeader,
                WalOpKind::SecBody,
                WalOpKind::ForceData,
                WalOpKind::SecHeader,
                WalOpKind::SecBody,
                WalOpKind::ForceData,
            ],
            "a section's force must complete before the next section's header is written — \
             recovery's mid-log-rot inference is sound only under that"
        );
    }

    #[test]
    fn the_section_header_is_written_before_its_body() {
        let dir = scratch("w1b");
        let base = dir.join("s.db");
        let trace = Trace::new();
        let s = open_traced(&base, DEFAULT_SEGMENT_BYTES, &trace).unwrap();
        trace.take();
        s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        let ev = trace.take();
        assert_eq!(ev[0].kind, WalOpKind::SecHeader);
        assert_eq!(ev[1].kind, WalOpKind::SecBody);
        assert_eq!(
            ev[1].off,
            ev[0].off + 25,
            "the body starts immediately after the 25-byte header"
        );
        assert_eq!(ev[2].off, ev[1].off + ev[1].len, "the force covers both");
        assert_eq!(ev[0].tag, TAG_SECTION);
    }

    // ------------------------------------------------------------------ W2/W3

    #[test]
    fn a_rollover_seals_with_a_full_force_before_the_successor_is_created() {
        let dir = scratch("w3");
        let base = dir.join("s.db");
        let trace = Trace::new();
        let s = open_traced(&base, MIN_SEGMENT_BYTES, &trace).unwrap();
        s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        trace.take();
        // This commit finds the segment over the threshold and rolls first.
        s.put(&2i64, &L).unwrap();
        s.commit().unwrap();
        assert_eq!(
            trace.kinds(),
            [
                // W3: seal the full segment, SIZE-persisting force...
                WalOpKind::ForceFull,
                // ...W2: create → header → force(true) → directory fsync...
                WalOpKind::Create,
                WalOpKind::SegHeader,
                WalOpKind::ForceFull,
                WalOpKind::DirSync,
                // ...and only then may a section land in it.
                WalOpKind::SecHeader,
                WalOpKind::SecBody,
                WalOpKind::ForceData,
            ]
        );
        assert_eq!(s.segment_seqs(), vec![1, 2]);
    }

    // -------------------------------------------------------------------- W7

    #[test]
    fn a_torn_tail_is_truncated_then_forced_then_rotated() {
        // The A1 review deferred this: W7's force ordering is unobservable
        // without an I/O seam, and the port had none. It has one now.
        let dir = scratch("w7");
        let base = dir.join("s.db");
        let len_after_first;
        {
            let s = StoreWAL::open(&base).unwrap();
            s.put(&1i64, &L).unwrap();
            s.commit().unwrap();
            len_after_first = std::fs::metadata(base.parent().unwrap().join(seg_name(&base, 1)))
                .unwrap()
                .len();
            s.put(&2i64, &L).unwrap();
            s.commit().unwrap();
            s.close().unwrap();
        }
        std::fs::OpenOptions::new()
            .write(true)
            .open(base.parent().unwrap().join(seg_name(&base, 1)))
            .unwrap()
            .set_len(len_after_first + 3)
            .unwrap();

        let trace = Trace::new();
        let s = open_traced(&base, DEFAULT_SEGMENT_BYTES, &trace).unwrap();
        assert_eq!(
            trace.kinds(),
            [
                WalOpKind::Truncate,
                WalOpKind::ForceFull,
                WalOpKind::Create,
                WalOpKind::SegHeader,
                WalOpKind::ForceFull,
                WalOpKind::DirSync,
            ],
            "truncate, then a SIZE-persisting force, then rotate — so no later append \
             reuses the torn segment's checksum domain"
        );
        assert_eq!(s.segment_seqs(), vec![1, 2]);
    }

    fn seg_name(base: &Path, seq: i64) -> String {
        format!(
            "{}.wal.{seq:016x}",
            base.file_name().unwrap().to_str().unwrap()
        )
    }

    // -------------------------------------------------------------------- W9

    /// Every point at which a section write can fail must fail the store CLOSED.
    /// v1 returned the error with the store open, and the caller's retry then
    /// wrote a complete, forced, ACKNOWLEDGED section after the partial bytes —
    /// which the next open read as mid-log garbage and discarded.
    #[test]
    fn a_failure_at_any_write_point_fails_the_store_closed() {
        for kind in [
            WalOpKind::SecHeader,
            WalOpKind::SecBody,
            WalOpKind::ForceData,
        ] {
            let dir = scratch("w9");
            let base = dir.join("s.db");
            let a;
            {
                let trace = Trace::new();
                let s = open_traced(&base, DEFAULT_SEGMENT_BYTES, &trace).unwrap();
                a = s.put(&11i64, &L).unwrap();
                s.commit().unwrap();
                trace.fail_when(move |e, _| e.kind == kind);
                s.put(&22i64, &L).unwrap();
                assert!(s.commit().is_err(), "{kind:?}: the commit must fail");
                assert!(s.is_closed(), "{kind:?}: and the store must be closed");
                // No retry can append after the partial bytes.
                assert!(matches!(s.commit(), Err(DbError::StoreClosed)));
                assert!(matches!(s.put(&33i64, &L), Err(DbError::StoreClosed)));
            }
            // The durable prefix is intact and reopens.
            let s = StoreWAL::open(&base).unwrap();
            assert_eq!(s.get(a, &L).unwrap(), Some(11), "{kind:?}");
            s.verify().unwrap();
        }
    }

    #[test]
    fn a_failed_rollover_fails_the_store_closed() {
        let dir = scratch("w9roll");
        let base = dir.join("s.db");
        let a;
        {
            let trace = Trace::new();
            let s = open_traced(&base, MIN_SEGMENT_BYTES, &trace).unwrap();
            a = s.put(&1i64, &L).unwrap();
            s.commit().unwrap();
            trace.fail_when(|e, _| e.kind == WalOpKind::Create);
            s.put(&2i64, &L).unwrap();
            assert!(s.commit().is_err());
            assert!(s.is_closed());
        }
        let s = StoreWAL::open(&base).unwrap();
        assert_eq!(s.get(a, &L).unwrap(), Some(1));
        assert_eq!(s.segment_seqs(), vec![1]);
        s.put(&2i64, &L).unwrap();
        s.commit().unwrap();
        s.checkpoint().unwrap();
        // Sequence 2 is handed out again, and that is correct rather than a
        // relaxation of W6: the failed create left NOTHING at that name — it
        // failed before the file existed, and a create that fails after it does
        // unlinks it — so no crash image can hold a partially created segment
        // there. W6's burn is what stops a name being reused while THIS state is
        // live; what makes a name permanently burnt is the file surviving in the
        // directory, where enumeration finds it and counts it (R2).
        assert_eq!(s.segment_seqs(), vec![1, 2]);
    }

    #[test]
    fn a_failed_directory_fsync_during_rollover_fails_closed() {
        // The commit acknowledgement rule is "the section is forced AND the
        // segment's directory entry is durable", so a failed directory fsync
        // cannot be swallowed: the segment could vanish with the commit in it.
        let dir = scratch("w9dir");
        let base = dir.join("s.db");
        let trace = Trace::new();
        let s = open_traced(&base, MIN_SEGMENT_BYTES, &trace).unwrap();
        s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        trace.fail_when(|e, _| e.kind == WalOpKind::DirSync);
        s.put(&2i64, &L).unwrap();
        assert!(s.commit().is_err());
        assert!(s.is_closed());
    }

    // --------------------------------------------------- the two-pass writer

    #[test]
    fn a_body_that_diverges_between_the_passes_is_refused_before_the_force() {
        use crate::store::wal_write::append_section;
        let dir = scratch("2pass");
        let base = dir.join("s.db");
        let mut set = WalSegmentSet::open(&base, false).unwrap();
        set.create_segment(1).unwrap();
        let closed = AtomicBool::new(false);
        let trace = Trace::new();
        let io: Option<Arc<dyn WalIo>> = Some(trace.clone());

        let mut pass = 0;
        let r = append_section(
            &mut set,
            DEFAULT_SEGMENT_BYTES,
            &io,
            &closed,
            TAG_SECTION,
            1,
            |sink| {
                pass += 1;
                // The measure pass sees four bytes, the write pass five.
                sink.write(&[1, 2, 3, 4])?;
                if pass == 2 {
                    sink.write(&[5])?;
                }
                Ok(())
            },
        );
        assert!(
            r.is_err(),
            "a nondeterministic body must not be acknowledged"
        );
        assert!(
            !trace.kinds().contains(&WalOpKind::ForceData),
            "the divergence check runs BEFORE the force, so nothing is acknowledged"
        );
    }

    #[test]
    fn the_two_passes_produce_the_length_and_crc_the_reader_verifies() {
        // Every entry framing path (the 64 KiB coalescing buffer, and payloads
        // that bypass it) in one section, read back through the real recovery.
        let dir = scratch("2passok");
        let base = dir.join("s.db");
        let mut want = Vec::new();
        {
            let s = StoreWAL::open(&base).unwrap();
            for len in [1usize, 63 << 10, 64 << 10, 200 << 10] {
                let v: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
                want.push((s.put(&v, &RawBytes).unwrap(), v));
            }
            s.commit().unwrap();
            s.close().unwrap();
        }
        let s = StoreWAL::open(&base).unwrap();
        for (r, v) in &want {
            assert_eq!(s.get(*r, &RawBytes).unwrap().as_ref(), Some(v));
        }
        s.verify().unwrap();
    }

    /// Content == value, so payloads round-trip byte-exactly through the log.
    struct RawBytes;
    impl crate::ser::Serializer<Vec<u8>> for RawBytes {
        fn serialize(&self, out: &mut DataOutput2, v: &Vec<u8>) {
            out.write_all(v);
        }
        fn deserialize(
            &self,
            input: &mut dyn crate::io::DataInput2,
            size: Option<usize>,
        ) -> Result<Vec<u8>> {
            let n = size.expect("framed size");
            let mut b = vec![0u8; n];
            input.read_fully(&mut b)?;
            Ok(b)
        }
        fn compare(&self, a: &Vec<u8>, b: &Vec<u8>) -> std::cmp::Ordering {
            a.cmp(b)
        }
        fn equals(&self, a: &Vec<u8>, b: &Vec<u8>) -> bool {
            a == b
        }
    }

    // ------------------------------------------------------ the identities

    #[test]
    fn an_append_is_stamped_with_the_lsn_of_the_image_it_extends() {
        // The delta cites its base by LSN; replay refuses a delta whose base is
        // absent, so a wrong stamp is a silent-loss channel. Here the base is
        // established at LSN 2 and the delta is written at LSN 3.
        let dir = scratch("ids");
        let base = dir.join("s.db");
        let s = StoreWAL::open(&base).unwrap();
        let r = s.put(&vec![7u8; 40], &RawBytes).unwrap();
        s.commit().unwrap(); // LSN 1: content image
        s.update_with_headroom(r, &vec![8u8; 40], &RawBytes, 64)
            .unwrap();
        s.commit().unwrap(); // LSN 2: the image the delta will cite
        assert_eq!(s.st.read().ids.content_base_lsn.get(&r.get()), Some(&2));
        s.append(r, &[9u8; 8]).unwrap();
        s.commit().unwrap(); // LSN 3: the delta
        assert_eq!(
            s.st.read().ids.content_base_lsn.get(&r.get()),
            Some(&2),
            "an append leaves the identity where it is: the base it extends is \
             still the one a later append must cite"
        );
        s.close().unwrap();

        let s = StoreWAL::open(&base).unwrap();
        let mut want = vec![8u8; 40];
        want.extend_from_slice(&[9u8; 8]);
        assert_eq!(s.get(r, &RawBytes).unwrap(), Some(want));
        assert_eq!(
            s.st.read().ids.content_base_lsn.get(&r.get()),
            Some(&2),
            "replay rebuilds the identity the writer held"
        );
    }

    #[test]
    fn a_delete_clears_the_identity_and_a_prealloc_leaves_no_content_base() {
        let dir = scratch("ids2");
        let base = dir.join("s.db");
        let s = StoreWAL::open(&base).unwrap();
        let r = s.put(&vec![1u8; 8], &RawBytes).unwrap();
        s.commit().unwrap();
        assert!(s.st.read().ids.content_base_lsn.contains_key(&r.get()));
        // A null-content update is self-contained but leaves NO content image:
        // keeping a stale base would let a later append cite a state in which
        // append is not valid.
        s.update(r, None::<&Vec<u8>>, &RawBytes).unwrap();
        s.commit().unwrap();
        assert!(!s.st.read().ids.content_base_lsn.contains_key(&r.get()));
        assert!(s.st.read().ids.state_lsn.contains_key(&r.get()));
        s.delete(r).unwrap();
        s.commit().unwrap();
        assert!(!s.st.read().ids.state_lsn.contains_key(&r.get()));
    }

    #[test]
    fn a_transaction_that_creates_and_deletes_one_recid_logs_nothing_for_it() {
        let dir = scratch("transient");
        let base = dir.join("s.db");
        let s = StoreWAL::open(&base).unwrap();
        let keep = s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        let gone = s.put(&2i64, &L).unwrap();
        s.delete(gone).unwrap();
        s.commit().unwrap();
        assert!(matches!(s.get(gone, &L), Err(DbError::GetVoid(_))));
        s.close().unwrap();
        let s = StoreWAL::open(&base).unwrap();
        assert_eq!(s.get(keep, &L).unwrap(), Some(1));
        assert!(matches!(s.get(gone, &L), Err(DbError::GetVoid(_))));
        s.verify().unwrap();
    }
}
