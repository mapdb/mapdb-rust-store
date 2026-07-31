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
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::direct::{STATE_LIVE, STATE_VOID};
use super::index_val as iv;
use super::lease::LeaseTable;
use super::wal_recover::{
    build_mark_body, parse_sec_hdr, recover, Identities, Recovered, SecIn, MARK_BODY_LEN, SEC_HDR,
    TAG_IMAGE, TAG_MARK, TAG_SECTION,
};
use super::wal_segments::{WalSegmentSet, SEG_HDR};
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
    read_only: bool,
    /// The two per-recid identities, maintained atomically with the committed
    /// apply of the entry that sets them — never before, never from staged
    /// state. Replay rebuilds them; the commit classifier reads them.
    ids: Identities,
    /// Committed self-contained entries over this store's lifetime — every one
    /// of which can obsolete an earlier image, which is what makes it the
    /// futility latch's staleness clock.
    committed_state_changes: i64,
    wal_io: Option<Arc<dyn WalIo>>,

    // ---- the cleaner (A3). Every field is touched under the write lock.
    /// The cycle in progress, or `None` when idle.
    cleaner: Option<Cleaner>,
    /// The active segment when the log first became due; no cycle may select at
    /// or above it, so reaching it means the whole log has been rewritten once.
    /// 0 = no episode in progress.
    clean_floor_seq: i64,
    /// The lifetime retired/written counters as the current episode began. Its
    /// achievement is `retired - written` over the episode — NET PROGRESS, not
    /// the change in log size, because concurrent commits move the log for
    /// reasons that have nothing to do with whether cleaning is working.
    episode_retired: i64,
    episode_written: i64,
    /// Images this episode has re-emitted, and the segments below the floor when
    /// its FIRST cycle opened — the range it set out to rewrite. The terminal is
    /// qualified against that range, not against what is left: a completed
    /// episode retires prefixes until nothing remains, so its final cycle is
    /// always as wide as the remainder.
    episode_records: i64,
    episode_segments: usize,
    /// Segments the NEXT cycle may retire in one go. A cycle that retires one
    /// segment pays for one mark, and a mark can cost more than a small segment
    /// holds: at the minimum segment size a cycle retires ~61 bytes and appends
    /// ~107, so one-at-a-time cleaning grows the log forever on a log a single
    /// wide pass would collapse.
    cycle_width: usize,
    cycle_retired_at: i64,
    cycle_written_at: i64,
    /// The cycle now open runs at the WIDEST width available to it. Only an
    /// episode whose last cycle was saturated may arm the latch: a futile narrow
    /// cycle is evidence about the width, not about the log.
    cycle_saturated: bool,
    last_cycle_saturated: bool,
    /// Log size when an episode completed its whole range WITHOUT shrinking the
    /// log — the configured ratio is unachievable. Latched so it is not retried
    /// on every commit. 0 = not latched.
    futile_at_bytes: u64,
    /// The target when the latch armed; a materially LOWER one re-arms cleaning.
    futile_at_target: u64,
    /// `committed_state_changes` when the latch armed, and the images the futile
    /// episode re-emitted. A latch is a proof about the log as it stood, and
    /// commits invalidate proofs: a mass delete of null-content or preallocated
    /// records obsoletes every image in the log while moving neither the log's
    /// size nor the target.
    futile_at_changes: i64,
    futile_records: i64,
    /// Lifetime cleaner accounting, both halves — bytes re-emitted, bytes retired.
    cleaner_bytes_written: i64,
    cleaner_bytes_retired: i64,
    /// Fault injection, and the only reason it exists: W10 is a check on phase
    /// 1's loop, so a suite that cannot make that loop DROP a record cannot tell
    /// a working W10 from one that passes because nothing ever fails it.
    /// Dropping a recid here is precisely the under-re-emission W10 is for.
    #[cfg(test)]
    drop_recid_from_publish: u64,
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
                read_only: opts.read_only,
                ids: identities,
                committed_state_changes: 0,
                wal_io: opts.wal_io,
                cleaner: None,
                clean_floor_seq: 0,
                episode_retired: 0,
                episode_written: 0,
                episode_records: 0,
                episode_segments: 0,
                cycle_width: 1,
                cycle_retired_at: 0,
                cycle_written_at: 0,
                cycle_saturated: false,
                last_cycle_saturated: false,
                futile_at_bytes: 0,
                futile_at_target: 0,
                futile_at_changes: 0,
                futile_records: 0,
                cleaner_bytes_written: 0,
                cleaner_bytes_retired: 0,
                #[cfg(test)]
                drop_recid_from_publish: 0,
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
        let mut st = self.st.write();
        st.min_log_bytes = bytes;
        // A configuration change invalidates every observation the current
        // episode made, latch included.
        st.abandon_episode();
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
        let mut st = self.st.write();
        st.space_amplification = factor;
        st.abandon_episode();
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
    /// Rolling first is what makes it a whole-log clean; the cycle then runs
    /// unbudgeted, because the caller asked for all of it. This is the
    /// incremental cleaner with its budget set to "everything" — the only sense
    /// in which a whole-store checkpoint still exists. The v1 whole-file
    /// rewrite, its `.ckpt` temp and its rename commit point are gone.
    ///
    /// Staged (uncommitted) mutations are untouched: they exist only in memory
    /// and are not part of any log.
    pub fn checkpoint(&self) -> Result<()> {
        let mut st = self.write_open()?;
        st.clean_whole_log(&self.closed)
    }

    /// Whether the futility latch is armed: the log is fully compacted and still
    /// cannot meet the configured space-amplification ratio, so automatic
    /// cleaning has stopped retrying it. Released by a quiet trigger, a
    /// configuration change, a further target's worth of growth, a materially
    /// lower target, or enough committed churn.
    pub fn cleaning_exhausted(&self) -> bool {
        self.st.read().cleaning_exhausted()
    }

    /// Lifetime cleaner accounting: `(bytes re-emitted, bytes retired)`.
    pub fn cleaner_bytes(&self) -> (i64, i64) {
        let st = self.st.read();
        (st.cleaner_bytes_written, st.cleaner_bytes_retired)
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

    // ---------------------------------------------------------- the trigger

    /// The size the log is allowed to reach.
    ///
    /// `live` is the inner store's `get_current_size()` — allocated bytes minus
    /// reclaimed — which is PAGE-GRANULAR: it includes the header and rounds to
    /// slices, so it reports about 2 MiB for a store holding 200 bytes. The
    /// ratio is therefore a log-versus-footprint ratio, exact at scale and
    /// conservative below a few MiB, where it DELAYS cleaning rather than
    /// hastening it. That direction is the safe one, but it means
    /// `set_min_log_bytes` is not an absolute cap on a tiny store: below ~2 MiB
    /// of footprint the amplification term, not the floor, decides.
    fn cleaning_target(&self) -> u64 {
        let live = self.inner.get_current_size();
        let scaled = live.saturating_mul(self.space_amplification as u64);
        self.min_log_bytes.max(scaled)
    }

    /// It bounds SPACE, not WRITE amplification, and the difference is not
    /// academic: cleaning strictly the oldest segment is FIFO, not cost-benefit,
    /// so for a cold-head workload that segment is ~100% live and re-emitting it
    /// buys nothing this cycle. Oldest-first is kept as an explicit trade-off
    /// with the pathological case named rather than hidden.
    fn cleaning_due(&self) -> bool {
        self.min_log_bytes > 0 && self.segs.log_bytes() > self.cleaning_target()
    }

    /// The hard ceiling: the log is past TWICE what the trigger allows, so
    /// bounding the pause has stopped being the priority and the committing
    /// writer participates until it is back under.
    fn cleaning_urgent(&self) -> bool {
        self.min_log_bytes > 0 && self.segs.log_bytes() > self.cleaning_target().saturating_mul(2)
    }

    /// Whether the futility latch is armed — a healthy, fully compacted log that
    /// cannot meet the configured ratio.
    fn cleaning_exhausted(&self) -> bool {
        self.futile_at_bytes > 0
    }

    /// Releases the futility latch, whatever armed it.
    fn clear_latch(&mut self) {
        self.futile_at_bytes = 0;
        self.futile_at_target = 0;
        self.futile_at_changes = 0;
        self.futile_records = 0;
    }

    /// Abandons the episode WITHOUT judging it — for a configuration change or
    /// an explicit `checkpoint()`, after which nothing it observed is still
    /// about the same store.
    fn abandon_episode(&mut self) {
        self.clear_latch();
        self.clean_floor_seq = 0;
        self.episode_retired = 0;
        self.episode_written = 0;
        self.episode_records = 0;
        self.episode_segments = 0;
        self.cycle_width = 1;
        self.last_cycle_saturated = false;
    }

    /// Ends the episode, having walked its whole range. Called ONLY on
    /// completion, because only a completed episode says anything: it is the
    /// bounded window the no-net-progress terminal is measured over.
    fn end_episode(&mut self) {
        let futile = self.clean_floor_seq != 0
            && !paid_for_itself(
                self.cleaner_bytes_retired - self.episode_retired,
                self.cleaner_bytes_written - self.episode_written,
            );
        if futile && self.last_cycle_saturated {
            self.futile_at_bytes = self.segs.log_bytes().max(1);
            self.futile_at_target = self.cleaning_target();
            self.futile_at_changes = self.committed_state_changes;
            self.futile_records = self.episode_records.max(1);
        }
        self.clean_floor_seq = 0;
        self.episode_retired = 0;
        self.episode_written = 0;
        self.episode_records = 0;
        self.episode_segments = 0;
        // Reset WITH the episode, not across it: a guard that outlives what it
        // describes would let an episode that did NO work (nothing below its
        // floor, so `futile` is trivially true) arm the terminal on the strength
        // of a PREVIOUS episode's wide last cycle. `cycle_width` does persist
        // deliberately — the width a log needs is a property of the log.
        self.last_cycle_saturated = false;
    }

    /// Opens a cycle over the oldest retirable segments IF the trigger is live.
    /// Returns whether a cycle is now open.
    ///
    /// An EPISODE begins by SEALING the active segment and taking its fresh
    /// successor as the floor. No cycle may select at or above that floor, so
    /// everything that existed when the episode began is retirable and nothing
    /// the episode itself writes is: once the lowest present segment reaches the
    /// floor, the episode has rewritten the whole log exactly once. Sealing is
    /// what makes that true — using the PRE-EXISTING active segment as the floor
    /// leaves it untouched, and it also subsumes the single-segment case, where
    /// a `segment_bytes` above the trigger would otherwise leave a log growing
    /// forever with no candidate at all.
    ///
    /// The terminal is FUTILITY, not reaching the floor. An episode that
    /// reclaimed bytes and ended is a success, and the right response is another
    /// episode; latching on "reached the floor" suppresses cleaning after every
    /// SUCCESSFUL one — including above the hard ceiling, where the writer is
    /// supposed to be made to participate, so the ceiling would not be one.
    fn begin_cycle_if_due(&mut self, closed: &AtomicBool) -> Result<bool> {
        if !self.cleaning_due() {
            // The trigger went quiet. The EPISODE is not over: keeping its floor
            // across a dip is what stops a workload hovering around the target
            // paying a fresh seal, create and directory fsync every few commits.
            // Only the latch is released, because a quiet trigger means the
            // situation changed.
            self.clear_latch();
            return Ok(false);
        }
        if self.futile_at_bytes > 0 {
            let room = self.cleaning_target();
            let retry = self.futile_at_bytes.saturating_add(room);
            // A MATERIAL drop, not any drop: the target is the inner store's
            // footprint, and an ordinary update moves it by a couple of hundred
            // bytes in either direction as the allocator reuses extents.
            let dropped = self.futile_at_target - (self.futile_at_target >> 3);
            let grew = self.segs.log_bytes() >= retry;
            let shrank = room <= dropped;
            // ...and neither of those can see a state-only mass delete, which
            // obsoletes every image in the log while moving neither number.
            let churned =
                self.committed_state_changes - self.futile_at_changes >= self.futile_records;
            if !grew && !shrank && !churned {
                return Ok(false);
            }
            self.clear_latch();
        }
        if self.clean_floor_seq == 0 {
            // The seal is CLEANING's cost, not the writer's: this rollover
            // exists only to give the episode a floor, and its 36-byte successor
            // header would otherwise be invisible to every tick.
            let (log_before, retired_before) = (self.segs.log_bytes(), self.cleaner_bytes_retired);
            self.roll_active_if_nonempty(closed)?;
            self.charge_cleaner(log_before, retired_before);
            self.clean_floor_seq = self.segs.active().expect("writable store").seq;
            self.episode_retired = self.cleaner_bytes_retired;
            self.episode_written = self.cleaner_bytes_written;
        }
        // BINARY search for the floor, not a walk: the walk is O(segments below
        // the floor) and runs at every cycle start, which is every commit or two.
        let all = self.segs.segments();
        let below = all.partition_point(|s| s.seq < self.clean_floor_seq);
        if below == 0 || below >= all.len() {
            self.end_episode(); // the episode has rewritten everything it could
            return Ok(false);
        }
        if self.episode_segments == 0 {
            self.episode_segments = below;
        }
        // One segment per cycle, or as many as the width search has reached. A
        // wide cycle is not a wider PAUSE — it is still driven in budgeted ticks
        // — and it is the only way to amortise one mark over many segments.
        let width = self.cycle_width.clamp(1, CYCLE_WIDTH_CAP).min(below);
        // SATURATED = as wide as this episode is ever allowed to go, measured
        // against the range it STARTED with. Against the remainder it would be
        // vacuous: a completed episode's final cycle always covers what is left.
        self.cycle_saturated = width >= CYCLE_WIDTH_CAP.min(self.episode_segments);
        let target = self.segs.segments()[width - 1].seq;
        self.start_cycle(target);
        Ok(true)
    }

    /// Opens a cycle retiring everything at or below `target_seq`. The caller
    /// must have established that a segment above it exists.
    ///
    /// **O(1).** Candidates are discovered by WALKING THE RETIRING RANGE itself,
    /// one bounded unit at a time — computing them from the `state_lsn` map
    /// would be O(live recids) under the write lock, and for a large store far
    /// more work than the segment being retired even contains.
    ///
    /// The walk finds every candidate and no others: a recid needs re-emission
    /// exactly when `state_lsn[R] <= boundary_lsn`, that value IS the LSN of its
    /// newest self-contained entry, and the retained log begins at
    /// `boundary_lsn + 1` — so that entry is inside the range and the recid
    /// appears in the walk. The FILTER stays over `state_lsn`, which is what
    /// keeps a recid merely allocated by an in-flight transaction out of the
    /// set: it has no committed entry and so no `state_lsn` at all.
    ///
    /// No surviving `T_APPEND` can be orphaned by this. The worry is a delta
    /// ABOVE the range whose base lies INSIDE it; it is unreachable, because
    /// `content_base_lsn[R] <= boundary < state_lsn[R]` cannot happen — every
    /// entry that raises `state_lsn` either moves `content_base_lsn` to the SAME
    /// LSN or clears it. So a delta whose base is in the range belongs to a
    /// candidate, which is re-emitted with that delta already folded into its
    /// content; replay then skips the stranded delta and the image supersedes
    /// it, which is what the skip audit is built to tolerate.
    fn start_cycle(&mut self, target_seq: i64) {
        self.cycle_retired_at = self.cleaner_bytes_retired;
        self.cycle_written_at = self.cleaner_bytes_written;
        let successor = self
            .segs
            .segments()
            .iter()
            .find(|s| s.seq > target_seq)
            // K4: a mark may not authorize removing its own segment, so a cycle
            // retiring everything has nowhere to record itself.
            .expect("a cycle always leaves a segment above its target");
        self.cleaner = Some(Cleaner::new(target_seq, successor.header_first_lsn()));
    }

    /// Charges one unit of cleaning and returns what it charged: what the log
    /// grew by, plus what the unit retired (which shrank it).
    ///
    /// Both halves of the accounting must be in the SAME unit, and the unit is
    /// FILE bytes. The sections a tick appends are not what it costs the log: an
    /// append that rolls over creates a segment header, and the mark that closes
    /// a cycle usually lands in a segment of its own. Charging section bytes
    /// against `retired`, which sums whole file lengths, reports progress on an
    /// episode that is growing the log — a treadmill that never reaches its
    /// terminal because it never stops "progressing".
    fn charge_cleaner(&mut self, log_before: u64, retired_before: i64) -> i64 {
        // Charges NOTHING once the store is closed: a section append can fail
        // the store from inside a unit, after which the segment set is empty and
        // the delta would be a large negative charge.
        if self.segs.segments().is_empty() && self.cleaner.is_none() {
            return 0;
        }
        let charge = self.segs.log_bytes() as i64 - log_before as i64
            + (self.cleaner_bytes_retired - retired_before);
        self.cleaner_bytes_written += charge;
        charge
    }

    // ------------------------------------------------------------ the phases

    /// Phase 1, one bounded unit: walk the retiring range and publish, as a
    /// single `'C'` section, an image of every record met whose state still
    /// lives inside it. Returns `(bytes written, entries walked)`.
    ///
    /// **Check, copy and publish are one serialized unit** — the whole method
    /// runs under the WAL write lock — and that is correctness, not style. Split
    /// them and the cleaner sees R live, copies image I, a committer writes
    /// update U, and the cleaner then appends a stale `C(R, I)` AFTER U: replay
    /// resurrects the old value.
    ///
    /// One section per unit, and every section is forced before the next is
    /// appended: recovery infers mid-log rot from "a valid section follows an
    /// invalid one", which is sound only while that holds.
    fn publish_unit(
        &mut self,
        closed: &AtomicBool,
        budget: &Budget,
        written_so_far: i64,
        records_so_far: usize,
    ) -> Result<(i64, usize)> {
        let byte_room = if budget.max_bytes > 0 {
            (budget.max_bytes as i64 - written_so_far).max(1) as u64
        } else {
            1 << 20
        };
        let rec_room = if budget.max_records > 0 {
            budget.max_records.saturating_sub(records_so_far).max(1)
        } else {
            SCAN_UNIT_ENTRIES
        };
        let cap = byte_room.min(1 << 20);
        let mut out = DataOutput2::with_capacity(cap.clamp(4096, 1 << 16) as usize);
        let lsn = self.next_lsn;
        // (recid, carries content) per emitted image, applied to the identities
        // only after the section is durable.
        let mut emitted: Vec<(u64, bool)> = Vec::new();
        // Recids already encoded into THIS section. A recid met again in a later
        // section of the range is normally filtered by its own raised
        // `state_lsn` — but the identities move only once the section is
        // durable, so within one unfinished batch that filter has not fired yet,
        // and the decoder's one-entry-per-recid-per-section rule would be
        // violated. Replay refuses such a section outright.
        let mut in_batch: HashSet<u64> = HashSet::new();

        // Disjoint borrows: the scan mutates the namespace (it releases handles)
        // while the visitor reads the identities, the staged set and the inner
        // store.
        #[cfg(test)]
        let drop_recid = self.drop_recid_from_publish;
        let WalState {
            segs,
            inner,
            ids,
            staged,
            cleaner,
            ..
        } = &mut *self;
        let c = cleaner.as_mut().expect("a cycle is open");
        let boundary = c.boundary_lsn;
        let steps = scan_unit(c, segs, rec_room, &mut |recid| {
            #[cfg(test)]
            if recid == drop_recid {
                return Ok(true);
            }
            if in_batch.contains(&recid) {
                return Ok(true);
            }
            // Across units no dedup is needed: a recid re-emitted by an earlier
            // unit has a `state_lsn` above the boundary, exactly like one a
            // concurrent commit re-homed. Both are simply not candidates.
            let Some(&sl) = ids.state_lsn.get(&recid) else {
                return Ok(true);
            };
            if sl > boundary {
                return Ok(true);
            }
            if staged.get(&recid).is_some_and(|s| s.created) {
                // A recid an in-flight transaction allocated has no committed
                // entry and therefore no `state_lsn`; the allocator cannot hand
                // out a recid that is committed-live. Both at once would mean
                // inner's slot has been overwritten with a preallocation while
                // committed content is still attested — re-emitting either way
                // would be a guess.
                return Err(DbError::corrupt_msg(format!(
                    "WAL cleaner: recid {recid} has committed state at LSN {sl} and is also \
                     allocated by an in-flight transaction"
                )));
            }
            let mut content = false;
            let live = inner.wal_snapshot_one(recid, |prealloc, cap_bytes, data| {
                if prealloc {
                    out.write_byte(T_PREALLOC as i32);
                    out.pack_long(recid);
                } else {
                    out.write_byte(T_RECORD as i32);
                    out.pack_long(recid);
                    out.pack_long(cap_bytes as u64);
                    match &data {
                        None => out.pack_long(0),
                        Some(d) => {
                            out.pack_long(d.len() as u64 + 1);
                            out.write_all(d);
                        }
                    }
                    content = data.is_some();
                }
                Ok(())
            })?;
            if !live {
                // `state_lsn` present means "committed non-void", and inner IS
                // the committed state, so this cannot happen without the
                // identity map having diverged from the store. Refuse rather
                // than retire a segment whose contents were not re-homed.
                return Err(DbError::corrupt_msg(format!(
                    "WAL cleaner: recid {recid} has committed state at LSN {sl} but the inner \
                     store holds nothing for it"
                )));
            }
            emitted.push((recid, content));
            in_batch.insert(recid);
            // An image larger than one unit's allowance still goes whole: a
            // record cannot be split across sections.
            Ok((out.buf.len() as u64) < cap)
        })?;

        let mut written = 0i64;
        if !emitted.is_empty() {
            let body = std::mem::take(&mut out.buf);
            let r = append_section(
                &mut self.segs,
                self.segment_bytes,
                &self.wal_io,
                closed,
                TAG_IMAGE,
                lsn,
                |sink| sink.write(&body),
            );
            if let Err(e) = r {
                self.fail_closed(closed);
                return Err(e);
            }
            self.next_lsn += 1;
            written = SEC_HDR as i64 + body.len() as i64;
            // IMAGES, not entries walked: the staleness clock compares the
            // store's committed self-contained entries against the live set the
            // futile episode had to preserve, and entries walked is neither — it
            // counts the garbage too.
            self.episode_records += emitted.len() as i64;
            // Identities move by the §4.2 row of each entry the section
            // contains, AFTER it is durable and atomically with it.
            for (recid, content) in &emitted {
                if *content {
                    self.ids.content(*recid, lsn);
                } else {
                    self.ids.state_only(*recid, lsn);
                }
            }
        }
        let c = self.cleaner.as_mut().expect("a cycle is open");
        if c.range_done {
            c.published = true;
            c.rewind(); // also clears range_done, for the verify walk
        }
        Ok((written, steps))
    }

    /// Phase 2 — W10, one bounded unit: re-walk the retiring range and assert
    /// that every recid it mentions has been re-homed above it.
    ///
    /// **A mark cannot be made self-verifying after the unlink**, because the
    /// evidence is exactly what is being deleted: a manifest of what was
    /// re-homed cannot prove completeness, since an omitted recid is omitted
    /// from the manifest too. The verifiable moment is here, while the segments
    /// still exist. What it buys is that an under-re-emission — a dropped
    /// `T_PREALLOC`, a dropped null-content record — fails loudly BEFORE the
    /// data is destroyed, instead of silently until the free-recid rebuild
    /// re-issues the recid and a later allocation collides with it. The skip
    /// audit cannot see this class at all: a record wholly contained in the
    /// range with no surviving append leaves no entry to skip.
    ///
    /// Chunking it across ticks is sound because the predicate is MONOTONE: once
    /// `state_lsn[R]` is absent-or-above, only a new self-contained entry at a
    /// still higher LSN can change it.
    ///
    /// Its boundary, stated so it is not over-trusted: W10 is sufficient for
    /// OMISSION, not for image FIDELITY. It asks "was this recid re-homed?", and
    /// a cleaner that emitted a CRC-valid but semantically wrong image raises
    /// `state_lsn` just the same and passes.
    fn verify_unit(&mut self, budget: &Budget, records_so_far: usize) -> Result<usize> {
        let rec_room = if budget.max_records > 0 {
            budget.max_records.saturating_sub(records_so_far).max(1)
        } else {
            SCAN_UNIT_ENTRIES
        };
        let WalState {
            segs, ids, cleaner, ..
        } = &mut *self;
        let c = cleaner.as_mut().expect("a cycle is open");
        let (boundary, target, log_start) = (c.boundary_lsn, c.target_seq, c.log_start_lsn);
        let steps = scan_unit(
            c,
            segs,
            rec_room,
            &mut |recid| match ids.state_lsn.get(&recid) {
                Some(&sl) if sl <= boundary => Err(DbError::corrupt_msg(format!(
                    "WAL cleaner would retire through segment {target} while recid {recid} still \
                     has its only self-contained entry at LSN {sl} (the log would begin at \
                     {log_start}): refusing to write the clean mark. Nothing has been deleted; \
                     the durable log is intact."
                ))),
                _ => Ok(true),
            },
        )?;
        if c.range_done {
            c.verified = true;
        }
        Ok(steps)
    }

    /// Closes a cycle: append the forced `'K'`, then unlink.
    ///
    /// **Ordering is the whole content of this method.** Every re-emitted image
    /// was forced as it was written and every rollover sealed its predecessor
    /// with a size-persisting force, so no mark ever attests bytes that were not
    /// forced (W1). The `'K'` is forced before the unlink (W5): a failed unlink
    /// is a leak the next open retries, never permission to advance an unproven
    /// mark. Every crash point in between is state-preserving — before the mark
    /// the retiring segments replay and cleaning simply re-runs, after it they
    /// are already superseded.
    fn finish_cycle(&mut self, closed: &AtomicBool) -> Result<i64> {
        let c = self.cleaner.as_ref().expect("a cycle is open");
        let (target, log_start) = (c.target_seq, c.log_start_lsn);
        self.append_mark(closed, target, log_start)?;
        let retired: u64 = self
            .segs
            .segments()
            .iter()
            .take_while(|s| s.seq <= target)
            .map(|s| s.file_len)
            .sum();
        if let Err(e) = self.segs.unlink_through(target) {
            self.fail_closed(closed);
            return Err(e);
        }
        self.cleaner_bytes_retired += retired as i64;
        self.cleaner = None;
        Ok(SEC_HDR as i64 + MARK_BODY_LEN)
    }

    /// One cleaning tick: re-emit, then verify (W10), then close the cycle — as
    /// far as `budget` allows, stopping at the first limit reached. Returns the
    /// bytes written. At most ONE cycle is closed per tick, so a caller driving
    /// this in a loop always sees the cycle boundary and can re-decide.
    fn clean_tick(&mut self, closed: &AtomicBool, budget: &Budget) -> Result<i64> {
        let t0 = std::time::Instant::now();
        let mut written = 0i64;
        let mut records = 0usize;
        let mut closed_cycle = false;
        let mut result = Ok(());
        while self.cleaner.is_some() {
            let (log_before, retired_before) = (self.segs.log_bytes(), self.cleaner_bytes_retired);
            let c = self.cleaner.as_ref().expect("checked");
            let (published, verified) = (c.published, c.verified);
            let step = if !published {
                self.publish_unit(closed, budget, written, records)
                    .map(|(_, steps)| steps)
            } else if !verified {
                self.verify_unit(budget, records)
            } else {
                match self.finish_cycle(closed) {
                    Ok(_) => {
                        closed_cycle = true;
                        written += self.charge_cleaner(log_before, retired_before);
                        break;
                    }
                    Err(e) => Err(e),
                }
            };
            match step {
                Ok(n) => records += n,
                Err(e) => {
                    // A unit refused — W10 caught an under-re-emission, or an
                    // identity map disagreed with the inner store. The cursor
                    // has ALREADY stepped past the entry that refused (it
                    // advances before the visitor runs), so a later tick would
                    // resume beyond it, find nothing wrong in what remains, and
                    // write the mark: the loud refusal would become exactly the
                    // silent loss it exists to prevent. Rewind, so any retry
                    // re-walks the range from the bottom and reaches the same
                    // verdict — or a genuinely different one, if a commit has
                    // since re-homed the recid, which makes the retirement safe
                    // for real.
                    if let Some(c) = self.cleaner.as_mut() {
                        c.rewind();
                    }
                    result = Err(e);
                    break;
                }
            }
            written += self.charge_cleaner(log_before, retired_before);
            if budget.max_records > 0 && records >= budget.max_records {
                break;
            }
            if budget.max_bytes > 0 && written >= budget.max_bytes as i64 {
                break;
            }
            if budget.max_nanos > 0 && t0.elapsed().as_nanos() as u64 >= budget.max_nanos {
                break;
            }
        }
        result?;
        if closed_cycle {
            // The cycle is closed and its whole cost is now charged, so this is
            // the first moment its net is knowable. THREE bands, not two:
            // halving on any gain oscillates around the break-even width,
            // because a cycle that barely pays is not evidence the width is too
            // big. Widen when it does not pay, hold when it pays modestly, and
            // give width back only when it pays HANDSOMELY.
            let cost = self.cleaner_bytes_written - self.cycle_written_at;
            let gain = self.cleaner_bytes_retired - self.cycle_retired_at - cost;
            if gain <= cost >> 3 {
                self.cycle_width = CYCLE_WIDTH_CAP.min(self.cycle_width.max(1) * 2);
            } else if gain > cost >> 1 {
                self.cycle_width = (self.cycle_width / 2).max(1);
            }
            self.last_cycle_saturated = self.cycle_saturated;
        }
        Ok(written)
    }

    /// Commit's inline clean. Gated on the TRIGGER alone, never on "a cycle is
    /// open": continuing an open cycle here regardless would mean the first
    /// commit after a background tick started one dragged it to completion
    /// synchronously, moving the work back onto the commit path. An abandoned
    /// cycle costs nothing durable — its images are forced and its retired
    /// segments simply stay, so the log carries duplicates until someone
    /// finishes it.
    fn auto_clean_locked(&mut self, closed: &AtomicBool) -> Result<()> {
        if !self.cleaning_due() {
            self.clear_latch(); // the floor outlives a dip; see begin_cycle_if_due
            return Ok(());
        }
        // ONE bounded slice per commit. This runs inside commit's write-lock
        // hold and cannot release it, so a loop here would be one uninterrupted
        // hold for the whole pass: the per-tick budget would bound an internal
        // iteration while the commit that triggered it still paid for all of
        // them, consecutively, with every reader and writer waiting.
        if self.cleaner.is_some() || self.begin_cycle_if_due(closed)? {
            self.clean_tick(closed, &FOREGROUND_BUDGET)?;
        }
        // The exception is the hard ceiling. Once the log has run away — past
        // twice its target — the writer participates until it is back under, and
        // the pause is accepted deliberately: an unbounded pause is the lesser
        // evil against an unbounded log.
        while self.cleaning_urgent()
            && (self.cleaner.is_some() || self.begin_cycle_if_due(closed)?)
        {
            self.clean_tick(closed, &FOREGROUND_BUDGET)?;
        }
        Ok(())
    }

    /// `checkpoint()`'s body: clean the log all the way down.
    ///
    /// Rolling first is what makes this a WHOLE-log clean — every
    /// section-bearing segment is then strictly below the active one, so a
    /// single cycle whose target is `active.seq - 1` retires all of them and its
    /// re-emission set is the whole committed store. One cycle, one mark, one
    /// unlink, through exactly the machinery a budgeted tick uses.
    fn clean_whole_log(&mut self, closed: &AtomicBool) -> Result<()> {
        while self.cleaner.is_some() {
            self.clean_tick(closed, &UNBOUNDED_BUDGET)?; // finish a partial cycle
        }
        let (log_before, retired_before) = (self.segs.log_bytes(), self.cleaner_bytes_retired);
        self.roll_active_if_nonempty(closed)?;
        self.charge_cleaner(log_before, retired_before);
        let target = self.segs.active().expect("writable store").seq - 1;
        // Below the first sequence number there is nothing to retire: the active
        // segment is the store's first and it is empty, so the log is already as
        // small as it can be.
        if target < super::wal_segments::FIRST_SEQ || self.segs.segments().len() < 2 {
            return Ok(());
        }
        self.start_cycle(target);
        while self.cleaner.is_some() {
            self.clean_tick(closed, &UNBOUNDED_BUDGET)?;
        }
        self.abandon_episode(); // an explicit full clean re-arms the automatic one
        Ok(())
    }

    /// Writes a `'K'` mark: the fact that everything at or below
    /// `cleaned_through_seq` may be removed, and where the retained log begins.
    /// Forced before any unlink (W5).
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

// ============================== the cleaner (A3) =============================

/// One cleaning tick's allowance. `0` means "no limit" in every field, which is
/// what an explicit `checkpoint()` runs under.
#[derive(Clone, Copy)]
struct Budget {
    max_records: usize,
    max_bytes: u64,
    max_nanos: u64,
}

/// The budget a COMMIT pays (D8, adopted from the reference verbatim).
///
/// These numbers are the deliverable, not a detail. Java measured the store-size
/// sweep that produced them: against the previous `(4096 records, 8 MiB, no time
/// bound)`, commits over 1 ms went 326 -> 1 and p99.9 744 µs -> 176 µs, for ~1%
/// of log high-water and ZERO extra device bytes. `max_nanos` is a SOFT ceiling
/// — checked between work units, so a single oversize image still runs whole —
/// which makes it a bound on how much work is STARTED, not a deadline.
const FOREGROUND_BUDGET: Budget = Budget {
    max_records: 256,
    max_bytes: 512 << 10,
    max_nanos: 500_000,
};

/// What `checkpoint()` runs under: no limit, because the caller asked for all of it.
const UNBOUNDED_BUDGET: Budget = Budget {
    max_records: 0,
    max_bytes: 0,
    max_nanos: 0,
};

/// Entries one scan unit walks when the budget names no record limit.
const SCAN_UNIT_ENTRIES: usize = 256;

/// Window for the two scans. Small on purpose: they read entry HEADERS and seek
/// over payloads, so a replay-sized window would read a megabyte to decode ten
/// bytes whenever entries are far apart, and the "bounded unit" would not be
/// bounded in device reads at all.
const SCAN_BUF: usize = 4096;

/// Ceiling on [`WalState::cycle_width`]. A cycle's CLOSE is not budgeted —
/// summing the retiring prefix, then closing, deleting and fsyncing every file
/// in it, all under the write lock — so an uncapped width buys mark amortisation
/// with an unbounded pause, which is the trade the incremental cleaner exists to
/// refuse.
const CYCLE_WIDTH_CAP: usize = 64;

/// A cleaning cycle in progress: retire every segment with `seq <= target_seq`
/// by re-emitting, above them, a self-contained image of every record whose
/// state still lives inside them. Resumable across ticks with any budget.
struct Cleaner {
    /// `cleanedThroughSeq` the closing `'K'` will attest.
    target_seq: i64,
    /// `logStartLsn` the closing `'K'` will attest — the successor's STATED
    /// start, read from its header rather than computed, so the number recovery
    /// compares against is the number the writer recorded.
    log_start_lsn: i64,
    /// The last LSN the retiring range accounts for, `log_start_lsn - 1`.
    ///
    /// Deriving the re-emission boundary from the number the mark will record —
    /// rather than from the retiring segment's own `last_lsn` — makes the
    /// writer's obligation and recovery's check two readings of ONE value, and
    /// it is total over the empty-segment case, where a `last_lsn` of 0 says
    /// nothing.
    boundary_lsn: i64,
    /// Phase 1 (re-emit) has walked the whole range.
    published: bool,
    /// Phase 2 (W10) has walked it again.
    verified: bool,

    // ---- the scan cursor: phase 1 uses it, then rewinds and phase 2 reuses it
    /// Index into `segs.segments()`; the retiring range is a prefix of it.
    seg: usize,
    /// Offset of the next SECTION to enter within `seg`.
    offset: u64,
    /// Offset of the next ENTRY inside the section being walked, or `None`
    /// between sections.
    entry_pos: Option<u64>,
    /// End of the section body being walked.
    body_end: u64,
    /// The current walk has reached the top of the retiring range.
    range_done: bool,
}

impl Cleaner {
    fn new(target_seq: i64, log_start_lsn: i64) -> Cleaner {
        Cleaner {
            target_seq,
            log_start_lsn,
            boundary_lsn: log_start_lsn - 1,
            published: false,
            verified: false,
            seg: 0,
            offset: 0,
            entry_pos: None,
            body_end: 0,
            range_done: false,
        }
    }

    /// Rewinds the cursor to the bottom of the range, for the second walk.
    fn rewind(&mut self) {
        self.seg = 0;
        self.offset = 0;
        self.entry_pos = None;
        self.range_done = false;
    }
}

/// Did re-emitting `written` bytes to retire `retired` pay for itself? A GAIN OF
/// AN EIGHTH of what was written, not merely a positive one.
///
/// Epsilon progress is not progress. At the minimum segment size, one-at-a-time
/// cleaning was measured re-emitting 174 KB to reclaim 179 KB — a 33x write
/// amplification for a 3% gain — and a strictly-positive test calls that
/// success, so the cleaner runs forever, the log grows at traffic rate, and the
/// terminal is never reached.
fn paid_for_itself(retired: i64, written: i64) -> bool {
    retired - written > (written >> 3)
}

/// Walks up to `max_steps` entries of the retiring range, handing each entry's
/// recid to `visit`, and stops early when `visit` returns `false`. Returns the
/// steps taken; sets `c.range_done` once the range is exhausted.
///
/// The unit is **an entry**, not a section. A section may be arbitrarily large —
/// a rollover happens only at a section boundary, so one commit can exceed
/// `segment_bytes` on its own — so "one section per tick" would hold the write
/// lock for an unbounded time, which is exactly the pause this cleaner removes.
/// Payloads are SEEKED over, not read, so the cost is proportional to the number
/// of entries rather than to the bytes they carry.
///
/// The reader is rebuilt per unit rather than carried in [`Cleaner`], which is a
/// deliberate difference from the reference: a `SecIn` borrows its segment's file
/// handle, and a cursor that owned one would make the WAL state self-referential.
/// It costs one window refill per unit — 256 entries — not per section.
fn scan_unit(
    c: &mut Cleaner,
    segs: &mut WalSegmentSet,
    max_steps: usize,
    visit: &mut dyn FnMut(u64) -> Result<bool>,
) -> Result<usize> {
    let mut steps = 0usize;
    'outer: while steps < max_steps {
        if c.seg >= segs.segments().len() || segs.segments()[c.seg].seq > c.target_seq {
            c.range_done = true;
            return Ok(steps);
        }
        segs.segments_mut()[c.seg].ensure_open()?;
        let seg = &segs.segments()[c.seg];
        let (valid_end, seq) = (seg.valid_end, seg.seq);
        let file = seg.file().expect("just opened");
        let mut r = SecIn::new(file, SCAN_BUF);
        // The HARD bound is the segment, set once; `rebound` then narrows the
        // soft bound per section without dropping the window.
        r.reset_hard(SEG_HDR, valid_end);
        if c.offset < SEG_HDR {
            c.offset = SEG_HDR;
        }
        let mut leave_segment = false;
        let mut stop = false;
        loop {
            if steps >= max_steps {
                break;
            }
            if c.entry_pos.is_none() {
                if c.offset >= valid_end {
                    leave_segment = true;
                    steps += 1;
                    break;
                }
                // The header is read THROUGH the window, and both bounds are
                // checked BEFORE the bytes are: this walk verifies no CRC —
                // the section was verified whole at open — so it says what it
                // trusts, and a header or body running past the validated end
                // would otherwise surface as a bare overrun out of a scan that
                // catches none.
                if valid_end - c.offset < SEC_HDR as u64 {
                    return Err(DbError::corrupt_msg(format!(
                        "WAL section header at offset {} in segment {seq:016x} does not fit \
                         before the segment's validated end {valid_end}",
                        c.offset
                    )));
                }
                r.rebound(c.offset, valid_end);
                let mut hdr = [0u8; SEC_HDR];
                r.read_fully(&mut hdr)?;
                let (tag, _lsn, body_len, _, _) = parse_sec_hdr(&hdr);
                let body_start = c.offset + SEC_HDR as u64;
                if body_len < 0 || body_len as u64 > valid_end - body_start {
                    return Err(DbError::corrupt_msg(format!(
                        "WAL section at offset {} in segment {seq:016x} claims a {body_len}-byte \
                         body, which runs past the segment's validated end {valid_end}",
                        c.offset
                    )));
                }
                c.body_end = body_start + body_len as u64;
                c.offset = c.body_end; // where the NEXT section begins
                                       // Entering a section costs a header read and is charged like an
                                       // entry. Without that, a range of mark-only or empty sections is
                                       // walked ENTIRELY within one unit at no budgeted cost — the same
                                       // unbounded-work-under-the-lock defect as a per-section unit,
                                       // with metadata instead of payload.
                steps += 1;
                if tag == TAG_MARK {
                    continue; // a 'K' body carries no entries
                }
                c.entry_pos = Some(body_start);
            }
            let entry_pos = c.entry_pos.expect("set above or carried in");
            r.rebound(entry_pos, c.body_end); // the section bound, window kept
            while r.pos() < c.body_end && steps < max_steps {
                let recid = next_entry_recid(&mut r, seq)?;
                c.entry_pos = Some(r.pos());
                steps += 1;
                if !visit(recid)? {
                    stop = true;
                    break;
                }
            }
            if c.entry_pos.is_some_and(|p| p >= c.body_end) {
                c.entry_pos = None; // section done
            }
            if stop {
                break;
            }
        }
        drop(r);
        if leave_segment {
            // Released the moment the walk leaves it: what keeps the descriptor
            // count O(1) rather than O(segments in the range).
            segs.segments_mut()[c.seg].release();
            c.seg += 1;
            c.offset = 0;
            c.entry_pos = None;
            continue;
        }
        if stop {
            break 'outer;
        }
    }
    Ok(steps)
}

/// Decodes one entry for its recid alone, seeking over the payload.
fn next_entry_recid(r: &mut SecIn, seq: i64) -> Result<u64> {
    let ty = r.read_byte()?;
    let recid = r.unpack_long()?;
    match ty {
        T_PREALLOC | T_DELETE => {}
        T_RECORD => {
            r.unpack_long()?; // capacity
            let len_plus = r.unpack_long()?;
            if len_plus != 0 {
                // The length goes in a local FIRST — written inline against a
                // `pos()` read in the same expression it would land short by the
                // width of the packed length and the walk would resume inside
                // the payload.
                let skip = len_plus - 1;
                let to = r.pos() + skip;
                r.seek(to);
            }
        }
        T_APPEND => {
            r.unpack_long()?; // base delta
            let len = r.unpack_long()?;
            let to = r.pos() + len;
            r.seek(to);
        }
        other => {
            return Err(DbError::corrupt_msg(format!(
                "bad WAL entry tag {other} in segment {seq:016x}"
            )))
        }
    }
    Ok(recid)
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
        // (The checkpoint above rolled into that reused name and then retired
        // segment 1 behind a mark, which is why only 2 is left.)
        assert_eq!(s.segment_seqs(), vec![2]);
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

    // --------------------------------------------------------- the cleaner

    #[test]
    fn a_gain_of_an_eighth_is_what_counts_as_paying_for_itself() {
        // Epsilon progress is not progress: re-emitting 174 KB to reclaim 179 KB
        // is a 33x write amplification for a 3% gain, and a strictly-positive
        // test calls that success — so the cleaner runs forever and the terminal
        // is never reached.
        assert!(!paid_for_itself(179_000, 174_000));
        assert!(!paid_for_itself(100, 100));
        assert!(
            !paid_for_itself(112, 100),
            "exactly an eighth is not enough"
        );
        assert!(paid_for_itself(113, 100));
        assert!(paid_for_itself(1000, 100));
    }

    #[test]
    fn w10_refuses_the_mark_when_a_record_was_not_re_homed() {
        // The check that cannot be deferred past the unlink: the evidence is
        // exactly what would be deleted. Fault injection drops one recid from
        // phase 1, which is the under-re-emission W10 exists for.
        let dir = scratch("w10");
        let base = dir.join("s.db");
        let s = StoreWAL::open_segment_bytes(&base, MIN_SEGMENT_BYTES).unwrap();
        let victim = s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        for i in 0..6i64 {
            s.put(&i, &L).unwrap();
            s.commit().unwrap();
        }
        let segs_before = s.segment_seqs();
        assert!(segs_before.len() > 2);
        s.st.write().drop_recid_from_publish = victim.get();

        let e = s.checkpoint().expect_err("W10 must refuse the mark");
        assert!(
            format!("{e:?}").contains(&format!("recid {}", victim.get())),
            "the refusal names the record it is protecting: {e:?}"
        );
        // Nothing was deleted and the durable log is intact.
        assert!(
            s.segment_seqs().len() >= segs_before.len(),
            "a refused cycle must not retire anything"
        );
        assert!(!s.is_closed(), "a refusal is not a store failure");
        assert_eq!(s.get(victim, &L).unwrap(), Some(1));

        // Removing the fault is NOT enough, and that is the reference's
        // behaviour rather than a port defect. `rewind` resets the cursor and
        // deliberately does not reset `published` (StoreWAL.java:2546-2552), so
        // the partial cycle a retry resumes is still past phase 1: it re-walks
        // the range in VERIFY, finds the same record un-re-homed, and refuses
        // again. Java reaches the identical state — its `checkpoint` also
        // finishes the partial cycle first (StoreWAL.java:2476) — so a port
        // that "fixed" this by re-publishing would diverge.
        s.st.write().drop_recid_from_publish = 0;
        s.checkpoint()
            .expect_err("a completed phase 1 is not re-run by a retry");

        // The documented escape is the one the rewind comment names: a COMMIT
        // that re-homes the recid, which makes the retirement safe for real
        // rather than merely re-attempted. Then the same cycle completes.
        s.update(victim, Some(&1i64), &L).unwrap();
        s.commit().unwrap();
        s.checkpoint().unwrap();
        assert!(s.segment_seqs().len() < segs_before.len());
        assert_eq!(s.get(victim, &L).unwrap(), Some(1));
        s.close().unwrap();
        let s = StoreWAL::open(&base).unwrap();
        assert_eq!(s.get(victim, &L).unwrap(), Some(1));
        s.verify().unwrap();
    }

    #[test]
    fn the_mark_is_forced_before_any_unlink() {
        // W5. A failed unlink is a leak the next open retries; an unlink before
        // the mark is durable is permission the log never gave.
        let dir = scratch("w5");
        let base = dir.join("s.db");
        let trace = Trace::new();
        let s = open_traced(&base, MIN_SEGMENT_BYTES, &trace).unwrap();
        for i in 0..4i64 {
            s.put(&i, &L).unwrap();
            s.commit().unwrap();
        }
        trace.take();
        s.checkpoint().unwrap();
        let kinds = trace.kinds();
        let mark_force = kinds
            .iter()
            .rposition(|k| *k == WalOpKind::ForceData)
            .expect("the mark is forced");
        let first_unlink = kinds
            .iter()
            .position(|k| *k == WalOpKind::Unlink)
            .expect("segments are retired");
        assert!(
            mark_force < first_unlink,
            "the 'K' must be forced before the first unlink: {kinds:?}"
        );
        assert_eq!(
            kinds.last(),
            Some(&WalOpKind::DirSync),
            "and the unlinks are made durable: {kinds:?}"
        );
    }

    #[test]
    fn the_closing_mark_states_the_successors_own_first_lsn() {
        // logStartLsn is READ from the successor's header, never computed, so the
        // number recovery compares against is the number the writer recorded.
        let dir = scratch("mark");
        let base = dir.join("s.db");
        let s = StoreWAL::open_segment_bytes(&base, MIN_SEGMENT_BYTES).unwrap();
        for i in 0..4i64 {
            s.put(&i, &L).unwrap();
            s.commit().unwrap();
        }
        {
            let mut st = s.st.write();
            let closed = AtomicBool::new(false);
            st.roll_active_if_nonempty(&closed).unwrap();
            let target = st.segs.active().unwrap().seq - 1;
            st.start_cycle(target);
            let c = st.cleaner.as_ref().unwrap();
            let want = st
                .segs
                .segments()
                .iter()
                .find(|x| x.seq > target)
                .unwrap()
                .header_first_lsn();
            assert_eq!(c.log_start_lsn, want);
            assert_eq!(c.boundary_lsn, want - 1);
            st.cleaner = None;
        }
        s.close().unwrap();
    }

    #[test]
    fn the_futility_latch_arms_only_on_a_saturated_completed_episode() {
        let dir = scratch("latch");
        let base = dir.join("s.db");
        let s = StoreWAL::open(&base).unwrap();
        let mut st = s.st.write();
        // An episode that gained nothing, from a cycle that was NOT as wide as
        // the episode allows, says something about the width — not about the log.
        st.clean_floor_seq = 1;
        st.episode_retired = 0;
        st.episode_written = 0;
        st.cleaner_bytes_retired = 100;
        st.cleaner_bytes_written = 100;
        st.last_cycle_saturated = false;
        st.end_episode();
        assert!(!st.cleaning_exhausted());
        // The same episode, concluded from a saturated cycle, is evidence.
        st.clean_floor_seq = 1;
        st.episode_retired = 0;
        st.episode_written = 0;
        st.last_cycle_saturated = true;
        st.end_episode();
        assert!(st.cleaning_exhausted());
        assert!(
            !st.last_cycle_saturated,
            "the guard is reset WITH the episode: an episode that did no work \
             must not arm the terminal on a previous one's wide cycle"
        );
        // A quiet trigger releases it — the situation changed.
        st.min_log_bytes = u64::MAX;
        let closed = AtomicBool::new(false);
        assert!(!st.begin_cycle_if_due(&closed).unwrap());
        assert!(!st.cleaning_exhausted());
    }

    #[test]
    fn an_in_flight_allocation_over_committed_state_refuses_rather_than_guesses() {
        let dir = scratch("inflight");
        let base = dir.join("s.db");
        let s = StoreWAL::open_segment_bytes(&base, MIN_SEGMENT_BYTES).unwrap();
        let r = s.put(&1i64, &L).unwrap();
        s.commit().unwrap();
        for i in 0..4i64 {
            s.put(&i, &L).unwrap();
            s.commit().unwrap();
        }
        // A recid that is BOTH committed-live and allocated by an in-flight
        // transaction cannot happen through the allocator; forge it, because the
        // cleaner's response to it is the point.
        s.st.write().staged.insert(r.get(), Staged::new(true));
        let e = s.checkpoint().expect_err("the cleaner must refuse");
        assert!(format!("{e:?}").contains("in-flight"), "{e:?}");
        s.st.write().staged.clear();
    }

    #[test]
    fn the_scan_seeks_over_payloads_instead_of_reading_them() {
        // The cost of a cleaning pass is proportional to the number of ENTRIES,
        // not to the bytes they carry: a scan that read payloads would make the
        // "bounded unit" unbounded in device reads.
        let dir = scratch("scancost");
        let base = dir.join("s.db");
        // The segment must hold all twenty: at 1 MiB the log rolls over after
        // ~10 and the walk below, which only ever enters segments()[0], sees
        // half of them — the test then measures the read cost of a walk it did
        // not perform.
        let s = StoreWAL::open_segment_bytes(&base, 8 << 20).unwrap();
        // Twenty big records in one segment: 20 entries, ~2 MB of payload.
        for i in 0..20u64 {
            s.put(&vec![(i & 0xff) as u8; 100_000], &RawBytes).unwrap();
            s.commit().unwrap();
        }
        let mut st = s.st.write();
        st.segs.segments_mut()[0].ensure_open().unwrap();
        let seg = &st.segs.segments()[0];
        let (valid_end, file) = (seg.valid_end, seg.file().unwrap());
        let mut r = SecIn::new(file, SCAN_BUF);
        r.reset_hard(SEG_HDR, valid_end);
        let mut off = SEG_HDR;
        let mut entries = 0;
        while off < valid_end {
            r.rebound(off, valid_end);
            let mut hdr = [0u8; SEC_HDR];
            r.read_fully(&mut hdr).unwrap();
            let (_, _, body_len, _, _) = parse_sec_hdr(&hdr);
            let body_start = off + SEC_HDR as u64;
            let body_end = body_start + body_len as u64;
            r.rebound(body_start, body_end);
            while r.pos() < body_end {
                next_entry_recid(&mut r, 1).unwrap();
                entries += 1;
            }
            off = body_end;
        }
        assert_eq!(entries, 20);
        assert!(
            r.reads() <= 3 * entries,
            "walking {entries} entries over ~2 MB must not read the payloads: {} reads",
            r.reads()
        );
    }
}
