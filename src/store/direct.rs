//! `StoreDirect` — durable direct store: recid index, free lists, allocator
//! metadata and record data all live on the volume (Java `StoreDirect`, spec 02
//! §5). Algorithms ported faithfully; on-volume format v1, magic "MDBS.SD1".
//!
//! v1 read path is the LOCKED baseline (accepted deviation D9.5): reads take the
//! segment read lock. Java's optimistic seqlock is a data race in Rust; the
//! atomic-copy optimistic mode is the gated M3b phase-2 work (spec 02 §5a).
//!
//! Incremental `compact_step` (roadmap R8) is stubbed pending port; full
//! `compact()` is implemented and is what the collections/tests rely on.

use super::index_val as iv;
use super::lease::LeaseTable;
use super::parity;
use super::segment_locks::SegmentLocks;
use super::volume::{Volume, SLICE_SIZE};
use super::{AppendResult, Recid, Record, RecordRead, Store, StoreDelta};
use crate::error::{DbError, Result};
use crate::io::{DataOutput2, SliceInput};
use crate::ser::Serializer;
use arc_swap::ArcSwap;
use parking_lot::{Mutex, RwLock};
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

// ---------- on-volume geometry ----------

const PAGE_SIZE: u64 = SLICE_SIZE;
/// "MDBS.SD1" big-endian.
const MAGIC: u64 = 0x4D44_4253_2E53_4431;

const O_FEATURES: u64 = 8;
const O_HEAD_CHECKSUM: u64 = 16;
const O_DATA_TAIL: u64 = 24;
const O_MAX_RECID: u64 = 32;
const O_FILE_TAIL: u64 = 40;
const O_FREE_RECID_STACK: u64 = 64;
const O_FREE_DATA_STACKS: u64 = 72;
const MAX_CAP_UNITS: u64 = iv::CAP_MAX_UNITS as u64;
const HEAD_END: u64 = O_FREE_DATA_STACKS + 8 * MAX_CAP_UNITS; // 524336
const ZERO_PAGE_LINK: u64 = HEAD_END;
const ZERO_SLOTS_START: u64 = HEAD_END + 16;
const RECIDS_PER_ZERO_PAGE: u64 = (PAGE_SIZE - ZERO_SLOTS_START) / 8; // 65528
const RECIDS_PER_PAGE: u64 = (PAGE_SIZE - 16) / 8; // 131070

const HEAD_CHECKSUM_SEED: i32 = 0x5D1B_A5E1u32 as i32;

const LONG_STACK_PREF_SIZE: u64 = 160;
const LONG_STACK_MAX_SIZE: u64 = 256;

const LINKED_CHUNK_HDR: usize = 12;
const MAX_CHUNK_DATA: usize = iv::MAX_CAPACITY - LINKED_CHUNK_HDR;
const MAX_VOLUME_SIZE: u64 = 1 << 44;

/// Record state for StoreWAL merge logic (WAL hooks; consumed by the WAL store).
#[allow(dead_code)]
pub(crate) const STATE_VOID: i32 = 0;
#[allow(dead_code)]
pub(crate) const STATE_NULL: i32 = 1;
#[allow(dead_code)]
pub(crate) const STATE_LIVE: i32 = 2;

pub struct StoreDirect {
    vol: Volume,
    /// Offsets of non-zero index pages, in chain order; copy-on-write.
    index_pages: ArcSwap<Vec<u64>>,
    thread_safe: bool,
    structural_lock: Mutex<()>,
    commit_lock: RwLock<()>,
    segs: SegmentLocks,
    closed: AtomicBool,
    poisoned: AtomicBool,
    /// Bytes on the free-data stacks (guarded by `structural_lock`; Relaxed OK).
    free_data_bytes: AtomicI64,
    #[allow(dead_code)] // read via StoreLease, used by the collection layer
    lease_table: Arc<LeaseTable>,
}

fn nz(recid: u64) -> Recid {
    NonZeroU64::new(recid).expect("recid 0 is never allocated")
}

impl StoreDirect {
    /// Anonymous heap-backed store.
    pub fn new_heap() -> Result<StoreDirect> {
        Self::new_heap_ts(true)
    }

    pub fn new_heap_ts(thread_safe: bool) -> Result<StoreDirect> {
        let s = Self::empty(Volume::heap(), thread_safe);
        s.init_create()?;
        Ok(s)
    }

    /// File-backed durable store (mmap volume).
    pub fn open_file(path: &Path) -> Result<StoreDirect> {
        Self::open_file_ts(path, true)
    }

    pub fn open_file_ts(path: &Path, thread_safe: bool) -> Result<StoreDirect> {
        let vol = Volume::open_file(path)?;
        let length = vol.length()?;
        let s = Self::empty(vol, thread_safe);
        let r = if length == 0 {
            s.init_create()
        } else if length < PAGE_SIZE {
            Err(DbError::corrupt("store file smaller than the header page"))
        } else {
            s.vol
                .ensure_available(PAGE_SIZE)
                .and_then(|_| s.init_open())
        };
        match r {
            Ok(()) => Ok(s),
            Err(e) => {
                let _ = s.vol.close(None);
                Err(e)
            }
        }
    }

    fn empty(vol: Volume, thread_safe: bool) -> StoreDirect {
        StoreDirect {
            vol,
            index_pages: ArcSwap::from_pointee(Vec::new()),
            thread_safe,
            structural_lock: Mutex::new(()),
            commit_lock: RwLock::new(()),
            segs: SegmentLocks::default_for(thread_safe),
            closed: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            free_data_bytes: AtomicI64::new(0),
            lease_table: LeaseTable::new(),
        }
    }

    // ---------- header init / open ----------

    fn init_create(&self) -> Result<()> {
        self.vol.ensure_available(PAGE_SIZE)?;
        self.vol.put_u64(0, MAGIC);
        self.vol.put_i32(O_FEATURES, 0);
        self.vol.put_i32(O_FEATURES + 4, 0);
        self.vol.put_i32(O_HEAD_CHECKSUM + 4, 0);
        self.set_data_tail(0);
        self.set_max_recid(0);
        self.set_file_tail(PAGE_SIZE);
        self.vol.put_u64(O_FREE_RECID_STACK, parity::p4set(0));
        for u in 1..=MAX_CAP_UNITS {
            self.vol.put_u64(master_link_offset(u), parity::p4set(0));
        }
        self.vol.put_u64(ZERO_PAGE_LINK, parity::p16set(0));
        self.vol.put_i32(O_HEAD_CHECKSUM, self.head_checksum());
        self.vol.sync()?;
        Ok(())
    }

    fn init_open(&self) -> Result<()> {
        if self.vol.length()? < PAGE_SIZE {
            return Err(DbError::corrupt("store file smaller than the header page"));
        }
        if self.vol.get_u64(0) != MAGIC {
            return Err(DbError::corrupt("not a MapDB StoreDirect file (bad magic)"));
        }
        if self.vol.get_i32(O_FEATURES) != 0 {
            return Err(DbError::corrupt("store uses unsupported feature bits"));
        }
        if self.vol.get_i32(O_HEAD_CHECKSUM) != self.head_checksum() {
            return Err(DbError::corrupt(
                "header checksum mismatch: store was not closed cleanly or is corrupted",
            ));
        }
        let ft = self.file_tail()?;
        if ft < PAGE_SIZE || ft % PAGE_SIZE != 0 {
            return Err(DbError::corrupt("bad fileTail"));
        }
        if self.vol.length()? < ft {
            return Err(DbError::corrupt("store file truncated"));
        }
        self.vol.ensure_available(ft)?;
        // dataTail must satisfy the same geometry verify_locked enforces BEFORE
        // the allocator can use it: parity alone lets a crafted file present e.g.
        // fileTail == dataTail == PAGE_SIZE, whose first allocation would take
        // the in-page branch, return PAGE_SIZE and write into slice 1 though only
        // slice 0 is mapped → panic. Reject a bad tail as DataCorruption (D4).
        let dt = self.data_tail()?; // validates parity
        if !data_tail_geometry_ok(dt, ft) {
            return Err(DbError::corrupt("bad dataTail"));
        }
        let mr = self.max_recid()?; // validates parity
        self.load_index_pages(ft)?;
        // maxRecid must have an addressable index slot in the loaded mirror BEFORE
        // the allocator trusts it: parity alone lets a crafted file present a
        // maxRecid with no index page plus a free-recid stack naming that recid.
        // The reuse branch of alloc_recid_locked would then hand it to index_set,
        // whose recid_to_offset returns None → expect() panic. Reject here (D4).
        // (checked after load_index_pages, which populates the mirror.)
        if !self.max_recid_geometry_ok(mr) {
            return Err(DbError::corrupt("bad maxRecid"));
        }
        self.recompute_free_data_bytes()?;
        Ok(())
    }

    /// The maxRecid geometry invariant, shared by `init_open` (guard a persisted
    /// value before the allocator trusts it) and `verify_locked` (oracle): a
    /// nonzero maxRecid must map to an existing index slot in the loaded page
    /// mirror. Requires `load_index_pages` to have run.
    fn max_recid_geometry_ok(&self, max_recid: u64) -> bool {
        max_recid == 0 || self.recid_to_offset(max_recid).is_some()
    }

    fn load_index_pages(&self, file_tail: u64) -> Result<()> {
        let mut pages = Vec::new();
        let mut ptr = ZERO_PAGE_LINK;
        loop {
            let page = parity::p16get(self.vol.get_u64(ptr))?;
            if page == 0 {
                break;
            }
            if page % PAGE_SIZE != 0 || page >= file_tail {
                return Err(DbError::corrupt("bad index page pointer"));
            }
            pages.push(page);
            if pages.len() > (1 << 24) {
                return Err(DbError::corrupt("index page chain loop"));
            }
            ptr = page + 8;
        }
        self.index_pages.store(Arc::new(pages));
        Ok(())
    }

    /// Mix of every header word the allocator depends on; stamped by commit/close.
    fn head_checksum(&self) -> i32 {
        let mut c = HEAD_CHECKSUM_SEED;
        let mut o = O_DATA_TAIL;
        while o < ZERO_SLOTS_START {
            let v = self.vol.get_u64(o);
            c = c.wrapping_mul(31).wrapping_add((v ^ (v >> 32)) as i32);
            o += 8;
        }
        c
    }

    // ---------- header accessors ----------

    fn data_tail(&self) -> Result<u64> {
        parity::p4get(self.vol.get_u64(O_DATA_TAIL))
    }
    fn set_data_tail(&self, v: u64) {
        self.vol.put_u64(O_DATA_TAIL, parity::p4set(v));
    }
    fn max_recid(&self) -> Result<u64> {
        Ok(parity::p4get(self.vol.get_u64(O_MAX_RECID))? >> 4)
    }
    fn set_max_recid(&self, v: u64) {
        self.vol.put_u64(O_MAX_RECID, parity::p4set(v << 4));
    }
    fn file_tail(&self) -> Result<u64> {
        parity::p16get(self.vol.get_u64(O_FILE_TAIL))
    }
    fn set_file_tail(&self, v: u64) {
        self.vol.put_u64(O_FILE_TAIL, parity::p16set(v));
    }

    // ---------- recid index ----------

    /// Volume offset of the recid's index slot, or `None` when its index page
    /// does not exist.
    fn recid_to_offset(&self, recid: u64) -> Option<u64> {
        let mut r0 = recid - 1;
        if r0 < RECIDS_PER_ZERO_PAGE {
            return Some(ZERO_SLOTS_START + r0 * 8);
        }
        r0 -= RECIDS_PER_ZERO_PAGE;
        let page = r0 / RECIDS_PER_PAGE;
        let pages = self.index_pages.load();
        if page as usize >= pages.len() {
            return None;
        }
        Some(pages[page as usize] + 16 + (r0 % RECIDS_PER_PAGE) * 8)
    }

    /// Raw (parity1-encoded) index slot; 0 when never allocated / out of range.
    fn raw_index_get(&self, recid: u64) -> u64 {
        if recid < 1 {
            return 0;
        }
        match self.recid_to_offset(recid) {
            None => 0,
            Some(off) => self.vol.get_u64(off),
        }
    }

    fn index_get_checked(&self, recid: u64) -> Result<u64> {
        let v = self.raw_index_get(recid);
        if v == 0 {
            return Err(DbError::GetVoid(recid));
        }
        if !iv_parity_ok(v) {
            return Err(DbError::corrupt("index slot parity broken"));
        }
        if iv::cap_units(v) == iv::CAP_DELETED {
            return Err(DbError::GetVoid(recid));
        }
        Ok(v)
    }

    fn index_set(&self, recid: u64, ivval: u64) {
        let off = self
            .recid_to_offset(recid)
            .expect("index slot not allocated for recid");
        self.vol.put_u64(off, parity::p1set(ivval));
    }

    /// structural_lock held. Allocate index pages until `recid` has a slot.
    fn ensure_index_capacity_locked(&self, recid: u64) -> Result<()> {
        while self.recid_to_offset(recid).is_none() {
            self.allocate_new_index_page_locked()?;
        }
        Ok(())
    }

    /// structural_lock held.
    fn allocate_new_index_page_locked(&self) -> Result<()> {
        let page = self.allocate_new_page_locked()?;
        self.vol.clear(page, page + PAGE_SIZE);
        self.vol.put_u64(page + 8, parity::p16set(0));
        let pages = self.index_pages.load_full();
        let ptr = if pages.is_empty() {
            ZERO_PAGE_LINK
        } else {
            pages[pages.len() - 1] + 8
        };
        self.vol.put_u64(ptr, parity::p16set(page));
        let mut grown = (*pages).clone();
        grown.push(page);
        self.index_pages.store(Arc::new(grown));
        Ok(())
    }

    // ---------- allocator (structural_lock held) ----------

    fn allocate_new_page_locked(&self) -> Result<u64> {
        let eof = self.file_tail()?;
        let new_eof = eof + PAGE_SIZE;
        if new_eof > MAX_VOLUME_SIZE {
            return Err(DbError::StoreFull);
        }
        self.vol.ensure_available(new_eof)?;
        self.set_file_tail(new_eof);
        Ok(eof)
    }

    fn alloc_recid_locked(&self) -> Result<u64> {
        let v = self.long_stack_take(O_FREE_RECID_STACK)?;
        if v != 0 {
            let recid = parity::p1get(v)? >> 1;
            // Persisted free-recid value: a freed recid is always <= maxRecid AND
            // has a live index slot. The reuse branch (unlike the fresh-recid
            // branch) does NOT call ensure_index_capacity_locked, so a recid with
            // no index page would reach index_set → recid_to_offset None → the
            // expect() panic. Require an addressable slot, not just the range, so
            // a corrupt free-recid value fails gracefully (D4).
            if recid == 0 || recid > self.max_recid()? || self.recid_to_offset(recid).is_none() {
                return Err(DbError::corrupt("free recid out of range"));
            }
            return Ok(recid);
        }
        let recid = self.max_recid()? + 1;
        self.ensure_index_capacity_locked(recid)?;
        self.set_max_recid(recid);
        Ok(recid)
    }

    fn free_recid_locked(&self, recid: u64) -> Result<()> {
        self.long_stack_put(O_FREE_RECID_STACK, parity::p1set(recid << 1))
    }

    /// structural_lock held. `cap_bytes` 16-aligned within [16, MAX_CAPACITY].
    fn allocate_data_locked(&self, cap_bytes: usize, recursive: bool) -> Result<u64> {
        debug_assert!(cap_bytes & 15 == 0 && (16..=iv::MAX_CAPACITY).contains(&cap_bytes));
        if !recursive {
            let v = self.long_stack_take(master_link_offset(cap_bytes as u64 / 16))?;
            if v != 0 {
                let off = parity::p1get(v)? << 3;
                // The popped free extent is persisted state that open() never
                // validated (recompute_free_data_bytes only counts entries), and
                // parity does not prove it still names a real extent. It becomes
                // a record offset that is written through immediately, so it
                // must satisfy the tiling invariants of its size class — in the
                // data area, 16-aligned, fully below fileTail, and inside one
                // page — or a corrupt value would clobber the header/live data
                // (later stamped clean by close). DataCorruption instead (D4).
                let file_tail = self.file_tail()?;
                // `off` is a packed-long-derived value (`p1get(v) << 3`): a
                // crafted parity-valid value can be near u64::MAX, so the
                // `off + cap_bytes` addition MUST be checked. Unchecked it
                // panics in debug and wraps in release — e.g. off == u64::MAX-15
                // in the 16-byte class wraps the sum to 0 AND passes the one-page
                // test, admitting a huge offset that then indexes a nonexistent
                // slice. checked_add + check_range make a bad value graceful (D4).
                let end = off.checked_add(cap_bytes as u64);
                if off < PAGE_SIZE
                    || off & 15 != 0
                    || (off % PAGE_SIZE) + cap_bytes as u64 > PAGE_SIZE
                    || match end {
                        None => true,
                        Some(e) => e > file_tail,
                    }
                {
                    return Err(DbError::corrupt("free-list extent out of range"));
                }
                // addressability backstop: prove [off, off+cap) is mapped in one
                // slice before it is written through (belt-and-suspenders vs the
                // logical file_tail check above).
                self.vol.check_range(off, cap_bytes as u64)?;
                self.free_data_bytes
                    .fetch_sub(cap_bytes as i64, Ordering::Relaxed);
                return Ok(off);
            }
        }
        let tail = self.data_tail()?;
        if tail == 0 {
            let page = self.allocate_new_page_locked()?;
            self.advance_data_tail(page, cap_bytes as u64)?;
            return Ok(page);
        }
        if (tail % PAGE_SIZE) + cap_bytes as u64 <= PAGE_SIZE {
            self.advance_data_tail(tail, cap_bytes as u64)?;
            return Ok(tail);
        }
        debug_assert!(!recursive, "chunk allocation must fit the current page");
        let rem = PAGE_SIZE - (tail % PAGE_SIZE);
        let page = self.allocate_new_page_locked()?;
        self.advance_data_tail(page, cap_bytes as u64)?;
        self.release_data_locked(rem, tail)?;
        Ok(page)
    }

    fn advance_data_tail(&self, start: u64, cap_bytes: u64) -> Result<()> {
        // `start` is a validated data offset and `cap_bytes` a bounded capacity
        // in normal operation, but a corrupt dataTail slipping past the open-time
        // guard must not wrap the sum (UB-adjacent / debug panic). checked_add
        // yields DataCorruption as a backstop (D4).
        let new_tail = start
            .checked_add(cap_bytes)
            .ok_or_else(|| DbError::corrupt("dataTail arithmetic overflow"))?;
        self.set_data_tail(if new_tail % PAGE_SIZE == 0 {
            0
        } else {
            new_tail
        });
        Ok(())
    }

    fn release_data_locked(&self, size_bytes: u64, offset: u64) -> Result<()> {
        debug_assert!(size_bytes & 15 == 0 && size_bytes >= 16 && size_bytes / 16 <= MAX_CAP_UNITS);
        debug_assert!(offset & 15 == 0 && offset >= PAGE_SIZE);
        self.long_stack_put(
            master_link_offset(size_bytes / 16),
            parity::p1set(offset >> 3),
        )?;
        self.free_data_bytes
            .fetch_add(size_bytes as i64, Ordering::Relaxed);
        Ok(())
    }

    // ---------- long stacks (structural_lock held) ----------

    fn put_packed_long(&self, offset: u64, v: u64) -> usize {
        let size = pack_long_size(v);
        let mut shift = (size - 1) * 7;
        let mut p = offset;
        while shift > 0 {
            self.vol.put_byte(p, ((v >> shift) & 0x7F) as u8);
            p += 1;
            shift -= 7;
        }
        self.vol.put_byte(p, ((v & 0x7F) | 0x80) as u8);
        size
    }

    /// Decode a packed long at `offset`, probing at most `limit` bytes (and
    /// never more than the 10-byte format maximum). The bytes come from a
    /// possibly-corrupt long-stack chunk, so the probe must not run past the
    /// extent the caller validated with `check_range`: on a corrupt chunk whose
    /// value bytes carry no terminator bit, an unbounded probe could step past
    /// the chunk into an unmapped slice and hit the `bound()` panic backstop.
    /// A missing terminator within `limit` is `DataCorruption` (D4/D5).
    fn get_packed_long(&self, offset: u64, limit: u64) -> Result<u64> {
        let mut ret = 0u64;
        for i in 0..limit.min(10) {
            let b = self.vol.get_u8(offset + i) as u64;
            ret = (ret << 7) | (b & 0x7F);
            if b & 0x80 != 0 {
                return Ok(ret);
            }
        }
        Err(DbError::corrupt("unterminated packed long"))
    }

    /// Validate and load the header of a long-stack chunk whose offset came
    /// from a PERSISTED master/prev link word. Parity only proves the word was
    /// written by us, not that it still points at a live chunk, and open() does
    /// not walk every stack (the free-recid stack in particular), so the first
    /// dereference can happen on the allocator hot path. Returns
    /// `(chunk_size, prev_chunk_offset)` after proving the offset is in the
    /// data area and 16-aligned, the size is a legal chunk size, and the whole
    /// chunk extent is addressable in one slice — so subsequent byte scans and
    /// in-chunk writes yield `DataCorruption` on a crafted file instead of
    /// clobbering unrelated bytes or panicking in `Slice::bound` (D4/D5).
    fn load_stack_chunk_checked(&self, chunk_offset: u64) -> Result<(u64, u64)> {
        self.check_stack_chunk_off(chunk_offset)?;
        let hdr = parity::p4get(self.vol.get_u64(chunk_offset))?;
        let chunk_size = hdr >> 48;
        if !(16..=LONG_STACK_MAX_SIZE).contains(&chunk_size) || chunk_size & 15 != 0 {
            return Err(DbError::corrupt("bad long stack chunk size"));
        }
        self.vol.check_range(chunk_offset, chunk_size)?;
        Ok((chunk_size, hdr & iv::MOFFSET))
    }

    fn long_stack_put(&self, master_link_offset: u64, value: u64) -> Result<()> {
        debug_assert!(value != 0 && (value >> 48) == 0);
        let master = parity::p4get(self.vol.get_u64(master_link_offset))?;
        if master == 0 {
            return self.long_stack_new_chunk(master_link_offset, 0, value);
        }
        let chunk_offset = master & iv::MOFFSET;
        let curr_pos = master >> 48;
        // The master link is persisted state: validate the chunk header and the
        // write position before touching bytes inside the chunk, so a corrupt
        // link yields DataCorruption instead of an out-of-extent write (D4).
        let (chunk_size, _prev) = self.load_stack_chunk_checked(chunk_offset)?;
        if curr_pos < 8 || curr_pos > chunk_size {
            return Err(DbError::corrupt("bad long stack position"));
        }
        let value_size = pack_long_size(value) as u64;
        if curr_pos + value_size > chunk_size {
            return self.long_stack_new_chunk(master_link_offset, chunk_offset, value);
        }
        self.put_packed_long(chunk_offset + curr_pos, value);
        self.vol.put_u64(
            master_link_offset,
            parity::p4set(((curr_pos + value_size) << 48) | chunk_offset),
        );
        Ok(())
    }

    /// `prev_chunk_offset` is either 0 or the caller's current chunk offset,
    /// already validated by the caller's `load_stack_chunk_checked`; it is only
    /// WRITTEN into the new chunk's header here, never dereferenced, so no
    /// further validation is needed. The new chunk offset itself comes from
    /// `allocate_data_locked` (trusted, freshly produced under the structural
    /// lock), not from persisted state.
    fn long_stack_new_chunk(
        &self,
        master_link_offset: u64,
        prev_chunk_offset: u64,
        value: u64,
    ) -> Result<()> {
        let tail = self.data_tail()?;
        let chunk_size = if tail == 0 {
            LONG_STACK_PREF_SIZE
        } else {
            (PAGE_SIZE - (tail % PAGE_SIZE)).min(LONG_STACK_PREF_SIZE)
        };
        let value_size = pack_long_size(value) as u64;
        debug_assert!(8 + value_size <= chunk_size);
        let chunk_offset = self.allocate_data_locked(chunk_size as usize, true)?;
        self.vol.clear(chunk_offset, chunk_offset + chunk_size);
        self.vol.put_u64(
            chunk_offset,
            parity::p4set((chunk_size << 48) | prev_chunk_offset),
        );
        self.put_packed_long(chunk_offset + 8, value);
        self.vol.put_u64(
            master_link_offset,
            parity::p4set(((8 + value_size) << 48) | chunk_offset),
        );
        Ok(())
    }

    /// Pop the most recent value (raw, still parity1-encoded), or 0 when empty.
    fn long_stack_take(&self, master_link_offset: u64) -> Result<u64> {
        let master = parity::p4get(self.vol.get_u64(master_link_offset))?;
        if master == 0 {
            return Ok(0);
        }
        let chunk_offset = master & iv::MOFFSET;
        // The master link is persisted state, and open() never walks this stack
        // when it is the free-recid stack — this may be the very first look at
        // these words. Validate the chunk header and the stored position before
        // the terminator back-scan / decode / clear touch bytes inside it, so a
        // corrupt link yields DataCorruption, not a bound() panic (D4/D5).
        let (chunk_size, prev_chunk_offset) = self.load_stack_chunk_checked(chunk_offset)?;
        let master_pos = master >> 48;
        if master_pos < 8 || master_pos > chunk_size {
            return Err(DbError::corrupt("bad long stack position"));
        }
        let mut pos = master_pos.saturating_sub(1).max(8);
        while pos > 8 && (self.vol.get_u8(chunk_offset + pos - 1) & 0x80) == 0 {
            pos -= 1;
        }
        // decode bounded by the validated chunk extent (see get_packed_long)
        let value = self.get_packed_long(chunk_offset + pos, chunk_size - pos)?;
        self.vol.clear(
            chunk_offset + pos,
            chunk_offset + pos + pack_long_size(value) as u64,
        );
        if pos > 8 {
            self.vol.put_u64(
                master_link_offset,
                parity::p4set((pos << 48) | chunk_offset),
            );
            return Ok(value);
        }
        // chunk emptied: relink master to the previous chunk, then free this one.
        // The prev link is persisted too — validate it before find_end scans it.
        let prev_pos = if prev_chunk_offset != 0 {
            let (prev_size, _) = self.load_stack_chunk_checked(prev_chunk_offset)?;
            self.long_stack_find_end(prev_chunk_offset, prev_size)
        } else {
            0
        };
        self.vol.put_u64(
            master_link_offset,
            parity::p4set((prev_pos << 48) | prev_chunk_offset),
        );
        self.release_data_locked(chunk_size, chunk_offset)?;
        Ok(value)
    }

    fn long_stack_find_end(&self, chunk_offset: u64, mut pos: u64) -> u64 {
        while pos > 8 && self.vol.get_u8(chunk_offset + pos - 1) == 0 {
            pos -= 1;
        }
        pos
    }

    // ---------- helpers ----------

    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(DbError::StoreClosed);
        }
        Ok(())
    }

    /// Enter a volume-touching op: shared commit barrier + closed re-check.
    fn mutate_enter(&self) -> Result<parking_lot::RwLockReadGuard<'_, ()>> {
        let g = self.commit_lock.read();
        if self.closed.load(Ordering::Acquire) {
            drop(g);
            return Err(DbError::StoreClosed);
        }
        Ok(g)
    }

    fn structural(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.structural_lock.lock()
    }

    fn linked_chain(&self, ivval: u64) -> Result<Vec<(u64, usize, usize)>> {
        // (offset, dataLen, capBytes)
        let mut chunks = Vec::new();
        let mut cap_units = iv::cap_units(ivval) as u64;
        let mut off = iv::offset(ivval);
        let mut total: u64 = 0;
        loop {
            let cap_bytes = cap_units * 16;
            if off < PAGE_SIZE || off & 15 != 0 {
                return Err(DbError::corrupt("linked chunk offset in header/misaligned"));
            }
            // header (len i32 + next u64) then the chunk's data must be in-slice.
            self.vol.check_range(off, cap_bytes)?;
            let len = self.vol.get_i32(off);
            if len < 0 || LINKED_CHUNK_HDR as u64 + len as u64 > cap_bytes {
                return Err(DbError::corrupt("linked chunk length out of range"));
            }
            chunks.push((off, len as usize, cap_bytes as usize));
            total += len as u64;
            if total > i32::MAX as u64 || chunks.len() > (1 << 22) {
                return Err(DbError::corrupt("linked chain too long"));
            }
            let next = parity::p1get(self.vol.get_u64(off + 4))?;
            if next == 0 {
                break;
            }
            cap_units = next >> 48;
            off = next & iv::MOFFSET;
            if !(1..=MAX_CAP_UNITS).contains(&cap_units) || off < PAGE_SIZE {
                return Err(DbError::corrupt("bad linked chunk pointer"));
            }
        }
        Ok(chunks)
    }

    fn linked_get(&self, ivval: u64) -> Result<Vec<u8>> {
        let chunks = self.linked_chain(ivval)?;
        let total: usize = chunks.iter().map(|c| c.1).sum();
        let mut out = vec![0u8; total];
        let mut p = 0;
        for (off, len, _) in chunks {
            self.vol
                .get_data(off + LINKED_CHUNK_HDR as u64, &mut out[p..p + len]);
            p += len;
        }
        Ok(out)
    }

    /// segment write lock held.
    fn write_new_data(&self, recid: u64, buf: &[u8], cap_bytes: usize, flags: u64) -> Result<()> {
        let off = {
            let _s = self.structural();
            self.allocate_data_locked(cap_bytes, false)?
        };
        self.vol.put_i32(off, buf.len() as i32);
        self.vol.put_data(off + 4, buf);
        self.index_set(recid, iv::compose(cap_bytes as u32 / 16, off, flags));
        Ok(())
    }

    /// segment write lock held. Oversize record → linked chunk chain, tail-first.
    fn write_new_linked(&self, recid: u64, buf: &[u8]) -> Result<()> {
        let len = buf.len();
        debug_assert!(needs_linked(len as u64));
        let mut tail_data = len % MAX_CHUNK_DATA;
        if tail_data == 0 {
            tail_data = MAX_CHUNK_DATA;
        }
        let mut pos = len - tail_data;
        let mut chunk_data_len = tail_data;
        let mut next_ptr = parity::p1set(0);
        loop {
            let cap_bytes = cap_bytes_for(LINKED_CHUNK_HDR as u64 + chunk_data_len as u64)?;
            let off = {
                let _s = self.structural();
                self.allocate_data_locked(cap_bytes, false)?
            };
            self.vol.put_i32(off, chunk_data_len as i32);
            self.vol.put_u64(off + 4, next_ptr);
            self.vol.put_data(
                off + LINKED_CHUNK_HDR as u64,
                &buf[pos..pos + chunk_data_len],
            );
            if pos == 0 {
                self.index_set(
                    recid,
                    iv::compose(cap_bytes as u32 / 16, off, iv::FLAG_LINKED),
                );
                return Ok(());
            }
            next_ptr = parity::p1set(((cap_bytes as u64 / 16) << 48) | off);
            chunk_data_len = MAX_CHUNK_DATA;
            pos -= MAX_CHUNK_DATA;
        }
    }

    /// segment write lock held. Free the data area of `iv` if it has one.
    fn release_old_data(&self, ivval: u64) -> Result<()> {
        let cap = iv::cap_units(ivval);
        if cap == iv::CAP_NULL || cap == iv::CAP_DELETED {
            return Ok(());
        }
        if iv::is_linked(ivval) {
            let chunks = self.linked_chain(ivval)?;
            let _s = self.structural();
            for (off, _len, cap_bytes) in chunks {
                self.release_data_locked(cap_bytes as u64, off)?;
            }
        } else {
            let _s = self.structural();
            self.release_data_locked(cap as u64 * 16, iv::offset(ivval))?;
        }
        Ok(())
    }

    /// Read record content of a non-linked live iv, validating `used`. The
    /// offset comes from a possibly-corrupt index value, so both the 4-byte
    /// header and the `[off, off+4+used)` extent are range-checked before any
    /// raw volume access (D4/D5).
    fn read_used(&self, ivval: u64) -> Result<(u64, usize)> {
        let off = iv::offset(ivval);
        if off < PAGE_SIZE || off & 15 != 0 {
            return Err(DbError::corrupt("record offset in header/misaligned"));
        }
        self.vol.check_range(off, 4)?;
        let used = self.vol.get_i32(off);
        let cap_bytes = iv::cap_units(ivval) as i64 * 16;
        if used < 0 || 4 + used as i64 > cap_bytes {
            return Err(DbError::corrupt("used beyond capacity"));
        }
        self.vol.check_range(off, 4 + used as u64)?;
        Ok((off, used as usize))
    }

    // ---------- durability ----------

    fn stamp_header_durable(&self) -> Result<()> {
        self.vol.sync()?;
        self.vol.put_i32(O_HEAD_CHECKSUM, self.head_checksum());
        self.vol.sync_header()?;
        Ok(())
    }

    fn recompute_free_data_bytes(&self) -> Result<()> {
        let mut total: i64 = 0;
        for u in 1..=MAX_CAP_UNITS {
            let mut count = 0i64;
            self.for_each_long_stack(master_link_offset(u), &mut |_v| {
                count += 1;
                Ok(())
            })?;
            total += count * (u as i64 * 16);
        }
        self.free_data_bytes.store(total, Ordering::Relaxed);
        Ok(())
    }

    // ---------- WAL hooks (crate-private) ----------

    #[allow(dead_code)]
    pub(crate) fn wal_prealloc(&self, recid: u64) -> Result<()> {
        let _c = self.mutate_enter()?;
        {
            let _s = self.structural();
            self.ensure_index_capacity_locked(recid)?;
            if recid > self.max_recid()? {
                self.set_max_recid(recid);
            }
        }
        let _wg = self.segs.write(recid);
        let ivval = self.raw_index_get(recid);
        if ivval == 0 || iv::cap_units(ivval) == iv::CAP_DELETED {
            self.index_set(recid, iv::compose(iv::CAP_NULL, 0, iv::FLAG_PREALLOC));
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn wal_put(
        &self,
        recid: u64,
        cap_bytes: usize,
        content: Option<&[u8]>,
    ) -> Result<()> {
        let _c = self.mutate_enter()?;
        {
            let _s = self.structural();
            self.ensure_index_capacity_locked(recid)?;
            if recid > self.max_recid()? {
                self.set_max_recid(recid);
            }
        }
        let _wg = self.segs.write(recid);
        let ivval = self.raw_index_get(recid);
        if ivval != 0 {
            if !iv_parity_ok(ivval) {
                return Err(DbError::corrupt("index slot parity broken"));
            }
            self.release_old_data(ivval)?;
        }
        match content {
            None => self.index_set(recid, iv::compose(iv::CAP_NULL, 0, 0)),
            Some(c) if needs_linked(c.len() as u64) => self.write_new_linked(recid, c)?,
            Some(c) => {
                let cap = if cap_bytes == 0 {
                    cap_bytes_for(4 + c.len() as u64)?
                } else {
                    cap_bytes
                };
                if cap > iv::MAX_CAPACITY || cap < 4 + c.len() || cap & 15 != 0 {
                    return Err(DbError::corrupt("bad record capacity"));
                }
                self.write_new_data(recid, c, cap, 0)?;
            }
        }
        Ok(())
    }

    /// Replay's delete: like [`Store::delete`] but a **no-op on a void or
    /// already-deleted recid** rather than `GetVoid`.
    ///
    /// That tolerance is the point. A v3 log may legitimately contain a
    /// `T_DELETE` for a recid whose creating section the cleaner already removed
    /// — the delete is then the only surviving mention of it, and refusing to
    /// replay it would turn a correctly cleaned log into an unopenable store.
    /// The strict `delete` stays strict for the API surface, where a void target
    /// really is a caller error.
    #[allow(dead_code)]
    pub(crate) fn wal_delete(&self, recid: u64) -> Result<()> {
        let _c = self.mutate_enter()?;
        let _wg = self.segs.write(recid);
        let ivval = self.raw_index_get(recid);
        if ivval == 0 {
            return Ok(());
        }
        if !iv_parity_ok(ivval) {
            return Err(DbError::corrupt("index slot parity broken"));
        }
        if iv::cap_units(ivval) == iv::CAP_DELETED {
            return Ok(());
        }
        self.release_old_data(ivval)?;
        {
            let _s = self.structural();
            self.free_recid_locked(recid)?;
        }
        self.index_set(recid, iv::compose(iv::CAP_DELETED, 0, 0));
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn rebuild_free_recids(&self) -> Result<()> {
        let _c = self.mutate_enter()?;
        self.rebuild_free_recids_inner()
    }

    fn rebuild_free_recids_inner(&self) -> Result<()> {
        let _s = self.structural();
        let master = parity::p4get(self.vol.get_u64(O_FREE_RECID_STACK))?;
        let mut chunk_offset = master & iv::MOFFSET;
        let mut chunks = Vec::new();
        while chunk_offset != 0 {
            // master/prev links are persisted state (open() does not walk the
            // free-recid stack); validate each chunk before reading its header
            // and re-releasing its extent to the free-data stacks (D4).
            let (size, prev) = self.load_stack_chunk_checked(chunk_offset)?;
            chunks.push((chunk_offset, size));
            chunk_offset = prev;
            if chunks.len() > (1 << 24) {
                return Err(DbError::corrupt("free recid stack loop"));
            }
        }
        self.vol.put_u64(O_FREE_RECID_STACK, parity::p4set(0));
        for (off, size) in chunks {
            self.release_data_locked(size, off)?;
        }
        let max = self.max_recid()?;
        for recid in 1..=max {
            let ivval = self.raw_index_get(recid);
            if ivval == 0 || iv::cap_units(ivval) == iv::CAP_DELETED {
                self.free_recid_locked(recid)?;
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn rec_state(&self, recid: u64) -> Result<i32> {
        let _c = self.mutate_enter()?;
        let _rg = self.segs.read(recid);
        let ivval = self.raw_index_get(recid);
        if ivval == 0 {
            return Ok(STATE_VOID);
        }
        if !iv_parity_ok(ivval) {
            return Err(DbError::corrupt("index slot parity broken"));
        }
        if iv::cap_units(ivval) == iv::CAP_DELETED {
            return Ok(STATE_VOID);
        }
        Ok(if iv::cap_units(ivval) == iv::CAP_NULL {
            STATE_NULL
        } else {
            STATE_LIVE
        })
    }

    /// Copy of record content, or None for null/P records. GetVoid on N/D.
    #[allow(dead_code)]
    pub(crate) fn raw_get(&self, recid: u64) -> Result<Option<Vec<u8>>> {
        let _c = self.mutate_enter()?;
        let _rg = self.segs.read(recid);
        let ivval = self.index_get_checked(recid)?;
        if iv::cap_units(ivval) == iv::CAP_NULL {
            return Ok(None);
        }
        if iv::is_linked(ivval) {
            return Ok(Some(self.linked_get(ivval)?));
        }
        let (off, used) = self.read_used(ivval)?;
        let mut r = vec![0u8; used];
        self.vol.get_data(off + 4, &mut r);
        Ok(Some(r))
    }

    /// Snapshot ONE recid for the WAL cleaner: `(prealloc, cap_bytes, content)`
    /// through `sink`, returning whether the record exists at all (`false` for a
    /// void or deleted slot, where the sink is not invoked).
    ///
    /// Per-recid rather than a whole-store walk, because the cleaner re-homes
    /// the records it MEETS in the segments it is retiring — a walk over every
    /// recid would be O(store) under the WAL write lock, which is the pause the
    /// incremental cleaner exists to remove. The sink runs after the per-recid
    /// lock is released, so the caller must hold its own barrier (the WAL write
    /// lock) if it needs check-copy-publish to be one serialized unit.
    pub(crate) fn wal_snapshot_one(
        &self,
        recid: u64,
        sink: impl FnOnce(bool, usize, Option<Vec<u8>>) -> Result<()>,
    ) -> Result<bool> {
        let _c = self.mutate_enter()?;
        let mut emit = false;
        let mut prealloc = false;
        let mut cap_bytes = 0usize;
        let mut content: Option<Vec<u8>> = None;
        {
            let _rg = self.segs.read(recid);
            let ivval = self.raw_index_get(recid);
            let cap = iv::cap_units(ivval);
            if ivval != 0 && cap != iv::CAP_DELETED {
                emit = true;
                if cap == iv::CAP_NULL {
                    prealloc = iv::is_prealloc(ivval);
                } else if iv::is_linked(ivval) {
                    content = Some(self.linked_get(ivval)?);
                } else {
                    let (off, used) = self.read_used(ivval)?;
                    let mut c = vec![0u8; used];
                    self.vol.get_data(off + 4, &mut c);
                    content = Some(c);
                    cap_bytes = cap as usize * 16;
                }
            }
        }
        if emit {
            sink(prealloc, cap_bytes, content)?;
        }
        Ok(emit)
    }

    // ---------- verify ----------

    /// Validate a long-stack chunk offset before dereferencing its header. The
    /// offset comes from a parity-valid but possibly-corrupt link word, so a
    /// crafted file must yield `DataCorruption` rather than the volume `bound`
    /// panic backstop (D4/D5). Used (via `load_stack_chunk_checked`) by both
    /// the open-time/verify traversal AND the allocator hot path — open() does
    /// not walk the free-recid stack, so take/put cannot assume validated links.
    fn check_stack_chunk_off(&self, off: u64) -> Result<()> {
        if off < PAGE_SIZE || off & 15 != 0 {
            return Err(DbError::corrupt(
                "long stack chunk offset in header/misaligned",
            ));
        }
        self.vol.check_range(off, 8)
    }

    fn for_each_long_stack(
        &self,
        master_link_offset: u64,
        check: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<Vec<(u64, u64)>> {
        // returns chunk extents (offset, size)
        let mut extents = Vec::new();
        let master = parity::p4get(self.vol.get_u64(master_link_offset))?;
        if master == 0 {
            return Ok(extents);
        }
        let mut chunk_offset = master & iv::MOFFSET;
        let mut pos = master >> 48;
        let mut guard = 0;
        while chunk_offset != 0 {
            guard += 1;
            if guard > (1 << 24) {
                return Err(DbError::VerifyFailed("long stack chunk loop".into()));
            }
            // header/size/extent validated as on the hot path (D4/D5).
            let (chunk_size, prev) = self.load_stack_chunk_checked(chunk_offset)?;
            if pos < 8 || pos > chunk_size {
                return Err(DbError::VerifyFailed("bad long stack position".into()));
            }
            extents.push((chunk_offset, chunk_size));
            let mut p = chunk_offset + 8;
            let end = chunk_offset + pos;
            while p < end {
                if self.vol.get_u8(p) == 0 {
                    return Err(DbError::VerifyFailed(
                        "zero byte in long stack value area".into(),
                    ));
                }
                // decode bounded by the remaining value area so a corrupt
                // unterminated value cannot probe past the validated extent.
                let raw = self.get_packed_long(p, end - p)?;
                p += pack_long_size(raw) as u64;
                if p > end {
                    return Err(DbError::VerifyFailed(
                        "long stack value overruns chunk".into(),
                    ));
                }
                check(parity::p1get(raw)?)?;
            }
            chunk_offset = prev;
            if prev != 0 {
                // find_end scans up to prev+prev_size — validate that extent now.
                let (prev_size, _) = self.load_stack_chunk_checked(prev)?;
                pos = self.long_stack_find_end(prev, prev_size);
            }
        }
        Ok(extents)
    }

    fn verify_locked(&self) -> Result<()> {
        let file_tail = self.file_tail()?;
        let data_tail = self.data_tail()?;
        let max_recid = self.max_recid()?;
        if file_tail < PAGE_SIZE || file_tail % PAGE_SIZE != 0 {
            return Err(DbError::VerifyFailed("bad fileTail".into()));
        }
        if !data_tail_geometry_ok(data_tail, file_tail) {
            return Err(DbError::VerifyFailed("bad dataTail".into()));
        }
        if !self.max_recid_geometry_ok(max_recid) {
            return Err(DbError::VerifyFailed("maxRecid beyond index pages".into()));
        }

        // index page chain must match the mirror
        let mut index_page_set = std::collections::HashSet::new();
        let mirror = self.index_pages.load();
        let mut ptr = ZERO_PAGE_LINK;
        let mut n = 0usize;
        loop {
            let page = parity::p16get(self.vol.get_u64(ptr))?;
            if page == 0 {
                break;
            }
            if n >= mirror.len() || mirror[n] != page {
                return Err(DbError::VerifyFailed(
                    "index page chain diverges from mirror".into(),
                ));
            }
            if page % PAGE_SIZE != 0 || page >= file_tail {
                return Err(DbError::VerifyFailed("index page out of range".into()));
            }
            if !index_page_set.insert(page) {
                return Err(DbError::VerifyFailed("index page loop".into()));
            }
            ptr = page + 8;
            n += 1;
        }
        if n != mirror.len() {
            return Err(DbError::VerifyFailed(
                "index page mirror longer than chain".into(),
            ));
        }

        let mut extents: Vec<(u64, u64)> = Vec::new();
        for recid in 1..=max_recid {
            let ivval = self.raw_index_get(recid);
            if ivval == 0 {
                continue;
            }
            if !iv_parity_ok(ivval) {
                return Err(DbError::VerifyFailed("index parity broken".into()));
            }
            let cap = iv::cap_units(ivval);
            if cap == iv::CAP_DELETED || cap == iv::CAP_NULL {
                if iv::offset(ivval) != 0 {
                    return Err(DbError::VerifyFailed(
                        "sentinel index value with offset".into(),
                    ));
                }
                continue;
            }
            if iv::is_linked(ivval) {
                for (off, _len, cap_bytes) in self.linked_chain(ivval)? {
                    extents.push((off, cap_bytes as u64));
                }
            } else {
                let (off, _used) = self.read_used(ivval)?;
                extents.push((off, cap as u64 * 16));
            }
        }

        // free recid stack: its chunks are extents; each value is a deleted recid.
        let mut free_recids = std::collections::HashSet::new();
        let recid_chunks = self.for_each_long_stack(O_FREE_RECID_STACK, &mut |v| {
            let recid = v >> 1;
            if recid < 1 || recid > max_recid {
                return Err(DbError::VerifyFailed("free recid out of range".into()));
            }
            let ivval = self.raw_index_get(recid);
            if ivval != 0 && (!iv_parity_ok(ivval) || iv::cap_units(ivval) != iv::CAP_DELETED) {
                return Err(DbError::VerifyFailed("free-list recid is live".into()));
            }
            if !free_recids.insert(recid) {
                return Err(DbError::VerifyFailed("duplicate free recid".into()));
            }
            Ok(())
        })?;
        extents.extend(recid_chunks);

        // free data stacks: chunks AND the freed value extents (value<<3, size) tile.
        let mut free_sum: i64 = 0;
        for u in 1..=MAX_CAP_UNITS {
            let size = u * 16;
            let mut value_offsets = Vec::new();
            let chunk_exts = self.for_each_long_stack(master_link_offset(u), &mut |v| {
                value_offsets.push(v << 3);
                Ok(())
            })?;
            extents.extend(chunk_exts);
            for off in value_offsets {
                extents.push((off, size));
                free_sum += size as i64;
            }
        }
        if free_sum != self.free_data_bytes.load(Ordering::Relaxed) {
            return Err(DbError::VerifyFailed("freeDataBytes drift".into()));
        }

        // geometry + exact tiling
        for &(off, size) in &extents {
            if off & 15 != 0 || size & 15 != 0 || size < 16 {
                return Err(DbError::VerifyFailed("unaligned extent".into()));
            }
            if off < PAGE_SIZE || off + size > file_tail {
                return Err(DbError::VerifyFailed("extent out of bounds".into()));
            }
            if (off % PAGE_SIZE) + size > PAGE_SIZE {
                return Err(DbError::VerifyFailed("extent crosses page boundary".into()));
            }
            let page = off - off % PAGE_SIZE;
            if index_page_set.contains(&page) {
                return Err(DbError::VerifyFailed("extent inside an index page".into()));
            }
        }
        let mut by_page: std::collections::HashMap<u64, Vec<(u64, u64)>> =
            std::collections::HashMap::new();
        for &(off, size) in &extents {
            by_page
                .entry(off - off % PAGE_SIZE)
                .or_default()
                .push((off, size));
        }
        let data_tail_page = if data_tail == 0 {
            u64::MAX
        } else {
            data_tail - data_tail % PAGE_SIZE
        };
        let mut page = PAGE_SIZE;
        while page < file_tail {
            if index_page_set.contains(&page) {
                page += PAGE_SIZE;
                continue;
            }
            let cover_end = if page == data_tail_page {
                data_tail
            } else {
                page + PAGE_SIZE
            };
            let mut cursor = page;
            if let Some(mut list) = by_page.remove(&page) {
                list.sort_by_key(|e| e.0);
                for (off, size) in list {
                    if off < cursor {
                        return Err(DbError::VerifyFailed("overlapping extents".into()));
                    }
                    if off > cursor {
                        return Err(DbError::VerifyFailed("lost extent: gap".into()));
                    }
                    cursor = off + size;
                }
            }
            if cursor != cover_end {
                return Err(DbError::VerifyFailed(
                    "lost extent: page not fully covered".into(),
                ));
            }
            page += PAGE_SIZE;
        }
        if !by_page.is_empty() {
            return Err(DbError::VerifyFailed("extents on unallocated pages".into()));
        }
        Ok(())
    }

    // ---------- full compact ----------

    fn compact_snapshot(&self) -> Result<Vec<CompactEntry>> {
        let mut entries = Vec::new();
        let max = self.max_recid()?;
        for recid in 1..=max {
            let ivval = self.raw_index_get(recid);
            if ivval == 0 {
                continue;
            }
            if !iv_parity_ok(ivval) {
                return Err(DbError::corrupt("index slot parity broken"));
            }
            let cap = iv::cap_units(ivval);
            if cap == iv::CAP_DELETED {
                continue;
            }
            if cap == iv::CAP_NULL {
                entries.push(CompactEntry {
                    recid,
                    prealloc: iv::is_prealloc(ivval),
                    cap_bytes: 0,
                    content: None,
                });
            } else if iv::is_linked(ivval) {
                entries.push(CompactEntry {
                    recid,
                    prealloc: false,
                    cap_bytes: 0,
                    content: Some(self.linked_get(ivval)?),
                });
            } else {
                let (off, used) = self.read_used(ivval)?;
                let mut content = vec![0u8; used];
                self.vol.get_data(off + 4, &mut content);
                entries.push(CompactEntry {
                    recid,
                    prealloc: false,
                    cap_bytes: cap as usize * 16,
                    content: Some(content),
                });
            }
        }
        Ok(entries)
    }

    /// Rebuild the store densely from a pre-taken `entries`/`max` snapshot.
    /// Everything here is past the crash barrier, so any error poisons the store
    /// (the caller's responsibility); the read-only snapshot is taken by the
    /// caller BEFORE this point so a corruption error there cannot poison.
    fn compact_inner(&self, entries: Vec<CompactEntry>, max: u64) -> Result<()> {
        // 0) crash barrier: invalidate the on-disk checksum durably.
        self.vol.put_i32(O_HEAD_CHECKSUM, !self.head_checksum());
        self.vol.sync()?;

        {
            let _s = self.structural();
            self.set_data_tail(0);
            self.set_file_tail(PAGE_SIZE);
            self.vol.put_u64(O_FREE_RECID_STACK, parity::p4set(0));
            for u in 1..=MAX_CAP_UNITS {
                self.vol.put_u64(master_link_offset(u), parity::p4set(0));
            }
            self.vol.put_u64(ZERO_PAGE_LINK, parity::p16set(0));
            self.vol.clear(ZERO_SLOTS_START, PAGE_SIZE);
            self.free_data_bytes.store(0, Ordering::Relaxed);
            self.index_pages.store(Arc::new(Vec::new()));
            self.set_max_recid(max);
            if max > 0 {
                self.ensure_index_capacity_locked(max)?;
            }
        }

        for e in &entries {
            match &e.content {
                None => self.index_set(
                    e.recid,
                    iv::compose(
                        iv::CAP_NULL,
                        0,
                        if e.prealloc { iv::FLAG_PREALLOC } else { 0 },
                    ),
                ),
                Some(c) if needs_linked(c.len() as u64) => self.write_new_linked(e.recid, c)?,
                Some(c) => self.write_new_data(e.recid, c, e.cap_bytes, 0)?,
            }
        }
        self.rebuild_free_recids_inner()?;
        self.stamp_header_durable()?;
        self.vol.truncate(self.file_tail()?)?;
        Ok(())
    }
}

// ---------- free functions ----------

fn iv_parity_ok(ivval: u64) -> bool {
    ivval.count_ones() & 1 == 1
}

/// The dataTail geometry invariant, shared by `init_open` (guard a persisted
/// value before the allocator trusts it) and `verify_locked` (oracle). A valid
/// dataTail is either 0 (no open page) or an offset that is 16-aligned, in the
/// data area, strictly below fileTail, and NOT page-aligned (a page-aligned
/// non-zero tail would be encoded as 0). Callers pass an already parity-decoded
/// `file_tail` known to be page-aligned and >= PAGE_SIZE.
fn data_tail_geometry_ok(data_tail: u64, file_tail: u64) -> bool {
    data_tail == 0
        || (data_tail.is_multiple_of(16)
            && !data_tail.is_multiple_of(PAGE_SIZE)
            && data_tail >= PAGE_SIZE
            && data_tail < file_tail)
}

fn master_link_offset(cap_units: u64) -> u64 {
    debug_assert!((1..=MAX_CAP_UNITS).contains(&cap_units));
    O_FREE_DATA_STACKS + 8 * (cap_units - 1)
}

fn pack_long_size(v: u64) -> usize {
    let mut c = 1;
    let mut v = v;
    loop {
        v >>= 7;
        if v == 0 {
            break;
        }
        c += 1;
    }
    c
}

fn needs_linked(content_len: u64) -> bool {
    4 + content_len > iv::MAX_CAPACITY as u64
}

fn check_size(cap_bytes: u64) -> Result<()> {
    if cap_bytes > iv::MAX_CAPACITY as u64 {
        return Err(DbError::RecordTooLarge);
    }
    Ok(())
}

fn cap_bytes_for(need: u64) -> Result<usize> {
    let rounded = need.checked_add(15).ok_or(DbError::RecordTooLarge)? & !15;
    check_size(rounded)?;
    Ok(rounded as usize)
}

/// Plain-record byte need = header(4) + content + headroom, as checked wide
/// arithmetic. A caller-supplied `headroom` near `usize::MAX` must map to
/// `RecordTooLarge`, never wrap into a small accepted capacity.
fn plain_need(content_len: u64, headroom: u64) -> Result<u64> {
    4u64.checked_add(content_len)
        .and_then(|n| n.checked_add(headroom))
        .ok_or(DbError::RecordTooLarge)
}

fn serialize<R>(value: &R, ser: &(impl Serializer<R> + Sync)) -> Vec<u8> {
    let mut out = DataOutput2::with_capacity(ser.size_hint() + 4);
    ser.serialize(&mut out, value);
    out.into_vec()
}

struct CompactEntry {
    recid: u64,
    prealloc: bool,
    cap_bytes: usize,
    content: Option<Vec<u8>>,
}

impl super::StoreLease for StoreDirect {
    fn lease_table(&self) -> &Arc<LeaseTable> {
        &self.lease_table
    }
}

impl Store for StoreDirect {
    fn preallocate(&self) -> Result<Recid> {
        self.check_closed()?;
        let _c = self.mutate_enter()?;
        let recid = {
            let _s = self.structural();
            self.alloc_recid_locked()?
        };
        {
            let _wg = self.segs.write(recid);
            self.index_set(recid, iv::compose(iv::CAP_NULL, 0, iv::FLAG_PREALLOC));
        }
        Ok(nz(recid))
    }

    fn put<R: Record>(&self, value: &R, ser: &(impl Serializer<R> + Sync)) -> Result<Recid> {
        self.check_closed()?;
        let buf = serialize(value, ser);
        let linked = needs_linked(buf.len() as u64);
        let cap_bytes = if linked {
            0
        } else {
            cap_bytes_for(4 + buf.len() as u64)?
        };
        let _c = self.mutate_enter()?;
        let recid = {
            let _s = self.structural();
            self.alloc_recid_locked()?
        };
        {
            let _wg = self.segs.write(recid);
            if linked {
                self.write_new_linked(recid, &buf)?;
            } else {
                self.write_new_data(recid, &buf, cap_bytes, 0)?;
            }
        }
        Ok(nz(recid))
    }

    fn get<R: Record>(&self, recid: Recid, ser: &(impl Serializer<R> + Sync)) -> Result<Option<R>> {
        let _c = self.mutate_enter()?;
        let _rg = self.segs.read(recid.get());
        let ivval = self.index_get_checked(recid.get())?;
        if iv::cap_units(ivval) == iv::CAP_NULL {
            return Ok(None);
        }
        if iv::is_linked(ivval) {
            let b = self.linked_get(ivval)?;
            let mut inp = SliceInput::new(&b);
            return Ok(Some(ser.deserialize(&mut inp, Some(b.len()))?));
        }
        let (off, used) = self.read_used(ivval)?;
        let v = self
            .vol
            .read_record(off + 4, used, |inp| ser.deserialize(inp, Some(used)))?;
        Ok(Some(v))
    }

    fn read(&self, recid: Recid, action: &mut dyn RecordRead) -> Result<i64> {
        let _c = self.mutate_enter()?;
        let _rg = self.segs.read(recid.get());
        let ivval = self.index_get_checked(recid.get())?;
        if iv::cap_units(ivval) == iv::CAP_NULL {
            return action.on_null();
        }
        if iv::is_linked(ivval) {
            let b = self.linked_get(ivval)?;
            let mut inp = SliceInput::new(&b);
            return action.on_bytes(&mut inp, b.len());
        }
        let (off, used) = self.read_used(ivval)?;
        self.vol
            .read_record(off + 4, used, |inp| action.on_bytes(inp, used))
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
        self.check_closed()?;
        let _c = self.mutate_enter()?;
        let _wg = self.segs.write(recid.get());
        let ivval = self.index_get_checked(recid.get())?;
        let current: Option<R> = if iv::cap_units(ivval) == iv::CAP_NULL {
            None
        } else if iv::is_linked(ivval) {
            let b = self.linked_get(ivval)?;
            let mut inp = SliceInput::new(&b);
            Some(ser.deserialize(&mut inp, Some(b.len()))?)
        } else {
            let (off, used) = self.read_used(ivval)?;
            Some(
                self.vol
                    .read_record(off + 4, used, |inp| ser.deserialize(inp, Some(used)))?,
            )
        };
        let eq = match (&current, expect) {
            (None, None) => true,
            (Some(c), Some(e)) => ser.equals(c, e),
            _ => false,
        };
        if !eq {
            return Ok(false);
        }
        let out = new.map(|v| serialize(v, ser));
        self.update_locked(recid.get(), ivval, out.as_deref(), 0)?;
        Ok(true)
    }

    fn delete(&self, recid: Recid) -> Result<()> {
        self.check_closed()?;
        let _c = self.mutate_enter()?;
        {
            let _wg = self.segs.write(recid.get());
            let ivval = self.index_get_checked(recid.get())?;
            self.release_old_data(ivval)?;
            {
                let _s = self.structural();
                self.free_recid_locked(recid.get())?;
            }
            self.index_set(recid.get(), iv::compose(iv::CAP_DELETED, 0, 0));
        }
        Ok(())
    }

    fn commit(&self) -> Result<()> {
        self.check_closed()?;
        let _c = self.commit_lock.write();
        self.check_closed()?;
        self.stamp_header_durable()
    }

    fn compact(&self) -> Result<()> {
        self.check_closed()?;
        let _c = self.commit_lock.write();
        self.check_closed()?;
        // Take the read-only snapshot BEFORE the crash barrier. A failure here
        // (e.g. corrupt index parity, out-of-range offset) must return an
        // ordinary error WITHOUT poisoning/closing the store or touching the
        // on-disk checksum — nothing has been modified yet.
        let entries = self.compact_snapshot()?;
        let max = self.max_recid()?;
        match self.compact_inner(entries, max) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.poisoned.store(true, Ordering::Release);
                self.closed.store(true, Ordering::Release);
                self.vol.put_i32(O_HEAD_CHECKSUM, !self.head_checksum());
                let _ = self.vol.sync();
                Err(e)
            }
        }
    }

    fn close(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) && !self.poisoned.load(Ordering::Acquire) {
            return Ok(());
        }
        let _c = self.commit_lock.write();
        if self.closed.load(Ordering::Acquire) && !self.poisoned.load(Ordering::Acquire) {
            return Ok(());
        }
        let stamp = !self.poisoned.load(Ordering::Acquire);
        self.closed.store(true, Ordering::Release);
        self.poisoned.store(false, Ordering::Release);
        if stamp {
            let tail = self.file_tail()?;
            self.stamp_header_durable()?;
            self.vol.close(Some(tail))?;
        } else {
            self.vol.close(None)?;
        }
        self.index_pages.store(Arc::new(Vec::new()));
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn is_thread_safe(&self) -> bool {
        self.thread_safe
    }

    fn verify(&self) -> Result<()> {
        self.check_closed()?;
        // Stop-the-world: verify() reads whole-store geometry (index slots, long
        // stacks, record headers) with raw-pointer accessors. The shared commit
        // barrier does NOT exclude in-place update/CAS/append (they hold only a
        // segment write lock), so a read-side barrier would race those writers
        // (UB in Rust). The exclusive commit lock is the correct oracle scope.
        let _c = self.commit_lock.write();
        self.check_closed()?;
        let _s = self.structural();
        match self.verify_locked() {
            Ok(()) => Ok(()),
            Err(DbError::DataCorruption(c)) => Err(DbError::VerifyFailed(c.to_string())),
            Err(e) => Err(e),
        }
    }

    fn get_all_recids(&self) -> Result<Vec<Recid>> {
        let _c = self.mutate_enter()?;
        let max = {
            let _s = self.structural();
            self.max_recid()?
        };
        let mut out = Vec::new();
        for recid in 1..=max {
            // Each index slot is written under its segment write lock; read it
            // under the matching read lock so this scan never races a concurrent
            // in-place update/CAS that rewrites the slot (D5).
            let ivval = {
                let _rg = self.segs.read(recid);
                self.raw_index_get(recid)
            };
            if ivval == 0 {
                continue;
            }
            let cap = iv::cap_units(ivval);
            if cap == iv::CAP_DELETED || iv::is_prealloc(ivval) {
                continue;
            }
            out.push(nz(recid));
        }
        Ok(out)
    }

    fn get_current_size(&self) -> u64 {
        let _s = self.structural();
        let ft = self.file_tail().unwrap_or(0) as i64;
        (ft - self.free_data_bytes.load(Ordering::Relaxed)).max(0) as u64
    }
}

impl StoreDirect {
    fn update_with_headroom_opt<R: Record>(
        &self,
        recid: Recid,
        value: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
        headroom: usize,
    ) -> Result<()> {
        self.check_closed()?;
        let out = value.map(|v| serialize(v, ser));
        // content that fits a plain record must also fit with its headroom
        if let Some(ref o) = out {
            if !needs_linked(o.len() as u64) {
                cap_bytes_for(plain_need(o.len() as u64, headroom as u64)?)?;
            }
        }
        let _c = self.mutate_enter()?;
        let _wg = self.segs.write(recid.get());
        let ivval = self.index_get_checked(recid.get())?;
        self.update_locked(recid.get(), ivval, out.as_deref(), headroom)
    }

    /// segment write lock held; `iv` is the current (checked) index value.
    fn update_locked(
        &self,
        recid: u64,
        ivval: u64,
        out: Option<&[u8]>,
        headroom: usize,
    ) -> Result<()> {
        let old_cap = iv::cap_units(ivval);
        let Some(buf) = out else {
            self.release_old_data(ivval)?;
            self.index_set(recid, iv::compose(iv::CAP_NULL, 0, 0));
            return Ok(());
        };
        if needs_linked(buf.len() as u64) {
            self.release_old_data(ivval)?;
            self.write_new_linked(recid, buf)?;
            return Ok(());
        }
        let need = plain_need(buf.len() as u64, headroom as u64)?;
        if !iv::is_linked(ivval) && old_cap != iv::CAP_NULL && need <= old_cap as u64 * 16 {
            // in-place: capacity retained. Validate the value-derived offset
            // BEFORE writing — a parity-valid but corrupt index could otherwise
            // point `off` at header/allocator words and this write would silently
            // clobber them (later stamped clean by close). D4/D5.
            let off = iv::offset(ivval);
            if off < PAGE_SIZE || off & 15 != 0 {
                return Err(DbError::corrupt("record offset in header/misaligned"));
            }
            self.vol.check_range(off, old_cap as u64 * 16)?;
            self.vol.put_i32(off, buf.len() as i32);
            self.vol.put_data(off + 4, buf);
            self.index_set(recid, iv::compose(old_cap, off, 0));
        } else {
            self.release_old_data(ivval)?;
            self.write_new_data(recid, buf, cap_bytes_for(need)?, 0)?;
        }
        Ok(())
    }
}

impl StoreDelta for StoreDirect {
    fn append(&self, recid: Recid, data: &[u8]) -> Result<AppendResult> {
        self.check_closed()?;
        let _c = self.mutate_enter()?;
        let _wg = self.segs.write(recid.get());
        let ivval = self.index_get_checked(recid.get())?;
        if iv::is_linked(ivval) {
            return Ok(AppendResult::Refused);
        }
        if iv::cap_units(ivval) == iv::CAP_NULL {
            // first append establishes the record: capacity == exactly what is needed
            if needs_linked(data.len() as u64) {
                self.write_new_linked(recid.get(), data)?;
            } else {
                let cap_bytes = cap_bytes_for(4 + data.len() as u64)?;
                self.write_new_data(recid.get(), data, cap_bytes, 0)?;
            }
            return Ok(AppendResult::NewSize(data.len()));
        }
        // read_used validates the offset (>= PAGE_SIZE, aligned, in-slice) and
        // `used`, so we never write through a corrupt value-derived offset.
        let (off, used) = self.read_used(ivval)?;
        let cap_bytes = iv::cap_units(ivval) as usize * 16;
        if 4 + used + data.len() > cap_bytes {
            return Ok(AppendResult::Refused);
        }
        self.vol.check_range(off, (4 + used + data.len()) as u64)?;
        self.vol.put_data(off + 4 + used as u64, data);
        self.vol.put_i32(off, (used + data.len()) as i32);
        Ok(AppendResult::NewSize(used + data.len()))
    }

    fn capacity_remaining(&self, recid: Recid) -> Result<usize> {
        let _c = self.mutate_enter()?;
        let _rg = self.segs.read(recid.get());
        let ivval = self.index_get_checked(recid.get())?;
        if iv::cap_units(ivval) == iv::CAP_NULL || iv::is_linked(ivval) {
            return Ok(0);
        }
        let (_off, used) = self.read_used(ivval)?;
        Ok((iv::cap_units(ivval) as usize * 16).saturating_sub(4 + used))
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
