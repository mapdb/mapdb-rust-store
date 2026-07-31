//! Paged volume: 1 MiB slices, heap or file-backed mmap (Java `ByteBufferVol`,
//! spec 02 §3). Records never cross a slice boundary (allocator invariant).
//!
//! # Unsafe policy (decision D5)
//! This is the ONLY module with `unsafe` for StoreDirect. Slice bytes are
//! accessed through raw pointers. Soundness rests on the store's lock discipline
//! (invariant, enforced by the caller):
//! *a byte range is written only while its record's segment write lock (or the
//! structural lock, for allocator/header words) is held, and read only under the
//! matching read lock — so no plain read ever races a write to the same range.*
//! Scalar accessors copy out immediately (no borrow escapes); only
//! [`Volume::read_record`] hands out a `&[u8]`, and only over record content
//! (offset ≥ PAGE_SIZE) held under the segment read lock, disjoint from any
//! concurrent writer's range. Growth republishes the slice table via `ArcSwap`;
//! existing slices are never mutated below published length and never moved.

use crate::error::{DbError, Result};
use crate::io::SliceInput;
use arc_swap::ArcSwap;
use memmap2::{MmapMut, MmapOptions};
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const SLICE_SHIFT: u32 = 20;
pub const SLICE_SIZE: u64 = 1 << SLICE_SHIFT; // 1 MiB
pub const SLICE_MASK: u64 = SLICE_SIZE - 1;

/// Owns the backing allocation so `Slice::ptr` stays valid; the bytes are
/// reached through the pointer, so the fields are intentionally "unread".
#[allow(dead_code)]
enum SliceBacking {
    Heap(Box<[u8]>),
    Mmap(MmapMut),
}

/// One 1 MiB slice. `ptr` addresses the start of the backing region (stable:
/// heap/mmap allocations don't move when the enum is moved).
struct Slice {
    ptr: *mut u8,
    backing: SliceBacking,
}

// SAFETY: access to the pointed-at bytes is serialized by the store's lock
// discipline (see module docs); the pointer is stable for the slice's life.
unsafe impl Send for Slice {}
unsafe impl Sync for Slice {}

impl Slice {
    fn heap() -> Slice {
        let mut b = vec![0u8; SLICE_SIZE as usize].into_boxed_slice();
        let ptr = b.as_mut_ptr();
        Slice {
            ptr,
            backing: SliceBacking::Heap(b),
        }
    }

    fn mmap(mut m: MmapMut) -> Slice {
        let ptr = m.as_mut_ptr();
        Slice {
            ptr,
            backing: SliceBacking::Mmap(m),
        }
    }

    // Every accessor bounds-checks `[off, off+len) ⊆ [0, SLICE_SIZE)`
    // UNCONDITIONALLY (not `debug_assert!`): the pointer arithmetic and
    // copy/`from_raw_parts` below are `unsafe`, so an out-of-range `off` derived
    // from a *corrupt* index value must never reach them, or it is UB. A corrupt
    // file thus panics here at worst (memory-safe); callers on value-derived
    // offsets pre-validate via [`Volume::check_range`] to return `DataCorruption`
    // instead of panicking (decision D4/D5).
    #[inline]
    fn bound(off: usize, len: usize) {
        assert!(
            off <= SLICE_SIZE as usize && len <= SLICE_SIZE as usize - off,
            "volume access [{off}, {off}+{len}) out of slice bounds",
        );
    }
    #[inline]
    fn write_u8(&self, off: usize, v: u8) {
        Self::bound(off, 1);
        // SAFETY: bounds proven above; write under the relevant lock, disjoint
        // from any concurrent read of this range.
        unsafe { *self.ptr.add(off) = v };
    }
    #[inline]
    fn read_u8(&self, off: usize) -> u8 {
        Self::bound(off, 1);
        unsafe { *self.ptr.add(off) }
    }
    #[inline]
    fn write_bytes(&self, off: usize, src: &[u8]) {
        Self::bound(off, src.len());
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(off), src.len()) };
    }
    #[inline]
    fn read_bytes(&self, off: usize, dst: &mut [u8]) {
        Self::bound(off, dst.len());
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.add(off), dst.as_mut_ptr(), dst.len()) };
    }
    #[inline]
    fn zero(&self, off: usize, len: usize) {
        Self::bound(off, len);
        unsafe { std::ptr::write_bytes(self.ptr.add(off), 0, len) };
    }
    #[inline]
    fn as_slice(&self, off: usize, len: usize) -> &[u8] {
        Self::bound(off, len);
        // SAFETY: [off, off+len) is in-bounds (proven above) and, per the lock
        // discipline, not being written concurrently for the borrow's duration.
        unsafe { std::slice::from_raw_parts(self.ptr.add(off), len) }
    }

    fn flush(&self) -> Result<()> {
        if let SliceBacking::Mmap(m) = &self.backing {
            m.flush().map_err(DbError::Io)?;
        }
        Ok(())
    }
}

/// The paged volume. `file` is `None` for an anonymous heap volume.
pub struct Volume {
    slices: ArcSwap<Vec<Arc<Slice>>>,
    file: Option<Mutex<File>>,
    path: Option<PathBuf>,
    grow_lock: Mutex<()>,
}

impl Volume {
    /// Anonymous heap-backed volume.
    pub fn heap() -> Volume {
        Volume {
            slices: ArcSwap::from_pointee(Vec::new()),
            file: None,
            path: None,
            grow_lock: Mutex::new(()),
        }
    }

    /// File-backed mmap volume (created if absent).
    pub fn open_file(path: &Path) -> Result<Volume> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        Ok(Volume {
            slices: ArcSwap::from_pointee(Vec::new()),
            file: Some(Mutex::new(file)),
            path: Some(path.to_path_buf()),
            grow_lock: Mutex::new(()),
        })
    }

    pub fn is_file_backed(&self) -> bool {
        self.file.is_some()
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Physical file length (file mode) or addressable mapped length (memory mode).
    pub fn length(&self) -> Result<u64> {
        match &self.file {
            None => Ok((self.slices.load().len() as u64) << SLICE_SHIFT),
            Some(f) => Ok(f.lock().metadata()?.len()),
        }
    }

    #[inline]
    fn slice_of(&self, offset: u64) -> Arc<Slice> {
        let slices = self.slices.load();
        let idx = (offset >> SLICE_SHIFT) as usize;
        Arc::clone(&slices[idx])
    }

    /// Validate that `[offset, offset+len)` is addressable and lies within a
    /// single slice, returning `DataCorruption` otherwise. Callers reading at an
    /// offset derived from a (possibly corrupt) index/link value MUST call this
    /// before the raw accessors, so corruption yields a graceful error rather
    /// than the `bound()` panic backstop (D4).
    pub fn check_range(&self, offset: u64, len: u64) -> Result<()> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| DbError::corrupt("volume offset arithmetic overflow"))?;
        let addressable = (self.slices.load().len() as u64) << SLICE_SHIFT;
        if end > addressable {
            return Err(DbError::corrupt("volume offset past addressable end"));
        }
        if (offset & SLICE_MASK) + len > SLICE_SIZE {
            return Err(DbError::corrupt("volume access crosses a slice boundary"));
        }
        Ok(())
    }

    /// Grow so bytes `[0, end_offset)` are addressable.
    pub fn ensure_available(&self, end_offset: u64) -> Result<()> {
        let needed = ((end_offset + SLICE_MASK) >> SLICE_SHIFT) as usize;
        let _g = self.grow_lock.lock();
        let cur = self.slices.load_full();
        if cur.len() >= needed {
            return Ok(());
        }
        let mut grown: Vec<Arc<Slice>> = (*cur).clone();
        for i in cur.len()..needed {
            let slice = match &self.file {
                None => Slice::heap(),
                Some(fmutex) => {
                    let f = fmutex.lock();
                    let end = ((i as u64) + 1) << SLICE_SHIFT;
                    if f.metadata()?.len() < end {
                        f.set_len(end)?;
                    }
                    // SAFETY: the file region [i*SLICE_SIZE, (i+1)*SLICE_SIZE) exists
                    // (set_len above) and is mapped READ_WRITE exactly once.
                    let m = unsafe {
                        MmapOptions::new()
                            .offset((i as u64) << SLICE_SHIFT)
                            .len(SLICE_SIZE as usize)
                            .map_mut(&*f)
                            .map_err(DbError::Io)?
                    };
                    Slice::mmap(m)
                }
            };
            grown.push(Arc::new(slice));
        }
        self.slices.store(Arc::new(grown));
        Ok(())
    }

    // ---- scalar accessors (value copies out; no borrow escapes) ----

    pub fn put_i32(&self, offset: u64, v: i32) {
        let s = self.slice_of(offset);
        s.write_bytes((offset & SLICE_MASK) as usize, &v.to_be_bytes());
    }
    pub fn get_i32(&self, offset: u64) -> i32 {
        let s = self.slice_of(offset);
        let mut b = [0u8; 4];
        s.read_bytes((offset & SLICE_MASK) as usize, &mut b);
        i32::from_be_bytes(b)
    }
    pub fn put_u64(&self, offset: u64, v: u64) {
        let s = self.slice_of(offset);
        s.write_bytes((offset & SLICE_MASK) as usize, &v.to_be_bytes());
    }
    pub fn get_u64(&self, offset: u64) -> u64 {
        let s = self.slice_of(offset);
        let mut b = [0u8; 8];
        s.read_bytes((offset & SLICE_MASK) as usize, &mut b);
        u64::from_be_bytes(b)
    }
    pub fn put_byte(&self, offset: u64, v: u8) {
        let s = self.slice_of(offset);
        s.write_u8((offset & SLICE_MASK) as usize, v);
    }
    pub fn get_u8(&self, offset: u64) -> u8 {
        let s = self.slice_of(offset);
        s.read_u8((offset & SLICE_MASK) as usize)
    }

    /// Absolute put of `src` at `offset`; must not cross a slice boundary.
    pub fn put_data(&self, offset: u64, src: &[u8]) {
        debug_assert!((offset & SLICE_MASK) + src.len() as u64 <= SLICE_SIZE);
        let s = self.slice_of(offset);
        s.write_bytes((offset & SLICE_MASK) as usize, src);
    }

    /// Absolute get into `dst`; must not cross a slice boundary.
    pub fn get_data(&self, offset: u64, dst: &mut [u8]) {
        debug_assert!((offset & SLICE_MASK) + dst.len() as u64 <= SLICE_SIZE);
        let s = self.slice_of(offset);
        s.read_bytes((offset & SLICE_MASK) as usize, dst);
    }

    /// Zero `[from, to)`; may span slices.
    pub fn clear(&self, from: u64, to: u64) {
        let mut p = from;
        while p < to {
            let s = self.slice_of(p);
            let off = (p & SLICE_MASK) as usize;
            let n = ((to - p).min(SLICE_SIZE - off as u64)) as usize;
            s.zero(off, n);
            p += n as u64;
        }
    }

    /// Run `f` over a `SliceInput` covering record content `[offset, offset+size)`
    /// as a 0-based buffer. Must not cross a slice boundary. The borrow does not
    /// escape `f`.
    pub fn read_record<T>(
        &self,
        offset: u64,
        size: usize,
        f: impl FnOnce(&mut SliceInput<'_>) -> Result<T>,
    ) -> Result<T> {
        debug_assert!((offset & SLICE_MASK) + size as u64 <= SLICE_SIZE);
        let s = self.slice_of(offset);
        let bytes = s.as_slice((offset & SLICE_MASK) as usize, size);
        let mut inp = SliceInput::new(bytes);
        f(&mut inp)
    }

    // ---- durability ----

    /// Full durability point: flush every mapped slice, then fsync. No-op in memory mode.
    pub fn sync(&self) -> Result<()> {
        let Some(f) = &self.file else { return Ok(()) };
        for s in self.slices.load().iter() {
            s.flush()?;
        }
        f.lock().sync_all()?;
        Ok(())
    }

    /// Header-page-only durability (slice 0), for the second phase of commit.
    pub fn sync_header(&self) -> Result<()> {
        let Some(f) = &self.file else { return Ok(()) };
        let slices = self.slices.load();
        if let Some(s0) = slices.first() {
            s0.flush()?;
        }
        f.lock().sync_all()?;
        Ok(())
    }

    /// Shrink the addressable volume to `truncate_to` (page-aligned).
    pub fn truncate(&self, truncate_to: u64) -> Result<()> {
        if truncate_to & SLICE_MASK != 0 {
            return Err(DbError::corrupt("truncate target not page-aligned"));
        }
        let _g = self.grow_lock.lock();
        let needed = (truncate_to >> SLICE_SHIFT) as usize;
        let cur = self.slices.load_full();
        if cur.len() > needed {
            let shrunk: Vec<Arc<Slice>> = cur[..needed].to_vec();
            self.slices.store(Arc::new(shrunk));
        }
        if let Some(f) = &self.file {
            let f = f.lock();
            if truncate_to < f.metadata()?.len() {
                f.set_len(truncate_to)?;
            }
            f.sync_all()?;
        }
        Ok(())
    }

    /// Release all slices; in file mode optionally shrink to `truncate_to` (>= 0).
    pub fn close(&self, truncate_to: Option<u64>) -> Result<()> {
        let _g = self.grow_lock.lock();
        self.slices.store(Arc::new(Vec::new()));
        if let Some(f) = &self.file {
            let f = f.lock();
            if let Some(t) = truncate_to {
                if t < f.metadata()?.len() {
                    f.set_len(t)?;
                }
            }
            f.sync_all()?;
        }
        Ok(())
    }
}
