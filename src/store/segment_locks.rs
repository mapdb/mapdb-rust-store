//! `SegmentLocks` — a fixed array of cache-line-padded reader/writer locks keyed
//! by recid low bits (Java `SegmentLocks`, spec 02 §5). In single-threaded mode
//! the locks are no-ops (an `enum` guard keeps the branch out of hot paths).

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Default number of segments (Java `SegmentLocks.DEFAULT_COUNT`).
pub const DEFAULT_COUNT: usize = 64;

/// Cache-line padding to avoid false sharing between adjacent segment locks.
#[repr(align(64))]
struct Padded(RwLock<()>);

/// A bank of per-segment RW locks, or a no-op bank for single-threaded stores.
pub struct SegmentLocks {
    locks: Option<Box<[Padded]>>,
    mask: usize,
}

impl SegmentLocks {
    /// `count` must be a power of two. `thread_safe == false` builds a no-op bank.
    pub fn new(count: usize, thread_safe: bool) -> Self {
        assert!(
            count.is_power_of_two(),
            "segment count must be power of two"
        );
        if thread_safe {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(Padded(RwLock::new(())));
            }
            SegmentLocks {
                locks: Some(v.into_boxed_slice()),
                mask: count - 1,
            }
        } else {
            SegmentLocks {
                locks: None,
                mask: count - 1,
            }
        }
    }

    /// Convenience constructor with the default segment count.
    pub fn default_for(thread_safe: bool) -> Self {
        Self::new(DEFAULT_COUNT, thread_safe)
    }

    #[inline]
    fn index(&self, recid: u64) -> usize {
        (recid as usize) & self.mask
    }

    /// Acquire the read guard for `recid`'s segment.
    #[inline]
    pub fn read(&self, recid: u64) -> SegReadGuard<'_> {
        match &self.locks {
            Some(l) => SegReadGuard::Real(l[self.index(recid)].0.read()),
            None => SegReadGuard::NoOp,
        }
    }

    /// Acquire the write guard for `recid`'s segment.
    #[inline]
    pub fn write(&self, recid: u64) -> SegWriteGuard<'_> {
        match &self.locks {
            Some(l) => SegWriteGuard::Real(l[self.index(recid)].0.write()),
            None => SegWriteGuard::NoOp,
        }
    }
}

/// RAII read guard; `NoOp` in single-threaded mode.
pub enum SegReadGuard<'a> {
    Real(RwLockReadGuard<'a, ()>),
    NoOp,
}

/// RAII write guard; `NoOp` in single-threaded mode.
pub enum SegWriteGuard<'a> {
    Real(RwLockWriteGuard<'a, ()>),
    NoOp,
}
