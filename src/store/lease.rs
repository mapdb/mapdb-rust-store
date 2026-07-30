//! One-live-handle lease (decision D12). Each store instance embeds a
//! [`LeaseTable`]; collections acquire an RAII [`LeaseGuard`] at open and release
//! it on last-clone drop. Hard exclusion (release builds included):
//! - `open(ReadWrite)` fails if ANY lease exists on the header recid;
//! - `open(ReadOnly)` fails if an RW lease exists; RO+RO is allowed.

use crate::error::{DbError, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// Requested access mode for a lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseKind {
    /// Exclusive: excludes every other lease on the header recid.
    ReadWrite,
    /// Shared read: multiple RO allowed; excluded while an RW lease is live.
    ReadOnly,
}

#[derive(Debug)]
enum Entry {
    Rw,
    Ro(usize),
}

/// Per-store-instance lease registry keyed by header/root recid. One map op per
/// open, zero per access.
#[derive(Default)]
pub struct LeaseTable {
    map: Mutex<HashMap<u64, Entry>>,
}

impl LeaseTable {
    pub fn new() -> Arc<LeaseTable> {
        Arc::new(LeaseTable {
            map: Mutex::new(HashMap::new()),
        })
    }

    /// Acquire a lease. `Err(AlreadyOpen)` on a conflicting existing lease.
    pub fn acquire(self: &Arc<Self>, header_recid: u64, kind: LeaseKind) -> Result<LeaseGuard> {
        let mut map = self.map.lock();
        match (map.get_mut(&header_recid), kind) {
            (None, LeaseKind::ReadWrite) => {
                map.insert(header_recid, Entry::Rw);
            }
            (None, LeaseKind::ReadOnly) => {
                map.insert(header_recid, Entry::Ro(1));
            }
            (Some(Entry::Ro(n)), LeaseKind::ReadOnly) => {
                *n += 1;
            }
            // RW-while-anything, RO-while-RW, RW-while-RO → conflict
            _ => return Err(DbError::AlreadyOpen { header_recid }),
        }
        Ok(LeaseGuard {
            table: Arc::clone(self),
            header_recid,
            kind,
        })
    }
}

/// RAII lease. Releases the entry on drop; carries an `Arc<LeaseTable>` so it
/// outlives the collection's `MapState` correctly.
pub struct LeaseGuard {
    table: Arc<LeaseTable>,
    header_recid: u64,
    kind: LeaseKind,
}

impl LeaseGuard {
    pub fn kind(&self) -> LeaseKind {
        self.kind
    }
    pub fn header_recid(&self) -> u64 {
        self.header_recid
    }
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        let mut map = self.table.map.lock();
        match map.get_mut(&self.header_recid) {
            Some(Entry::Ro(n)) if *n > 1 => *n -= 1,
            Some(_) => {
                map.remove(&self.header_recid);
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_exclusion_rules() {
        let t = LeaseTable::new();
        // double RW rejected
        let rw = t.acquire(1, LeaseKind::ReadWrite).unwrap();
        assert!(matches!(
            t.acquire(1, LeaseKind::ReadWrite),
            Err(DbError::AlreadyOpen { header_recid: 1 })
        ));
        // RO-while-RW rejected
        assert!(t.acquire(1, LeaseKind::ReadOnly).is_err());
        drop(rw);
        // after drop, reopen succeeds
        let ro1 = t.acquire(1, LeaseKind::ReadOnly).unwrap();
        // RO+RO ok
        let ro2 = t.acquire(1, LeaseKind::ReadOnly).unwrap();
        // RW-while-RO rejected
        assert!(t.acquire(1, LeaseKind::ReadWrite).is_err());
        drop(ro1);
        // still one RO live → RW still rejected
        assert!(t.acquire(1, LeaseKind::ReadWrite).is_err());
        drop(ro2);
        // all released → RW ok
        assert!(t.acquire(1, LeaseKind::ReadWrite).is_ok());
        // different header independent
        let _a = t.acquire(2, LeaseKind::ReadWrite).unwrap();
    }
}
