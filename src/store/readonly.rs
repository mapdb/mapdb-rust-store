//! [`StoreReadOnlyWrapper`] — a [`Store`] decorator that rejects every mutating
//! operation, exposing a delegate store as logically read-only (Java
//! `org.mapdb.store.StoreReadOnlyWrapper`).
//!
//! Read/inspection calls (`get`, `read`, `get_all_recids`, `verify`,
//! `is_closed`, `is_thread_safe`, `get_current_size`) pass straight through;
//! mutators (`preallocate`, `put`, `update`, `delete`, `compare_and_swap`,
//! `compact`, `append`, `update_with_headroom`) return
//! [`DbError::ReadOnly`](crate::error::DbError::ReadOnly). `commit` is a
//! harmless no-op (a read-only DB may still call it) and `close` closes the
//! delegate.
//!
//! **Logical guard, not an OS-level mode** (as in Java): this rejects mutations
//! at the [`Store`] API only; it does not downgrade the underlying file mapping
//! to read-only. It is not a [`StoreTx`](super::StoreTx) — a read-only view has
//! nothing to roll back.

use crate::error::{DbError, Result};
use crate::ser::Serializer;
use crate::store::lease::LeaseTable;
use crate::store::{AppendResult, Recid, Record, RecordRead, Store, StoreDelta, StoreLease};
use std::sync::Arc;

/// Wraps any [`Store`] and rejects mutations. Generic over the delegate so the
/// core stays monomorphized (decision D1).
#[derive(Debug)]
pub struct StoreReadOnlyWrapper<S> {
    delegate: S,
}

impl<S> StoreReadOnlyWrapper<S> {
    pub fn new(delegate: S) -> Self {
        Self { delegate }
    }

    /// Borrow the wrapped store (reads only — mutating through it bypasses the
    /// guard, so callers must not).
    pub fn delegate(&self) -> &S {
        &self.delegate
    }

    /// Unwrap, returning the delegate store.
    pub fn into_inner(self) -> S {
        self.delegate
    }
}

impl<S: Store> Store for StoreReadOnlyWrapper<S> {
    // ---- mutators: rejected -------------------------------------------------

    fn preallocate(&self) -> Result<Recid> {
        Err(DbError::ReadOnly)
    }

    fn put<R: Record>(&self, _value: &R, _ser: &(impl Serializer<R> + Sync)) -> Result<Recid> {
        Err(DbError::ReadOnly)
    }

    fn update<R: Record>(
        &self,
        _recid: Recid,
        _value: Option<&R>,
        _ser: &(impl Serializer<R> + Sync),
    ) -> Result<()> {
        Err(DbError::ReadOnly)
    }

    fn compare_and_swap<R: Record>(
        &self,
        _recid: Recid,
        _expect: Option<&R>,
        _new: Option<&R>,
        _ser: &(impl Serializer<R> + Sync),
    ) -> Result<bool> {
        Err(DbError::ReadOnly)
    }

    fn delete(&self, _recid: Recid) -> Result<()> {
        Err(DbError::ReadOnly)
    }

    fn compact(&self) -> Result<()> {
        Err(DbError::ReadOnly)
    }

    // ---- reads / inspection: delegated -------------------------------------

    fn get<R: Record>(&self, recid: Recid, ser: &(impl Serializer<R> + Sync)) -> Result<Option<R>> {
        self.delegate.get(recid, ser)
    }

    fn read(&self, recid: Recid, action: &mut dyn RecordRead) -> Result<i64> {
        self.delegate.read(recid, action)
    }

    fn get_all_recids(&self) -> Result<Vec<Recid>> {
        self.delegate.get_all_recids()
    }

    fn verify(&self) -> Result<()> {
        self.delegate.verify()
    }

    fn is_closed(&self) -> bool {
        self.delegate.is_closed()
    }

    fn is_thread_safe(&self) -> bool {
        self.delegate.is_thread_safe()
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn get_current_size(&self) -> u64 {
        self.delegate.get_current_size()
    }

    fn is_tx(&self) -> bool {
        self.delegate.is_tx()
    }

    fn structural_generation(&self) -> u64 {
        self.delegate.structural_generation()
    }

    // ---- lifecycle ----------------------------------------------------------

    /// No-op: a read-only view has nothing to make durable. Tolerated so callers
    /// may `commit()`.
    fn commit(&self) -> Result<()> {
        Ok(())
    }

    /// Closes the underlying store so its resources are released.
    fn close(&self) -> Result<()> {
        self.delegate.close()
    }
}

/// Delegate the lease table so read-only collections can still be opened over
/// the wrapper (decision D12).
impl<S: StoreLease> StoreLease for StoreReadOnlyWrapper<S> {
    fn lease_table(&self) -> &Arc<LeaseTable> {
        self.delegate.lease_table()
    }
}

/// Delta surface: capacity queries pass through, growth is rejected.
impl<S: StoreDelta> StoreDelta for StoreReadOnlyWrapper<S> {
    fn append(&self, _recid: Recid, _data: &[u8]) -> Result<AppendResult> {
        Err(DbError::ReadOnly)
    }

    fn capacity_remaining(&self, recid: Recid) -> Result<usize> {
        self.delegate.capacity_remaining(recid)
    }

    fn update_with_headroom<R: Record>(
        &self,
        _recid: Recid,
        _value: &R,
        _ser: &(impl Serializer<R> + Sync),
        _headroom: usize,
    ) -> Result<()> {
        Err(DbError::ReadOnly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ser::serializers::LongSer;
    use crate::store::StoreByteArray;

    fn wrapped() -> (StoreReadOnlyWrapper<StoreByteArray>, Recid) {
        // Build a store with a record, then wrap it read-only.
        let inner = StoreByteArray::new(true);
        let recid = inner.put(&42i64, &LongSer).unwrap();
        (StoreReadOnlyWrapper::new(inner), recid)
    }

    #[test]
    fn reads_pass_through() {
        let (ro, recid) = wrapped();
        assert!(ro.is_read_only());
        assert_eq!(ro.get(recid, &LongSer).unwrap(), Some(42i64));
        assert!(!ro.is_closed());
        ro.verify().unwrap();
        assert_eq!(ro.get_all_recids().unwrap(), vec![recid]);
    }

    #[test]
    fn mutations_rejected_with_readonly() {
        let (ro, recid) = wrapped();
        assert!(matches!(ro.preallocate(), Err(DbError::ReadOnly)));
        assert!(matches!(ro.put(&1i64, &LongSer), Err(DbError::ReadOnly)));
        assert!(matches!(
            ro.update(recid, Some(&7i64), &LongSer),
            Err(DbError::ReadOnly)
        ));
        assert!(matches!(
            ro.compare_and_swap(recid, Some(&42i64), Some(&7i64), &LongSer),
            Err(DbError::ReadOnly)
        ));
        assert!(matches!(ro.delete(recid), Err(DbError::ReadOnly)));
        assert!(matches!(ro.compact(), Err(DbError::ReadOnly)));
        // The record is unchanged after every rejected mutation.
        assert_eq!(ro.get(recid, &LongSer).unwrap(), Some(42i64));
    }

    #[test]
    fn commit_is_noop_and_delta_rejects_growth() {
        let (ro, recid) = wrapped();
        ro.commit().unwrap();
        assert!(matches!(ro.append(recid, b"x"), Err(DbError::ReadOnly)));
        assert!(matches!(
            ro.update_with_headroom(recid, &9i64, &LongSer, 16),
            Err(DbError::ReadOnly)
        ));
        // capacity_remaining passes through without error.
        let _ = ro.capacity_remaining(recid).unwrap();
    }
}
