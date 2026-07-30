//! `ConfiguredStore` — a closed enum of the backends the runtime [`DBMaker`]
//! (crate::db::maker) can produce, so a runtime-configured builder can return one
//! concrete `DB<ConfiguredStore>` type.
//!
//! The enum implements the serializer-generic [`Store`] and [`StoreLease`] traits
//! by `match`-forwarding to the wrapped backend. `Store` is not object-safe, but
//! a generic method *is* legal on an enum, and each collection still monomorphizes
//! exactly once over `ConfiguredStore`. This is the one place per store operation
//! that pays a branch; it lives only at the convenience-facade edge, never in the
//! collection core (which keeps using the fully-generic typed constructors).
//!
//! Read-only variants wrap the backend in [`StoreReadOnlyWrapper`]; `Wal` has no
//! read-only variant because read-only + WAL is rejected by the maker.
//!
//! `ConfiguredStore` intentionally implements ONLY [`Store`] and [`StoreLease`]
//! (plus [`DbRollback`](crate::db::db::DbRollback) via its inherent `rollback`).
//! It deliberately does NOT re-expose the concrete backend traits (`StoreTx`,
//! pump/columnar/verify surfaces): the facade only ever needs the serializer-
//! generic store contract, and hiding the rest keeps callers off backend-specific
//! behavior that is not uniform across the variants.

use crate::error::{DbError, Result};
use crate::ser::Serializer;
use crate::store::{
    AppendResult, LeaseTable, Recid, Record, RecordRead, Store, StoreByteArray, StoreDelta,
    StoreDirect, StoreLease, StoreOnHeap, StoreReadOnlyWrapper, StoreTx, StoreWAL,
};
use std::sync::Arc;

/// The backend a runtime-configured [`DB`](crate::db::DB) is built over.
pub enum ConfiguredStore {
    Heap(StoreOnHeap),
    ByteArray(StoreByteArray),
    Direct(StoreDirect),
    Wal(StoreWAL),
    ReadOnlyHeap(StoreReadOnlyWrapper<StoreOnHeap>),
    ReadOnlyByteArray(StoreReadOnlyWrapper<StoreByteArray>),
    ReadOnlyDirect(StoreReadOnlyWrapper<StoreDirect>),
}

/// Expand `$m` to a match over every variant, calling method `$method` with `$args`.
macro_rules! forward {
    ($self:expr, $method:ident ( $($args:expr),* )) => {
        match $self {
            ConfiguredStore::Heap(s) => s.$method($($args),*),
            ConfiguredStore::ByteArray(s) => s.$method($($args),*),
            ConfiguredStore::Direct(s) => s.$method($($args),*),
            ConfiguredStore::Wal(s) => s.$method($($args),*),
            ConfiguredStore::ReadOnlyHeap(s) => s.$method($($args),*),
            ConfiguredStore::ReadOnlyByteArray(s) => s.$method($($args),*),
            ConfiguredStore::ReadOnlyDirect(s) => s.$method($($args),*),
        }
    };
}

impl Store for ConfiguredStore {
    fn preallocate(&self) -> Result<Recid> {
        forward!(self, preallocate())
    }
    fn preallocate_many(&self, into: &mut [Recid]) -> Result<()> {
        forward!(self, preallocate_many(into))
    }
    fn put<R: Record>(&self, value: &R, ser: &(impl Serializer<R> + Sync)) -> Result<Recid> {
        forward!(self, put(value, ser))
    }
    fn get<R: Record>(&self, recid: Recid, ser: &(impl Serializer<R> + Sync)) -> Result<Option<R>> {
        forward!(self, get(recid, ser))
    }
    fn read(&self, recid: Recid, action: &mut dyn RecordRead) -> Result<i64> {
        forward!(self, read(recid, action))
    }
    fn update<R: Record>(
        &self,
        recid: Recid,
        value: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
    ) -> Result<()> {
        forward!(self, update(recid, value, ser))
    }
    fn compare_and_swap<R: Record>(
        &self,
        recid: Recid,
        expect: Option<&R>,
        new: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
    ) -> Result<bool> {
        forward!(self, compare_and_swap(recid, expect, new, ser))
    }
    fn delete(&self, recid: Recid) -> Result<()> {
        forward!(self, delete(recid))
    }
    fn commit(&self) -> Result<()> {
        forward!(self, commit())
    }
    fn compact(&self) -> Result<()> {
        forward!(self, compact())
    }
    fn close(&self) -> Result<()> {
        forward!(self, close())
    }
    fn is_closed(&self) -> bool {
        forward!(self, is_closed())
    }
    fn verify(&self) -> Result<()> {
        forward!(self, verify())
    }
    fn get_all_recids(&self) -> Result<Vec<Recid>> {
        forward!(self, get_all_recids())
    }
    fn is_thread_safe(&self) -> bool {
        forward!(self, is_thread_safe())
    }
    fn is_read_only(&self) -> bool {
        forward!(self, is_read_only())
    }
    fn get_current_size(&self) -> u64 {
        forward!(self, get_current_size())
    }
    fn is_tx(&self) -> bool {
        forward!(self, is_tx())
    }
    fn structural_generation(&self) -> u64 {
        forward!(self, structural_generation())
    }
}

impl StoreLease for ConfiguredStore {
    fn lease_table(&self) -> &Arc<LeaseTable> {
        forward!(self, lease_table())
    }
}

impl ConfiguredStore {
    /// Roll back the transaction, if this backend is transactional (only WAL).
    /// Mirrors Java's `UnsupportedOperationException` for non-tx stores.
    pub fn rollback(&self) -> Result<()> {
        match self {
            ConfiguredStore::Wal(s) => s.rollback(),
            _ => Err(DbError::Unsupported(
                "rollback on a non-transactional store",
            )),
        }
    }

    /// In-place append; only byte-backed non-read-only backends support it. Used
    /// by no in-scope collection, but kept for completeness of the store surface.
    pub fn try_append(&self, recid: Recid, data: &[u8]) -> Result<AppendResult> {
        match self {
            ConfiguredStore::ByteArray(s) => s.append(recid, data),
            ConfiguredStore::Direct(s) => s.append(recid, data),
            ConfiguredStore::Wal(s) => s.append(recid, data),
            _ => Err(DbError::Unsupported("append on this backend")),
        }
    }
}
