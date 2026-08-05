//! `store` layer — the `Store` trait chain and its implementations (spec 02).
//!
//! The trait is **serializer-generic and NOT object-safe** (decision D1):
//! collections are statically generic over `S: Store`, monomorphized. This lets
//! `StoreOnHeap` keep live objects and dispatch the object dialect, and
//! preserves serializer-defined *logical* CAS equality on byte stores.
//!
//! Interface chain `Store ← StoreDelta ← StoreTx`; impls: [`StoreOnHeap`],
//! [`StoreByteArray`] (the reference oracle), and later StoreDirect / StoreWAL /
//! StoreAppendOnly.

use crate::error::Result;
use crate::io::SliceInput;
use crate::ser::Serializer;
use std::any::Any;
use std::num::NonZeroU64;

pub mod lease;
pub mod locks;
pub mod segment_locks;

pub mod bytearray;
pub mod direct;
pub mod heap;
pub mod index_val;
pub mod parity;
pub mod readonly;
pub mod volume;
pub mod wal;
/// WAL format v3 codec and recovery state machine (tables S, K, R).
pub(crate) mod wal_recover;
/// WAL format v3 namespace layer: files, the store lock, tables N and H.
pub(crate) mod wal_segments;
/// WAL format v3 section writer (table W) and the durability event seam.
pub(crate) mod wal_write;

/// Cross-port fixture harness shared by the in-crate `ro` executor and the
/// integration tests (`#[path]`-included there). See C-D3.
#[cfg(test)]
mod xfix;
/// The schema-v2 `ro` cells, in-crate because `open_cfg` is `pub(crate)`.
#[cfg(test)]
mod xfix_ro;

pub use bytearray::StoreByteArray;
pub use direct::StoreDirect;
pub use heap::StoreOnHeap;
pub use lease::{LeaseGuard, LeaseKind, LeaseTable};
pub use readonly::StoreReadOnlyWrapper;
pub use wal::StoreWAL;

/// Record identifier. Recid 0 is never allocated (universal "no link"
/// sentinel), so a `NonZeroU64` gives niche-packed `Option<Recid>` for free
/// (decision D8). DirTree/htree use raw `u64` with 0 as an in-band absent
/// sentinel at those call sites.
pub type Recid = NonZeroU64;

/// Bound alias for storable values (spec 02 §1). Required on every typed trait
/// method: an impl cannot strengthen bounds, and `StoreOnHeap` clones the value
/// into `Arc<dyn Any + Send + Sync>`.
pub trait Record: Clone + Send + Sync + 'static {}
impl<T: Clone + Send + Sync + 'static> Record for T {}

/// Result of [`StoreDelta::append`]: the new content size, or a capacity
/// refusal (Java `REFUSED = -1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendResult {
    NewSize(usize),
    Refused,
}

/// Sealed capability marking a decode action as torn-safe (decision D4). User
/// code cannot name the bound, so only audited built-in actions reach the
/// phase-2 optimistic path. See [`StoreTornRead`].
pub(crate) mod sealed {
    /// Implemented only by audited built-in decode paths (checked seek/arith,
    /// capped varints, bounded allocation). Guards panic-freedom on garbage
    /// bytes, not memory safety.
    pub trait TornSafeDecode {}
}

/// Push-down read action (spec 02 §1). The store resolves the recid under its
/// own locks and dispatches exactly one method. Return values are opaque `i64`s
/// passed through bit-exactly.
///
/// Contract (load-bearing): re-invocable; may be handed torn/garbage bytes and
/// re-run; must fully reset output state per invocation; must bounds-clamp every
/// decoded length; must not call back into the store; must not run user
/// callbacks (emit only after a validated read). In v1 (locked baseline) actions
/// never actually see torn bytes.
pub trait RecordRead {
    /// Record is byte-resident. Input positioned at content start; `size` =
    /// content length.
    fn on_bytes(&mut self, input: &mut SliceInput<'_>, size: usize) -> Result<i64>;
    /// Record is object-resident (heap store / materialized cache entry).
    fn on_object(&mut self, _obj: &dyn Any) -> Result<i64> {
        Err(crate::error::DbError::corrupt(
            "action does not support object handles",
        ))
    }
    /// Record exists but is null (preallocated, or explicit null).
    fn on_null(&mut self) -> Result<i64> {
        Ok(0)
    }
}

/// Store4 core interface. Maps recids to records; structure-blind. Not
/// object-safe (generic methods); collections are generic over `S: Store`.
pub trait Store {
    /// Reserve a recid with null content (Preallocated state). `get` returns
    /// `None`; `update` fills it.
    fn preallocate(&self) -> Result<Recid>;

    /// Batch preallocate (bulk-build fast path).
    fn preallocate_many(&self, into: &mut [Recid]) -> Result<()> {
        for slot in into.iter_mut() {
            *slot = self.preallocate()?;
        }
        Ok(())
    }

    /// Store a (non-null) record, returning its new recid. Serialization for
    /// byte stores happens outside store locks.
    fn put<R: Record>(&self, value: &R, ser: &(impl Serializer<R> + Sync)) -> Result<Recid>;

    /// Read a record. `None` for null/preallocated content; `Err(GetVoid)` for
    /// void/deleted recids.
    fn get<R: Record>(&self, recid: Recid, ser: &(impl Serializer<R> + Sync)) -> Result<Option<R>>;

    /// Push-down read (always-locked path). Returns the action's value.
    fn read(&self, recid: Recid, action: &mut dyn RecordRead) -> Result<i64>;

    /// Replace the content of an existing recid. `None` writes null content.
    fn update<R: Record>(
        &self,
        recid: Recid,
        value: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
    ) -> Result<()>;

    /// Atomic (per recid) logical compare-and-swap using `ser.equals` under the
    /// record lock. `None` matches/writes null content.
    fn compare_and_swap<R: Record>(
        &self,
        recid: Recid,
        expect: Option<&R>,
        new: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
    ) -> Result<bool>;

    /// Delete a record (recid may be reused).
    fn delete(&self, recid: Recid) -> Result<()>;

    /// Make preceding mutations durable (no-op for non-durable stores).
    fn commit(&self) -> Result<()>;

    /// Reclaim obsolete storage where supported.
    fn compact(&self) -> Result<()> {
        Ok(())
    }

    fn close(&self) -> Result<()>;
    fn is_closed(&self) -> bool;

    /// Check store invariants; `Err(VerifyFailed)` on inconsistency. The TCK
    /// calls this after every mutation.
    fn verify(&self) -> Result<()>;

    /// Live recids, sorted, excluding preallocated records.
    fn get_all_recids(&self) -> Result<Vec<Recid>>;

    fn is_thread_safe(&self) -> bool {
        true
    }

    /// True iff this store rejects mutations at the API surface (Java
    /// `Store.isReadOnly()` default `false`). Only [`StoreReadOnlyWrapper`]
    /// overrides it to `true`; the mutators then return
    /// [`DbError::ReadOnly`](crate::error::DbError::ReadOnly).
    fn is_read_only(&self) -> bool {
        false
    }

    /// Approximate byte footprint (for byte-budget cache eviction). Must
    /// decrease on delete. `0` = unsupported.
    fn get_current_size(&self) -> u64 {
        0
    }

    /// True for transactional stores (rollback can void recids; disables the
    /// btree root cache — spec 03 §2).
    fn is_tx(&self) -> bool {
        false
    }

    /// Monotonic counter bumped whenever a structural revert (a `rollback`) may
    /// have invalidated a collection's cached structure — e.g. the btree's
    /// left-edge spine, which is otherwise append-only and can be left too long
    /// by a rollback that shrinks the tree height. Non-tx stores never change
    /// it. A collection caches the last-seen value and rebuilds its derived
    /// structure only when this advances, so the common (no-rollback) path pays
    /// nothing. Not a commit counter: only reverts need to invalidate caches.
    fn structural_generation(&self) -> u64 {
        0
    }
}

/// Crate-private companion: the torn-safe push-down entry point (decision D4).
/// Sealed. Default body delegates to the locked `read`; StoreDirect's phase-2
/// override runs the atomic copy. No blanket impl — each store writes
/// `impl StoreTornRead for X {}` explicitly.
#[allow(dead_code)]
pub(crate) trait StoreTornRead: Store {
    fn read_torn_safe<A: RecordRead + sealed::TornSafeDecode>(
        &self,
        recid: Recid,
        action: &mut A,
    ) -> Result<i64> {
        self.read(recid, action)
    }
}

/// Delta capability (spec 02 §1): in-place record growth with capacity refusal.
/// Implemented by byte-backed stores only.
pub trait StoreDelta: Store {
    /// Extend record content. `AppendResult::Refused` when capacity is
    /// insufficient (the caller then splits). Appending to a preallocated/null
    /// record establishes it.
    fn append(&self, recid: Recid, data: &[u8]) -> Result<AppendResult>;

    /// Capacity hint; may be stale (`append` is authoritative).
    fn capacity_remaining(&self, recid: Recid) -> Result<usize>;

    /// `update` provisioning at least `headroom` appendable bytes.
    fn update_with_headroom<R: Record>(
        &self,
        recid: Recid,
        value: &R,
        ser: &(impl Serializer<R> + Sync),
        headroom: usize,
    ) -> Result<()>;
}

/// Transactional capability (spec 02 §1).
pub trait StoreTx: Store {
    /// Discard all uncommitted mutations, including appends.
    fn rollback(&self) -> Result<()>;
}

/// Crate-private lease machinery (decision D12). Each store embeds a
/// [`LeaseTable`]; the provided `acquire_lease` does the work. Collection
/// constructors bound `S: Store + StoreLease`.
#[allow(dead_code)] // consumed by the collection layer
pub(crate) trait StoreLease {
    fn lease_table(&self) -> &std::sync::Arc<LeaseTable>;
    fn acquire_lease(&self, header_recid: u64, kind: LeaseKind) -> Result<LeaseGuard> {
        self.lease_table().acquire(header_recid, kind)
    }
}
