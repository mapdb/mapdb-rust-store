//! Runtime map-mutation listeners and the `MapExtra` surface, ported from
//! MapDB 3 (`org.mapdb.MapModificationListener`,
//! `SynchronousMapModificationListener`, `ModificationAwareMap`, `MapExtra`).
//!
//! These traits are consumed by `BTreeMap`. They are defined here, at the
//! crate root, because they are collection-agnostic — a persistent concurrent
//! map implements them regardless of its backing structure.
//!
//! ## Async vs. synchronous delivery
//! A plain [`MapModificationListener`] is fired **after** the covering leaf/
//! segment write lock is released (deferred), which avoids re-entrancy and
//! lock-ordering deadlocks but lets two same-key mutations deliver their events
//! in the opposite order under contention. A
//! [`SynchronousMapModificationListener`] is the marker refinement that requests
//! delivery **while the covering lock is still held**, preserving per-key event
//! order for order-sensitive consumers (secondary indexes; see the Java `Bind`).
//!
//! Because Rust has no `instanceof`, the two modes are **compile-time distinct
//! registrations**, not a runtime `synchronous()` flag that could disagree with
//! the marker: a listener registered through the ordinary path is deferred, and a
//! listener registered through the sync path is bounded on the
//! [`SynchronousMapModificationListener`] marker (so a non-marker type cannot be
//! registered synchronously, and a sync listener cannot be *silently* demoted to
//! deferred). See `BTreeMap::modification_listener_add` /
//! `modification_listener_add_sync`.

use crate::error::Result;
use std::sync::Arc;

/// Runtime map-mutation callback, compatible with MapDB 3.
///
/// `triggered` is `true` for automatic expiry/eviction and `false` for
/// user-requested mutations. `old_value`/`new_value` are `None` where Java would
/// pass `null` (insert has `old_value == None`; remove has `new_value == None`).
///
/// Listeners must be `Send + Sync` because they are shared across the threads
/// that mutate the map.
pub trait MapModificationListener<K, V>: Send + Sync {
    /// Fired after the mutation (and its size-counter update) has committed.
    ///
    /// Returns `Err` to signal listener FAILURE — the Rust-idiomatic stand-in for
    /// Java's throwing listener. A failure never rolls back the
    /// already-published primary mutation; the map still delivers the event to the
    /// remaining listeners, then surfaces the first failure to the caller (see the
    /// fire-point semantics in `BTreeMap`).
    fn modify(
        &self,
        key: &K,
        old_value: Option<&V>,
        new_value: Option<&V>,
        triggered: bool,
    ) -> Result<()>;
}

/// Marker refinement of [`MapModificationListener`] whose events must fire
/// synchronously, while the map still holds the covering leaf/segment write lock
/// that serialized the mutation (Java `SynchronousMapModificationListener`).
///
/// Register such a listener via the sync registration path
/// (`BTreeMap::modification_listener_add_sync`), which is bounded on this marker
/// — so sync-ness is chosen at the type level and cannot be silently lost.
///
/// A synchronous listener runs UNDER the map's covering leaf lock, so its body
/// must not re-enter the primary map, and must not mutate another lock-holding
/// MapDB map in a topology that can invert lock order.
///
/// RE-ENTRANCY / LOCK-ORDER CAVEAT (deviation): the external-value
/// read barrier is a `parking_lot::RwLock`, which — unlike Java's
/// `ReentrantReadWriteLock` — is **not** write→read re-entrant. A synchronous
/// listener that calls back into the *same* external-value map's `get`/iterator
/// while the map holds the barrier's write lock (remove path) will deadlock,
/// whereas Java would re-enter. Reciprocal/transitive `Bind` bindings are the
/// canonical hazard; keep synchronous listeners free of primary-map re-entry.
///
/// It fires after the primary mutation has committed, so a failing (or
/// panicking) listener leaves the primary consistent; the map fires the
/// remaining listeners and surfaces the first failure afterwards (see the
/// fire-point semantics).
pub trait SynchronousMapModificationListener<K, V>: MapModificationListener<K, V> {}

/// Blanket adapter so any `Fn(&K, Option<&V>, Option<&V>, bool)` can be used as
/// an async listener without a bespoke type.
pub struct FnListener<F>(pub F);

impl<K, V, F> MapModificationListener<K, V> for FnListener<F>
where
    F: Fn(&K, Option<&V>, Option<&V>, bool) -> Result<()> + Send + Sync,
{
    fn modify(
        &self,
        key: &K,
        old_value: Option<&V>,
        new_value: Option<&V>,
        triggered: bool,
    ) -> Result<()> {
        (self.0)(key, old_value, new_value, triggered)
    }
}

/// A map that supports runtime-only modification listeners (Java
/// `ModificationAwareMap`). Registration and removal are by shared handle;
/// removal matches by pointer identity ([`Arc::ptr_eq`]).
pub trait ModificationAwareMap<K, V> {
    fn modification_listener_add(&self, listener: Arc<dyn MapModificationListener<K, V>>);
    fn modification_listener_remove(&self, listener: &Arc<dyn MapModificationListener<K, V>>);
}

/// MapDB 3-compatible extensions shared by persistent concurrent maps (Java
/// `MapExtra`). The `ConcurrentMap` operations Java inherits live as inherent
/// methods on the concrete map types (std-shaped naming, decision D13), so this
/// trait carries only the genuinely-extra surface.
///
/// The element serializers are exposed as trait objects because the concrete
/// maps are generic over `GroupFormat` (decision D2) and the serializer is the
/// format's `element()`.
pub trait MapExtra<K, V>: ModificationAwareMap<K, V> {
    /// 64-bit size (Java `sizeLong`); never int-saturated (decision D9.4).
    fn size_long(&self) -> Result<u64>;

    /// True once the backing handle is closed (Java `isClosed`).
    fn is_closed(&self) -> bool;

    /// The key element serializer (Java `keySerializer`).
    fn key_serializer(&self) -> &dyn crate::ser::Serializer<K>;

    /// The value element serializer (Java `valueSerializer`).
    fn value_serializer(&self) -> &dyn crate::ser::Serializer<V>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn fn_listener_is_async_by_default_and_fires() {
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        let listener: Arc<dyn MapModificationListener<i64, i64>> = Arc::new(FnListener(
            move |_k: &i64, _o: Option<&i64>, _n: Option<&i64>, _t| {
                h.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        ));
        listener.modify(&1, None, Some(&2), false).unwrap();
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    struct SyncListener;
    impl MapModificationListener<i64, i64> for SyncListener {
        fn modify(&self, _k: &i64, _o: Option<&i64>, _n: Option<&i64>, _t: bool) -> Result<()> {
            Ok(())
        }
    }
    // A sync listener is classified by the marker trait at registration time,
    // not by any runtime method — implementing the marker is all that is needed.
    impl SynchronousMapModificationListener<i64, i64> for SyncListener {}

    #[test]
    fn sync_marker_is_a_map_listener() {
        let l = SyncListener;
        // usable as a plain listener (delivers events) and as the sync marker.
        l.modify(&1, None, Some(&2), false).unwrap();
        fn assert_marker<L: SynchronousMapModificationListener<i64, i64>>(_l: &L) {}
        assert_marker(&l);
    }
}
