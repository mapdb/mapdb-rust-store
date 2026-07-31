//! `BTreeMap` — B-link tree over a Store4 store (spec 03 §1–2). Lock-free
//! push-down readers + Lehman-Yao concurrent writers, ported from Java
//! `org.mapdb.btree.BTreeMap`.
//!
//! The map surface is Rust-idiomatic (`Result`-returning, `K`/`V` by value so
//! null-rejection is unrepresentable). The concurrency protocol, node CoW
//! discipline, and split ordering are ported faithfully; see the module and
//! method docs for the invariants readers rely on.
//!
//! `StoreLease` (D12) is a crate-private capability, so the `where S:
//! StoreLease` bound on this public map leaks a private trait — allowed here:
//! the bound is only satisfiable by in-crate stores, which is the intent.
#![allow(private_bounds)]

use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2, SliceInput};
use crate::listener::{MapExtra, MapModificationListener, ModificationAwareMap};
use crate::ser::long::LongFormat;
use crate::ser::serializers::LONG;
use crate::ser::{GroupFormat, SearchResult, Serializer};
use crate::store::lease::{LeaseGuard, LeaseKind};
use crate::store::{Recid, RecordRead, Store, StoreLease};
use arc_swap::ArcSwap;
use std::any::Any;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Duration;

use super::node::{Node, NodeBody, NodeSerializer, DIR, LEFT, RIGHT};

/// Fixed-8-byte-BE recid group codec for external-value leaves (Java `LongFormat`).
static NODE_RECID_FORMAT: LongFormat = LongFormat;

/// Legal `maxNodeSize` range, shared by the create paths (below) and the DB
/// catalog validator (`db::db::validate_catalog`). Java requires `>= 4` (the
/// split bound) and has no upper limit; the port caps the top so a create can
/// never persist a value the reopen validator would reject and brick the DB
/// See PORTING-GAPS.md for the Java-file caveat.
pub const MIN_MAX_NODE_SIZE: usize = 4;
pub const MAX_MAX_NODE_SIZE: usize = 1 << 20;

#[inline]
fn nz(x: u64) -> Recid {
    NonZeroU64::new(x).expect("btree recid must be non-zero")
}

/// Adapter exposing a `GroupFormat`'s element serializer as a standalone,
/// `Sync` [`Serializer`] so external value records can be stored/read through
/// the store's generic API (the store wants `impl Serializer + Sync`, but
/// `GroupFormat::element()` yields `&dyn Serializer`). Delegates every method.
struct ElemSer<'a, F: GroupFormat>(&'a F);

impl<'a, F: GroupFormat> Serializer<F::Elem> for ElemSer<'a, F> {
    fn serialize(&self, out: &mut DataOutput2, value: &F::Elem) {
        self.0.element().serialize(out, value)
    }
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<F::Elem> {
        self.0.element().deserialize(input, size)
    }
    fn fixed_size(&self) -> Option<usize> {
        self.0.element().fixed_size()
    }
    fn size_hint(&self) -> usize {
        self.0.element().size_hint()
    }
    fn compare(&self, a: &F::Elem, b: &F::Elem) -> std::cmp::Ordering {
        self.0.element().compare(a, b)
    }
    fn equals(&self, a: &F::Elem, b: &F::Elem) -> bool {
        self.0.element().equals(a, b)
    }
    fn natural_order(&self) -> bool {
        self.0.element().natural_order()
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        self.0.element().equals_by_serialized_bytes()
    }
}

/// Insert `value` into a cloned recid vec at `pos` (external-value leaves).
fn insert_i64(arr: &[i64], pos: usize, value: i64) -> Vec<i64> {
    let mut r = Vec::with_capacity(arr.len() + 1);
    r.extend_from_slice(&arr[..pos]);
    r.push(value);
    r.extend_from_slice(&arr[pos..]);
    r
}

fn delete_i64(arr: &[i64], pos: usize) -> Vec<i64> {
    let mut r = Vec::with_capacity(arr.len() - 1);
    r.extend_from_slice(&arr[..pos]);
    r.extend_from_slice(&arr[pos + 1..]);
    r
}

/// `childIdx` from a search result: the found position, or the insertion point.
#[inline]
fn search_idx(r: SearchResult) -> usize {
    match r {
        Ok(p) => p,
        Err(e) => e,
    }
}

// ===================== node lock table =====================

const LOCK_STRIPES: usize = 64;

/// Per-node write locks keyed by EXACT recid (mapdb1/2/3 lineage). A striped
/// set of locked recids: distinct recids are distinct set entries, sharded
/// across `LOCK_STRIPES` mutexes only to reduce insert/remove contention.
///
/// DEADLOCK-FREEDOM. Writers follow Sagiv 1986, not Lehman-Yao's 3-lock
/// discipline: every acquisition happens while holding NO node lock —
/// move-right releases before re-locking (`lock_covering`), a split releases
/// the child before locking the parent, root grow holds only the root-pointer
/// recid. Hold-and-wait therefore never occurs and no waits-for cycle can
/// form, with no lock-ordering argument needed (Sagiv Thm 1). Conditional on:
/// exception-safe release (NodeGuard), no listener/codec re-entry and acyclic
/// cross-map bindings, the store no-upcall contract, and recid stability.
///
/// NON-REENTRANT by design; the debug assert on self-relock is the tripwire.
/// Reentrancy is banned because it MASKS the two bug classes this table must
/// surface: (a) aliasing — a fixed-size/striped lock ARRAY maps parent and
/// child to one lock, which reentrancy silently absorbs single-threaded while
/// enabling cross-thread order inversion (the historical mapdb store ran
/// reentrant 128/16/8-way modulo stripes for a decade, and mapdb3's
/// CC.PARANOID SingleEntryLock existed precisely to unmask re-entry); (b) any
/// future acquisition-while-holding in the wrong direction. Note the assert
/// catches SELF-RELOCK only — the zero-held invariant itself is checked by
/// the debug held-set checker, not this primitive.
///
/// The protocol reserves capacity for up to THREE overlapping locks (Sagiv
/// compression: parent → child → right sibling, TOP-DOWN then left-to-right)
/// — unexercised by current code. Adopting it permanently forbids holding a
/// child while locking its parent (Sagiv p. 277: top-down compression
/// deadlocks against bottom-up L&Y inserters).
struct NodeLockTable {
    stripes: Vec<parking_lot::Mutex<std::collections::HashMap<u64, std::thread::ThreadId>>>,
    enabled: bool,
}

impl NodeLockTable {
    fn new(enabled: bool) -> Self {
        let mut stripes = Vec::with_capacity(LOCK_STRIPES);
        for _ in 0..LOCK_STRIPES {
            stripes.push(parking_lot::Mutex::new(std::collections::HashMap::new()));
        }
        NodeLockTable { stripes, enabled }
    }

    #[inline]
    fn stripe(
        &self,
        recid: u64,
    ) -> &parking_lot::Mutex<std::collections::HashMap<u64, std::thread::ThreadId>> {
        &self.stripes[(recid as usize) % LOCK_STRIPES]
    }

    fn lock(&self, recid: u64) {
        if !self.enabled {
            return;
        }
        let me = std::thread::current().id();
        loop {
            {
                let mut g = self.stripe(recid).lock();
                match g.get(&recid) {
                    None => {
                        g.insert(recid, me);
                        return;
                    }
                    Some(owner) => {
                        debug_assert_ne!(*owner, me, "reentrant node lock: {recid}");
                    }
                }
            }
            std::thread::park_timeout(Duration::from_nanos(10));
        }
    }

    fn unlock(&self, recid: u64) {
        if !self.enabled {
            return;
        }
        let prev = self.stripe(recid).lock().remove(&recid);
        debug_assert_eq!(
            prev,
            Some(std::thread::current().id()),
            "node lock {recid} unlocked by non-owner"
        );
        let _ = prev;
    }
}

/// RAII node-lock guard: `Drop` releases the exact recid so a `?` early-return
/// inside a locked critical section (a corrupt-node load, an update I/O error, a
/// split-write failure) never leaks the lock — which would park a later writer
/// on that recid forever. Move-right and split propagation
/// release explicitly via [`NodeGuard::release`] and re-lock the next recid.
struct NodeGuard<'a> {
    table: &'a NodeLockTable,
    recid: u64,
    held: bool,
}

impl<'a> NodeGuard<'a> {
    #[inline]
    fn recid(&self) -> u64 {
        self.recid
    }
    /// Release the lock now (idempotent). Used for the one-lock-at-a-time
    /// move-right hop and to drop the child lock before locking the parent.
    #[inline]
    fn release(&mut self) {
        if self.held {
            self.table.unlock(self.recid);
            self.held = false;
        }
    }
}

impl<'a> Drop for NodeGuard<'a> {
    fn drop(&mut self) {
        self.release();
    }
}

// ----- cycle / recid robustness -----

/// Cheap cycle detector for traversals over persisted recid graphs. Below the
/// soft threshold it costs one increment; above it (never reached by a valid
/// tree) it records visited recids and reports `DataCorruption` on repetition,
/// so a crafted child/link cycle terminates instead of hanging.
struct CycleGuard {
    steps: u64,
    soft: u64,
    seen: Option<std::collections::HashSet<u64>>,
}

impl CycleGuard {
    fn new(soft: u64) -> Self {
        CycleGuard {
            steps: 0,
            soft,
            seen: None,
        }
    }
    #[inline]
    fn visit(&mut self, recid: u64) -> Result<()> {
        self.steps += 1;
        if self.steps > self.soft {
            let set = self.seen.get_or_insert_with(std::collections::HashSet::new);
            if !set.insert(recid) {
                return Err(DbError::corrupt("cycle in btree recid graph"));
            }
        }
        Ok(())
    }
}

/// Valid tree depth / move-right hops are tiny; anything past this is a cycle.
const CYCLE_DESCENT_SOFT: u64 = 4096;
/// Leaf-link scans can legitimately be long; only enormous chains (past this many
/// leaves) begin tracking visited recids, so a crafted leaf-link cycle terminates
/// with `DataCorruption` at bounded extra memory instead of looping forever.
const CYCLE_SCAN_SOFT: u64 = 1 << 16;

/// Convert an externally-derived recid to `Recid`, mapping a `0` (which a valid
/// tree never produces) to `DataCorruption` instead of a `nz()` panic.
#[inline]
fn recid_or_corrupt(x: u64) -> Result<Recid> {
    NonZeroU64::new(x).ok_or_else(|| DbError::corrupt("btree recid is zero"))
}

// ===================== the map =====================

/// Newly-built leaf value group in either representation, kept abstract so the
/// split/insert paths handle inline values and external recids uniformly.
enum Vals<VF: GroupFormat> {
    Inline(VF::Group),
    External(Vec<i64>),
}

impl<VF: GroupFormat> Vals<VF> {
    fn into_body<KF: GroupFormat>(self, fence: Option<KF::Group>) -> NodeBody<KF, VF> {
        match self {
            Vals::Inline(values) => NodeBody::Leaf { values, fence },
            Vals::External(recids) => NodeBody::ExternalLeaf { recids, fence },
        }
    }
}

/// Shared list of runtime modification listeners (Java `CopyOnWriteArrayList`),
/// published via `ArcSwap` for lock-free firing.
type Listeners<K, V> = ArcSwap<Vec<Arc<dyn MapModificationListener<K, V>>>>;

struct Inner<S, KF: GroupFormat, VF: GroupFormat> {
    store: Arc<S>,
    key_format: KF,
    value_format: VF,
    max_node_size: usize,
    root_recid_recid: u64,
    /// `false` ⇔ external-value map (values in separate store records; leaves
    /// hold value recids).
    value_inline: bool,
    /// O(1) size counter recid (Feature A); `0` = disabled. A `Long` record
    /// updated by a CAS loop in the SAME store as the tree mutation.
    counter_recid: u64,
    /// External-value read barrier. Readers hold the
    /// READ lock across the value-record `store.get`; `remove` holds the WRITE
    /// lock across the whole delete, so a lock-free reader can never observe a
    /// value recid that a concurrent remove deleted and the store reused. Unused
    /// (never contended) for inline maps.
    external_lock: parking_lot::RwLock<()>,
    /// SYNCHRONOUS listeners — fired under the covering leaf lock (order-sensitive
    /// `Bind` secondaries). See fire-point docs on the mutation methods.
    sync_listeners: Listeners<K<KF>, V<VF>>,
    /// Ordinary (deferred) listeners — fired after unlock + split propagation.
    deferred_listeners: Listeners<K<KF>, V<VF>>,
    root_cacheable: bool,
    locks: NodeLockTable,
    /// Per-level left-edge recid, index 0 = leaf level, last = root (Java
    /// `volatile long[]`). Stable by construction; only root splits append.
    left_edges: ArcSwap<Vec<u64>>,
    /// Cached root recid (0 = not loaded). A stale value is harmless (an old
    /// root still covers the whole key space via right-links).
    cached_root: AtomicU64,
    /// Set if a split published a leaf/child but then FAILED to complete its
    /// upward propagation (e.g. a store I/O error mid root-grow). Such a tree is
    /// structurally advanced but its `left_edges` may lack a level a later
    /// writer waits on — so every op fails fast instead of parking forever.
    poisoned: std::sync::atomic::AtomicBool,
    /// Last `store.structural_generation()` the `left_edges` cache was known
    /// consistent with. Advances only when a tx-store rollback bumps the store's
    /// generation; a mismatch triggers a one-shot rebuild (see
    /// `refresh_left_edges_if_tx`). Always `0` for non-tx stores.
    last_struct_gen: AtomicU64,
    _lease: LeaseGuard,
}

/// B-link tree map. Cheap to clone (shares one `Arc<Inner>` = one open lease,
/// D12): every clone and derived iterator observes the same writer state.
pub struct BTreeMap<S, KF: GroupFormat, VF: GroupFormat> {
    inner: Arc<Inner<S, KF, VF>>,
}

impl<S, KF: GroupFormat, VF: GroupFormat> Clone for BTreeMap<S, KF, VF> {
    fn clone(&self) -> Self {
        BTreeMap {
            inner: Arc::clone(&self.inner),
        }
    }
}

type K<KF> = <KF as GroupFormat>::Elem;
type V<VF> = <VF as GroupFormat>::Elem;
/// One descending scan step: the visited leaf's in-range entries (ASCENDING;
/// drain from the back) + the next inclusive upper bound (`None` = exhausted).
type DescendStep<KF, VF> = (Vec<(K<KF>, V<VF>)>, Option<K<KF>>);
/// Retained descending-descent frame: `(recid, dir-node snapshot, entry
/// separator)` — the subtree below holds only keys strictly above the
/// separator (`None` = unbounded left edge).
type RevFrame<KF, VF> = (u64, Node<KF, VF>, Option<K<KF>>);

impl<S, KF, VF> BTreeMap<S, KF, VF>
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    fn node_ser(&self) -> NodeSerializer<'_, KF, VF> {
        NodeSerializer::new_mode(
            &self.inner.key_format,
            &self.inner.value_format,
            self.inner.max_node_size,
            self.inner.value_inline,
        )
    }

    /// Element serializer for external value records.
    fn elem_ser(&self) -> ElemSer<'_, VF> {
        ElemSer(&self.inner.value_format)
    }

    fn kf(&self) -> &KF {
        &self.inner.key_format
    }
    fn vf(&self) -> &VF {
        &self.inner.value_format
    }

    /// Recid of the root-pointer record; persist this to reopen the map.
    pub fn root_recid_recid(&self) -> u64 {
        self.inner.root_recid_recid
    }

    /// True iff `other` is a clone of this map (shares one `Arc<Inner>`, hence one
    /// open lease and one set of locks). Used by `Bind` to reject self-binding.
    pub fn shares_state_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// A stable, type-erased identity for this map's shared state (the address of
    /// its `Arc<Inner>`). Two handles compare equal here iff they are clones of one
    /// map. `Bind` uses this to reject a self-bind even when the primary and the
    /// secondary handle have different generic value types (the secondary trait
    /// object cannot name the primary's concrete type). Safe: only the pointer's
    /// address is read, never dereferenced.
    pub fn state_id(&self) -> usize {
        Arc::as_ptr(&self.inner) as *const () as usize
    }

    pub fn max_node_size(&self) -> usize {
        self.inner.max_node_size
    }

    /// Total key order (== `keyFormat.compare`); used for view bound checks.
    pub fn compare_keys(&self, a: &K<KF>, b: &K<KF>) -> std::cmp::Ordering {
        self.kf().compare(a, b)
    }

    /// True iff keys use their natural order (JDK null-comparator convention).
    pub fn key_natural_order(&self) -> bool {
        self.kf().natural_order()
    }

    /// Logical value equality (value format's element equals, not byte equality).
    pub fn value_equals(&self, a: &V<VF>, b: &V<VF>) -> bool {
        self.vf().element().equals(a, b)
    }

    // ---------------- accessors (Feature A/B, external values) ----------------

    /// Recid of the O(1) size-counter record (Feature A), or `0` when disabled.
    pub fn counter_recid(&self) -> u64 {
        self.inner.counter_recid
    }

    /// True when values are encoded directly in leaf nodes; false when leaves
    /// store value recids (external values).
    pub fn value_inline(&self) -> bool {
        self.inner.value_inline
    }

    /// True once the backing store is closed (Java `isClosed`).
    pub fn is_closed(&self) -> bool {
        self.inner.store.is_closed()
    }

    /// The key element serializer (Java `keySerializer`).
    pub fn key_serializer(&self) -> &dyn Serializer<K<KF>> {
        self.kf().element()
    }

    /// The value element serializer (Java `valueSerializer`).
    pub fn value_serializer(&self) -> &dyn Serializer<V<VF>> {
        self.vf().element()
    }

    // ---------------- size counter (Feature A) ----------------

    /// Apply `delta` to the shared counter record via a CAS retry loop (mirrors
    /// Java `addToCounter`). Called AFTER the structural mutation commits, so the
    /// counter reflects an already-applied change; the CAS loop serializes
    /// concurrent updates without holding any node lock. No-op when disabled.
    fn add_to_counter(&self, delta: i64) -> Result<()> {
        let cr = self.inner.counter_recid;
        if cr == 0 {
            return Ok(());
        }
        let recid = nz(cr);
        loop {
            let cur = self
                .inner
                .store
                .get(recid, &LONG)?
                .ok_or_else(|| DbError::corrupt("btree size counter record missing"))?;
            // Java `long` arithmetic wraps; use wrapping_add so a (pathological)
            // overflow neither panics in debug under the leaf lock nor differs
            // from Java in release. `size_long` guards a negative persisted value.
            let next = cur.wrapping_add(delta);
            if self
                .inner
                .store
                .compare_and_swap(recid, Some(&cur), Some(&next), &LONG)?
            {
                return Ok(());
            }
        }
    }

    /// Mark this in-memory handle poisoned: `size_long` and every further op then
    /// fail fast. Used when a secondary update (the O(1) counter) fails AFTER the
    /// primary node mutation committed, so a silently-wrong counter can never be
    /// observed as authoritative.
    #[inline]
    fn poison(&self) {
        self.inner.poisoned.store(true, AtomicOrdering::Release);
    }

    // ---------------- modification listeners (Feature B) ----------------

    /// Register an ORDINARY (deferred) modification listener — fired after the
    /// covering leaf lock is released and split propagation completes. Duplicate
    /// registration (same `Arc`) is ignored (Java `addIfAbsent`).
    pub fn modification_listener_add(
        &self,
        listener: Arc<dyn MapModificationListener<K<KF>, V<VF>>>,
    ) {
        Self::push_listener(&self.inner.deferred_listeners, listener);
    }

    /// Register a SYNCHRONOUS modification listener — fired under the covering
    /// leaf lock (order-sensitive `Bind` secondaries). Compile-time bounded on the
    /// [`SynchronousMapModificationListener`] marker, so a non-marker listener
    /// cannot be registered here and a sync listener cannot be silently demoted to
    /// deferred (misclassification is impossible).
    pub fn modification_listener_add_sync<L>(&self, listener: Arc<L>)
    where
        L: crate::listener::SynchronousMapModificationListener<K<KF>, V<VF>> + 'static,
    {
        let dynamic: Arc<dyn MapModificationListener<K<KF>, V<VF>>> = listener;
        Self::push_listener(&self.inner.sync_listeners, dynamic);
    }

    fn push_listener(
        list: &Listeners<K<KF>, V<VF>>,
        listener: Arc<dyn MapModificationListener<K<KF>, V<VF>>>,
    ) {
        list.rcu(|cur| {
            let mut v: Vec<_> = (**cur).clone();
            if !v.iter().any(|x| Arc::ptr_eq(x, &listener)) {
                v.push(listener.clone());
            }
            Arc::new(v)
        });
    }

    /// Deregister a previously-added listener (by `Arc` identity), from either
    /// the sync or the deferred list.
    pub fn modification_listener_remove(
        &self,
        listener: &Arc<dyn MapModificationListener<K<KF>, V<VF>>>,
    ) {
        for list in [&self.inner.sync_listeners, &self.inner.deferred_listeners] {
            list.rcu(|cur| {
                let v: Vec<_> = cur
                    .iter()
                    .filter(|x| !Arc::ptr_eq(x, listener))
                    .cloned()
                    .collect();
                Arc::new(v)
            });
        }
    }

    /// Invoke `listener.modify`, catching a panic (Java catches
    /// `RuntimeException|Error`) and converting it to a listener
    /// error so it flows through the SAME captured-error recovery path (leaf lock
    /// released, split propagation completed) instead of unwinding through it.
    fn invoke_listener(
        l: &Arc<dyn MapModificationListener<K<KF>, V<VF>>>,
        key: &K<KF>,
        old: Option<&V<VF>>,
        new: Option<&V<VF>>,
    ) -> Result<()> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            l.modify(key, old, new, false)
        })) {
            Ok(res) => res,
            Err(_) => Err(DbError::corrupt_msg("modification listener panicked")),
        }
    }

    /// Fire the SYNCHRONOUS listeners while the covering leaf lock is still held.
    /// Per-listener continuation: every listener sees the event even if an earlier
    /// one fails; the FIRST failure is returned (later ones dropped). The caller
    /// releases the lock and (for a split) completes propagation before surfacing
    /// this error.
    fn fire_sync(&self, key: &K<KF>, old: Option<&V<VF>>, new: Option<&V<VF>>) -> Result<()> {
        let ls = self.inner.sync_listeners.load();
        if ls.is_empty() {
            return Ok(());
        }
        let mut first: Option<DbError> = None;
        for l in ls.iter() {
            if let Err(e) = Self::invoke_listener(l, key, old, new) {
                if first.is_none() {
                    first = Some(e);
                }
            }
        }
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Fire the ordinary (deferred) listeners AFTER unlock + split propagation.
    fn fire_deferred(&self, key: &K<KF>, old: Option<&V<VF>>, new: Option<&V<VF>>) -> Result<()> {
        let ls = self.inner.deferred_listeners.load();
        if ls.is_empty() {
            return Ok(());
        }
        let mut first: Option<DbError> = None;
        for l in ls.iter() {
            if let Err(e) = Self::invoke_listener(l, key, old, new) {
                if first.is_none() {
                    first = Some(e);
                }
            }
        }
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    // ---------------- leaf value access (inline / external) ----------------

    /// Read the value at leaf position `pos`, expanding a value recid via the
    /// store in external mode. Callers on the read path hold `external_lock` read.
    fn leaf_value_at(&self, body: &NodeBody<KF, VF>, pos: usize) -> Result<V<VF>> {
        match body {
            NodeBody::Leaf { values, .. } => Ok(self.vf().get(values, pos)),
            NodeBody::ExternalLeaf { recids, .. } => {
                let recid = recids[pos];
                self.inner
                    .store
                    .get(recid_or_corrupt(recid as u64)?, &self.elem_ser())?
                    .ok_or_else(|| DbError::corrupt("external value record missing"))
            }
            NodeBody::Dir { .. } => Err(DbError::corrupt("value access on a directory node")),
        }
    }

    /// Publish an in-place value SET at `pos`. Inline: rewrite the leaf node.
    /// External: update the value record; the node is unchanged (Java `setValue`).
    fn publish_leaf_set(
        &self,
        current: u64,
        n: &Node<KF, VF>,
        pos: usize,
        value: V<VF>,
    ) -> Result<()> {
        match &n.body {
            NodeBody::Leaf { values, fence } => {
                let new_vals = self.vf().set(values, pos, value);
                let updated = Node {
                    flags: n.flags,
                    link: n.link,
                    keys: n.keys.clone(),
                    body: NodeBody::Leaf {
                        values: new_vals,
                        fence: fence.clone(),
                    },
                };
                self.store_update(current, &updated)
            }
            NodeBody::ExternalLeaf { recids, .. } => {
                let recid = recids[pos];
                self.inner.store.update(
                    recid_or_corrupt(recid as u64)?,
                    Some(&value),
                    &self.elem_ser(),
                )
            }
            NodeBody::Dir { .. } => Err(DbError::corrupt("value set on a directory node")),
        }
    }

    /// Build the new value group after inserting `value` at `ip`, allocating an
    /// external value record when needed. Returns the inserted representation.
    fn insert_leaf_vals(
        &self,
        body: &NodeBody<KF, VF>,
        ip: usize,
        value: V<VF>,
    ) -> Result<Vals<VF>> {
        match body {
            NodeBody::Leaf { values, .. } => Ok(Vals::Inline(self.vf().insert(values, ip, value))),
            NodeBody::ExternalLeaf { recids, .. } => {
                let recid = self.inner.store.put(&value, &self.elem_ser())?.get() as i64;
                Ok(Vals::External(insert_i64(recids, ip, recid)))
            }
            NodeBody::Dir { .. } => Err(DbError::corrupt("value insert on a directory node")),
        }
    }

    // ---------------- construction ----------------

    /// Create a fresh empty tree over `store` (no size counter); returns the map
    /// (holds an RW lease on its root-pointer recid).
    pub fn create(
        store: Arc<S>,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
    ) -> Result<Self> {
        Self::create_mode(store, key_format, value_format, max_node_size, false, true)
    }

    /// Create a fresh empty tree, optionally with an O(1) size counter (Feature
    /// A). When `counter_enable` a dedicated `Long` record (initial value 0) is
    /// allocated; its recid is exposed via [`Self::counter_recid`].
    pub fn create_with_counter(
        store: Arc<S>,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
        counter_enable: bool,
    ) -> Result<Self> {
        Self::create_mode(
            store,
            key_format,
            value_format,
            max_node_size,
            counter_enable,
            true,
        )
    }

    /// Create a map whose leaves hold value RECIDS rather than value bytes
    /// (external values); the values live in separate store
    /// records. Optionally enables the O(1) size counter.
    pub fn create_external_values(
        store: Arc<S>,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
        counter_enable: bool,
    ) -> Result<Self> {
        Self::create_mode(
            store,
            key_format,
            value_format,
            max_node_size,
            counter_enable,
            false,
        )
    }

    fn create_mode(
        store: Arc<S>,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
        counter_enable: bool,
        value_inline: bool,
    ) -> Result<Self> {
        // Create-time API arg (never from stored bytes): reject out-of-range as a
        // configuration error rather than panicking (Java throws
        // IllegalArgumentException). The SAME bound is enforced by the catalog
        // validator on reopen, so a create can never persist a value that would
        // brick the DB at the next open (R3).
        if !(MIN_MAX_NODE_SIZE..=MAX_MAX_NODE_SIZE).contains(&max_node_size) {
            return Err(DbError::wrong_config("maxNodeSize must be in 4..=1048576"));
        }
        let ns = NodeSerializer::new_mode(&key_format, &value_format, max_node_size, value_inline);
        // An empty leaf's value group is empty in both representations.
        let body: NodeBody<KF, VF> = if value_inline {
            NodeBody::Leaf {
                values: value_format.empty(),
                fence: None,
            }
        } else {
            NodeBody::ExternalLeaf {
                recids: Vec::new(),
                fence: None,
            }
        };
        let empty_leaf: Node<KF, VF> = Node {
            flags: LEFT | RIGHT,
            link: 0,
            keys: key_format.empty(),
            body,
        };
        let root_recid = store.put(&empty_leaf, &ns)?;
        let rrr = store.put(&(root_recid.get() as i64), &LONG)?;
        let counter_recid = if counter_enable {
            store.put(&0i64, &LONG)?.get()
        } else {
            0
        };
        Self::open_mode(
            store,
            rrr.get(),
            key_format,
            value_format,
            max_node_size,
            counter_recid,
            value_inline,
        )
    }

    /// Reopen an inline tree (no counter) whose root-pointer recid is `root_recid_recid`.
    pub fn open(
        store: Arc<S>,
        root_recid_recid: u64,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
    ) -> Result<Self> {
        Self::open_mode(
            store,
            root_recid_recid,
            key_format,
            value_format,
            max_node_size,
            0,
            true,
        )
    }

    /// Reopen an inline tree, wiring up its O(1) size counter when
    /// `counter_recid > 0` (Feature A). `counter_recid == 0` means "no counter".
    pub fn open_with_counter(
        store: Arc<S>,
        root_recid_recid: u64,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
        counter_recid: u64,
    ) -> Result<Self> {
        Self::open_mode(
            store,
            root_recid_recid,
            key_format,
            value_format,
            max_node_size,
            counter_recid,
            true,
        )
    }

    /// Reopen a map created by [`Self::create_external_values`].
    pub fn open_external_values(
        store: Arc<S>,
        root_recid_recid: u64,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
        counter_recid: u64,
    ) -> Result<Self> {
        Self::open_mode(
            store,
            root_recid_recid,
            key_format,
            value_format,
            max_node_size,
            counter_recid,
            false,
        )
    }

    fn open_mode(
        store: Arc<S>,
        root_recid_recid: u64,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
        counter_recid: u64,
        value_inline: bool,
    ) -> Result<Self> {
        // `max_node_size` reaches this path from the stored catalog on reopen; a
        // hostile/corrupt out-of-range value must be a corruption error, never a
        // panic (C2). The facade validator also range-checks it up front (R3).
        if !(MIN_MAX_NODE_SIZE..=MAX_MAX_NODE_SIZE).contains(&max_node_size) {
            return Err(DbError::corrupt(
                "stored maxNodeSize out of range 4..=1048576",
            ));
        }
        // The map dereferences `root_recid_recid` via `nz()`; a 0 argument (API
        // misuse) must be a clean error, not a panic.
        if root_recid_recid == 0 {
            return Err(DbError::corrupt("root_recid_recid must be a valid recid"));
        }
        let lease = store.acquire_lease(root_recid_recid, LeaseKind::ReadWrite)?;
        let thread_safe = store.is_thread_safe();
        let root_cacheable = !store.is_tx();
        let init_gen = store.structural_generation();
        let inner = Inner {
            store,
            key_format,
            value_format,
            max_node_size,
            root_recid_recid,
            value_inline,
            counter_recid,
            external_lock: parking_lot::RwLock::new(()),
            sync_listeners: ArcSwap::from_pointee(Vec::new()),
            deferred_listeners: ArcSwap::from_pointee(Vec::new()),
            root_cacheable,
            locks: NodeLockTable::new(thread_safe),
            left_edges: ArcSwap::from_pointee(Vec::new()),
            cached_root: AtomicU64::new(0),
            poisoned: std::sync::atomic::AtomicBool::new(false),
            last_struct_gen: AtomicU64::new(init_gen),
            _lease: lease,
        };
        let map = BTreeMap {
            inner: Arc::new(inner),
        };
        // Validate the O(1) counter record up front: it must exist and be
        // non-negative, so a bad counter recid is rejected at open — before the
        // first mutation commits against a phantom/negative counter.
        if counter_recid != 0 {
            let v = map
                .inner
                .store
                .get(nz(counter_recid), &LONG)?
                .ok_or_else(|| DbError::corrupt("btree size counter record missing"))?;
            if v < 0 {
                return Err(DbError::corrupt("btree size counter record is negative"));
            }
        }
        let edges = map.build_left_edges()?;
        map.inner.left_edges.store(Arc::new(edges));
        Ok(map)
    }

    /// Walk the leftmost spine root→leaf; result index 0 = leaf level.
    fn build_left_edges(&self) -> Result<Vec<u64>> {
        let mut cyc = CycleGuard::new(CYCLE_DESCENT_SOFT);
        let mut spine = Vec::new(); // root-first
        let mut current = self.load_root_recid()?;
        spine.push(current);
        let mut n = self.load(current)?;
        // A healthy tree's root node is ALWAYS both LEFT and RIGHT (a single-leaf
        // root, or a grown DIR root). The one way the root pointer names a node
        // that isn't is a root-grow that published the child split but failed
        // before updating the pointer: the old root was
        // republished LEFT-only. Detect that unrecoverable partial state at open
        // instead of parking a later writer forever in `left_edge`. (Open runs
        // with no concurrent writer — D12 RW lease — so this never races the
        // transient mid-root-split window, which is held under the root lock.)
        if n.flags & (LEFT | RIGHT) != (LEFT | RIGHT) {
            return Err(DbError::corrupt(
                "btree root is not root-shaped (incomplete root-grow); store is damaged",
            ));
        }
        while n.is_dir() {
            cyc.visit(current)?;
            current = n.children()[0]; // deserialize guarantees a dir has >=1 child
            spine.push(current);
            n = self.load(current)?;
        }
        spine.reverse(); // leaf-first
        Ok(spine)
    }

    fn load_root_recid(&self) -> Result<u64> {
        let r = self
            .inner
            .store
            .get(nz(self.inner.root_recid_recid), &LONG)?
            .ok_or_else(|| DbError::corrupt("btree root pointer is null"))?;
        // A crafted root pointer of 0 / negative would panic in `nz()` on the
        // next load; reject it here (recids are positive).
        if r <= 0 {
            return Err(DbError::corrupt("btree root pointer is not a valid recid"));
        }
        Ok(r as u64)
    }

    /// Resync the `left_edges` structural cache with the tx-visible tree when a
    /// rollback may have shrunk it. `left_edges` is normally append-only and
    /// always current, but a tx store's tree can be reverted out-of-band by a
    /// `rollback()` that shrinks its height while this map object (and its longer
    /// cached vector) stays open; the next root grow would then append onto a
    /// stale vector whose entries name deleted/reused recids (found in review). Gated on the store's `structural_generation`, so
    /// this is a cheap load-and-compare on the common (no-rollback) path and only
    /// rebuilds the one time after each rollback — never per put.
    ///
    /// Concurrency contract: a transactional store is a **single global writer**
    /// (the conventional WAL model; no concurrent-writer tests exist for tx
    /// stores, and the D12 lease is RW-exclusive). `rollback`/`commit` are
    /// transaction boundaries that must not race in-flight mutations — they
    /// revert/publish exactly those mutations — so the post-rollback rebuild here
    /// never runs concurrently with a root grow. (Concurrent writers over a tx
    /// store would additionally need a structural mutex shared with root grow to
    /// exclude the transient LEFT-only-root window — out of scope for v1.)
    /// No-op (and zero cost) for non-tx stores, mirroring `cached_root`.
    fn refresh_left_edges_if_tx(&self) -> Result<()> {
        if self.inner.root_cacheable {
            return Ok(()); // non-tx: append-only cache is authoritative
        }
        let gen = self.inner.store.structural_generation();
        if gen == self.inner.last_struct_gen.load(AtomicOrdering::Acquire) {
            return Ok(()); // no rollback since last resync — cache is current
        }
        self.check_poison()?;
        let edges = self.build_left_edges()?;
        self.inner.left_edges.store(Arc::new(edges));
        self.inner
            .last_struct_gen
            .store(gen, AtomicOrdering::Release);
        Ok(())
    }

    #[inline]
    fn check_poison(&self) -> Result<()> {
        if self.inner.poisoned.load(AtomicOrdering::Acquire) {
            return Err(DbError::corrupt(
                "btree poisoned by a failed structural update; reopen the store \
                 (a root-grow failure surfaces as corruption at open)",
            ));
        }
        Ok(())
    }

    /// Authoritative "is `recid` the current root?" — reads the root-pointer
    /// record FRESH (not the possibly-stale cache). Root growth is gated on this
    /// AND the node's `LEFT|RIGHT` flags (see the split sites): the flag test is
    /// the concurrency serialization (a splitter republishes the root LEFT-only
    /// under its lock before releasing), and this identity test rejects a crafted
    /// descendant falsely flagged root-shaped so it cannot replace the real tree.
    fn is_current_root(&self, recid: u64) -> Result<bool> {
        Ok(self.load_root_recid()? == recid)
    }

    /// Every op starts at the root, so a poison check here fails all ops fast.
    fn root_recid(&self) -> Result<u64> {
        self.check_poison()?;
        let r = self.inner.cached_root.load(AtomicOrdering::Acquire);
        if r != 0 {
            return Ok(r);
        }
        let r = self.load_root_recid()?;
        if self.inner.root_cacheable {
            self.inner.cached_root.store(r, AtomicOrdering::Release);
        }
        Ok(r)
    }

    fn load(&self, recid: u64) -> Result<Node<KF, VF>> {
        self.inner
            .store
            .get(recid_or_corrupt(recid)?, &self.node_ser())?
            .ok_or_else(|| DbError::corrupt("btree node record is null"))
    }

    /// Acquire an RAII lock on `recid` (auto-releases on drop / `?` unwind).
    fn lock_guard(&self, recid: u64) -> NodeGuard<'_> {
        self.inner.locks.lock(recid);
        NodeGuard {
            table: &self.inner.locks,
            recid,
            held: true,
        }
    }

    // ---------------- read path (push-down) ----------------

    /// Lock-free push-down lookup: `Ok(Some)` if present, `Ok(None)` if absent.
    /// External maps hold the value read barrier (`external_lock` read) across the
    /// WHOLE descent + value-record expansion, so a concurrent remove (write lock)
    /// can never delete and let the store reuse the recid we just read (74b9963).
    pub fn get(&self, key: &K<KF>) -> Result<Option<V<VF>>> {
        if self.inner.value_inline {
            self.do_get(key)
        } else {
            let _barrier = self.inner.external_lock.read();
            self.do_get(key)
        }
    }

    fn do_get(&self, key: &K<KF>) -> Result<Option<V<VF>>> {
        let mut action = GetAction {
            key,
            kf: self.kf(),
            vf: self.vf(),
            value_inline: self.inner.value_inline,
            value: None,
            stored_recid: None,
            found: false,
        };
        let mut cyc = CycleGuard::new(CYCLE_DESCENT_SOFT);
        let mut current = self.root_recid()?;
        while current != 0 {
            cyc.visit(current)?;
            current = self
                .inner
                .store
                .read(recid_or_corrupt(current)?, &mut action)? as u64;
        }
        if !action.found {
            return Ok(None);
        }
        if self.inner.value_inline {
            Ok(action.value)
        } else {
            // expand the captured value recid (still under the read barrier). A
            // missing record here is corruption (the barrier keeps a concurrent
            // remove out), NOT "key absent" — matching the write path's
            // `leaf_value_at` so a barrier regression fails loudly, not silently.
            let recid = action
                .stored_recid
                .ok_or_else(|| DbError::corrupt("external get: no value recid captured"))?;
            match self
                .inner
                .store
                .get(recid_or_corrupt(recid as u64)?, &self.elem_ser())?
            {
                Some(v) => Ok(Some(v)),
                None => Err(DbError::corrupt("external value record missing")),
            }
        }
    }

    pub fn contains_key(&self, key: &K<KF>) -> Result<bool> {
        let mut action = GetAction {
            key,
            kf: self.kf(),
            vf: self.vf(),
            value_inline: self.inner.value_inline,
            value: None,
            stored_recid: None,
            found: false,
        };
        let mut cyc = CycleGuard::new(CYCLE_DESCENT_SOFT);
        let mut current = self.root_recid()?;
        while current != 0 {
            cyc.visit(current)?;
            current = self
                .inner
                .store
                .read(recid_or_corrupt(current)?, &mut action)? as u64;
        }
        Ok(action.found)
    }

    // ---------------- writer helpers ----------------

    /// Route within a dir node; `None` = follow link.
    fn route_child(&self, dir: &Node<KF, VF>, key: &K<KF>) -> Option<usize> {
        let child_idx = search_idx(self.kf().search(&dir.keys, key));
        if child_idx >= dir.children().len() {
            None
        } else {
            Some(child_idx)
        }
    }

    /// Leaf coverage: true when `key` lies beyond this leaf's inclusive fence.
    fn beyond_leaf(&self, leaf: &Node<KF, VF>, key: &K<KF>) -> bool {
        if leaf.is_right() {
            return false;
        }
        // search on the 1-element fence group: Err(1) == key > fence.
        match leaf.leaf_fence() {
            Some(f) => matches!(self.kf().search(f, key), Err(1)),
            None => false,
        }
    }

    /// Dir coverage: true when `key` lies beyond this dir's last key (its bound).
    fn beyond_dir(&self, dir: &Node<KF, VF>, key: &K<KF>) -> bool {
        matches!(self.kf().search(&dir.keys, key), Err(ins) if ins == self.kf().size(&dir.keys))
    }

    /// Lock `recid`, load it, and move right (hand-over-hand, one lock held)
    /// until the node covers `key`. Returns the covering node; its recid is put
    /// in `cursor` and its lock is HELD by the caller.
    fn lock_covering(
        &self,
        recid: u64,
        key: &K<KF>,
        dir_level: bool,
    ) -> Result<(Node<KF, VF>, NodeGuard<'_>)> {
        let mut cyc = CycleGuard::new(CYCLE_DESCENT_SOFT);
        let mut guard = self.lock_guard(recid);
        let mut n = self.load(guard.recid())?;
        loop {
            let beyond = if dir_level {
                !n.is_right() && self.beyond_dir(&n, key)
            } else {
                self.beyond_leaf(&n, key)
            };
            if !beyond {
                break;
            }
            let next = n.link; // nonzero: a non-right node has a validated link
            cyc.visit(next)?;
            guard.release(); // one lock at a time: release current, then lock next
            guard = self.lock_guard(next);
            n = self.load(next)?;
        }
        Ok((n, guard))
    }

    /// Unlocked descent to the leaf routing `key`, then `lock_covering` to the
    /// real owner. Returns that LOCKED leaf; its recid is left in `cursor` and
    /// the caller MUST `unlock_node(*cursor)` (unless it hands off to split
    /// propagation, which releases the lock itself). `parent_stack`, when
    /// `Some`, is filled with the covered parent path for split propagation.
    fn lock_leaf(
        &self,
        key: &K<KF>,
        mut parent_stack: Option<&mut Vec<u64>>,
    ) -> Result<(Node<KF, VF>, NodeGuard<'_>)> {
        let mut cyc = CycleGuard::new(CYCLE_DESCENT_SOFT);
        let mut current = self.root_recid()?;
        let mut n = self.load(current)?;
        while n.is_dir() {
            cyc.visit(current)?;
            match self.route_child(&n, key) {
                None => {
                    current = n.link;
                }
                Some(child_idx) => {
                    if let Some(stack) = parent_stack.as_deref_mut() {
                        stack.push(current);
                    }
                    current = n.children()[child_idx];
                }
            }
            n = self.load(current)?;
        }
        self.lock_covering(current, key, false)
    }

    // ---------------- put / putIfAbsent ----------------

    pub fn put(&self, key: K<KF>, value: V<VF>) -> Result<Option<V<VF>>> {
        self.put_internal(key, value, false)
    }

    pub fn put_if_absent(&self, key: K<KF>, value: V<VF>) -> Result<Option<V<VF>>> {
        self.put_internal(key, value, true)
    }

    /// Blind put (API parity): insert or overwrite without returning the old value.
    pub fn put_only(&self, key: K<KF>, value: V<VF>) -> Result<()> {
        self.put_internal(key, value, false)?;
        Ok(())
    }

    fn put_internal(
        &self,
        key: K<KF>,
        value: V<VF>,
        only_if_absent: bool,
    ) -> Result<Option<V<VF>>> {
        // Tx stores only: resync the structural cache with the tx-visible tree
        // (a prior rollback may have shrunk it) before a split/grow can consult
        // or extend it. No-op for non-tx stores.
        self.refresh_left_edges_if_tx()?;
        let mut stack: Vec<u64> = Vec::new();
        // `guard` auto-unlocks on any `?`/return below.
        let (n, guard) = self.lock_leaf(&key, Some(&mut stack))?;
        let current = guard.recid();
        let pos = self.kf().search(&n.keys, &key);
        if matches!(n.body, NodeBody::Dir { .. }) {
            return Err(DbError::corrupt("put reached a directory node"));
        }
        if let Ok(p) = pos {
            // key present
            let old = self.leaf_value_at(&n.body, p)?;
            if only_if_absent {
                return Ok(Some(old)); // no mutation: no counter change, no listener
            }
            // publish the value change (inline: rewrite node; external: update record)
            self.publish_leaf_set(current, &n, p, value.clone())?;
            // fire sync listeners UNDER the leaf lock (preserves same-key order),
            // then release the lock, then (if sync ok) fire deferred listeners.
            let sync_res = self.fire_sync(&key, Some(&old), Some(&value));
            drop(guard);
            sync_res?;
            self.fire_deferred(&key, Some(&old), Some(&value))?;
            return Ok(Some(old));
        }
        let ip = search_idx(pos);
        let new_keys = self.kf().insert(&n.keys, ip, key.clone());
        let new_vals = self.insert_leaf_vals(&n.body, ip, value.clone())?;
        if self.kf().size(&new_keys) <= self.inner.max_node_size {
            let fence = n.leaf_fence().cloned();
            let updated = Node {
                flags: n.flags,
                link: n.link,
                keys: new_keys,
                body: new_vals.into_body(fence),
            };
            self.store_update(current, &updated)?;
            // counter BEFORE listeners: the mutation is committed, so a failing
            // listener must not desync `size_long` from the tree contents. If the
            // counter update itself fails AFTER the node committed, poison the
            // handle so a silently-wrong O(1) count is never trusted.
            if let Err(e) = self.add_to_counter(1) {
                self.poison();
                return Err(e);
            }
            let sync_res = self.fire_sync(&key, None, Some(&value));
            drop(guard);
            sync_res?;
            self.fire_deferred(&key, None, Some(&value))?;
            return Ok(None);
        }
        // overfull: split (fires counter + sync listeners under the leaf lock,
        // completes propagation, then surfaces any listener error).
        self.split_leaf_and_propagate(guard, &n, new_keys, new_vals, &mut stack, &key, &value)?;
        // deferred listeners only after a successful split (a sync failure returns
        // above via `?`, skipping this — matching Java's fire-point semantics).
        self.fire_deferred(&key, None, Some(&value))?;
        Ok(None)
    }

    fn store_update(&self, recid: u64, node: &Node<KF, VF>) -> Result<()> {
        self.inner
            .store
            .update(nz(recid), Some(node), &self.node_ser())
    }

    fn store_put(&self, node: &Node<KF, VF>) -> Result<u64> {
        Ok(self.inner.store.put(node, &self.node_ser())?.get())
    }

    /// Split the (locked) overfull leaf and propagate separators upward. The
    /// right sibling B is written FIRST (referent before referrer), the left
    /// half republished with `link=q` (from that instant the split is fully
    /// searchable through the link), and only THEN is the child lock released.
    ///
    /// SYNC-LISTENER FIRE POINT: after the
    /// left half republishes (the insert's searchable commit point) and after the
    /// O(1) counter bump, but BEFORE this leaf's lock is released. Even when the
    /// inserted key landed in B, every locking path to B still goes through A's
    /// link until the separator reaches the parent, and `lock_covering` is
    /// hand-over-hand — so no competing same-key writer can publish first. THROW
    /// RECOVERY: a listener failure must NOT skip `propagate_split` (skipping the
    /// ROOT split's propagation would leave a later B-split spinning forever on a
    /// level-1 left edge that was never created). So the listener error is
    /// captured, the lock released, propagation COMPLETED, and only then is the
    /// listener error returned. A structural propagation error is primary; the
    /// listener error is secondary (dropped).
    #[allow(clippy::too_many_arguments)]
    fn split_leaf_and_propagate(
        &self,
        mut guard: NodeGuard<'_>,
        orig: &Node<KF, VF>,
        keys: KF::Group,
        values: Vals<VF>,
        stack: &mut Vec<u64>,
        key: &K<KF>,
        value: &V<VF>,
    ) -> Result<()> {
        let recid = guard.recid();
        // Root-growth gate — BOTH conditions, evaluated before any publication:
        //   (a) `orig` still carries LEFT|RIGHT (root shape). This is the
        //       CONCURRENCY serialization: the first thread to split the root
        //       republishes it LEFT-only under its lock before releasing, so a
        //       second splitter that then locks it sees the flipped flag and
        //       does NOT also grow a root.
        //   (b) the root pointer authoritatively names this recid. This rejects
        //       a CRAFTED descendant falsely flagged LEFT|RIGHT.
        // The cheap flag test short-circuits the store read for ordinary
        // (non-root) splits.
        let was_root =
            (orig.flags & (LEFT | RIGHT)) == (LEFT | RIGHT) && self.is_current_root(recid)?;
        let total = self.kf().size(&keys);
        let h = total / 2;
        let orig_fence = orig.leaf_fence().cloned();
        let b_flags = orig.flags & !LEFT; // B keeps RIGHT status of the original
        let b_vals = self.split_vals(&values, h, total);
        let b = Node {
            flags: b_flags,
            link: orig.link,
            keys: self.kf().copy_range(&keys, h, total),
            body: b_vals.into_body(orig_fence),
        };
        // On any `?` before publication `guard` drops → leaf lock released; the
        // orphaned `b` is harmless garbage (nothing references it yet).
        let q = self.store_put(&b)?;
        let sep = self.kf().get(&keys, h - 1);
        let a_fence = self.kf().insert(&self.kf().empty(), 0, sep.clone());
        let a_vals = self.split_vals(&values, 0, h);
        let a = Node {
            flags: orig.flags & !RIGHT,
            link: q,
            keys: self.kf().copy_range(&keys, 0, h),
            body: a_vals.into_body(Some(a_fence)),
        };
        self.store_update(recid, &a)?;
        // The split is now published (searchable via the link). Bump the counter
        // and fire the sync listeners UNDER the still-held leaf lock. Both are
        // SECONDARY to structural completion: whatever happens here, separator/root
        // propagation MUST still run, or a later split of B parks forever on a
        // level-1 left edge that was never created / rejects the LEFT-only root.
        let counter_res = self.add_to_counter(1);
        let listener_res = self.fire_sync(key, None, Some(value));
        guard.release(); // release AFTER the sync fire (Java's finally)
                         // ALWAYS complete propagation, even after a counter or listener failure.
        let prop_res = self.propagate_split(recid, q, sep, was_root, stack, 1);
        if let Err(e) = prop_res {
            self.poison();
            return Err(e); // structural error is primary
        }
        // Propagation completed. Surface a secondary failure (counter first — it
        // desyncs the O(1) count, so poison; then the listener error).
        if let Err(e) = counter_res {
            self.poison();
            return Err(e);
        }
        listener_res // listener error is secondary
    }

    /// Copy a `Vals` half for a split (`[from, to)`), preserving representation.
    fn split_vals(&self, vals: &Vals<VF>, from: usize, to: usize) -> Vals<VF> {
        match vals {
            Vals::Inline(g) => Vals::Inline(self.vf().copy_range(g, from, to)),
            Vals::External(r) => Vals::External(r[from..to].to_vec()),
        }
    }

    /// Insert (`sep` → `new_child` right of `old_child`) into the parent level;
    /// split upward as needed. `level` counts from 0 = leaf. No lock held on entry.
    fn propagate_split(
        &self,
        mut old_child: u64,
        mut new_child: u64,
        mut sep: K<KF>,
        mut child_was_root: bool,
        stack: &mut Vec<u64>,
        mut level: usize,
    ) -> Result<()> {
        loop {
            if child_was_root {
                // grow the tree by one level
                let root_keys = self.kf().insert(&self.kf().empty(), 0, sep);
                let new_root = Node {
                    flags: DIR | LEFT | RIGHT,
                    link: 0,
                    keys: root_keys,
                    body: NodeBody::Dir {
                        children: vec![old_child, new_child],
                    },
                };
                // `_rguard` auto-unlocks the root-pointer recid on any `?`/return.
                let _rguard = self.lock_guard(self.inner.root_recid_recid);
                let new_root_recid = self.store_put(&new_root)?;
                self.inner.store.update(
                    nz(self.inner.root_recid_recid),
                    Some(&(new_root_recid as i64)),
                    &LONG,
                )?;
                if self.inner.root_cacheable {
                    self.inner
                        .cached_root
                        .store(new_root_recid, AtomicOrdering::Release);
                }
                let le = self.inner.left_edges.load();
                // Root grow appends exactly one level, so the cache must describe
                // a tree of height `level`. A mismatch means the cache drifted
                // from the live tree (e.g. a crafted uneven-depth tree, or a
                // tx-store `left_edges` left stale by a rollback that shrank the
                // tree height); fail hard instead of a debug-only panic / a
                // silent append onto a stale vector whose entries name deleted or
                // reused recids.
                if le.len() != level {
                    self.inner.poisoned.store(true, AtomicOrdering::Release);
                    return Err(DbError::corrupt(
                        "btree leftEdges/level mismatch (stale structural cache); reopen the store",
                    ));
                }
                let mut grown = Vec::with_capacity(le.len() + 1);
                grown.extend_from_slice(&le);
                grown.push(new_root_recid);
                self.inner.left_edges.store(Arc::new(grown));
                return Ok(());
            }
            let start = if stack.is_empty() {
                self.left_edge(level)?
            } else {
                stack.pop().unwrap()
            };
            let (n, mut guard) = self.lock_covering(start, &sep, true)?;
            let current = guard.recid();
            // Same dual gate as the leaf split (rounds 4 + 5): LEFT|RIGHT flag
            // (concurrency serialization via the under-lock flag flip) AND
            // authoritative root-pointer identity (crafted-flag protection),
            // flag test first so ordinary dir splits skip the store read.
            let current_is_root =
                (n.flags & (LEFT | RIGHT)) == (LEFT | RIGHT) && self.is_current_root(current)?;
            let pos = self.kf().search(&n.keys, &sep);
            // A separator is the max key of a freshly-split left half: strictly
            // inside the covering dir and distinct from every existing separator
            // (sibling key spaces are disjoint). A crafted parent that already
            // contains it must be a corruption error, not a duplicate insert or
            // a debug-only panic.
            let ip = match pos {
                Err(ip) => ip,
                Ok(_) => return Err(DbError::corrupt("duplicate parent separator")),
            };
            let children = match &n.body {
                NodeBody::Dir { children } => children,
                NodeBody::Leaf { .. } | NodeBody::ExternalLeaf { .. } => {
                    return Err(DbError::corrupt("propagate reached a leaf node"))
                }
            };
            let new_keys = self.kf().insert(&n.keys, ip, sep.clone());
            let new_children = insert_long(children, ip + 1, new_child);
            let keys_len = self.kf().size(&new_keys);
            if keys_len <= self.inner.max_node_size {
                let updated = Node {
                    flags: n.flags,
                    link: n.link,
                    keys: new_keys,
                    body: NodeBody::Dir {
                        children: new_children,
                    },
                };
                self.store_update(current, &updated)?;
                return Ok(());
            }
            // split dir node
            let hh = keys_len / 2;
            let b = Node {
                flags: n.flags & !LEFT,
                link: n.link,
                keys: self.kf().copy_range(&new_keys, hh, keys_len),
                body: NodeBody::Dir {
                    children: new_children[hh..].to_vec(),
                },
            };
            let q = self.store_put(&b)?;
            let parent_sep = self.kf().get(&new_keys, hh - 1);
            let a = Node {
                flags: n.flags & !RIGHT,
                link: q,
                keys: self.kf().copy_range(&new_keys, 0, hh),
                body: NodeBody::Dir {
                    children: new_children[..hh].to_vec(),
                },
            };
            self.store_update(current, &a)?;
            child_was_root = current_is_root;
            guard.release();
            old_child = current;
            new_child = q;
            sep = parent_sep;
            level += 1;
        }
    }

    /// Left-edge recid of `level`; spins while a concurrent root split creating
    /// this level is between publishing the child and appending here. Bails with
    /// an error if the map was poisoned by a failed root-grow, so a level that
    /// will never be published cannot park a writer forever.
    fn left_edge(&self, level: usize) -> Result<u64> {
        loop {
            let le = self.inner.left_edges.load();
            if level < le.len() {
                return Ok(le[level]);
            }
            self.check_poison()?;
            std::thread::park_timeout(Duration::from_nanos(100));
        }
    }

    // ---------------- remove / replace ----------------

    pub fn remove(&self, key: &K<KF>) -> Result<Option<V<VF>>> {
        self.remove_internal(key, None)
    }

    pub fn remove_if(&self, key: &K<KF>, value: &V<VF>) -> Result<bool> {
        Ok(self.remove_internal(key, Some(value))?.is_some())
    }

    /// Blind remove (API parity): returns only whether an entry existed.
    pub fn remove_only(&self, key: &K<KF>) -> Result<bool> {
        Ok(self.remove_internal(key, None)?.is_some())
    }

    fn remove_internal(&self, key: &K<KF>, expected: Option<&V<VF>>) -> Result<Option<V<VF>>> {
        // External maps take the value read barrier's WRITE lock across the whole
        // remove, so no lock-free reader can observe a value recid this remove
        // deletes and the store then reuses (74b9963 / bd43aa1). The counter and
        // sync listeners fire inside the barrier (under the node lock); deferred
        // listeners fire after both locks release.
        let (old, sync_res) = if self.inner.value_inline {
            self.remove_barrier(key, expected)?
        } else {
            let _barrier = self.inner.external_lock.write();
            self.remove_barrier(key, expected)?
        };
        match old {
            None => Ok(None),
            Some(o) => {
                sync_res?; // a sync-listener failure skips the deferred fire
                self.fire_deferred(key, Some(&o), None)?;
                Ok(Some(o))
            }
        }
    }

    /// The remove core, run under the external write barrier (external maps). The
    /// node mutation, external value-record delete, counter adjustment and sync
    /// listeners all happen here (the sync fire result is returned so the caller
    /// can release the external barrier before surfacing it).
    fn remove_barrier(
        &self,
        key: &K<KF>,
        expected: Option<&V<VF>>,
    ) -> Result<(Option<V<VF>>, Result<()>)> {
        let (n, guard) = self.lock_leaf(key, None)?;
        let current = guard.recid();
        let pos = self.kf().search(&n.keys, key);
        if matches!(n.body, NodeBody::Dir { .. }) {
            return Err(DbError::corrupt("remove reached a directory node"));
        }
        let p = match pos {
            Ok(p) => p,
            Err(_) => return Ok((None, Ok(()))),
        };
        let old = self.leaf_value_at(&n.body, p)?;
        if let Some(exp) = expected {
            if !self.vf().element().equals(&old, exp) {
                return Ok((None, Ok(())));
            }
        }
        // no merging/rebalance (mapdb3 semantics): fence retained.
        let fence = n.leaf_fence().cloned();
        let new_keys = self.kf().delete(&n.keys, p);
        match &n.body {
            NodeBody::Leaf { values, .. } => {
                let updated = Node {
                    flags: n.flags,
                    link: n.link,
                    keys: new_keys,
                    body: NodeBody::Leaf {
                        values: self.vf().delete(values, p),
                        fence,
                    },
                };
                self.store_update(current, &updated)?;
            }
            NodeBody::ExternalLeaf { recids, .. } => {
                let recid = recids[p];
                let updated = Node {
                    flags: n.flags,
                    link: n.link,
                    keys: new_keys,
                    body: NodeBody::ExternalLeaf {
                        recids: delete_i64(recids, p),
                        fence,
                    },
                };
                self.store_update(current, &updated)?;
                // AFTER the node no longer references it (Java order): free the
                // value record. The write barrier keeps concurrent readers out.
                self.inner.store.delete(recid_or_corrupt(recid as u64)?)?;
            }
            NodeBody::Dir { .. } => unreachable!(),
        }
        // counter BEFORE listeners (mutation committed): a failing listener must
        // not desync `size_long`. If the counter update fails after the node
        // committed, poison the handle so a wrong O(1) count is never trusted.
        if let Err(e) = self.add_to_counter(-1) {
            self.poison();
            return Err(e);
        }
        let sync_res = self.fire_sync(key, Some(&old), None);
        drop(guard);
        Ok((Some(old), sync_res))
    }

    pub fn replace(&self, key: &K<KF>, value: V<VF>) -> Result<Option<V<VF>>> {
        self.replace_internal(key, None, value)
    }

    pub fn replace_if(&self, key: &K<KF>, old_value: &V<VF>, new_value: V<VF>) -> Result<bool> {
        Ok(self
            .replace_internal(key, Some(old_value), new_value)?
            .is_some())
    }

    fn replace_internal(
        &self,
        key: &K<KF>,
        expected: Option<&V<VF>>,
        new_value: V<VF>,
    ) -> Result<Option<V<VF>>> {
        let (n, guard) = self.lock_leaf(key, None)?;
        let current = guard.recid();
        let pos = self.kf().search(&n.keys, key);
        if matches!(n.body, NodeBody::Dir { .. }) {
            return Err(DbError::corrupt("replace reached a directory node"));
        }
        let p = match pos {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };
        let old = self.leaf_value_at(&n.body, p)?;
        if let Some(exp) = expected {
            if !self.vf().element().equals(&old, exp) {
                return Ok(None);
            }
        }
        // replace of an existing key: counter unchanged.
        self.publish_leaf_set(current, &n, p, new_value.clone())?;
        let sync_res = self.fire_sync(key, Some(&old), Some(&new_value));
        drop(guard);
        sync_res?;
        self.fire_deferred(key, Some(&old), Some(&new_value))?;
        Ok(Some(old))
    }

    // ---------------- iteration ----------------

    /// Leaf that routes `lo` (leftmost leaf when `None`), reached by the same
    /// unlocked routing as the writers.
    fn first_leaf_for_lower_bound(&self, lo: Option<&K<KF>>) -> Result<Node<KF, VF>> {
        let mut cyc = CycleGuard::new(CYCLE_DESCENT_SOFT);
        let mut current = self.root_recid()?;
        let mut n = self.load(current)?;
        while n.is_dir() {
            cyc.visit(current)?;
            let child_idx = match lo {
                None => Some(0),
                Some(k) => self.route_child(&n, k),
            };
            current = match child_idx {
                None => n.link,
                Some(i) => n.children()[i],
            };
            n = self.load(current)?;
        }
        Ok(n)
    }

    fn first_leaf_recid_for_lower_bound(&self, lo: Option<&K<KF>>) -> Result<u64> {
        let mut cyc = CycleGuard::new(CYCLE_DESCENT_SOFT);
        let mut current = self.root_recid()?;
        let mut n = self.load(current)?;
        while n.is_dir() {
            cyc.visit(current)?;
            let child_idx = match lo {
                None => Some(0),
                Some(k) => self.route_child(&n, k),
            };
            current = match child_idx {
                None => n.link,
                Some(i) => n.children()[i],
            };
            n = self.load(current)?;
        }
        Ok(current)
    }

    /// Bounded ascending entry iterator over `[lo,hi]` (`None` bound = open),
    /// weakly consistent. Yields `Result` items: a mid-scan load error surfaces
    /// as `Some(Err(..))`, after which the iterator is fused.
    pub fn entry_iter(
        &self,
        lo: Option<K<KF>>,
        lo_inc: bool,
        hi: Option<K<KF>>,
        hi_inc: bool,
    ) -> Result<EntryIter<S, KF, VF>> {
        if !self.inner.value_inline {
            // External leaves contain value RECIDS, so retaining a decoded leaf
            // after releasing `external_lock` would be unsafe: a remove may delete
            // and reuse one of those recids before the next call. The iterator
            // therefore stores only its last emitted KEY. Every `next()` takes the
            // read barrier, freshly resumes from that key, and expands exactly one
            // value before releasing the barrier (the Java reference algorithm).
            return Ok(EntryIter {
                map: self.clone(),
                done: false,
                hi,
                hi_inc,
                state: EntryIterState::External {
                    resume: lo,
                    resume_inc: lo_inc,
                },
            });
        }
        let start_leaf = self.first_leaf_for_lower_bound(lo.as_ref())?;
        let sp = match &lo {
            None => 0,
            Some(k) => match self.kf().search(&start_leaf.keys, k) {
                Ok(p) => {
                    if lo_inc {
                        p
                    } else {
                        p + 1
                    }
                }
                Err(ins) => ins,
            },
        };
        Ok(EntryIter {
            map: self.clone(),
            done: false,
            hi,
            hi_inc,
            state: EntryIterState::Inline {
                leaf: Some(start_leaf),
                pos: sp,
                lo_pending: lo.is_some(),
                lo,
                lo_inc,
                cyc: CycleGuard::new(CYCLE_SCAN_SOFT),
            },
        })
    }

    /// Return the first external-value entry satisfying the bounds. The caller
    /// MUST hold `external_lock` read across this entire call: descent loads a
    /// leaf containing value recids, and `leaf_value_at` must expand the chosen
    /// recid before a concurrent remove can delete/reuse it.
    ///
    /// Deliberately one FRESH entry per call (the Java reference algorithm),
    /// NOT leaf-batched: the external ascending iterator's contract is
    /// per-entry freshness — concurrent removes/updates between pulls are
    /// observed, and a closed store errors on the very next pull (both pinned
    /// by `btree_external_values` tests). The streaming DESCENDING iterator
    /// makes the opposite trade (leaf batches, entries at most one leaf stale)
    /// because its predecessor materialized the whole range eagerly.
    fn first_external_entry(
        &self,
        lo: Option<&K<KF>>,
        lo_inc: bool,
        hi: Option<&K<KF>>,
        hi_inc: bool,
    ) -> Result<Option<(K<KF>, V<VF>)>> {
        use std::cmp::Ordering;
        let mut cyc = CycleGuard::new(CYCLE_SCAN_SOFT);
        let mut leaf = self.first_leaf_for_lower_bound(lo)?;
        let mut pos = match lo {
            None => 0,
            Some(k) => match self.kf().search(&leaf.keys, k) {
                Ok(p) => {
                    if lo_inc {
                        p
                    } else {
                        p + 1
                    }
                }
                Err(ins) => ins,
            },
        };
        let lo_pending = lo.is_some();
        loop {
            while pos >= self.kf().size(&leaf.keys) {
                let link = leaf.link;
                if link == 0 {
                    return Ok(None);
                }
                cyc.visit(link)?;
                leaf = self.load(link)?;
                pos = 0;
            }
            let k = self.kf().get(&leaf.keys, pos);
            if lo_pending {
                if let Some(lo_k) = lo {
                    let c = self.kf().compare(&k, lo_k);
                    if c == Ordering::Less || (c == Ordering::Equal && !lo_inc) {
                        pos += 1;
                        continue;
                    }
                }
            }
            if let Some(hi_k) = hi {
                let c = self.kf().compare(&k, hi_k);
                if c == Ordering::Greater || (c == Ordering::Equal && !hi_inc) {
                    return Ok(None);
                }
            }
            let v = self.leaf_value_at(&leaf.body, pos)?;
            return Ok(Some((k, v)));
        }
    }

    /// Whole-map ascending iterator.
    pub fn iter(&self) -> Result<EntryIter<S, KF, VF>> {
        self.entry_iter(None, true, None, true)
    }

    /// One DESCENDING scan step. Descends to the leaf covering the current
    /// upper bound and returns `(batch, next)`: `batch` = that leaf's in-range
    /// entries in ASCENDING order (drain it from the back), `next` = the upper
    /// bound `(key, inclusive)` for the following step, or `None` when nothing
    /// smaller can qualify.
    ///
    /// Navigation left of a leaf needs no back-links: while routing down, every
    /// time the descent passes a dir child `i > 0` (or follows a B-link right),
    /// the separator immediately to the LEFT of the taken branch (`keys[i-1]`,
    /// an inclusive subtree high bound) is recorded; the tightest such
    /// separator is exactly the greatest possible key strictly below the
    /// reached leaf's coverage. The routing invariant makes each step's
    /// separator strictly smaller than that step's upper bound, so the walk
    /// provably terminates; the iterator additionally enforces the strict
    /// decrease and reports `corrupt` instead of looping on a mangled tree.
    ///
    /// `stack` retains the descent path across steps — frames of
    /// `(recid, dir-node snapshot, entry separator)` from the root down. A
    /// refill pops only the frames whose subtree (keys strictly above the entry
    /// separator) can no longer contain the tightened bound and re-descends
    /// from the deepest surviving frame, so a full reverse scan loads each dir
    /// node O(1) times amortized instead of once per visited leaf. Retained
    /// frames are stale snapshots — exactly the weak consistency the ascending
    /// iterator's retained leaf links already have; recids of tree nodes are
    /// never reused, so a stale frame can mis-route only as far as a link chase
    /// or an error, never into unrelated data.
    ///
    /// External-value maps: the CALLER must hold `external_lock` read across
    /// the whole call — values are expanded before return, so the batch owns
    /// plain values and never retains recids past the barrier.
    fn last_leaf_batch(
        &self,
        stack: &mut Vec<RevFrame<KF, VF>>,
        lo: Option<&K<KF>>,
        lo_inc: bool,
        hi: Option<&K<KF>>,
        hi_inc: bool,
    ) -> Result<DescendStep<KF, VF>> {
        use std::cmp::Ordering;
        let mut cyc = CycleGuard::new(CYCLE_DESCENT_SOFT);
        // Drop the path frames the tightened bound has moved left of. A frame
        // covers only keys strictly ABOVE its EFFECTIVE lower bound: its own
        // entry separator, or — for a child-0 frame (`None`) — the deepest
        // separator among its ancestors. So pop by the deepest retained
        // `Some`: if the bound is not strictly above it, that owner frame and
        // every deeper (inheriting) frame are all left of the bound.
        match hi {
            None => stack.clear(),
            Some(b) => loop {
                match stack.iter().rposition(|(_, _, s)| s.is_some()) {
                    // Only left-edge frames remain: they cover (-inf, ..] ∋ b.
                    None => break,
                    Some(pos) => {
                        let s = stack[pos].2.as_ref().expect("rposition found Some");
                        if self.kf().compare(b, s) == Ordering::Greater {
                            break;
                        }
                        stack.truncate(pos);
                    }
                }
            },
        }
        if stack.is_empty() {
            let root = self.root_recid()?;
            let node = self.load(root)?;
            stack.push((root, node, None));
        }
        // Tightest separator left of the (partially retained) descent path.
        let mut step_down: Option<K<KF>> = stack.iter().rev().find_map(|(_, _, s)| s.clone());
        // Descend from the deepest surviving frame to the covering leaf. Dir
        // frames are pushed; the leaf itself is never retained.
        let mut n = loop {
            let top_is_dir = match stack.last() {
                Some((_, node, _)) => node.is_dir(),
                None => return Err(DbError::corrupt("descending descent lost its root")),
            };
            if !top_is_dir {
                // Single-node tree: the root is a leaf; process it directly.
                let (_, node, _) = stack.pop().expect("nonempty stack");
                break node;
            }
            let (recid, idx, link, last_key) = {
                let (recid, node, _) = stack.last().expect("nonempty stack");
                let nkeys = self.kf().size(&node.keys);
                let last_key = if nkeys > 0 {
                    Some(self.kf().get(&node.keys, nkeys - 1))
                } else {
                    None
                };
                let idx = match hi {
                    None if node.link != 0 => None, // rightmost path: ride links right
                    None => match node.children().len().checked_sub(1) {
                        Some(last) => Some(last),
                        None => return Err(DbError::corrupt("dir node with no children")),
                    },
                    Some(k) => self.route_child(node, k),
                };
                (*recid, idx, node.link, last_key)
            };
            cyc.visit(recid)?;
            match idx {
                None => {
                    // B-link right: the sibling REPLACES this frame at the same
                    // level; its subtree holds keys above this node's last key.
                    if last_key.is_some() {
                        step_down = last_key.clone();
                    }
                    let node = self.load(link)?;
                    stack.pop();
                    stack.push((link, node, last_key));
                }
                Some(i) => {
                    let (child_recid, sep) = {
                        let (_, node, _) = stack.last().expect("nonempty stack");
                        let sep = if i > 0 {
                            Some(self.kf().get(&node.keys, i - 1))
                        } else {
                            None
                        };
                        (node.children()[i], sep)
                    };
                    if sep.is_some() {
                        step_down = sep.clone();
                    }
                    let child = self.load(child_recid)?;
                    if child.is_dir() {
                        stack.push((child_recid, child, sep));
                    } else {
                        break child;
                    }
                }
            }
        };
        // Leaf level: ride links right while the bound is beyond this leaf's
        // fence (concurrent split), or all the way for an open upper bound.
        loop {
            let follow = match hi {
                None => n.link != 0,
                Some(k) => n.link != 0 && self.beyond_leaf(&n, k),
            };
            if !follow {
                break;
            }
            let sz = self.kf().size(&n.keys);
            if sz > 0 {
                step_down = Some(self.kf().get(&n.keys, sz - 1));
            }
            cyc.visit(n.link)?;
            n = self.load(n.link)?;
        }
        let sz = self.kf().size(&n.keys);
        let mut batch = Vec::new();
        let mut saw_below_lo = false;
        for pos in 0..sz {
            let k = self.kf().get(&n.keys, pos);
            if let Some(lo_k) = lo {
                let c = self.kf().compare(&k, lo_k);
                if c == Ordering::Less || (c == Ordering::Equal && !lo_inc) {
                    saw_below_lo = true;
                    continue;
                }
            }
            if let Some(hi_k) = hi {
                let c = self.kf().compare(&k, hi_k);
                if c == Ordering::Greater || (c == Ordering::Equal && !hi_inc) {
                    break;
                }
            }
            let v = self.leaf_value_at(&n.body, pos)?;
            batch.push((k, v));
        }
        // A key below `lo` in this leaf ⇒ every leaf further left is entirely
        // below `lo`: the scan is complete once this batch drains.
        let next = if saw_below_lo { None } else { step_down };
        Ok((batch, next))
    }

    /// Bounded descending iterator over `[lo,hi]` (`None` bound = open),
    /// weakly consistent, STREAMING (spec 03 §7 second cut): one root descent
    /// per visited leaf, O(leaf) memory — never materializes the range. Yields
    /// `Result` items and fuses after the first error. External-value maps take
    /// the reclamation read barrier once per LEAF (batch expansion), not per
    /// entry, and never hold it while control is in user code.
    pub fn descending_entry_iter(
        &self,
        lo: Option<K<KF>>,
        lo_inc: bool,
        hi: Option<K<KF>>,
        hi_inc: bool,
    ) -> Result<DescendingEntryIter<S, KF, VF>> {
        Ok(DescendingEntryIter {
            map: self.clone(),
            lo,
            lo_inc,
            upper_open: hi.is_none(),
            upper: hi,
            upper_inc: hi_inc,
            buf: Vec::new(),
            stack: Vec::new(),
            done: false,
        })
    }

    /// Atomically remove and return the LEAST in-range entry, or `None` when
    /// empty. Retry loop over the first ascending candidate + conditional
    /// remove; the successful conditional remove is the mutation point, so poll
    /// never removes a value it did not return (selection is weakly consistent).
    pub fn poll_first_entry(
        &self,
        lo: Option<K<KF>>,
        lo_inc: bool,
        hi: Option<K<KF>>,
        hi_inc: bool,
    ) -> Result<Option<(K<KF>, V<VF>)>> {
        loop {
            let mut it = self.entry_iter(lo.clone(), lo_inc, hi.clone(), hi_inc)?;
            let first = match it.next() {
                None => return Ok(None),
                Some(r) => r?,
            };
            if self.remove_internal(&first.0, Some(&first.1))?.is_some() {
                return Ok(Some(first));
            }
        }
    }

    /// Atomically remove and return the GREATEST in-range entry, or `None` when
    /// empty. Mirror of `poll_first_entry`: each attempt takes the first
    /// element of the streaming descending iterator (O(log n + leaf) per
    /// attempt, replacing the old O(range) ascending scan-keep-last) and
    /// conditionally removes it.
    pub fn poll_last_entry(
        &self,
        lo: Option<K<KF>>,
        lo_inc: bool,
        hi: Option<K<KF>>,
        hi_inc: bool,
    ) -> Result<Option<(K<KF>, V<VF>)>> {
        loop {
            let mut it = self.descending_entry_iter(lo.clone(), lo_inc, hi.clone(), hi_inc)?;
            let last = match it.next() {
                None => return Ok(None),
                Some(r) => r?,
            };
            if self.remove_internal(&last.0, Some(&last.1))?.is_some() {
                return Ok(Some(last));
            }
        }
    }

    /// Walk in-range keys ascending, WITHOUT expanding external value records —
    /// only the keys group is read, so no external read barrier is needed (a
    /// deleted value recid cannot affect a key-only scan). `f` returns `Ok(false)`
    /// to stop early. Weakly consistent, like the entry iterators. Used by
    /// `size`/`clear`/`is_empty` so an external map never materializes O(range)
    /// value records just to count or collect keys.
    fn walk_keys<F>(
        &self,
        lo: Option<&K<KF>>,
        lo_inc: bool,
        hi: Option<&K<KF>>,
        hi_inc: bool,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(&K<KF>) -> Result<bool>,
    {
        use std::cmp::Ordering;
        let mut cyc = CycleGuard::new(CYCLE_SCAN_SOFT);
        let mut leaf = self.first_leaf_for_lower_bound(lo)?;
        let mut pos = match lo {
            None => 0,
            Some(k) => match self.kf().search(&leaf.keys, k) {
                Ok(p) => {
                    if lo_inc {
                        p
                    } else {
                        p + 1
                    }
                }
                Err(ins) => ins,
            },
        };
        let mut lo_pending = lo.is_some();
        loop {
            while pos >= self.kf().size(&leaf.keys) {
                let link = leaf.link;
                if link == 0 {
                    return Ok(());
                }
                cyc.visit(link)?;
                leaf = self.load(link)?;
                pos = 0;
            }
            let k = self.kf().get(&leaf.keys, pos);
            if lo_pending {
                if let Some(lo_k) = lo {
                    let c = self.kf().compare(&k, lo_k);
                    if c == Ordering::Less || (c == Ordering::Equal && !lo_inc) {
                        pos += 1;
                        continue;
                    }
                }
                lo_pending = false;
            }
            if let Some(hi_k) = hi {
                let c = self.kf().compare(&k, hi_k);
                if c == Ordering::Greater || (c == Ordering::Equal && !hi_inc) {
                    return Ok(());
                }
            }
            if !f(&k)? {
                return Ok(());
            }
            pos += 1;
        }
    }

    /// Entry count. When the O(1) size counter is enabled (Feature A) this reads
    /// the counter record in O(1); otherwise it walks the leaf chain (keys only).
    pub fn size_long(&self) -> Result<u64> {
        if self.inner.counter_recid != 0 {
            self.check_poison()?;
            let v = self
                .inner
                .store
                .get(nz(self.inner.counter_recid), &LONG)?
                .ok_or_else(|| DbError::corrupt("btree size counter record missing"))?;
            if v < 0 {
                // A negative persisted counter would cast to ~1.8e19 as u64 — a
                // wrong-config/corruption signal, not a real size.
                return Err(DbError::corrupt("btree size counter record is negative"));
            }
            return Ok(v as u64);
        }
        self.size_long_range(None, true, None, true)
    }

    /// Count in-range entries by walking KEYS only (external maps expand no value
    /// records here — see [`Self::walk_keys`]).
    pub fn size_long_range(
        &self,
        lo: Option<K<KF>>,
        lo_inc: bool,
        hi: Option<K<KF>>,
        hi_inc: bool,
    ) -> Result<u64> {
        let mut count = 0u64;
        self.walk_keys(lo.as_ref(), lo_inc, hi.as_ref(), hi_inc, |_k| {
            count += 1;
            Ok(true)
        })?;
        Ok(count)
    }

    pub fn is_empty(&self) -> Result<bool> {
        // Key-only early-exit scan: never materializes an external value record.
        let mut any = false;
        self.walk_keys(None, true, None, true, |_k| {
            any = true;
            Ok(false) // stop at the first key
        })?;
        Ok(!any)
    }

    /// Remove every entry (leaving empty leaves linked, mapdb3 semantics).
    pub fn clear(&self) -> Result<()> {
        // Collect KEYS only (no external value expansion), then remove each.
        let keys: Vec<K<KF>> = {
            let mut v = Vec::new();
            self.walk_keys(None, true, None, true, |k| {
                v.push(k.clone());
                Ok(true)
            })?;
            v
        };
        for k in &keys {
            self.remove(k)?;
        }
        Ok(())
    }

    /// Collect all entries (test/utility helper).
    pub fn entries(&self) -> Result<Vec<(K<KF>, V<VF>)>> {
        let mut v = Vec::new();
        for e in self.iter()? {
            v.push(e?);
        }
        Ok(v)
    }
}

// ===================== listener / MapExtra trait surface =====================

impl<S, KF, VF> ModificationAwareMap<K<KF>, V<VF>> for BTreeMap<S, KF, VF>
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    fn modification_listener_add(&self, listener: Arc<dyn MapModificationListener<K<KF>, V<VF>>>) {
        BTreeMap::modification_listener_add(self, listener)
    }
    fn modification_listener_remove(
        &self,
        listener: &Arc<dyn MapModificationListener<K<KF>, V<VF>>>,
    ) {
        BTreeMap::modification_listener_remove(self, listener)
    }
}

impl<S, KF, VF> MapExtra<K<KF>, V<VF>> for BTreeMap<S, KF, VF>
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    fn size_long(&self) -> Result<u64> {
        BTreeMap::size_long(self)
    }
    fn is_closed(&self) -> bool {
        BTreeMap::is_closed(self)
    }
    fn key_serializer(&self) -> &dyn Serializer<K<KF>> {
        self.kf().element()
    }
    fn value_serializer(&self) -> &dyn Serializer<V<VF>> {
        self.vf().element()
    }
}

// ===================== GetAction (push-down read) =====================

struct GetAction<'a, KF: GroupFormat, VF: GroupFormat> {
    key: &'a KF::Elem,
    kf: &'a KF,
    vf: &'a VF,
    /// `false` ⇔ external map: the leaf value slot holds a recid, captured in
    /// `stored_recid` and expanded by `do_get` after the read (never inside the
    /// action — the store contract forbids re-entrant store calls).
    value_inline: bool,
    value: Option<VF::Elem>,
    stored_recid: Option<i64>,
    found: bool,
}

impl<'a, KF: GroupFormat + Send + Sync + 'static, VF: GroupFormat + Send + Sync + 'static>
    RecordRead for GetAction<'a, KF, VF>
{
    fn on_bytes(&mut self, input: &mut SliceInput<'_>, size: usize) -> Result<i64> {
        self.found = false;
        self.value = None;
        self.stored_recid = None;
        let h = input.unpack_int()?;
        let flags = h & 0xF;
        let keys_len = ((h as u32) >> 4) as usize;
        // every key occupies >= 1 serialized byte, so keysLen > size is corrupt
        if keys_len > size {
            return Err(DbError::corrupt("node header keysLen exceeds record size"));
        }
        // `on_bytes` bypasses `NodeSerializer::deserialize`, so it must enforce
        // the same recid invariants: a non-rightmost node carries a
        // nonzero link, and every dir child is nonzero — otherwise a crafted
        // record could yield a spurious 0 "terminal" sentinel (silent absent) or
        // a `nz(0)` panic downstream.
        let link: u64 = if flags & RIGHT != 0 {
            0
        } else {
            let l = input.unpack_long()?;
            if l == 0 {
                return Err(DbError::corrupt("non-rightmost node with zero link"));
            }
            l
        };

        let pos: SearchResult = if self.kf.supports_binary() {
            self.kf.binary_search(self.key, input, keys_len)?
        } else {
            let g = self.kf.deserialize(input, keys_len)?;
            self.kf.search(&g, self.key)
        };

        if flags & DIR != 0 {
            let child_idx = search_idx(pos);
            let child_count = keys_len + if flags & RIGHT != 0 { 1 } else { 0 };
            // Same structural check as `NodeSerializer::deserialize`: a dir with
            // no children can never route; without this a crafted empty non-right
            // dir would silently "go right" instead of erroring.
            if child_count == 0 {
                return Err(DbError::corrupt("directory node with no children"));
            }
            if child_idx >= child_count {
                return Ok(link as i64); // beyond high bound: right sibling (nonzero)
            }
            input.unpack_long_skip(child_idx)?;
            let child = input.unpack_long()?;
            if child == 0 {
                return Err(DbError::corrupt("directory child recid is zero"));
            }
            return Ok(child as i64);
        }
        // leaf
        if let Ok(p) = pos {
            self.found = true;
            if self.value_inline {
                self.value = Some(if self.vf.supports_binary() {
                    self.vf.binary_get(input, keys_len, p)?
                } else {
                    let g = self.vf.deserialize(input, keys_len)?;
                    self.vf.get(&g, p)
                });
            } else {
                // external leaf: value slots are 8-byte-BE recids (Java LongFormat).
                self.stored_recid = Some(NODE_RECID_FORMAT.binary_get(input, keys_len, p)?);
            }
            return Ok(0);
        }
        let ip = search_idx(pos);
        if ip >= keys_len && link != 0 {
            return Ok(link as i64);
        }
        Ok(0)
    }

    fn on_object(&mut self, obj: &dyn Any) -> Result<i64> {
        self.found = false;
        self.value = None;
        self.stored_recid = None;
        let n = obj
            .downcast_ref::<Node<KF, VF>>()
            .ok_or_else(|| DbError::corrupt("btree GetAction: object is not a Node"))?;
        let pos = self.kf.search(&n.keys, self.key);
        if n.is_dir() {
            let child_idx = search_idx(pos);
            let children = n.children();
            if child_idx >= children.len() {
                return Ok(n.link as i64);
            }
            return Ok(children[child_idx] as i64);
        }
        if let Ok(p) = pos {
            self.found = true;
            match &n.body {
                NodeBody::Leaf { values, .. } => self.value = Some(self.vf.get(values, p)),
                NodeBody::ExternalLeaf { recids, .. } => self.stored_recid = Some(recids[p]),
                NodeBody::Dir { .. } => {}
            }
            return Ok(0);
        }
        let ip = search_idx(pos);
        if ip >= self.kf.size(&n.keys) && n.link != 0 {
            return Ok(n.link as i64);
        }
        Ok(0)
    }

    fn on_null(&mut self) -> Result<i64> {
        // A dir child / leaf link / root recid that resolves to a null (or
        // preallocated) record is a structurally impossible tree, not a "key
        // absent" answer: mirror `load()`'s null → corrupt so `get`/`contains`
        // agree with the write & iteration paths instead of silently reporting
        // a present-by-write key as absent (D5).
        Err(DbError::corrupt("btree node record is null"))
    }
}

// ===================== EntryIter =====================

/// Representation-specific cursor state. Keeping this as an enum makes it
/// impossible for external iteration to accidentally retain an inline leaf full
/// of reclaimable value recids between pull steps.
enum EntryIterState<KF: GroupFormat, VF: GroupFormat> {
    Inline {
        leaf: Option<Node<KF, VF>>,
        pos: usize,
        lo: Option<KF::Elem>,
        lo_inc: bool,
        lo_pending: bool,
        /// Detects a crafted leaf-link cycle (would otherwise loop forever
        /// emitting duplicate entries) in the retained inline leaf walk.
        cyc: CycleGuard,
    },
    External {
        /// Initial lower bound, then the last emitted key. It becomes exclusive
        /// after the first emission, so keys are never repeated.
        resume: Option<KF::Elem>,
        resume_inc: bool,
    },
}

/// Ascending, weakly-consistent leaf-link iterator (Java `entryIterator`'s
/// anonymous `Iterator`). Yields `Result<(K, V)>`; fused after the first error.
/// Inline values retain and drain one decoded leaf at a time. External values
/// resume from the last emitted key on every step, holding the reclamation read
/// barrier only through that step's fresh descent and value expansion; an idle
/// or abandoned iterator therefore never blocks removal.
pub struct EntryIter<S, KF: GroupFormat, VF: GroupFormat> {
    map: BTreeMap<S, KF, VF>,
    done: bool,
    hi: Option<KF::Elem>,
    hi_inc: bool,
    state: EntryIterState<KF, VF>,
}

impl<S, KF, VF> EntryIter<S, KF, VF>
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    fn advance(&mut self) -> Result<Option<(KF::Elem, VF::Elem)>> {
        if self.done {
            return Ok(None);
        }
        match &mut self.state {
            EntryIterState::External { resume, resume_inc } => {
                // The guard is deliberately local: never retain it (or a leaf full
                // of external recids) while control is in user code. `remove` takes
                // the write side across leaf unlink + value-record deletion, so the
                // selected recid cannot disappear or be reused before expansion.
                let next = {
                    let _barrier = self.map.inner.external_lock.read();
                    self.map.first_external_entry(
                        resume.as_ref(),
                        *resume_inc,
                        self.hi.as_ref(),
                        self.hi_inc,
                    )?
                };
                match next {
                    Some((k, v)) => {
                        // Resume EXCLUSIVELY from the emitted key. Concurrent inserts
                        // behind it are skipped; inserts ahead may be observed, which
                        // is the intended weakly-consistent Java behavior.
                        *resume = Some(k.clone());
                        *resume_inc = false;
                        Ok(Some((k, v)))
                    }
                    None => {
                        self.done = true;
                        Ok(None)
                    }
                }
            }
            EntryIterState::Inline {
                leaf,
                pos,
                lo,
                lo_inc,
                lo_pending,
                cyc,
            } => {
                let kf = &self.map.inner.key_format;
                let vf = &self.map.inner.value_format;
                loop {
                    // skip exhausted leaves, following links
                    loop {
                        match leaf.as_ref() {
                            Some(current) if *pos >= kf.size(&current.keys) => {
                                let link = current.link;
                                *leaf = if link == 0 {
                                    None
                                } else {
                                    cyc.visit(link)?;
                                    Some(self.map.load(link)?)
                                };
                                *pos = 0;
                            }
                            _ => break,
                        }
                    }
                    let current = match leaf.as_ref() {
                        None => {
                            self.done = true;
                            return Ok(None);
                        }
                        Some(current) => current,
                    };
                    let k = kf.get(&current.keys, *pos);
                    if *lo_pending {
                        let lower = lo.as_ref().unwrap();
                        let c = kf.compare(&k, lower);
                        if c == std::cmp::Ordering::Less
                            || (c == std::cmp::Ordering::Equal && !*lo_inc)
                        {
                            *pos += 1;
                            continue; // below lo: skip
                        }
                        *lo_pending = false;
                    }
                    if let Some(hi) = self.hi.as_ref() {
                        let c = kf.compare(&k, hi);
                        if c == std::cmp::Ordering::Greater
                            || (c == std::cmp::Ordering::Equal && !self.hi_inc)
                        {
                            self.done = true;
                            return Ok(None);
                        }
                    }
                    let v = match &current.body {
                        NodeBody::Leaf { values, .. } => vf.get(values, *pos),
                        NodeBody::ExternalLeaf { .. } | NodeBody::Dir { .. } => {
                            self.done = true;
                            return Err(DbError::corrupt(
                                "entry iterator reached a non-inline-leaf node",
                            ));
                        }
                    };
                    *pos += 1;
                    return Ok(Some((k, v)));
                }
            }
        }
    }
}

impl<S, KF, VF> Iterator for EntryIter<S, KF, VF>
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    type Item = Result<(KF::Elem, VF::Elem)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.advance() {
            Ok(Some(e)) => Some(Ok(e)),
            Ok(None) => None,
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

/// Descending, weakly-consistent streaming iterator (spec 03 §7 second cut).
/// One root descent per visited leaf, buffering exactly that leaf's in-range
/// entries; `upper` tightens strictly every step (enforced — a mangled tree
/// yields `corrupt`, never a loop). Yields `Result<(K, V)>`; fused after the
/// first error. External-value maps take the reclamation read barrier once per
/// leaf batch and release it before any entry is handed to user code.
pub struct DescendingEntryIter<S, KF: GroupFormat, VF: GroupFormat> {
    map: BTreeMap<S, KF, VF>,
    lo: Option<KF::Elem>,
    lo_inc: bool,
    /// Current upper bound key; meaningless while `upper_open`.
    upper: Option<KF::Elem>,
    upper_inc: bool,
    /// True until the first step of an open-upper-bound (`hi = None`) scan.
    upper_open: bool,
    /// Current leaf's in-range entries, ascending; drained from the back.
    buf: Vec<(KF::Elem, VF::Elem)>,
    /// Retained descent path (dir frames only) — see `last_leaf_batch`.
    stack: Vec<RevFrame<KF, VF>>,
    done: bool,
}

impl<S, KF, VF> DescendingEntryIter<S, KF, VF>
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    fn advance(&mut self) -> Result<Option<(KF::Elem, VF::Elem)>> {
        loop {
            if let Some(e) = self.buf.pop() {
                return Ok(Some(e));
            }
            if self.done {
                return Ok(None);
            }
            let hi = if self.upper_open {
                None
            } else {
                match &self.upper {
                    Some(k) => Some(k),
                    // Non-open scan with no bound left: exhausted.
                    None => {
                        self.done = true;
                        return Ok(None);
                    }
                }
            };
            let (batch, next) = if self.map.inner.value_inline {
                self.map.last_leaf_batch(
                    &mut self.stack,
                    self.lo.as_ref(),
                    self.lo_inc,
                    hi,
                    self.upper_inc,
                )?
            } else {
                // Barrier held across descent + whole-leaf value expansion,
                // released before any entry reaches user code. The retained
                // stack holds only dir-node snapshots (keys + child recids of
                // never-reused tree nodes) — no value recid outlives the
                // barrier.
                let _barrier = self.map.inner.external_lock.read();
                self.map.last_leaf_batch(
                    &mut self.stack,
                    self.lo.as_ref(),
                    self.lo_inc,
                    hi,
                    self.upper_inc,
                )?
            };
            self.buf = batch;
            match next {
                None => self.done = true,
                Some(next_k) => {
                    // The next bound must strictly decrease, or a corrupt tree
                    // (mis-sorted separators / aliased subtrees) could loop or
                    // re-emit entries forever.
                    if !self.upper_open {
                        if let Some(prev) = &self.upper {
                            if self.map.kf().compare(&next_k, prev) != std::cmp::Ordering::Less {
                                self.done = true;
                                self.buf.clear();
                                return Err(DbError::corrupt(
                                    "descending scan bound did not decrease",
                                ));
                            }
                        }
                    }
                    self.upper = Some(next_k);
                    self.upper_inc = true;
                    self.upper_open = false;
                }
            }
            if self.upper_open {
                // First step of an open scan reached the rightmost leaf and
                // recorded a real bound (or finished); never repeat the open
                // descent.
                self.upper_open = false;
            }
        }
    }
}

impl<S, KF, VF> Iterator for DescendingEntryIter<S, KF, VF>
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    type Item = Result<(KF::Elem, VF::Elem)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.advance() {
            Ok(Some(e)) => Some(Ok(e)),
            Ok(None) => None,
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

// ===================== navigable view integration =====================

use super::view::{OrderedMapAdapter, RangeView};

impl<S, KF, VF> OrderedMapAdapter for BTreeMap<S, KF, VF>
where
    S: Store + StoreLease + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    type Key = K<KF>;
    type Val = V<VF>;

    fn compare(&self, a: &K<KF>, b: &K<KF>) -> std::cmp::Ordering {
        self.compare_keys(a, b)
    }
    fn natural_order(&self) -> bool {
        self.key_natural_order()
    }
    fn value_equals(&self, a: &V<VF>, b: &V<VF>) -> bool {
        BTreeMap::value_equals(self, a, b)
    }
    fn get(&self, k: &K<KF>) -> Result<Option<V<VF>>> {
        BTreeMap::get(self, k)
    }
    fn contains_key(&self, k: &K<KF>) -> Result<bool> {
        BTreeMap::contains_key(self, k)
    }
    fn put(&self, k: K<KF>, v: V<VF>) -> Result<Option<V<VF>>> {
        BTreeMap::put(self, k, v)
    }
    fn remove(&self, k: &K<KF>) -> Result<Option<V<VF>>> {
        BTreeMap::remove(self, k)
    }
    fn remove_if(&self, k: &K<KF>, v: &V<VF>) -> Result<bool> {
        BTreeMap::remove_if(self, k, v)
    }
    fn put_if_absent(&self, k: K<KF>, v: V<VF>) -> Result<Option<V<VF>>> {
        BTreeMap::put_if_absent(self, k, v)
    }
    fn replace(&self, k: &K<KF>, v: V<VF>) -> Result<Option<V<VF>>> {
        BTreeMap::replace(self, k, v)
    }
    fn replace_if(&self, k: &K<KF>, ov: &V<VF>, nv: V<VF>) -> Result<bool> {
        BTreeMap::replace_if(self, k, ov, nv)
    }

    fn entry_iter_range<'a>(
        &'a self,
        lo: Option<K<KF>>,
        lo_inc: bool,
        hi: Option<K<KF>>,
        hi_inc: bool,
    ) -> Result<Box<dyn Iterator<Item = Result<(K<KF>, V<VF>)>> + 'a>> {
        Ok(Box::new(self.entry_iter(lo, lo_inc, hi, hi_inc)?))
    }

    fn descending_entry_iter_range<'a>(
        &'a self,
        lo: Option<K<KF>>,
        lo_inc: bool,
        hi: Option<K<KF>>,
        hi_inc: bool,
    ) -> Result<Box<dyn Iterator<Item = Result<(K<KF>, V<VF>)>> + 'a>> {
        Ok(Box::new(
            self.descending_entry_iter(lo, lo_inc, hi, hi_inc)?,
        ))
    }

    fn poll_first_range(
        &self,
        lo: Option<K<KF>>,
        lo_inc: bool,
        hi: Option<K<KF>>,
        hi_inc: bool,
    ) -> Result<Option<(K<KF>, V<VF>)>> {
        self.poll_first_entry(lo, lo_inc, hi, hi_inc)
    }

    fn poll_last_range(
        &self,
        lo: Option<K<KF>>,
        lo_inc: bool,
        hi: Option<K<KF>>,
        hi_inc: bool,
    ) -> Result<Option<(K<KF>, V<VF>)>> {
        self.poll_last_entry(lo, lo_inc, hi, hi_inc)
    }

    fn size_long_range(
        &self,
        lo: Option<K<KF>>,
        lo_inc: bool,
        hi: Option<K<KF>>,
        hi_inc: bool,
    ) -> Result<u64> {
        BTreeMap::size_long_range(self, lo, lo_inc, hi, hi_inc)
    }
}

impl<S, KF, VF> BTreeMap<S, KF, VF>
where
    S: Store + StoreLease + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    /// Full open-bounds ascending navigable view backing the whole navigable
    /// surface (nav queries, sub-maps, descending). Cheap (clones the `Arc`).
    pub fn view(&self) -> RangeView<Self> {
        RangeView::full(self.clone())
    }

    /// `[from, to)` half-open ascending sub-view (std-`BTreeMap::range` shape).
    pub fn range(&self, from: K<KF>, to: K<KF>) -> RangeView<Self> {
        self.view().sub_map(from, true, to, false)
    }

    /// Inclusive/exclusive-flagged sub-view.
    pub fn sub_map(&self, from: K<KF>, from_inc: bool, to: K<KF>, to_inc: bool) -> RangeView<Self> {
        self.view().sub_map(from, from_inc, to, to_inc)
    }

    pub fn head_map(&self, to: K<KF>, inc: bool) -> RangeView<Self> {
        self.view().head_map(to, inc)
    }

    pub fn tail_map(&self, from: K<KF>, inc: bool) -> RangeView<Self> {
        self.view().tail_map(from, inc)
    }

    /// Descending full view.
    pub fn descending(&self) -> RangeView<Self> {
        self.view().descending()
    }

    pub fn first_entry(&self) -> Result<Option<(K<KF>, V<VF>)>> {
        self.view().first_entry()
    }
    pub fn last_entry(&self) -> Result<Option<(K<KF>, V<VF>)>> {
        self.view().last_entry()
    }
    pub fn floor_entry(&self, k: &K<KF>) -> Result<Option<(K<KF>, V<VF>)>> {
        self.view().floor_entry(k)
    }
    pub fn ceiling_entry(&self, k: &K<KF>) -> Result<Option<(K<KF>, V<VF>)>> {
        self.view().ceiling_entry(k)
    }
    pub fn lower_entry(&self, k: &K<KF>) -> Result<Option<(K<KF>, V<VF>)>> {
        self.view().lower_entry(k)
    }
    pub fn higher_entry(&self, k: &K<KF>) -> Result<Option<(K<KF>, V<VF>)>> {
        self.view().higher_entry(k)
    }
    /// std-`BTreeMap::pop_first` shape (atomic, weakly-consistent selection).
    pub fn pop_first(&self) -> Result<Option<(K<KF>, V<VF>)>> {
        self.poll_first_entry(None, true, None, true)
    }
    pub fn pop_last(&self) -> Result<Option<(K<KF>, V<VF>)>> {
        self.poll_last_entry(None, true, None, true)
    }
}

// ===================== columnar single-column scan (R7) =====================

use crate::ser::columnar::ColumnarValueFormat;
use crate::ser::Value;

impl<S, KF> BTreeMap<S, KF, ColumnarValueFormat>
where
    S: Store + StoreLease + 'static,
    KF: GroupFormat + Send + Sync + 'static,
{
    /// Scan ONE value column over the ascending key range `[from, to]` (a `None`
    /// bound is open), invoking `f(key, cell)` with each in-range key paired with
    /// that column's value — WITHOUT materializing whole value rows on the byte
    /// path (reads only the requested column's contiguous bytes via
    /// `column_cursor`). Weakly consistent, exactly like [`Self::entry_iter`];
    /// keys are delivered ascending. The callback runs OUTSIDE the `RecordRead`
    /// (after validation), never inside it.
    pub fn for_each_value_column<F>(
        &self,
        from: Option<K<KF>>,
        from_inc: bool,
        to: Option<K<KF>>,
        to_inc: bool,
        column: usize,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(&K<KF>, &Value),
    {
        // Column scan is unavailable for external values (Java throws
        // UnsupportedOperationException — external leaves hold recids, not rows).
        // A healthy store: `Unsupported`, not `DataCorruption`.
        if !self.inner.value_inline {
            return Err(DbError::Unsupported(
                "column scan is unavailable for external values",
            ));
        }
        let cf = self.vf();
        assert!(
            column < cf.column_count(),
            "column {column} out of range (columns={})",
            cf.column_count()
        );
        let mut action = LeafColumnScan {
            cf,
            kf: self.kf(),
            column,
            lo_pending_in: from.is_some(),
            from,
            from_inc,
            to,
            to_inc,
            keys: Vec::new(),
            vals: Vec::new(),
            lo_pending_out: false,
            done: false,
        };
        let mut cyc = CycleGuard::new(CYCLE_SCAN_SOFT);
        let mut recid = self.first_leaf_recid_for_lower_bound(action.from.as_ref())?;
        while recid != 0 {
            cyc.visit(recid)?; // crafted leaf-link cycle → error, not infinite duplicate emits
            recid = self
                .inner
                .store
                .read(recid_or_corrupt(recid)?, &mut action)? as u64;
            // emit AFTER the validated read — never run user code inside a RecordRead.
            for i in 0..action.keys.len() {
                f(&action.keys[i], &action.vals[i]);
            }
            if action.done {
                return Ok(());
            }
            action.lo_pending_in = action.lo_pending_out;
        }
        Ok(())
    }
}

/// Per-leaf push-down action for [`BTreeMap::for_each_value_column`]: collects
/// one leaf's in-range `(key, column-cell)` pairs, reading only the requested
/// column's bytes on the byte path, returning the next leaf recid to visit (or 0
/// to STOP). Every invocation fully resets output state before decoding.
struct LeafColumnScan<'a, KF: GroupFormat> {
    cf: &'a ColumnarValueFormat,
    kf: &'a KF,
    column: usize,
    from: Option<KF::Elem>,
    from_inc: bool,
    to: Option<KF::Elem>,
    to_inc: bool,
    lo_pending_in: bool,
    keys: Vec<KF::Elem>,
    vals: Vec<Value>,
    lo_pending_out: bool,
    done: bool,
}

impl<'a, KF: GroupFormat + Send + Sync + 'static> LeafColumnScan<'a, KF> {
    fn reset_outputs(&mut self) {
        self.keys.clear();
        self.vals.clear();
        self.lo_pending_out = self.lo_pending_in;
        self.done = false;
    }

    /// First position satisfying the lower bound (mirrors entry_iter's startPos).
    fn lower_pos(&self, key_group: &KF::Group) -> usize {
        let from = self.from.as_ref().unwrap();
        match self.kf.search(key_group, from) {
            Ok(p) => {
                if self.from_inc {
                    p
                } else {
                    p + 1
                }
            }
            Err(ins) => ins,
        }
    }

    /// Exclusive end position for the upper bound; sets `done` when no later leaf
    /// can contain an in-range key.
    fn upper_pos(&mut self, key_group: &KF::Group, keys_len: usize) -> usize {
        let to = match &self.to {
            None => return keys_len,
            Some(t) => t,
        };
        let tp = match self.kf.search(key_group, to) {
            Ok(p) => {
                self.done = true; // found the bound key: later leaves are all greater
                if self.to_inc {
                    p + 1
                } else {
                    p
                }
            }
            Err(ins) => {
                self.done = ins < keys_len; // an existing key beyond the bound lives here
                ins
            }
        };
        tp.min(keys_len)
    }
}

impl<'a, KF: GroupFormat + Send + Sync + 'static> RecordRead for LeafColumnScan<'a, KF> {
    fn on_bytes(&mut self, input: &mut SliceInput<'_>, size: usize) -> Result<i64> {
        self.reset_outputs();
        let h = input.unpack_int()?;
        let flags = h & 0xF;
        let keys_len = ((h as u32) >> 4) as usize;
        if flags & DIR != 0 {
            return Err(DbError::corrupt("column scan reached a directory node"));
        }
        if keys_len > size {
            return Err(DbError::corrupt("node header keysLen exceeds record size"));
        }
        let link: u64 = if flags & RIGHT != 0 {
            0
        } else {
            let l = input.unpack_long()?;
            if l == 0 {
                return Err(DbError::corrupt("non-rightmost leaf with zero link"));
            }
            l
        };
        let key_group = self.kf.deserialize(input, keys_len)?; // leaves `input` at value-group start
        let lo_pos = if self.lo_pending_in && self.from.is_some() {
            self.lower_pos(&key_group)
        } else {
            0
        };
        let to_pos = self.upper_pos(&key_group, keys_len);
        self.lo_pending_out = self.lo_pending_in && lo_pos >= keys_len;
        let from_pos = lo_pos.min(to_pos);

        let mut vc = self
            .cf
            .column_cursor(input, keys_len, self.column, from_pos, to_pos)?;
        let mut i = from_pos;
        while vc.next()? {
            self.keys.push(self.kf.get(&key_group, i));
            self.vals.push(vc.value());
            i += 1;
        }
        drop(vc);
        Ok(if self.done { 0 } else { link as i64 })
    }

    fn on_object(&mut self, obj: &dyn Any) -> Result<i64> {
        self.reset_outputs();
        let n = obj
            .downcast_ref::<Node<KF, ColumnarValueFormat>>()
            .ok_or_else(|| DbError::corrupt("column scan: object is not a Node"))?;
        if n.is_dir() {
            return Err(DbError::corrupt("column scan reached a directory node"));
        }
        let keys_len = self.kf.size(&n.keys);
        let lo_pos = if self.lo_pending_in && self.from.is_some() {
            self.lower_pos(&n.keys)
        } else {
            0
        };
        let to_pos = self.upper_pos(&n.keys, keys_len);
        self.lo_pending_out = self.lo_pending_in && lo_pos >= keys_len;
        let values = match &n.body {
            NodeBody::Leaf { values, .. } => values,
            // Columnar scan is inline-only (Java: unavailable for external values).
            NodeBody::ExternalLeaf { .. } | NodeBody::Dir { .. } => {
                return Err(DbError::corrupt("column scan on a non-inline-leaf node"))
            }
        };
        for i in lo_pos.min(to_pos)..to_pos {
            self.keys.push(self.kf.get(&n.keys, i));
            let row = self.cf.get(values, i); // materialized row fallback
            self.vals.push(row[self.column].clone());
        }
        Ok(if self.done { 0 } else { n.link as i64 })
    }

    fn on_null(&mut self) -> Result<i64> {
        // A leaf-chain link (or first-leaf recid) pointing at a null/preallocated
        // record must error, not fall through: the default `on_null` returns 0
        // WITHOUT resetting outputs, which would re-emit the previous leaf's
        // batch to the user callback and then silently truncate the scan (D5).
        Err(DbError::corrupt("btree leaf chain reached a null record"))
    }
}

fn insert_long(arr: &[u64], pos: usize, value: u64) -> Vec<u64> {
    let mut r = Vec::with_capacity(arr.len() + 1);
    r.extend_from_slice(&arr[..pos]);
    r.push(value);
    r.extend_from_slice(&arr[pos..]);
    r
}

// ===================== bulk build (TreePump) =====================

use super::pump::{NodeSink, TreePump};

/// TreePump sink for a BTreeMap: materializes a node and writes it to its
/// preallocated recid. Mirrors the anonymous Java `NodeSink` in
/// `createFromSorted`.
struct BTreeSink<'a, S: Store, KF: GroupFormat, VF: GroupFormat> {
    store: &'a S,
    kf: &'a KF,
    vf: &'a VF,
    max_node_size: usize,
}

impl<
        'a,
        S: Store,
        KF: GroupFormat + Send + Sync + 'static,
        VF: GroupFormat + Send + Sync + 'static,
    > NodeSink for BTreeSink<'a, S, KF, VF>
{
    type Key = KF::Elem;
    type Val = VF::Elem;

    fn compare_keys(&self, a: &KF::Elem, b: &KF::Elem) -> std::cmp::Ordering {
        self.kf.compare(a, b)
    }

    fn write_leaf(
        &self,
        recid: u64,
        flags: i32,
        link: u64,
        keys: Vec<KF::Elem>,
        values: Vec<VF::Elem>,
    ) -> Result<()> {
        // non-rightmost leaf: fence = last key (its inclusive high bound).
        let fence = if flags & RIGHT == 0 {
            Some(self.kf.from_slice(&[keys[keys.len() - 1].clone()]))
        } else {
            None
        };
        let node = Node {
            flags,
            link,
            keys: self.kf.from_slice(&keys),
            body: NodeBody::Leaf {
                values: self.vf.from_slice(&values),
                fence,
            },
        };
        self.store.update(
            nz(recid),
            Some(&node),
            &NodeSerializer::new(self.kf, self.vf, self.max_node_size),
        )
    }

    fn write_dir(
        &self,
        recid: u64,
        flags: i32,
        link: u64,
        keys: Vec<KF::Elem>,
        children: Vec<u64>,
    ) -> Result<()> {
        let node: Node<KF, VF> = Node {
            flags,
            link,
            keys: self.kf.from_slice(&keys),
            body: NodeBody::Dir { children },
        };
        self.store.update(
            nz(recid),
            Some(&node),
            &NodeSerializer::new(self.kf, self.vf, self.max_node_size),
        )
    }
}

impl<S, KF, VF> BTreeMap<S, KF, VF>
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    /// Bulk build with the default pump fill (3/4 of `max_node_size`), no counter.
    /// Bulk builds are always INLINE (external mode is rejected — Java's `Sink`
    /// only builds inline maps).
    pub fn create_from_sorted<I>(
        store: Arc<S>,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
        entries: I,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = (K<KF>, V<VF>)>,
    {
        let fill = TreePump::<S, BTreeSink<S, KF, VF>>::default_fill(max_node_size);
        Self::create_from_sorted_fill(
            store,
            key_format,
            value_format,
            max_node_size,
            fill,
            entries,
            false,
        )
    }

    /// Bulk build with the default pump fill, optionally enabling an O(1) size
    /// counter (Feature A) initialized to the number of entries written.
    pub fn create_from_sorted_counter<I>(
        store: Arc<S>,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
        entries: I,
        counter_enable: bool,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = (K<KF>, V<VF>)>,
    {
        let fill = TreePump::<S, BTreeSink<S, KF, VF>>::default_fill(max_node_size);
        Self::create_from_sorted_fill(
            store,
            key_format,
            value_format,
            max_node_size,
            fill,
            entries,
            counter_enable,
        )
    }

    /// Bulk build from STRICTLY ascending entries (`NotSorted` on misorder or
    /// duplicate). Single-threaded; the caller commits. When `counter_enable`,
    /// an O(1) size counter is allocated and initialized to the entry count.
    pub fn create_from_sorted_fill<I>(
        store: Arc<S>,
        key_format: KF,
        value_format: VF,
        max_node_size: usize,
        node_fill: usize,
        entries: I,
        counter_enable: bool,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = (K<KF>, V<VF>)>,
    {
        // Create-time API arg (never from stored bytes): configuration error (R3).
        if !(MIN_MAX_NODE_SIZE..=MAX_MAX_NODE_SIZE).contains(&max_node_size) {
            return Err(DbError::wrong_config("maxNodeSize must be in 4..=1048576"));
        }
        let (root_recid_recid, count) = {
            let sink = BTreeSink {
                store: &*store,
                kf: &key_format,
                vf: &value_format,
                max_node_size,
            };
            let mut pump = TreePump::new(&*store, &sink, max_node_size, node_fill);
            let mut count = 0i64;
            for (k, v) in entries {
                pump.put(k, v)?;
                count = count.wrapping_add(1); // Java `long` wraps; size_long guards negative
            }
            let root_recid = pump.finish()?;
            (store.put(&(root_recid as i64), &LONG)?.get(), count)
        };
        let counter_recid = if counter_enable {
            store.put(&count, &LONG)?.get()
        } else {
            0
        };
        Self::open_mode(
            store,
            root_recid_recid,
            key_format,
            value_format,
            max_node_size,
            counter_recid,
            true,
        )
    }
}
