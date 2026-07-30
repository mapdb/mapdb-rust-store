#![allow(private_bounds)]
//! `Bind` — secondary indexes and derived views over a primary `BTreeMap`
//! (Java `org.mapdb.Bind`).
//!
//! Two listener classes, exactly as Java:
//! - **Order-sensitive** (`secondary_value`, `secondary_values`, `secondary_key`,
//!   `secondary_keys`, `map_inverse`) install a SYNCHRONOUS listener, fired under
//!   the covering leaf lock that serialized the mutation, so same-key events are
//!   totally ordered with the mutations (last writer wins in the index too).
//! - **Ordinary/deferred** (`size`, `histogram`, `map_put_after_delete`) install a
//!   plain listener, fired after the leaf lock is released and split propagation
//!   completes.
//!
//! ## Initial population
//! Every order-sensitive binding (and `histogram`) first checks whether its
//! secondary is empty and, if so, iterates the primary's EXISTING entries and
//! replays the derive function, exactly as Java (`if (secondary.isEmpty()) for
//! (Entry e : primary.entrySet()) ...`). The initial scan and the listener
//! registration are NOT atomic against concurrent writers, so install every
//! binding while the primary is quiescent.
//!
//! ## Secondary containers
//! Secondaries are accepted through the [`SecMap`] / [`SecSet`] traits, so a
//! binding can target either the in-memory [`SecondaryMap`] / [`SecondarySet`]
//! (the Rust stand-in for Java's `ConcurrentMap` / `Set`) OR a persistent
//! [`BTreeMap`] (Java accepts any `Map`). `map_put_after_delete` and `histogram`
//! keep their concrete [`SecondaryMap`] parameter (Java takes `Map` /
//! `ConcurrentMap`).
//!
//! ## Self-binding and lock cycles
//! Direct self-binding is rejected: the standalone [`reject_self_bind`] compares
//! two same-typed handles, and every order-sensitive install ALSO rejects a
//! secondary whose [`SecMap::sec_identity`] matches the primary's
//! [`state_id`](BTreeMap::state_id) — a persistent secondary that is the primary
//! itself. Binding a map to a clone of itself would re-enter the covering leaf
//! lock a synchronous listener already holds. This does NOT detect
//! longer/transitive cycles: a synchronous callback mutates a secondary while
//! holding the primary leaf lock, so reciprocal or transitive bindings can
//! deadlock. Install bindings while the primary is quiescent.
//!
//! ## Exception hazard
//! A binding listener that throws (`secondary_key` / `map_inverse` on a duplicate
//! derived key) does so AFTER the primary has already been mutated; the primary
//! and any already-updated secondaries are left mutated — the error only signals
//! the constraint violation (Java parity).

use crate::btree::BTreeMap;
use crate::db::atomic::AtomicLong;
use crate::error::{DbError, Result};
use crate::listener::{MapModificationListener, SynchronousMapModificationListener};
use crate::ser::GroupFormat;
use crate::store::{Store, StoreLease};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

type K<F> = <F as GroupFormat>::Elem;
type V<F> = <F as GroupFormat>::Elem;

// ============================ secondary containers ============================

/// A shareable secondary map (Java `ConcurrentMap`).
pub struct SecondaryMap<Kd, Vd>(Arc<Mutex<HashMap<Kd, Vd>>>);

impl<Kd, Vd> Clone for SecondaryMap<Kd, Vd> {
    fn clone(&self) -> Self {
        SecondaryMap(Arc::clone(&self.0))
    }
}

impl<Kd: Eq + Hash + Clone, Vd: Clone> SecondaryMap<Kd, Vd> {
    pub fn new() -> Self {
        SecondaryMap(Arc::new(Mutex::new(HashMap::new())))
    }
    pub fn get(&self, k: &Kd) -> Option<Vd> {
        self.0.lock().get(k).cloned()
    }
    pub fn contains_key(&self, k: &Kd) -> bool {
        self.0.lock().contains_key(k)
    }
    pub fn len(&self) -> usize {
        self.0.lock().len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.lock().is_empty()
    }
    pub fn keys(&self) -> Vec<Kd> {
        self.0.lock().keys().cloned().collect()
    }
}

impl<Kd: Eq + Hash + Clone, Vd: Clone> Default for SecondaryMap<Kd, Vd> {
    fn default() -> Self {
        Self::new()
    }
}

/// A shareable secondary set (Java `Set`).
pub struct SecondarySet<T>(Arc<Mutex<HashSet<T>>>);

impl<T> Clone for SecondarySet<T> {
    fn clone(&self) -> Self {
        SecondarySet(Arc::clone(&self.0))
    }
}

impl<T: Eq + Hash + Clone> SecondarySet<T> {
    pub fn new() -> Self {
        SecondarySet(Arc::new(Mutex::new(HashSet::new())))
    }
    pub fn contains(&self, t: &T) -> bool {
        self.0.lock().contains(t)
    }
    pub fn len(&self) -> usize {
        self.0.lock().len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.lock().is_empty()
    }
}

impl<T: Eq + Hash + Clone> Default for SecondarySet<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================ SecMap / SecSet traits ============================

/// A map-like secondary target (Java `Map`): either the in-memory [`SecondaryMap`]
/// or a persistent [`BTreeMap`]. Bindings mutate it under the primary's leaf lock.
pub trait SecMap<Kd, Vd>: Send + Sync + 'static {
    fn sec_is_empty(&self) -> Result<bool>;
    fn sec_get(&self, k: &Kd) -> Result<Option<Vd>>;
    fn sec_put(&self, k: Kd, v: Vd) -> Result<()>;
    /// Java `ConcurrentMap.putIfAbsent`: atomically insert `(k, v)` iff `k` is
    /// absent, returning the value already present (if any). ONE critical section,
    /// so a unique-index install cannot lose a race between check and insert (R4).
    fn sec_put_if_absent(&self, k: Kd, v: Vd) -> Result<Option<Vd>>;
    fn sec_remove(&self, k: &Kd) -> Result<()>;
    /// Java `Map.remove(key, value)`: remove only if `k` currently maps to `v`.
    fn sec_remove_if_value(&self, k: &Kd, v: &Vd) -> Result<()>;
    /// The primary's [`state_id`](BTreeMap::state_id) when this secondary is a
    /// persistent map (so a self-bind can be detected), else `None`.
    fn sec_identity(&self) -> Option<usize>;
}

/// A set-like secondary target (Java `Set`) for the one-to-many indexes.
pub trait SecSet<T>: Send + Sync + 'static {
    fn sec_is_empty(&self) -> Result<bool>;
    fn sec_add(&self, t: T) -> Result<()>;
    fn sec_remove(&self, t: &T) -> Result<()>;
    fn sec_identity(&self) -> Option<usize>;
}

impl<Kd, Vd> SecMap<Kd, Vd> for SecondaryMap<Kd, Vd>
where
    Kd: Eq + Hash + Clone + Send + Sync + 'static,
    Vd: Clone + PartialEq + Send + Sync + 'static,
{
    fn sec_is_empty(&self) -> Result<bool> {
        Ok(self.0.lock().is_empty())
    }
    fn sec_get(&self, k: &Kd) -> Result<Option<Vd>> {
        Ok(self.0.lock().get(k).cloned())
    }
    fn sec_put(&self, k: Kd, v: Vd) -> Result<()> {
        self.0.lock().insert(k, v);
        Ok(())
    }
    fn sec_put_if_absent(&self, k: Kd, v: Vd) -> Result<Option<Vd>> {
        use std::collections::hash_map::Entry;
        let mut m = self.0.lock();
        match m.entry(k) {
            Entry::Occupied(e) => Ok(Some(e.get().clone())),
            Entry::Vacant(e) => {
                e.insert(v);
                Ok(None)
            }
        }
    }
    fn sec_remove(&self, k: &Kd) -> Result<()> {
        self.0.lock().remove(k);
        Ok(())
    }
    fn sec_remove_if_value(&self, k: &Kd, v: &Vd) -> Result<()> {
        let mut m = self.0.lock();
        if m.get(k).map_or(false, |ex| ex == v) {
            m.remove(k);
        }
        Ok(())
    }
    fn sec_identity(&self) -> Option<usize> {
        None
    }
}

impl<T> SecSet<T> for SecondarySet<T>
where
    T: Eq + Hash + Clone + Send + Sync + 'static,
{
    fn sec_is_empty(&self) -> Result<bool> {
        Ok(self.0.lock().is_empty())
    }
    fn sec_add(&self, t: T) -> Result<()> {
        self.0.lock().insert(t);
        Ok(())
    }
    fn sec_remove(&self, t: &T) -> Result<()> {
        self.0.lock().remove(t);
        Ok(())
    }
    fn sec_identity(&self) -> Option<usize> {
        None
    }
}

/// A persistent [`BTreeMap`] used as a secondary map (Java accepts any `Map`).
/// `sec_identity` exposes the map's shared-state address so a self-bind — the
/// secondary handle being the primary itself — is detected and rejected.
impl<S, KF, VF> SecMap<K<KF>, V<VF>> for BTreeMap<S, KF, VF>
where
    S: Store + StoreLease + Send + Sync + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    fn sec_is_empty(&self) -> Result<bool> {
        BTreeMap::is_empty(self)
    }
    fn sec_get(&self, k: &K<KF>) -> Result<Option<V<VF>>> {
        BTreeMap::get(self, k)
    }
    fn sec_put(&self, k: K<KF>, v: V<VF>) -> Result<()> {
        BTreeMap::put(self, k, v).map(|_| ())
    }
    fn sec_put_if_absent(&self, k: K<KF>, v: V<VF>) -> Result<Option<V<VF>>> {
        BTreeMap::put_if_absent(self, k, v)
    }
    fn sec_remove(&self, k: &K<KF>) -> Result<()> {
        BTreeMap::remove(self, k).map(|_| ())
    }
    fn sec_remove_if_value(&self, k: &K<KF>, v: &V<VF>) -> Result<()> {
        BTreeMap::remove_if(self, k, v).map(|_| ())
    }
    fn sec_identity(&self) -> Option<usize> {
        Some(self.state_id())
    }
}

// ============================ self-bind guard ============================

/// Reject binding a map to a clone of itself (Java `Bind` self-cycle guard),
/// comparing two handles of the SAME concrete type.
pub fn reject_self_bind<S, KF, VF>(
    primary: &BTreeMap<S, KF, VF>,
    secondary: &BTreeMap<S, KF, VF>,
) -> Result<()>
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    if primary.shares_state_with(secondary) {
        return Err(DbError::wrong_config(
            "cannot bind a map to itself (direct self-cycle)",
        ));
    }
    Ok(())
}

/// Reject a secondary whose type-erased identity is the primary itself (used at
/// every order-sensitive install, so a persistent-map self-bind is caught even
/// though the secondary's generic value type differs from the primary's).
fn reject_self_bind_id<S, KF, VF>(
    primary: &BTreeMap<S, KF, VF>,
    secondary_id: Option<usize>,
) -> Result<()>
where
    S: Store + StoreLease + Send + Sync + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
{
    if secondary_id == Some(primary.state_id()) {
        return Err(DbError::wrong_config(
            "cannot bind a map to itself (direct self-cycle)",
        ));
    }
    Ok(())
}

// ---------------- synchronous listener adapters ----------------

struct SyncFn<Ke, Ve, F> {
    f: F,
    _marker: std::marker::PhantomData<fn(&Ke, &Ve)>,
}

impl<Ke, Ve, F> MapModificationListener<Ke, Ve> for SyncFn<Ke, Ve, F>
where
    Ke: Send + Sync,
    Ve: Send + Sync,
    F: Fn(&Ke, Option<&Ve>, Option<&Ve>) -> Result<()> + Send + Sync,
{
    fn modify(&self, k: &Ke, old: Option<&Ve>, new: Option<&Ve>, _t: bool) -> Result<()> {
        (self.f)(k, old, new)
    }
}
impl<Ke, Ve, F> SynchronousMapModificationListener<Ke, Ve> for SyncFn<Ke, Ve, F>
where
    Ke: Send + Sync,
    Ve: Send + Sync,
    F: Fn(&Ke, Option<&Ve>, Option<&Ve>) -> Result<()> + Send + Sync,
{
}

fn add_sync<S, KF, VF, F>(primary: &BTreeMap<S, KF, VF>, f: F)
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
    K<KF>: Send + Sync,
    V<VF>: Send + Sync,
    F: Fn(&K<KF>, Option<&V<VF>>, Option<&V<VF>>) -> Result<()> + Send + Sync + 'static,
{
    let l = Arc::new(SyncFn {
        f,
        _marker: std::marker::PhantomData,
    });
    primary.modification_listener_add_sync(l);
}

fn add_deferred<S, KF, VF, F>(primary: &BTreeMap<S, KF, VF>, f: F)
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
    K<KF>: Send + Sync,
    V<VF>: Send + Sync,
    F: Fn(&K<KF>, Option<&V<VF>>, Option<&V<VF>>) -> Result<()> + Send + Sync + 'static,
{
    let l: Arc<dyn MapModificationListener<K<KF>, V<VF>>> = Arc::new(SyncFn {
        f,
        _marker: std::marker::PhantomData,
    });
    primary.modification_listener_add(l);
}

/// Java `Bind.putUnique`: insert `derived -> primary_key`, tolerating a duplicate
/// that already maps to the SAME primary key, but rejecting one mapping to a
/// different key.
fn put_unique<Dk, Pk, Sec>(sec: &Sec, derived: Dk, primary_key: &Pk) -> Result<()>
where
    Pk: PartialEq + Clone,
    Sec: SecMap<Dk, Pk>,
{
    // Atomic check-and-insert (`putIfAbsent`): two writers on different primary
    // leaves whose derived keys collide can no longer both observe "absent" and
    // both insert — exactly one wins, the other sees the winner and errors (R4).
    match sec.sec_put_if_absent(derived, primary_key.clone())? {
        None => Ok(()),
        Some(existing) => {
            if &existing != primary_key {
                Err(DbError::wrong_config(
                    "unique secondary index: duplicate derived key",
                ))
            } else {
                Ok(())
            }
        }
    }
}

// ---------------- order-sensitive bindings (synchronous) ----------------

/// `secondary[key] = derive(key, value)`, removed when the primary key is removed.
pub fn secondary_value<S, KF, VF, Dv, Sec, F>(
    primary: &BTreeMap<S, KF, VF>,
    secondary: Sec,
    derive: F,
) -> Result<()>
where
    S: Store + StoreLease + Send + Sync + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
    K<KF>: Clone + Send + Sync,
    V<VF>: Send + Sync,
    Dv: Send + Sync + 'static,
    Sec: SecMap<K<KF>, Dv>,
    F: Fn(&K<KF>, &V<VF>) -> Dv + Send + Sync + 'static,
{
    reject_self_bind_id(primary, secondary.sec_identity())?;
    if secondary.sec_is_empty()? {
        for (k, v) in primary.entries()? {
            let d = derive(&k, &v);
            secondary.sec_put(k, d)?;
        }
    }
    add_sync(primary, move |k, _old, new| {
        match new {
            Some(v) => secondary.sec_put(k.clone(), derive(k, v))?,
            None => secondary.sec_remove(k)?,
        }
        Ok(())
    });
    Ok(())
}

/// A one-to-many value index: `secondary` holds `(primary_key, derived_value)`
/// tuples for every value `derive(key, value)` yields (Java `secondaryValues`).
pub fn secondary_values<S, KF, VF, Dv, Sec, F>(
    primary: &BTreeMap<S, KF, VF>,
    secondary: Sec,
    derive: F,
) -> Result<()>
where
    S: Store + StoreLease + Send + Sync + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
    K<KF>: Clone + Send + Sync,
    V<VF>: Send + Sync,
    Dv: Send + Sync + 'static,
    Sec: SecSet<(K<KF>, Dv)>,
    F: Fn(&K<KF>, &V<VF>) -> Vec<Dv> + Send + Sync + 'static,
{
    reject_self_bind_id(primary, secondary.sec_identity())?;
    if secondary.sec_is_empty()? {
        for (k, v) in primary.entries()? {
            for dv in derive(&k, &v) {
                secondary.sec_add((k.clone(), dv))?;
            }
        }
    }
    add_sync(primary, move |k, old, new| {
        if let Some(ov) = old {
            for dv in derive(k, ov) {
                secondary.sec_remove(&(k.clone(), dv))?;
            }
        }
        if let Some(nv) = new {
            for dv in derive(k, nv) {
                secondary.sec_add((k.clone(), dv))?;
            }
        }
        Ok(())
    });
    Ok(())
}

/// A UNIQUE single-key secondary index: `secondary[derive(key,value)] = key`.
/// A derived key that already maps to a different primary key is rejected
/// (Java throws `IllegalArgumentException` from the listener).
pub fn secondary_key<S, KF, VF, Dk, Sec, F>(
    primary: &BTreeMap<S, KF, VF>,
    secondary: Sec,
    derive: F,
) -> Result<()>
where
    S: Store + StoreLease + Send + Sync + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
    K<KF>: PartialEq + Clone + Send + Sync,
    V<VF>: Send + Sync,
    Dk: Send + Sync + 'static,
    Sec: SecMap<Dk, K<KF>>,
    F: Fn(&K<KF>, &V<VF>) -> Dk + Send + Sync + 'static,
{
    reject_self_bind_id(primary, secondary.sec_identity())?;
    if secondary.sec_is_empty()? {
        for (k, v) in primary.entries()? {
            put_unique(&secondary, derive(&k, &v), &k)?;
        }
    }
    add_sync(primary, move |k, old, new| {
        if let Some(ov) = old {
            let dk = derive(k, ov);
            secondary.sec_remove_if_value(&dk, k)?;
        }
        if let Some(nv) = new {
            let dk = derive(k, nv);
            put_unique(&secondary, dk, k)?;
        }
        Ok(())
    });
    Ok(())
}

/// A multi-key secondary index: `secondary` holds `(derived_key, primary_key)`
/// tuples for every derived key of a value (Java `secondaryKeys`).
pub fn secondary_keys<S, KF, VF, Dk, Sec, F>(
    primary: &BTreeMap<S, KF, VF>,
    secondary: Sec,
    derive: F,
) -> Result<()>
where
    S: Store + StoreLease + Send + Sync + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
    K<KF>: Clone + Send + Sync,
    V<VF>: Send + Sync,
    Dk: Send + Sync + 'static,
    Sec: SecSet<(Dk, K<KF>)>,
    F: Fn(&K<KF>, &V<VF>) -> Vec<Dk> + Send + Sync + 'static,
{
    reject_self_bind_id(primary, secondary.sec_identity())?;
    if secondary.sec_is_empty()? {
        for (k, v) in primary.entries()? {
            for dk in derive(&k, &v) {
                secondary.sec_add((dk, k.clone()))?;
            }
        }
    }
    add_sync(primary, move |k, old, new| {
        if let Some(ov) = old {
            for dk in derive(k, ov) {
                secondary.sec_remove(&(dk, k.clone()))?;
            }
        }
        if let Some(nv) = new {
            for dk in derive(k, nv) {
                secondary.sec_add((dk, k.clone()))?;
            }
        }
        Ok(())
    });
    Ok(())
}

/// Inverse index: `inverse[value] = key`. Values must be UNIQUE — a value already
/// mapping to a different key is rejected (Java `mapInverse` delegates to
/// `secondaryKey` with the identity value projection).
pub fn map_inverse<S, KF, VF, Sec>(primary: &BTreeMap<S, KF, VF>, inverse: Sec) -> Result<()>
where
    S: Store + StoreLease + Send + Sync + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
    K<KF>: PartialEq + Clone + Send + Sync,
    V<VF>: Clone + Send + Sync + 'static,
    Sec: SecMap<V<VF>, K<KF>>,
{
    secondary_key(primary, inverse, |_k, v: &V<VF>| v.clone())
}

// ---------------- deferred bindings ----------------

/// Maintain a running size in an [`AtomicLong`] (Java `Bind.size`). Seeds the
/// counter from the primary's current size ONLY when the counter is 0 (Java's
/// `if (counter.get() == 0)`), then increments on insert / decrements on remove.
pub fn size<S, KF, VF>(primary: &BTreeMap<S, KF, VF>, counter: &AtomicLong<S>) -> Result<()>
where
    S: Store + StoreLease + Send + Sync + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
    K<KF>: Send + Sync,
    V<VF>: Send + Sync,
{
    if counter.get()? == 0 {
        counter.set(primary.size_long()? as i64)?;
    }
    let counter = counter.clone();
    add_deferred(primary, move |_k, old, new| {
        match (old, new) {
            (None, Some(_)) => {
                counter.increment_and_get()?;
            }
            (Some(_), None) => {
                counter.decrement_and_get()?;
            }
            _ => {}
        }
        Ok(())
    });
    Ok(())
}

/// A category histogram (Java `Bind.histogram`): counts values by `category`. A
/// category whose count reaches EXACTLY 0 is removed; negative counts are kept
/// (Java `addCount`: `value == 0 ? null : value`).
pub fn histogram<S, KF, VF, C, F>(
    primary: &BTreeMap<S, KF, VF>,
    hist: SecondaryMap<C, i64>,
    category: F,
) -> Result<()>
where
    S: Store + StoreLease + Send + Sync + 'static,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
    K<KF>: Send + Sync,
    V<VF>: Send + Sync,
    C: Eq + Hash + Clone + Send + Sync + 'static,
    F: Fn(&K<KF>, &V<VF>) -> C + Send + Sync + 'static,
{
    // Initial population over the primary's existing entries. `category` is
    // evaluated BEFORE the histogram lock is held (Java evaluates `category.apply`
    // before `ConcurrentMap.compute`) so a closure that reads the same histogram
    // cannot self-deadlock the non-reentrant mutex (R5).
    if hist.is_empty() {
        for (k, v) in primary.entries()? {
            let c = category(&k, &v);
            let mut m = hist.0.lock();
            let n = m.get(&c).copied().unwrap_or(0).wrapping_add(1);
            if n == 0 {
                m.remove(&c);
            } else {
                m.insert(c, n);
            }
        }
    }
    add_deferred(primary, move |k, old, new| {
        // Evaluate categories BEFORE taking the lock (R5): reentrancy-safe.
        let old_c = old.map(|ov| category(k, ov));
        let new_c = new.map(|nv| category(k, nv));
        let mut m = hist.0.lock();
        if let Some(c) = old_c {
            // Java `long` wraps on overflow; keep parity with wrapping_sub (R5).
            let n = m.get(&c).copied().unwrap_or(0).wrapping_sub(1);
            if n == 0 {
                m.remove(&c);
            } else {
                m.insert(c, n);
            }
        }
        if let Some(c) = new_c {
            let n = m.get(&c).copied().unwrap_or(0).wrapping_add(1);
            if n == 0 {
                m.remove(&c);
            } else {
                m.insert(c, n);
            }
        }
        Ok(())
    });
    Ok(())
}

/// Capture the removed value of every deleted key (Java `Bind.mapPutAfterDelete`).
/// Accepts any [`SecMap`] target (in-memory or a persistent [`BTreeMap`]).
pub fn map_put_after_delete<S, KF, VF, Sec>(primary: &BTreeMap<S, KF, VF>, deleted: Sec)
where
    S: Store + StoreLease,
    KF: GroupFormat + Send + Sync + 'static,
    VF: GroupFormat + Send + Sync + 'static,
    K<KF>: Clone + Send + Sync,
    V<VF>: Clone + Send + Sync,
    Sec: SecMap<K<KF>, V<VF>>,
{
    add_deferred(primary, move |k, old, new| {
        if new.is_none() {
            if let Some(ov) = old {
                deleted.sec_put(k.clone(), ov.clone())?;
            }
        }
        Ok(())
    });
}
