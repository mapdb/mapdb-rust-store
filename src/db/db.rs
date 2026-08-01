#![allow(private_bounds)]
//! The `DB` facade (Java `org.mapdb.db.DB`): the name catalog at recid 1, typed
//! collection makers, a DB-owned per-name instance cache, and the close/Drop
//! lifecycle. The reference semantics are the Java implementation's
//! `org.mapdb.db.DB` in the mapdb-java-store repository.

use crate::btree::BTreeMap;
use crate::db::atomic::{AtomicBoolean, AtomicInteger, AtomicLong, AtomicString, AtomicVar};
use crate::db::catalog::{NameCatalog, CATALOG_SER, RECID_CATALOG};
use crate::db::descriptor::{
    group_descriptor_or_custom, ser_descriptor_or_custom, verify_group, verify_ser,
    GroupDescriptor, SerDescriptor,
};
use crate::db::set::{NavigableSet, NoValueFormat};
use crate::error::{DbError, Result};
use crate::listener::MapModificationListener;
use crate::queue::blocking::{Mode, PersistentBlockingQueue};
use crate::ser::families::BOOLEAN;
use crate::ser::serializers::{INT, LONG};
use crate::ser::{GroupFormat, Serializer};
use crate::store::{
    Recid, Store, StoreByteArray, StoreDirect, StoreLease, StoreOnHeap, StoreReadOnlyWrapper,
    StoreWAL,
};
use parking_lot::Mutex;
use std::any::Any;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

type K<F> = <F as GroupFormat>::Elem;
type V<F> = <F as GroupFormat>::Elem;

// DB lifecycle states.
const OPEN: u8 = 0;
const CLOSING: u8 = 1;
const CLOSED: u8 = 2;

/// A store that the `DB` facade can roll back if it is transactional. Non-tx
/// stores return `Unsupported` (Java `UnsupportedOperationException`).
pub trait DbRollback {
    fn try_rollback(&self) -> Result<()>;
}

impl DbRollback for StoreOnHeap {
    fn try_rollback(&self) -> Result<()> {
        Err(DbError::Unsupported(
            "rollback on a non-transactional store",
        ))
    }
}
impl DbRollback for StoreByteArray {
    fn try_rollback(&self) -> Result<()> {
        Err(DbError::Unsupported(
            "rollback on a non-transactional store",
        ))
    }
}
impl DbRollback for StoreDirect {
    fn try_rollback(&self) -> Result<()> {
        Err(DbError::Unsupported(
            "rollback on a non-transactional store",
        ))
    }
}
impl DbRollback for StoreWAL {
    fn try_rollback(&self) -> Result<()> {
        use crate::store::StoreTx;
        self.rollback()
    }
}
impl<S> DbRollback for StoreReadOnlyWrapper<S> {
    fn try_rollback(&self) -> Result<()> {
        Err(DbError::Unsupported("rollback on a read-only store"))
    }
}
impl DbRollback for crate::db::store_kind::ConfiguredStore {
    fn try_rollback(&self) -> Result<()> {
        self.rollback()
    }
}

#[inline]
fn nz(v: u64) -> Result<Recid> {
    NonZeroU64::new(v).ok_or_else(|| DbError::corrupt("recid must be non-zero"))
}

/// A cached open instance: a boxed shared-state clone of the public handle plus an
/// optional teardown hook (queues wake blocked waiters on close/delete).
struct CacheEntry {
    handle: Box<dyn Any + Send + Sync>,
    close_hook: Option<Box<dyn Fn() + Send + Sync>>,
    /// Best-effort data-record teardown for `delete()` (Java `obj.clear()` /
    /// collection teardown). Maps/sets clear their ENTRY/NODE records (the
    /// structural root + counter records LEAK, per Java — freeing them would let
    /// the store reuse a recid a still-live map clone keeps writing, review C1);
    /// queues free their node records AND the header record (all queue clones
    /// share one globally-closable handle). Atomics have no teardown — `delete`
    /// frees their single `#recid` directly.
    teardown: Option<Box<dyn FnOnce() -> Result<()> + Send + Sync>>,
}

/// State serialized by the DB administrative lock: catalog + instance cache.
struct Admin {
    catalog: NameCatalog,
    cache: HashMap<String, CacheEntry>,
}

/// The DB facade over a store `S`. Not `Clone`: it owns the store's lifecycle.
/// Collections hold their own `Arc<S>`, so they outlive the `DB` value but fail
/// with `StoreClosed` once the DB closes the store.
pub struct DB<S> {
    store: Arc<S>,
    admin: Mutex<Admin>,
    state: AtomicU8,
    /// Files removed after the store closes (temp DB / delete-after-close).
    /// NON-TRANSACTIONAL backends only: a WAL store owns a segment namespace
    /// rather than a file and deletes it itself, inside `close`, while it still
    /// holds the store lock (D2).
    cleanup_paths: Vec<PathBuf>,
    /// Closes the backing store. Captured at construction so `Drop` (which cannot
    /// carry `S: Store` bounds) can close without those bounds. NO auto-commit.
    close_store: Box<dyn Fn() -> Result<()> + Send + Sync>,
}

// ============================ construction / init ============================

impl<S> DB<S>
where
    S: Store + StoreLease + DbRollback + Send + Sync + 'static,
{
    /// Wrap a store, initializing or validating the name catalog at recid 1.
    ///
    /// Fresh (empty) store: `put(empty catalog)` must land on recid 1 (else a
    /// foreign writer already occupies it → error, best-effort undo), then the
    /// initial empty catalog is committed (the sole hidden facade commit).
    ///
    /// Non-empty store: recid 1 must already hold a valid catalog; a missing
    /// record is a wrong store, malformed bytes are corruption — never repaired.
    pub fn new(store: Arc<S>) -> Result<Self> {
        Self::with_cleanup(store, Vec::new())
    }

    pub(crate) fn with_cleanup(store: Arc<S>, cleanup_paths: Vec<PathBuf>) -> Result<Self> {
        let is_empty = store.get_all_recids()?.is_empty();
        let catalog = if is_empty {
            if store.is_read_only() {
                return Err(DbError::wrong_config(
                    "cannot open an empty store read-only: the catalog cannot be written",
                ));
            }
            let empty = NameCatalog::new();
            let recid = store.put(&empty, &CATALOG_SER)?;
            if recid.get() != RECID_CATALOG {
                // A foreign writer already occupies recid 1: best-effort undo.
                let _ = store.delete(recid);
                return Err(DbError::wrong_config(
                    "store is not a fresh MapDB store: recid 1 is already taken",
                ));
            }
            // The sole hidden facade commit: persist the fresh empty catalog.
            store.commit()?;
            empty
        } else {
            // Recid 1 must decode as a valid catalog.
            match store.get(nz(RECID_CATALOG)?, &CATALOG_SER) {
                Ok(Some(cat)) => cat,
                Ok(None) => {
                    return Err(DbError::wrong_config(
                        "store has data but no name catalog at recid 1 (wrong store?)",
                    ))
                }
                Err(DbError::GetVoid(_)) => {
                    return Err(DbError::wrong_config(
                        "store has data but recid 1 is not allocated (wrong store?)",
                    ))
                }
                Err(e) => return Err(e),
            }
        };
        // Reject a hostile / malformed catalog at open, before any collection can
        // be built over it.
        validate_catalog(&catalog)?;
        let close_store: Box<dyn Fn() -> Result<()> + Send + Sync> = {
            let s = Arc::clone(&store);
            Box::new(move || s.close())
        };
        Ok(DB {
            store,
            admin: Mutex::new(Admin {
                catalog,
                cache: HashMap::new(),
            }),
            state: AtomicU8::new(OPEN),
            cleanup_paths,
            close_store,
        })
    }

    /// The backing store (shared handle).
    pub fn store(&self) -> &Arc<S> {
        &self.store
    }
}

// ============================ typed constructors ============================

impl DB<StoreOnHeap> {
    /// A DB over a fresh heap store (Java `DBMaker.heapDB()`).
    pub fn make_heap() -> Result<Self> {
        DB::new(Arc::new(StoreOnHeap::new(true)))
    }
}

impl DB<StoreByteArray> {
    /// A DB over a fresh in-memory byte-array store (Java `DBMaker.memoryByteArrayDB()`).
    pub fn make_byte_array() -> Result<Self> {
        DB::new(Arc::new(StoreByteArray::new(true)))
    }
}

impl DB<StoreDirect> {
    /// A DB over an in-memory StoreDirect (Java `DBMaker.memoryDB()` / `memoryDirectDB()`).
    pub fn make_memory_direct() -> Result<Self> {
        DB::new(Arc::new(StoreDirect::new_heap()?))
    }
    /// A DB over a file-backed StoreDirect (Java `DBMaker.fileDB(f).make()`).
    pub fn make_direct(path: &std::path::Path) -> Result<Self> {
        DB::new(Arc::new(StoreDirect::open_file(path)?))
    }
}

impl DB<StoreWAL> {
    /// A DB over a file-backed WAL store (Java `DBMaker.fileDB(f).transactionEnable()`).
    pub fn make_wal(path: &std::path::Path) -> Result<Self> {
        DB::new(Arc::new(StoreWAL::open(path)?))
    }
}

// ============================ catalog / admin helpers ============================

impl<S> DB<S>
where
    S: Store + StoreLease + DbRollback + Send + Sync + 'static,
{
    /// Acquire the administrative lock, THEN recheck lifecycle: an operation that
    /// queued behind an in-flight `close`/`rollback` must observe `StoreClosed`
    /// once it wins the lock, so it cannot mutate/create into a closing DB.
    /// All catalog/cache mutations funnel through here.
    fn lock_admin(&self) -> Result<parking_lot::MutexGuard<'_, Admin>> {
        let guard = self.admin.lock();
        if self.state.load(Ordering::Acquire) != OPEN {
            return Err(DbError::StoreClosed);
        }
        Ok(guard)
    }

    /// Persist a candidate catalog to recid 1 (Java DB does NOT auto-commit
    /// named-object catalog edits). The caller installs it in `admin.catalog`
    /// only after this returns `Ok`, so a save failure leaves the in-memory
    /// catalog untouched (CRITICAL review #3).
    fn save_catalog(&self, catalog: &NameCatalog) -> Result<()> {
        self.store
            .update(nz(RECID_CATALOG)?, Some(catalog), &CATALOG_SER)
    }

    /// Run every cached instance's teardown hook (wakes blocked queue waiters),
    /// then clear the cache. Shared by `close`, `rollback`, and `delete`-all.
    fn wake_and_clear_cache(admin: &mut Admin) {
        for entry in admin.cache.values() {
            if let Some(hook) = &entry.close_hook {
                hook();
            }
        }
        admin.cache.clear();
    }

    /// Stage a catalog mutation on a COPY, persist it FIRST, and install it in
    /// `admin.catalog` only after the save succeeds (CRITICAL review #3). A save
    /// failure leaves the in-memory catalog and the cache untouched.
    fn publish_catalog(
        &self,
        admin: &mut Admin,
        mutate: impl FnOnce(&mut NameCatalog),
    ) -> Result<()> {
        let mut staged = admin.catalog.clone();
        mutate(&mut staged);
        self.save_catalog(&staged)?;
        admin.catalog = staged;
        Ok(())
    }

    /// A snapshot copy of the whole name catalog (Java `getNameCatalog`). Errors
    /// `StoreClosed` on a closed DB (Java `checkOpen`, MINOR review #13).
    pub fn get_name_catalog(&self) -> Result<NameCatalog> {
        let admin = self.lock_admin()?;
        Ok(admin.catalog.clone())
    }

    /// True if a named object exists (Java `exists`). Errors `StoreClosed` on a
    /// closed DB (Java `checkOpen`).
    pub fn exists(&self, name: &str) -> Result<bool> {
        let admin = self.lock_admin()?;
        Ok(admin.catalog.contains_key(&format!("{name}#type")))
    }

    /// The stored `#type` of `name`, or `None` (Java `getType`). Errors
    /// `StoreClosed` on a closed DB (Java `checkOpen`).
    pub fn get_type(&self, name: &str) -> Result<Option<String>> {
        let admin = self.lock_admin()?;
        Ok(admin.catalog.get(&format!("{name}#type")).cloned())
    }

    /// All named objects (Java `getAllNames`), ascending. Errors `StoreClosed` on
    /// a closed DB (Java `checkOpen`).
    pub fn get_all_names(&self) -> Result<Vec<String>> {
        let admin = self.lock_admin()?;
        let mut names = Vec::new();
        for k in admin.catalog.keys() {
            if let Some(name) = k.strip_suffix("#type") {
                names.push(name.to_string());
            }
        }
        Ok(names)
    }

    /// Commit the backing store (Java `commit`). No-op on a read-only store.
    /// Serialized with catalog mutation / close via the admin lock (§4).
    pub fn commit(&self) -> Result<()> {
        let _admin = self.lock_admin()?;
        if self.store.is_read_only() {
            return Ok(());
        }
        self.store.commit()
    }

    /// Roll back the backing store and clear catalog-derived handles (Java
    /// `rollback`). Fails `Unsupported` on a non-transactional store. After a
    /// successful rollback every cached instance's teardown hook runs (waking
    /// blocked queue waiters — CRITICAL review #1), the cache is cleared, and the
    /// catalog is reloaded + revalidated from recid 1:
    /// externally held old handles remain memory-safe but must not be reused, and
    /// D12 rejects a fresh independent open until they drop.
    pub fn rollback(&self) -> Result<()> {
        let mut admin = self.lock_admin()?;
        self.store.try_rollback()?;
        // Wake + drop all cached handles BEFORE reloading (Java closeRuntimeHandle
        // on every instance): a queue's external Arc holds no D12 lease, so a
        // reopen would otherwise build a second lock domain over the same header.
        Self::wake_and_clear_cache(&mut admin);
        // Reload the catalog from the reverted store and revalidate it.
        let reloaded = self
            .store
            .get(nz(RECID_CATALOG)?, &CATALOG_SER)?
            .unwrap_or_default();
        validate_catalog(&reloaded)?;
        admin.catalog = reloaded;
        Ok(())
    }

    /// Reclaim obsolete storage where supported (Java `compact`). Serialized via
    /// the admin lock (§4).
    pub fn compact(&self) -> Result<()> {
        let _admin = self.lock_admin()?;
        self.store.compact()
    }

    /// True once the DB has begun/finished closing.
    pub fn is_closed(&self) -> bool {
        self.state.load(Ordering::Acquire) != OPEN
    }

    /// Rename a named object (Java `rename`): `old` must exist, `new#type` must
    /// not; rewrites every `old#...` key to `new#...` without touching data recids,
    /// saves once, and moves the cache entry. Java does NOT commit.
    pub fn rename(&self, old: &str, new: &str) -> Result<()> {
        validate_name(old)?;
        validate_name(new)?;
        let mut admin = self.lock_admin()?;
        if !admin.catalog.contains_key(&format!("{old}#type")) {
            return Err(DbError::wrong_config(format!("no such name: {old}")));
        }
        if admin.catalog.contains_key(&format!("{new}#type")) {
            return Err(DbError::wrong_config(format!(
                "cannot rename onto existing name: {new}"
            )));
        }
        // Stage the rewrite on a COPY; install only after the save succeeds so a
        // save failure cannot leave a half-renamed in-memory catalog (CRITICAL #3).
        let old_prefix = format!("{old}#");
        let mut staged = admin.catalog.clone();
        let moved: Vec<(String, String)> = staged
            .iter()
            .filter(|(k, _)| k.starts_with(&old_prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (k, v) in moved {
            let suffix = &k[old_prefix.len()..];
            staged.remove(&k);
            staged.insert(format!("{new}#{suffix}"), v);
        }
        self.save_catalog(&staged)?;
        admin.catalog = staged;
        // Cache move only after the catalog is durably restated.
        if let Some(entry) = admin.cache.remove(old) {
            admin.cache.insert(new.to_string(), entry);
        }
        Ok(())
    }

    /// Delete a named object (Java `delete`): unlink all `name#...` catalog keys
    /// and save FIRST, drop the cache/lease association, then best-effort free the
    /// object's data. Returns `false` if the name did not exist. Java does NOT
    /// commit. A teardown failure leaves an unlinked leak, never a catalog pointer
    /// to destroyed data.
    pub fn delete(&self, name: &str) -> Result<bool> {
        let mut admin = self.lock_admin()?;
        let type_key = format!("{name}#type");
        let stored_type = match admin.catalog.get(&type_key) {
            Some(t) => t.clone(),
            None => return Ok(false),
        };
        // Capture the atomic's single record recid BEFORE unlinking (params vanish
        // on strip). ONLY atomics free a record by recid here — a collection's
        // structural root / counter / header records are NEVER freed off the
        // catalog, because a still-live handle clone keeps using them and the
        // store would hand the recid to a new object (review C1). Collections free
        // their ENTRY/NODE records through the cached handle's teardown instead.
        let atomic_recid = if is_atomic_type(&stored_type) {
            admin
                .catalog
                .get(&format!("{name}#recid"))
                .and_then(|s| s.parse::<u64>().ok())
        } else {
            None
        };
        // Stage the unlink on a COPY; install only after the save succeeds so a
        // save failure cannot leave a phantom-deleted in-memory catalog (CRITICAL #3).
        let prefix = format!("{name}#");
        let mut staged = admin.catalog.clone();
        staged.retain(|k, _| !k.starts_with(&prefix));
        self.save_catalog(&staged)?;
        admin.catalog = staged;
        // Unlinked and durable. Drop the cached handle. Run the close hook FIRST
        // (a queue marks itself closed and wakes blocked waiters, so no live clone
        // can operate on the records the teardown is about to free), THEN the
        // data-record teardown (map/set `clear()`; queue node+header free). Both
        // are best-effort: a failure now only leaks records — it can never leave a
        // catalog pointer to freed data.
        if let Some(entry) = admin.cache.remove(name) {
            if let Some(hook) = entry.close_hook {
                hook();
            }
            if let Some(teardown) = entry.teardown {
                let _ = teardown();
            }
        }
        // An atomic's single record is safe to free: it has no shared structural
        // root that a clone would keep mutating (Java `store.delete(atomicRecid)`).
        if let Some(v) = atomic_recid {
            if let Ok(r) = nz(v) {
                let _ = self.store.delete(r);
            }
        }
        Ok(true)
    }

    // ---------- cache primitives ----------

    fn cache_lookup<T: Clone + Send + Sync + 'static>(
        admin: &Admin,
        name: &str,
    ) -> Option<Result<T>> {
        admin.cache.get(name).map(|e| {
            e.handle.downcast_ref::<T>().cloned().ok_or_else(|| {
                DbError::CachedTypeMismatch(format!(
                    "'{name}' is already open as a different concrete type"
                ))
            })
        })
    }

    fn cache_insert<T: Clone + Send + Sync + 'static>(
        admin: &mut Admin,
        name: &str,
        handle: T,
        close_hook: Option<Box<dyn Fn() + Send + Sync>>,
    ) {
        Self::cache_insert_full(admin, name, handle, close_hook, None)
    }

    fn cache_insert_full<T: Clone + Send + Sync + 'static>(
        admin: &mut Admin,
        name: &str,
        handle: T,
        close_hook: Option<Box<dyn Fn() + Send + Sync>>,
        teardown: Option<Box<dyn FnOnce() -> Result<()> + Send + Sync>>,
    ) {
        admin.cache.insert(
            name.to_string(),
            CacheEntry {
                handle: Box::new(handle),
                close_hook,
                teardown,
            },
        );
    }

    // ============================ makers ============================

    /// Begin a typed tree-map open/create (Java `treeMap`).
    pub fn tree_map<KF, VF>(
        &self,
        name: &str,
        key_format: KF,
        value_format: VF,
    ) -> TreeMapMaker<'_, S, KF, VF>
    where
        KF: GroupFormat,
        VF: GroupFormat,
    {
        TreeMapMaker {
            db: self,
            name: name.to_string(),
            key_format,
            value_format,
            max_node_size: 32,
            counter_enable: false,
            values_outside: false,
            listeners: Vec::new(),
        }
    }

    /// Begin a typed tree-set open/create (Java `treeSet`).
    pub fn tree_set<KF>(&self, name: &str, key_format: KF) -> TreeSetMaker<'_, S, KF>
    where
        KF: GroupFormat,
    {
        TreeSetMaker {
            db: self,
            name: name.to_string(),
            key_format,
            max_node_size: 32,
            counter_enable: false,
        }
    }

    /// Begin an atomic-long open/create (Java `atomicLong`).
    pub fn atomic_long(&self, name: &str) -> AtomicLongMaker<'_, S> {
        AtomicLongMaker {
            db: self,
            name: name.to_string(),
            initial: 0,
        }
    }
    /// Atomic-long with an explicit initial value.
    pub fn atomic_long_init(&self, name: &str, initial: i64) -> AtomicLongMaker<'_, S> {
        AtomicLongMaker {
            db: self,
            name: name.to_string(),
            initial,
        }
    }

    /// Begin an atomic-integer open/create (Java `atomicInteger`).
    pub fn atomic_integer(&self, name: &str) -> AtomicIntegerMaker<'_, S> {
        AtomicIntegerMaker {
            db: self,
            name: name.to_string(),
            initial: 0,
        }
    }
    pub fn atomic_integer_init(&self, name: &str, initial: i32) -> AtomicIntegerMaker<'_, S> {
        AtomicIntegerMaker {
            db: self,
            name: name.to_string(),
            initial,
        }
    }

    /// Begin an atomic-boolean open/create (Java `atomicBoolean`).
    pub fn atomic_boolean(&self, name: &str) -> AtomicBooleanMaker<'_, S> {
        AtomicBooleanMaker {
            db: self,
            name: name.to_string(),
            initial: false,
        }
    }
    pub fn atomic_boolean_init(&self, name: &str, initial: bool) -> AtomicBooleanMaker<'_, S> {
        AtomicBooleanMaker {
            db: self,
            name: name.to_string(),
            initial,
        }
    }

    /// Begin an atomic-string open/create (Java `atomicString`). Default value null.
    pub fn atomic_string(&self, name: &str) -> AtomicStringMaker<'_, S> {
        AtomicStringMaker {
            db: self,
            name: name.to_string(),
            initial: None,
        }
    }
    pub fn atomic_string_init(&self, name: &str, initial: &str) -> AtomicStringMaker<'_, S> {
        AtomicStringMaker {
            db: self,
            name: name.to_string(),
            initial: Some(initial.to_string()),
        }
    }

    /// Begin an atomic-var open/create (Java `atomicVar`).
    pub fn atomic_var<E, Se>(
        &self,
        name: &str,
        serializer: Se,
        initial: Option<E>,
    ) -> AtomicVarMaker<'_, S, E, Se>
    where
        Se: Serializer<E> + SerDescriptor + Sync,
    {
        AtomicVarMaker {
            db: self,
            name: name.to_string(),
            serializer,
            initial,
        }
    }

    /// Begin a FIFO persistent blocking queue maker (Java `queue`).
    pub fn queue<E, Se>(&self, name: &str, serializer: Se) -> QueueMaker<'_, S, E, Se>
    where
        Se: Serializer<E> + SerDescriptor + Sync,
    {
        QueueMaker {
            db: self,
            name: name.to_string(),
            serializer,
            mode: Mode::Fifo,
            catalog_type: "Queue",
            capacity: i64::MAX as u64,
            _marker: std::marker::PhantomData,
        }
    }
    /// Begin a LIFO persistent blocking stack maker (Java `stack`).
    pub fn stack<E, Se>(&self, name: &str, serializer: Se) -> QueueMaker<'_, S, E, Se>
    where
        Se: Serializer<E> + SerDescriptor + Sync,
    {
        QueueMaker {
            db: self,
            name: name.to_string(),
            serializer,
            mode: Mode::Lifo,
            catalog_type: "Stack",
            capacity: i64::MAX as u64,
            _marker: std::marker::PhantomData,
        }
    }
    /// Begin a circular (overwrite-on-full) blocking queue maker (Java `circularQueue`).
    pub fn circular_queue<E, Se>(
        &self,
        name: &str,
        serializer: Se,
        capacity: u64,
    ) -> QueueMaker<'_, S, E, Se>
    where
        Se: Serializer<E> + SerDescriptor + Sync,
    {
        QueueMaker {
            db: self,
            name: name.to_string(),
            serializer,
            mode: Mode::Circular,
            catalog_type: "CircularQueue",
            capacity,
            _marker: std::marker::PhantomData,
        }
    }

    // ============================ close / lifecycle ============================

    /// Close the DB (Java `close`): idempotent. Wakes blocked queue waiters, closes
    /// the store, then runs delete-after-close cleanup (even if store close errored;
    /// both errors are preserved with the store error primary). NO auto-commit — a
    /// WAL's uncommitted changes are intentionally discarded.
    pub fn close(&self) -> Result<()> {
        // OPEN -> CLOSING. If another thread already won the transition, park-wait
        // until it publishes CLOSED before returning: a returning `close()` must
        // guarantee the store is actually closed, not merely that someone is
        // closing it (MINOR review #11). Then report idempotent success.
        if self
            .state
            .compare_exchange(OPEN, CLOSING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            while self.state.load(Ordering::Acquire) != CLOSED {
                std::thread::yield_now();
            }
            return Ok(());
        }
        // 1. Wake blocked queue waiters, then drop cached handles.
        {
            let mut admin = self.admin.lock();
            for entry in admin.cache.values() {
                if let Some(hook) = &entry.close_hook {
                    hook();
                }
            }
            admin.cache.clear();
        }
        // 2. Close the store (capture error but keep going).
        let store_res = (self.close_store)();
        // 3. Delete-after-close cleanup — runs even if the store close errored.
        let cleanup_res = self.run_cleanup();
        self.state.store(CLOSED, Ordering::Release);
        // Neither error may be dropped: report the store-close error as primary,
        // the cleanup error as secondary, and combine BOTH when both failed.
        combine_close_errors(store_res, cleanup_res)
    }
}

/// Fold the store-close and delete-after-close cleanup results into one, keeping
/// the offending variant when only one failed and preserving both messages when
/// both failed (MINOR review #11 — never silently drop the cleanup error).
fn combine_close_errors(store_res: Result<()>, cleanup_res: Result<()>) -> Result<()> {
    match (store_res, cleanup_res) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(e), Ok(())) => Err(e),
        (Ok(()), Err(e)) => Err(e),
        (Err(se), Err(ce)) => Err(DbError::corrupt_msg(format!(
            "close failed on two counts: store close [{se}]; cleanup [{ce}]"
        ))),
    }
}

// Unbounded impl so `Drop` (which cannot carry `S: Store` bounds) can reuse this.
impl<S> DB<S> {
    fn run_cleanup(&self) -> Result<()> {
        let mut first_err: Option<DbError> = None;
        for p in &self.cleanup_paths {
            if p.exists() {
                if let Err(e) = std::fs::remove_file(p) {
                    if first_err.is_none() {
                        first_err = Some(DbError::Io(e));
                    }
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

impl<S> Drop for DB<S> {
    fn drop(&mut self) {
        // Best-effort idempotent close. NO auto-commit (critical for WAL). The
        // one-shot state guard means an explicit close already ran the teardown,
        // so this cannot double-delete.
        if self
            .state
            .compare_exchange(OPEN, CLOSING, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            {
                let admin = self.admin.lock();
                for entry in admin.cache.values() {
                    if let Some(hook) = &entry.close_hook {
                        hook();
                    }
                }
            }
            let _ = (self.close_store)();
            let _ = self.run_cleanup();
            self.state.store(CLOSED, Ordering::Release);
        }
    }
}

// ============================ name validation ============================

/// Validate a collection name: non-empty and only `[A-Za-z0-9._-]` (Java's
/// `checkName`). `#` is forbidden (it separates a name from a parameter).
pub(crate) fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(DbError::wrong_config("collection name must be non-empty"));
    }
    for c in name.chars() {
        if !(c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-') {
            return Err(DbError::wrong_config(format!(
                "illegal collection name '{name}': only [A-Za-z0-9._-] allowed"
            )));
        }
    }
    Ok(())
}

/// The catalog `#type` values that name a single-record atomic (Java
/// `DB.isAtomicType`). Only these have their record freed by `delete()`.
fn is_atomic_type(t: &str) -> bool {
    matches!(
        t,
        "AtomicLong" | "AtomicInteger" | "AtomicBoolean" | "AtomicString" | "AtomicVar"
    )
}

fn parse_recid(cat: &NameCatalog, key: &str) -> Result<u64> {
    let s = cat
        .get(key)
        .ok_or_else(|| DbError::corrupt_msg(format!("catalog missing {key}")))?;
    s.parse::<u64>()
        .map_err(|_| DbError::corrupt_msg(format!("catalog {key} is not a decimal recid")))
}

fn parse_recid_default0(cat: &NameCatalog, key: &str) -> Result<u64> {
    match cat.get(key) {
        None => Ok(0), // legacy default (absent counterRecid → 0)
        Some(s) => s
            .parse::<u64>()
            .map_err(|_| DbError::corrupt_msg(format!("catalog {key} is not a decimal recid"))),
    }
}

/// The legal range of `maxNodeSize`, shared with `BTreeMap`'s create/open guards
/// so the create and reopen bounds can never drift.
use crate::btree::map::{MAX_MAX_NODE_SIZE, MIN_MAX_NODE_SIZE};

/// Fully validate a decoded name catalog beyond MDBC syntax
/// group by the first `#`, require a legal name and a known
/// `#type`, require the EXACT required/allowed field set per type (reject unknown
/// fields), validate codec descriptors, parse recids (`≥ 1`; `counterRecid == 0`
/// allowed as disabled), booleans, and the `maxNodeSize` range. Called at open
/// and after a rollback reload — before any collection is built over the catalog.
pub(crate) fn validate_catalog(cat: &NameCatalog) -> Result<()> {
    use std::collections::BTreeMap as Map;
    // Group `name#param -> value` by name.
    let mut groups: Map<&str, Map<&str, &str>> = Map::new();
    for (k, v) in cat.iter() {
        let hash = k
            .find('#')
            .ok_or_else(|| DbError::corrupt_msg(format!("catalog key '{k}' has no '#'")))?;
        let (name, param) = (&k[..hash], &k[hash + 1..]);
        if param.contains('#') {
            return Err(DbError::corrupt_msg(format!(
                "catalog key '{k}' has more than one '#'"
            )));
        }
        groups.entry(name).or_default().insert(param, v.as_str());
    }
    for (name, fields) in &groups {
        validate_name(name).map_err(|_| {
            DbError::corrupt_msg(format!("catalog has illegal object name '{name}'"))
        })?;
        let ty = fields
            .get("type")
            .ok_or_else(|| DbError::corrupt_msg(format!("catalog object '{name}' has no #type")))?;
        // required (present + exact), optional (may be absent), and a descriptor
        // validator per field.
        let check = |required: &[&str], optional: &[&str], descr: &[(&str, bool)]| -> Result<()> {
            for r in required {
                if !fields.contains_key(*r) {
                    return Err(DbError::corrupt_msg(format!(
                        "catalog '{name}' ({ty}) missing required field '{r}'"
                    )));
                }
            }
            let allowed: std::collections::HashSet<&str> = ["type"]
                .iter()
                .chain(required.iter())
                .chain(optional.iter())
                .copied()
                .collect();
            for f in fields.keys() {
                if !allowed.contains(*f) {
                    return Err(DbError::corrupt_msg(format!(
                        "catalog '{name}' ({ty}) has unknown field '{f}'"
                    )));
                }
            }
            // recid fields ending in "Recid" (except counterRecid) must be ≥ 1.
            for (field, is_group_descriptor) in descr {
                if let Some(val) = fields.get(*field) {
                    let ok = if *is_group_descriptor {
                        crate::db::descriptor::is_valid_group_descriptor(val)
                    } else {
                        crate::db::descriptor::is_valid_ser_descriptor(val)
                    };
                    if !ok {
                        return Err(DbError::corrupt_msg(format!(
                            "catalog '{name}' field '{field}' has an invalid codec descriptor '{val}'"
                        )));
                    }
                }
            }
            Ok(())
        };
        let recid_ge1 = |field: &str| -> Result<()> {
            if let Some(v) = fields.get(field) {
                let r: u64 = v.parse().map_err(|_| {
                    DbError::corrupt_msg(format!("catalog '{name}' {field} is not a decimal recid"))
                })?;
                if r < 1 {
                    return Err(DbError::corrupt_msg(format!(
                        "catalog '{name}' {field} must be ≥ 1"
                    )));
                }
            }
            Ok(())
        };
        let counter_ok = |field: &str| -> Result<()> {
            if let Some(v) = fields.get(field) {
                v.parse::<u64>().map_err(|_| {
                    DbError::corrupt_msg(format!("catalog '{name}' {field} is not a decimal recid"))
                })?; // 0 allowed (disabled)
            }
            Ok(())
        };
        let max_node = || -> Result<()> {
            if let Some(v) = fields.get("maxNodeSize") {
                let m: usize = v.parse().map_err(|_| {
                    DbError::corrupt_msg(format!("catalog '{name}' maxNodeSize is not an integer"))
                })?;
                if !(MIN_MAX_NODE_SIZE..=MAX_MAX_NODE_SIZE).contains(&m) {
                    return Err(DbError::corrupt_msg(format!(
                        "catalog '{name}' maxNodeSize {m} out of range [{MIN_MAX_NODE_SIZE}, {MAX_MAX_NODE_SIZE}]"
                    )));
                }
            }
            Ok(())
        };
        let boolean = |field: &str| -> Result<()> {
            if let Some(v) = fields.get(field) {
                if *v != "true" && *v != "false" {
                    return Err(DbError::corrupt_msg(format!(
                        "catalog '{name}' {field} is not a boolean"
                    )));
                }
            }
            Ok(())
        };
        match *ty {
            "TreeMap" => {
                check(
                    &[
                        "keySerializer",
                        "valueSerializer",
                        "rootRecidRecid",
                        "maxNodeSize",
                    ],
                    &["counterRecid", "valueInline"],
                    &[("keySerializer", true), ("valueSerializer", true)],
                )?;
                recid_ge1("rootRecidRecid")?;
                counter_ok("counterRecid")?;
                max_node()?;
                boolean("valueInline")?;
            }
            "TreeSet" => {
                check(
                    &["serializer", "rootRecidRecid", "maxNodeSize"],
                    &["counterRecid"],
                    &[("serializer", true)],
                )?;
                recid_ge1("rootRecidRecid")?;
                counter_ok("counterRecid")?;
                max_node()?;
            }
            "AtomicLong" | "AtomicInteger" | "AtomicBoolean" | "AtomicString" => {
                check(&["recid"], &[], &[])?;
                recid_ge1("recid")?;
            }
            "AtomicVar" => {
                check(&["recid", "serializer"], &[], &[("serializer", false)])?;
                recid_ge1("recid")?;
            }
            "Queue" | "Stack" | "CircularQueue" => {
                check(
                    &["headerRecid", "serializer"],
                    &[],
                    &[("serializer", false)],
                )?;
                recid_ge1("headerRecid")?;
            }
            other => {
                return Err(DbError::corrupt_msg(format!(
                    "catalog object '{name}' has unknown #type '{other}'"
                )))
            }
        }
    }
    Ok(())
}

// ============================ tree map maker ============================

/// Builder for a typed tree map (Java `DB.TreeMapMaker`).
pub struct TreeMapMaker<'db, S, KF: GroupFormat, VF: GroupFormat> {
    db: &'db DB<S>,
    name: String,
    key_format: KF,
    value_format: VF,
    max_node_size: usize,
    counter_enable: bool,
    values_outside: bool,
    listeners: Vec<Arc<dyn MapModificationListener<K<KF>, V<VF>>>>,
}

impl<'db, S, KF, VF> TreeMapMaker<'db, S, KF, VF>
where
    S: Store + StoreLease + DbRollback + Send + Sync + 'static,
    KF: GroupFormat + GroupDescriptor + Clone + Send + Sync + 'static,
    VF: GroupFormat + GroupDescriptor + Clone + Send + Sync + 'static,
{
    pub fn max_node_size(mut self, n: usize) -> Self {
        self.max_node_size = n;
        self
    }
    pub fn counter_enable(mut self) -> Self {
        self.counter_enable = true;
        self
    }
    pub fn values_outside_nodes_enable(mut self) -> Self {
        self.values_outside = true;
        self
    }
    /// Register a deferred modification listener (applied on every open of the
    /// cached handle; duplicate registration of the same `Arc` is ignored).
    pub fn modification_listener(
        mut self,
        listener: Arc<dyn MapModificationListener<K<KF>, V<VF>>>,
    ) -> Self {
        self.listeners.push(listener);
        self
    }

    fn apply_listeners(&self, map: &BTreeMap<S, KF, VF>) {
        for l in &self.listeners {
            map.modification_listener_add(l.clone());
        }
    }

    fn cache_and_return(&self, admin: &mut Admin, map: BTreeMap<S, KF, VF>) -> BTreeMap<S, KF, VF> {
        // Teardown for `delete()`: clear entry/node records (root + counter LEAK,
        // per Java — never freed while a clone may still use them, C1).
        let td_map = map.clone();
        DB::<S>::cache_insert_full(
            admin,
            &self.name,
            map.clone(),
            None,
            Some(Box::new(move || td_map.clear())),
        );
        self.apply_listeners(&map);
        map
    }

    pub fn create(self) -> Result<BTreeMap<S, KF, VF>> {
        validate_name(&self.name)?;
        let mut admin = self.db.lock_admin()?;
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            return Err(DbError::wrong_config(format!(
                "name already exists: {}",
                self.name
            )));
        }
        let map = self.build_map()?;
        self.db
            .publish_catalog(&mut admin, |c| self.write_catalog(c, &map))?;
        Ok(self.cache_and_return(&mut admin, map))
    }

    pub fn open(self) -> Result<BTreeMap<S, KF, VF>> {
        let mut admin = self.db.lock_admin()?;
        self.verify_catalog(&admin.catalog)?;
        if let Some(res) = DB::<S>::cache_lookup::<BTreeMap<S, KF, VF>>(&admin, &self.name) {
            let map = res?;
            self.apply_listeners(&map);
            return Ok(map);
        }
        let map = self.open_from_catalog(&admin.catalog)?;
        Ok(self.cache_and_return(&mut admin, map))
    }

    pub fn create_or_open(self) -> Result<BTreeMap<S, KF, VF>> {
        let mut admin = self.db.lock_admin()?;
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            self.verify_catalog(&admin.catalog)?;
            if let Some(res) = DB::<S>::cache_lookup::<BTreeMap<S, KF, VF>>(&admin, &self.name) {
                let map = res?;
                self.apply_listeners(&map);
                return Ok(map);
            }
            let map = self.open_from_catalog(&admin.catalog)?;
            return Ok(self.cache_and_return(&mut admin, map));
        }
        validate_name(&self.name)?;
        let map = self.build_map()?;
        self.db
            .publish_catalog(&mut admin, |c| self.write_catalog(c, &map))?;
        Ok(self.cache_and_return(&mut admin, map))
    }

    /// Bulk build from a strictly ascending iterator (Java `createFrom`).
    pub fn create_from<I>(self, entries: I) -> Result<BTreeMap<S, KF, VF>>
    where
        I: IntoIterator<Item = (K<KF>, V<VF>)>,
    {
        validate_name(&self.name)?;
        if self.values_outside {
            return Err(DbError::Unsupported(
                "createFrom rejects external-value maps (Java behavior)",
            ));
        }
        let mut admin = self.db.lock_admin()?;
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            return Err(DbError::wrong_config(format!(
                "name already exists: {}",
                self.name
            )));
        }
        // The pump builder can fail NotSorted; that must not register the name.
        let map = BTreeMap::create_from_sorted_counter(
            self.db.store.clone(),
            clone_format(&self.key_format),
            clone_format(&self.value_format),
            self.max_node_size,
            entries,
            self.counter_enable,
        )?;
        self.db
            .publish_catalog(&mut admin, |c| self.write_catalog(c, &map))?;
        Ok(self.cache_and_return(&mut admin, map))
    }

    fn build_map(&self) -> Result<BTreeMap<S, KF, VF>> {
        let store = self.db.store.clone();
        let kf = clone_format(&self.key_format);
        let vf = clone_format(&self.value_format);
        if self.values_outside {
            BTreeMap::create_external_values(store, kf, vf, self.max_node_size, self.counter_enable)
        } else {
            BTreeMap::create_with_counter(store, kf, vf, self.max_node_size, self.counter_enable)
        }
    }

    fn write_catalog(&self, catalog: &mut NameCatalog, map: &BTreeMap<S, KF, VF>) {
        let n = &self.name;
        catalog.insert(format!("{n}#type"), "TreeMap".into());
        catalog.insert(
            format!("{n}#keySerializer"),
            group_descriptor_or_custom(&self.key_format),
        );
        catalog.insert(
            format!("{n}#valueSerializer"),
            group_descriptor_or_custom(&self.value_format),
        );
        catalog.insert(
            format!("{n}#rootRecidRecid"),
            map.root_recid_recid().to_string(),
        );
        catalog.insert(format!("{n}#maxNodeSize"), self.max_node_size.to_string());
        catalog.insert(format!("{n}#counterRecid"), map.counter_recid().to_string());
        catalog.insert(format!("{n}#valueInline"), map.value_inline().to_string());
    }

    fn verify_catalog(&self, catalog: &NameCatalog) -> Result<()> {
        let n = &self.name;
        match catalog.get(&format!("{n}#type")) {
            None => return Err(DbError::wrong_config(format!("no such name: {n}"))),
            Some(t) if t != "TreeMap" => {
                return Err(DbError::wrong_config(format!(
                    "name {n} is a {t}, not a TreeMap"
                )))
            }
            Some(_) => {}
        }
        let kd = catalog
            .get(&format!("{n}#keySerializer"))
            .ok_or_else(|| DbError::corrupt("catalog missing keySerializer"))?;
        verify_group(kd, &self.key_format)?;
        let vd = catalog
            .get(&format!("{n}#valueSerializer"))
            .ok_or_else(|| DbError::corrupt("catalog missing valueSerializer"))?;
        verify_group(vd, &self.value_format)?;
        Ok(())
    }

    fn open_from_catalog(&self, catalog: &NameCatalog) -> Result<BTreeMap<S, KF, VF>> {
        let n = &self.name;
        let root = parse_recid(catalog, &format!("{n}#rootRecidRecid"))?;
        let mns = catalog
            .get(&format!("{n}#maxNodeSize"))
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| DbError::corrupt("catalog maxNodeSize invalid"))?;
        let counter = parse_recid_default0(catalog, &format!("{n}#counterRecid"))?;
        // absent valueInline → true (legacy default)
        let value_inline = match catalog.get(&format!("{n}#valueInline")) {
            None => true,
            Some(s) if s == "true" => true,
            Some(s) if s == "false" => false,
            Some(_) => return Err(DbError::corrupt("catalog valueInline is not a boolean")),
        };
        let store = self.db.store.clone();
        let kf = clone_format(&self.key_format);
        let vf = clone_format(&self.value_format);
        if value_inline {
            BTreeMap::open_with_counter(store, root, kf, vf, mns, counter)
        } else {
            BTreeMap::open_external_values(store, root, kf, vf, mns, counter)
        }
    }
}

// ============================ tree set maker ============================

/// Builder for a typed tree set (Java `DB.TreeSetMaker`).
pub struct TreeSetMaker<'db, S, KF> {
    db: &'db DB<S>,
    name: String,
    key_format: KF,
    max_node_size: usize,
    counter_enable: bool,
}

impl<'db, S, KF> TreeSetMaker<'db, S, KF>
where
    S: Store + StoreLease + DbRollback + Send + Sync + 'static,
    KF: GroupFormat + GroupDescriptor + Clone + Send + Sync + 'static,
{
    /// Cache a set handle with a `delete()` teardown that clears its element/node
    /// records (the structural root + counter records LEAK, per Java — never freed
    /// while a clone may still use them, C1).
    fn cache_set(admin: &mut Admin, name: &str, set: &NavigableSet<S, KF>) {
        let td = set.clone();
        DB::<S>::cache_insert_full(
            admin,
            name,
            set.clone(),
            None,
            Some(Box::new(move || td.clear())),
        );
    }

    pub fn max_node_size(mut self, n: usize) -> Self {
        self.max_node_size = n;
        self
    }
    pub fn counter_enable(mut self) -> Self {
        self.counter_enable = true;
        self
    }

    pub fn create(self) -> Result<NavigableSet<S, KF>> {
        validate_name(&self.name)?;
        let mut admin = self.db.lock_admin()?;
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            return Err(DbError::wrong_config(format!(
                "name already exists: {}",
                self.name
            )));
        }
        let map = BTreeMap::create_with_counter(
            self.db.store.clone(),
            clone_format(&self.key_format),
            NoValueFormat,
            self.max_node_size,
            self.counter_enable,
        )?;
        self.db
            .publish_catalog(&mut admin, |c| self.write_catalog(c, &map))?;
        let set = NavigableSet::from_map(map);
        Self::cache_set(&mut admin, &self.name, &set);
        Ok(set)
    }

    pub fn open(self) -> Result<NavigableSet<S, KF>> {
        let mut admin = self.db.lock_admin()?;
        self.open_locked(&mut admin)
    }

    /// Open using an already-held admin guard (no TOCTOU release/re-lock).
    fn open_locked(&self, admin: &mut Admin) -> Result<NavigableSet<S, KF>> {
        self.verify_catalog(&admin.catalog)?;
        if let Some(res) = DB::<S>::cache_lookup::<NavigableSet<S, KF>>(admin, &self.name) {
            return res;
        }
        let set = self.open_from_catalog(&admin.catalog)?;
        Self::cache_set(admin, &self.name, &set);
        Ok(set)
    }

    pub fn create_or_open(self) -> Result<NavigableSet<S, KF>> {
        let mut admin = self.db.lock_admin()?;
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            return self.open_locked(&mut admin);
        }
        validate_name(&self.name)?;
        let map = BTreeMap::create_with_counter(
            self.db.store.clone(),
            clone_format(&self.key_format),
            NoValueFormat,
            self.max_node_size,
            self.counter_enable,
        )?;
        self.db
            .publish_catalog(&mut admin, |c| self.write_catalog(c, &map))?;
        let set = NavigableSet::from_map(map);
        Self::cache_set(&mut admin, &self.name, &set);
        Ok(set)
    }

    pub fn create_from<I>(self, elements: I) -> Result<NavigableSet<S, KF>>
    where
        I: IntoIterator<Item = K<KF>>,
    {
        validate_name(&self.name)?;
        let mut admin = self.db.lock_admin()?;
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            return Err(DbError::wrong_config(format!(
                "name already exists: {}",
                self.name
            )));
        }
        let entries = elements.into_iter().map(|k| (k, ()));
        let map = BTreeMap::create_from_sorted_counter(
            self.db.store.clone(),
            clone_format(&self.key_format),
            NoValueFormat,
            self.max_node_size,
            entries,
            self.counter_enable,
        )?;
        self.db
            .publish_catalog(&mut admin, |c| self.write_catalog(c, &map))?;
        let set = NavigableSet::from_map(map);
        Self::cache_set(&mut admin, &self.name, &set);
        Ok(set)
    }

    fn write_catalog(&self, catalog: &mut NameCatalog, map: &BTreeMap<S, KF, NoValueFormat>) {
        let n = &self.name;
        catalog.insert(format!("{n}#type"), "TreeSet".into());
        catalog.insert(
            format!("{n}#serializer"),
            group_descriptor_or_custom(&self.key_format),
        );
        catalog.insert(
            format!("{n}#rootRecidRecid"),
            map.root_recid_recid().to_string(),
        );
        catalog.insert(format!("{n}#maxNodeSize"), self.max_node_size.to_string());
        catalog.insert(format!("{n}#counterRecid"), map.counter_recid().to_string());
    }

    fn verify_catalog(&self, catalog: &NameCatalog) -> Result<()> {
        let n = &self.name;
        match catalog.get(&format!("{n}#type")) {
            None => return Err(DbError::wrong_config(format!("no such name: {n}"))),
            Some(t) if t != "TreeSet" => {
                return Err(DbError::wrong_config(format!(
                    "name {n} is a {t}, not a TreeSet"
                )))
            }
            Some(_) => {}
        }
        let sd = catalog
            .get(&format!("{n}#serializer"))
            .ok_or_else(|| DbError::corrupt("catalog missing serializer"))?;
        verify_group(sd, &self.key_format)
    }

    fn open_from_catalog(&self, catalog: &NameCatalog) -> Result<NavigableSet<S, KF>> {
        let n = &self.name;
        let root = parse_recid(catalog, &format!("{n}#rootRecidRecid"))?;
        let mns = catalog
            .get(&format!("{n}#maxNodeSize"))
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| DbError::corrupt("catalog maxNodeSize invalid"))?;
        let counter = parse_recid_default0(catalog, &format!("{n}#counterRecid"))?;
        let map = BTreeMap::open_with_counter(
            self.db.store.clone(),
            root,
            clone_format(&self.key_format),
            NoValueFormat,
            mns,
            counter,
        )?;
        Ok(NavigableSet::from_map(map))
    }
}

// ============================ atomic makers ============================

macro_rules! numeric_atomic_maker {
    ($maker:ident, $atomic:ident, $prim:ty, $ser:expr, $type_str:literal) => {
        pub struct $maker<'db, S> {
            db: &'db DB<S>,
            name: String,
            initial: $prim,
        }
        impl<'db, S> $maker<'db, S>
        where
            S: Store + StoreLease + DbRollback + Send + Sync + 'static,
        {
            pub fn create(self) -> Result<$atomic<S>> {
                validate_name(&self.name)?;
                let mut admin = self.db.lock_admin()?;
                self.create_locked(&mut admin)
            }
            fn create_locked(&self, admin: &mut Admin) -> Result<$atomic<S>> {
                if admin.catalog.contains_key(&format!("{}#type", self.name)) {
                    return Err(DbError::wrong_config(format!(
                        "name already exists: {}",
                        self.name
                    )));
                }
                let recid = self.db.store.put(&self.initial, &$ser)?;
                let n = self.name.clone();
                self.db.publish_catalog(admin, |c| {
                    c.insert(format!("{n}#type"), $type_str.into());
                    c.insert(format!("{n}#recid"), recid.get().to_string());
                })?;
                let atomic = $atomic::new(self.db.store.clone(), recid);
                DB::<S>::cache_insert(admin, &self.name, atomic.clone(), None);
                Ok(atomic)
            }
            pub fn open(self) -> Result<$atomic<S>> {
                let mut admin = self.db.lock_admin()?;
                self.open_locked(&mut admin)
            }
            fn open_locked(&self, admin: &mut Admin) -> Result<$atomic<S>> {
                self.verify_type(&admin.catalog)?;
                if let Some(res) = DB::<S>::cache_lookup::<$atomic<S>>(admin, &self.name) {
                    return res;
                }
                let recid = nz(parse_recid(
                    &admin.catalog,
                    &format!("{}#recid", self.name),
                )?)?;
                let atomic = $atomic::new(self.db.store.clone(), recid);
                DB::<S>::cache_insert(admin, &self.name, atomic.clone(), None);
                Ok(atomic)
            }
            pub fn create_or_open(self) -> Result<$atomic<S>> {
                validate_name(&self.name)?;
                let mut admin = self.db.lock_admin()?;
                if admin.catalog.contains_key(&format!("{}#type", self.name)) {
                    self.open_locked(&mut admin)
                } else {
                    self.create_locked(&mut admin)
                }
            }
            fn verify_type(&self, catalog: &NameCatalog) -> Result<()> {
                match catalog.get(&format!("{}#type", self.name)) {
                    None => Err(DbError::wrong_config(format!(
                        "no such name: {}",
                        self.name
                    ))),
                    Some(t) if t != $type_str => Err(DbError::wrong_config(format!(
                        "name {} is a {t}, not a {}",
                        self.name, $type_str
                    ))),
                    Some(_) => Ok(()),
                }
            }
        }
    };
}

numeric_atomic_maker!(AtomicLongMaker, AtomicLong, i64, LONG, "AtomicLong");
numeric_atomic_maker!(AtomicIntegerMaker, AtomicInteger, i32, INT, "AtomicInteger");
numeric_atomic_maker!(
    AtomicBooleanMaker,
    AtomicBoolean,
    bool,
    BOOLEAN,
    "AtomicBoolean"
);

/// Atomic-string maker (nullable; a null default still writes a present record
/// whose content is the `0x00` null marker, never a store-level null record).
pub struct AtomicStringMaker<'db, S> {
    db: &'db DB<S>,
    name: String,
    initial: Option<String>,
}

impl<'db, S> AtomicStringMaker<'db, S>
where
    S: Store + StoreLease + DbRollback + Send + Sync + 'static,
{
    pub fn create(self) -> Result<AtomicString<S>> {
        validate_name(&self.name)?;
        let mut admin = self.db.lock_admin()?;
        self.create_locked(&mut admin)
    }
    fn create_locked(&self, admin: &mut Admin) -> Result<AtomicString<S>> {
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            return Err(DbError::wrong_config(format!(
                "name already exists: {}",
                self.name
            )));
        }
        // Always write a (present) record whose content encodes null-ness
        // (Java STRING_NULLABLE), never a store-level null record.
        let recid = self
            .db
            .store
            .put(&self.initial, &crate::db::atomic::STRING_NULLABLE)?;
        let n = self.name.clone();
        self.db.publish_catalog(admin, |c| {
            c.insert(format!("{n}#type"), "AtomicString".into());
            c.insert(format!("{n}#recid"), recid.get().to_string());
        })?;
        let atomic = AtomicString::new(self.db.store.clone(), recid);
        DB::<S>::cache_insert(admin, &self.name, atomic.clone(), None);
        Ok(atomic)
    }
    pub fn open(self) -> Result<AtomicString<S>> {
        let mut admin = self.db.lock_admin()?;
        self.open_locked(&mut admin)
    }
    fn open_locked(&self, admin: &mut Admin) -> Result<AtomicString<S>> {
        match admin.catalog.get(&format!("{}#type", self.name)) {
            None => {
                return Err(DbError::wrong_config(format!(
                    "no such name: {}",
                    self.name
                )))
            }
            Some(t) if t != "AtomicString" => {
                return Err(DbError::wrong_config(format!(
                    "name {} is a {t}, not an AtomicString",
                    self.name
                )))
            }
            Some(_) => {}
        }
        if let Some(res) = DB::<S>::cache_lookup::<AtomicString<S>>(admin, &self.name) {
            return res;
        }
        let recid = nz(parse_recid(
            &admin.catalog,
            &format!("{}#recid", self.name),
        )?)?;
        let atomic = AtomicString::new(self.db.store.clone(), recid);
        DB::<S>::cache_insert(admin, &self.name, atomic.clone(), None);
        Ok(atomic)
    }
    pub fn create_or_open(self) -> Result<AtomicString<S>> {
        validate_name(&self.name)?;
        let mut admin = self.db.lock_admin()?;
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            self.open_locked(&mut admin)
        } else {
            self.create_locked(&mut admin)
        }
    }
}

/// Atomic-var maker (nullable, arbitrary element serializer).
pub struct AtomicVarMaker<'db, S, E, Se: Serializer<E> + Sync> {
    db: &'db DB<S>,
    name: String,
    serializer: Se,
    initial: Option<E>,
}

impl<'db, S, E, Se> AtomicVarMaker<'db, S, E, Se>
where
    S: Store + StoreLease + DbRollback + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
    Se: Serializer<E> + SerDescriptor + Sync + Send + 'static,
{
    pub fn create(self) -> Result<AtomicVar<S, E, Se>> {
        validate_name(&self.name)?;
        let mut admin = self.db.lock_admin()?;
        self.create_with(&mut admin)
    }
    fn create_with(self, admin: &mut Admin) -> Result<AtomicVar<S, E, Se>> {
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            return Err(DbError::wrong_config(format!(
                "name already exists: {}",
                self.name
            )));
        }
        let recid = match &self.initial {
            None => self.db.store.preallocate()?,
            Some(v) => self.db.store.put(v, &self.serializer)?,
        };
        let n = self.name.clone();
        let ser_desc = ser_descriptor_or_custom(&self.serializer);
        self.db.publish_catalog(admin, |c| {
            c.insert(format!("{n}#type"), "AtomicVar".into());
            c.insert(format!("{n}#recid"), recid.get().to_string());
            c.insert(format!("{n}#serializer"), ser_desc);
        })?;
        let atomic = AtomicVar::new(self.db.store.clone(), recid, Arc::new(self.serializer));
        DB::<S>::cache_insert(admin, &n, atomic.clone(), None);
        Ok(atomic)
    }
    pub fn open(self) -> Result<AtomicVar<S, E, Se>> {
        let mut admin = self.db.lock_admin()?;
        self.open_with(&mut admin)
    }
    fn open_with(self, admin: &mut Admin) -> Result<AtomicVar<S, E, Se>> {
        let n = self.name.clone();
        match admin.catalog.get(&format!("{n}#type")) {
            None => return Err(DbError::wrong_config(format!("no such name: {n}"))),
            Some(t) if t != "AtomicVar" => {
                return Err(DbError::wrong_config(format!(
                    "name {n} is a {t}, not an AtomicVar"
                )))
            }
            Some(_) => {}
        }
        let sd = admin
            .catalog
            .get(&format!("{n}#serializer"))
            .ok_or_else(|| DbError::corrupt("catalog missing AtomicVar serializer"))?;
        verify_ser(sd, &self.serializer)?;
        if let Some(res) = DB::<S>::cache_lookup::<AtomicVar<S, E, Se>>(admin, &n) {
            return res;
        }
        let recid = nz(parse_recid(&admin.catalog, &format!("{n}#recid"))?)?;
        let atomic = AtomicVar::new(self.db.store.clone(), recid, Arc::new(self.serializer));
        DB::<S>::cache_insert(admin, &n, atomic.clone(), None);
        Ok(atomic)
    }
    pub fn create_or_open(self) -> Result<AtomicVar<S, E, Se>> {
        validate_name(&self.name)?;
        let mut admin = self.db.lock_admin()?;
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            self.open_with(&mut admin)
        } else {
            self.create_with(&mut admin)
        }
    }
}

// ============================ queue maker ============================

/// Builder for a persistent blocking queue / stack / circular queue.
pub struct QueueMaker<'db, S, E, Se: Serializer<E> + Sync> {
    db: &'db DB<S>,
    name: String,
    serializer: Se,
    mode: Mode,
    catalog_type: &'static str,
    capacity: u64,
    _marker: std::marker::PhantomData<fn() -> E>,
}

impl<'db, S, E, Se> QueueMaker<'db, S, E, Se>
where
    S: Store + StoreLease + DbRollback + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
    Se: Serializer<E> + SerDescriptor + Sync + Send + 'static,
{
    pub fn create(self) -> Result<Arc<PersistentBlockingQueue<S, E, Se>>> {
        validate_name(&self.name)?;
        let mut admin = self.db.lock_admin()?;
        self.create_with(&mut admin)
    }
    fn create_with(self, admin: &mut Admin) -> Result<Arc<PersistentBlockingQueue<S, E, Se>>> {
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            return Err(DbError::wrong_config(format!(
                "name already exists: {}",
                self.name
            )));
        }
        let ser_desc = ser_descriptor_or_custom(&self.serializer);
        let queue = PersistentBlockingQueue::create(
            self.db.store.clone(),
            self.serializer,
            self.mode,
            self.capacity,
        )?;
        let header = queue.header_recid();
        let n = self.name.clone();
        let ty = self.catalog_type;
        self.db.publish_catalog(admin, |c| {
            c.insert(format!("{n}#type"), ty.into());
            c.insert(format!("{n}#headerRecid"), header.get().to_string());
            c.insert(format!("{n}#serializer"), ser_desc);
        })?;
        Ok(Self::cache_queue(admin, &self.name, queue))
    }

    pub fn open(self) -> Result<Arc<PersistentBlockingQueue<S, E, Se>>> {
        let mut admin = self.db.lock_admin()?;
        self.open_with(&mut admin)
    }
    fn open_with(self, admin: &mut Admin) -> Result<Arc<PersistentBlockingQueue<S, E, Se>>> {
        let n = self.name.clone();
        match admin.catalog.get(&format!("{n}#type")) {
            None => return Err(DbError::wrong_config(format!("no such name: {n}"))),
            Some(t) if t != self.catalog_type => {
                return Err(DbError::wrong_config(format!(
                    "name {n} is a {t}, not a {}",
                    self.catalog_type
                )))
            }
            Some(_) => {}
        }
        let sd = admin
            .catalog
            .get(&format!("{n}#serializer"))
            .ok_or_else(|| DbError::corrupt("catalog missing queue serializer"))?;
        verify_ser(sd, &self.serializer)?;
        if let Some(res) =
            DB::<S>::cache_lookup::<Arc<PersistentBlockingQueue<S, E, Se>>>(admin, &n)
        {
            // Recheck the retained handle's mode too (defense in depth, R6).
            let q = res?;
            if q.mode()? != self.mode {
                return Err(DbError::corrupt_msg(format!(
                    "queue '{n}' header mode does not match its catalog #type {}",
                    self.catalog_type
                )));
            }
            return Ok(q);
        }
        let header = nz(parse_recid(&admin.catalog, &format!("{n}#headerRecid"))?)?;
        let queue = PersistentBlockingQueue::open(self.db.store.clone(), header, self.serializer)?;
        // The header's stored mode must match the requested/catalog mode; a
        // mismatch is a corrupt catalog<->header pairing (Java `QueueMaker.open2`
        // compares `queue.mode()`), R6.
        if queue.mode()? != self.mode {
            return Err(DbError::corrupt_msg(format!(
                "queue '{n}' header mode does not match its catalog #type {}",
                self.catalog_type
            )));
        }
        Ok(Self::cache_queue(admin, &n, queue))
    }

    pub fn create_or_open(self) -> Result<Arc<PersistentBlockingQueue<S, E, Se>>> {
        validate_name(&self.name)?;
        let mut admin = self.db.lock_admin()?;
        if admin.catalog.contains_key(&format!("{}#type", self.name)) {
            self.open_with(&mut admin)
        } else {
            self.create_with(&mut admin)
        }
    }

    fn cache_queue(
        admin: &mut Admin,
        name: &str,
        queue: PersistentBlockingQueue<S, E, Se>,
    ) -> Arc<PersistentBlockingQueue<S, E, Se>> {
        let arc = Arc::new(queue);
        let hook_arc = Arc::clone(&arc);
        let close_hook: Box<dyn Fn() + Send + Sync> = Box::new(move || hook_arc.close_handle());
        // `delete()` teardown: free the queue's node records AND its header record
        // (safe — all clones share one globally-closable handle, C1).
        let td_arc = Arc::clone(&arc);
        let teardown: Box<dyn FnOnce() -> Result<()> + Send + Sync> =
            Box::new(move || td_arc.purge_records());
        admin.cache.insert(
            name.to_string(),
            CacheEntry {
                handle: Box::new(Arc::clone(&arc)),
                close_hook: Some(close_hook),
                teardown: Some(teardown),
            },
        );
        arc
    }
}

// ============================ helpers ============================

/// GroupFormats are stateless value types in the port; the makers hold one and
/// need a fresh copy per constructed collection. Built-in formats are `Copy`- or
/// `Clone`-free zero-sized types, but the generic maker cannot assume `Clone`, so
/// we require callers to hand ownership and re-derive via a transmute-free copy
/// through `Default` where possible. Instead we simply require the maker to own
/// the format and clone it via the `Clone` bound added at call sites.
fn clone_format<F: Clone>(f: &F) -> F {
    f.clone()
}
