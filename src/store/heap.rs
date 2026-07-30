//! `StoreOnHeap` — records are live objects, never serialized (spec 02 §4).
//! `read` dispatches `on_object`/`on_null`, never `on_bytes`. Sharded
//! `RwLock<HashMap>` for records (concurrent, not lock-free — accepted
//! difference for a test-oriented store), a global mutex for allocation.

use super::locks::{assert_not_in_action, ActionGuard};
use super::{LeaseTable, Recid, Record, RecordRead, Store};
use crate::error::{DbError, Result};
use crate::ser::Serializer;
use parking_lot::{Mutex, RwLock};
use std::any::Any;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const SHARDS: usize = 64;

enum HeapRec {
    Null,
    Prealloc,
    Live(Arc<dyn Any + Send + Sync>),
}

struct Alloc {
    max_recid: u64,
    free: Vec<u64>,
}

pub struct StoreOnHeap {
    shards: Box<[RwLock<HashMap<u64, HeapRec>>]>,
    alloc: Mutex<Alloc>,
    thread_safe: bool,
    closed: AtomicBool,
    #[allow(dead_code)] // read via StoreLease, used by the collection layer
    lease_table: Arc<LeaseTable>,
}

impl Default for StoreOnHeap {
    fn default() -> Self {
        Self::new(true)
    }
}

impl StoreOnHeap {
    pub fn new(thread_safe: bool) -> Self {
        let mut shards = Vec::with_capacity(SHARDS);
        for _ in 0..SHARDS {
            shards.push(RwLock::new(HashMap::new()));
        }
        StoreOnHeap {
            shards: shards.into_boxed_slice(),
            alloc: Mutex::new(Alloc {
                max_recid: 0,
                free: Vec::new(),
            }),
            thread_safe,
            closed: AtomicBool::new(false),
            lease_table: LeaseTable::new(),
        }
    }

    #[inline]
    fn shard(&self, recid: u64) -> &RwLock<HashMap<u64, HeapRec>> {
        &self.shards[(recid as usize) & (SHARDS - 1)]
    }

    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(DbError::StoreClosed);
        }
        Ok(())
    }

    fn alloc_recid(&self) -> u64 {
        let mut a = self.alloc.lock();
        if let Some(r) = a.free.pop() {
            r
        } else {
            a.max_recid += 1;
            a.max_recid
        }
    }
}

impl super::StoreLease for StoreOnHeap {
    fn lease_table(&self) -> &Arc<LeaseTable> {
        &self.lease_table
    }
}

// The heap store dispatches the object dialect; the default `read_torn_safe`
// body delegates to the locked `read` (D4). Objects are never torn.
impl super::StoreTornRead for StoreOnHeap {}

fn nz(recid: u64) -> Recid {
    NonZeroU64::new(recid).expect("recid 0 is never allocated")
}

impl Store for StoreOnHeap {
    fn preallocate(&self) -> Result<Recid> {
        assert_not_in_action("preallocate");
        self.check_closed()?;
        let recid = self.alloc_recid();
        self.shard(recid).write().insert(recid, HeapRec::Prealloc);
        Ok(nz(recid))
    }

    fn put<R: Record>(&self, value: &R, _ser: &(impl Serializer<R> + Sync)) -> Result<Recid> {
        assert_not_in_action("put");
        self.check_closed()?;
        let recid = self.alloc_recid();
        let obj: Arc<dyn Any + Send + Sync> = Arc::new(value.clone());
        self.shard(recid).write().insert(recid, HeapRec::Live(obj));
        Ok(nz(recid))
    }

    fn get<R: Record>(
        &self,
        recid: Recid,
        _ser: &(impl Serializer<R> + Sync),
    ) -> Result<Option<R>> {
        assert_not_in_action("get");
        self.check_closed()?;
        let guard = self.shard(recid.get()).read();
        let _a = ActionGuard::enter();
        match guard.get(&recid.get()) {
            None => Err(DbError::GetVoid(recid.get())),
            Some(HeapRec::Null) | Some(HeapRec::Prealloc) => Ok(None),
            Some(HeapRec::Live(o)) => match o.downcast_ref::<R>() {
                Some(v) => Ok(Some(v.clone())),
                None => Err(DbError::corrupt("heap record type mismatch")),
            },
        }
    }

    fn read(&self, recid: Recid, action: &mut dyn RecordRead) -> Result<i64> {
        assert_not_in_action("read");
        self.check_closed()?;
        let guard = self.shard(recid.get()).read();
        let _a = ActionGuard::enter();
        match guard.get(&recid.get()) {
            None => Err(DbError::GetVoid(recid.get())),
            Some(HeapRec::Null) | Some(HeapRec::Prealloc) => action.on_null(),
            Some(HeapRec::Live(o)) => action.on_object(o.as_ref()),
        }
    }

    fn update<R: Record>(
        &self,
        recid: Recid,
        value: Option<&R>,
        _ser: &(impl Serializer<R> + Sync),
    ) -> Result<()> {
        assert_not_in_action("update");
        self.check_closed()?;
        let rec = match value {
            None => HeapRec::Null,
            Some(v) => HeapRec::Live(Arc::new(v.clone())),
        };
        let mut guard = self.shard(recid.get()).write();
        if !guard.contains_key(&recid.get()) {
            return Err(DbError::GetVoid(recid.get()));
        }
        guard.insert(recid.get(), rec);
        Ok(())
    }

    fn compare_and_swap<R: Record>(
        &self,
        recid: Recid,
        expect: Option<&R>,
        new: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
    ) -> Result<bool> {
        assert_not_in_action("compare_and_swap");
        self.check_closed()?;
        let mut guard = self.shard(recid.get()).write();
        let current: Option<R> = match guard.get(&recid.get()) {
            None => return Err(DbError::GetVoid(recid.get())),
            Some(HeapRec::Null) | Some(HeapRec::Prealloc) => None,
            Some(HeapRec::Live(o)) => match o.downcast_ref::<R>() {
                Some(v) => Some(v.clone()),
                None => return Err(DbError::corrupt("heap record type mismatch")),
            },
        };
        let eq = match (&current, expect) {
            (None, None) => true,
            (Some(c), Some(e)) => ser.equals(c, e),
            _ => false,
        };
        if !eq {
            return Ok(false);
        }
        let rec = match new {
            None => HeapRec::Null,
            Some(v) => HeapRec::Live(Arc::new(v.clone())),
        };
        guard.insert(recid.get(), rec);
        Ok(true)
    }

    fn delete(&self, recid: Recid) -> Result<()> {
        assert_not_in_action("delete");
        self.check_closed()?;
        let mut guard = self.shard(recid.get()).write();
        if guard.remove(&recid.get()).is_none() {
            return Err(DbError::GetVoid(recid.get()));
        }
        drop(guard);
        self.alloc.lock().free.push(recid.get());
        Ok(())
    }

    fn commit(&self) -> Result<()> {
        self.check_closed()
    }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        for s in self.shards.iter() {
            s.write().clear();
        }
        let mut a = self.alloc.lock();
        a.free.clear();
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn is_thread_safe(&self) -> bool {
        self.thread_safe
    }

    fn verify(&self) -> Result<()> {
        self.check_closed()?;
        let a = self.alloc.lock();
        let max = a.max_recid;
        for &free in &a.free {
            if free > max {
                return Err(DbError::VerifyFailed(format!(
                    "free recid beyond maxRecid: {free}"
                )));
            }
            if self.shard(free).read().contains_key(&free) {
                return Err(DbError::VerifyFailed(format!("free recid is live: {free}")));
            }
        }
        for s in self.shards.iter() {
            for &recid in s.read().keys() {
                if recid < 1 || recid > max {
                    return Err(DbError::VerifyFailed(format!(
                        "recid out of range: {recid}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn get_all_recids(&self) -> Result<Vec<Recid>> {
        self.check_closed()?;
        let mut out = Vec::new();
        for s in self.shards.iter() {
            for (&recid, rec) in s.read().iter() {
                if !matches!(rec, HeapRec::Prealloc) {
                    out.push(nz(recid));
                }
            }
        }
        out.sort();
        Ok(out)
    }
}
