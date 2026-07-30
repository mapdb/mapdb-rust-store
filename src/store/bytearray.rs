//! `StoreByteArray` — reference `StoreDelta` (spec 02 §4): one `Vec<u8>` per
//! record, `buf.len()` is capacity, `used` is content length. Everything
//! explicit — this store doubles as the differential-fuzz oracle. Kept dumb;
//! its value is being obviously correct.
//!
//! Sharded `RwLock<HashMap>` (the segment lock *is* the shard lock) + a global
//! allocation mutex. Serialization happens outside the record lock.

use super::locks::assert_not_in_action;
use super::{AppendResult, LeaseTable, Recid, Record, RecordRead, Store, StoreDelta};
use crate::error::{DbError, Result};
use crate::io::{DataOutput2, SliceInput};
use crate::ser::Serializer;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const SHARDS: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RecState {
    Live,
    Null,
    Prealloc,
}

struct Rec {
    buf: Vec<u8>,
    used: usize,
    state: RecState,
}

struct Alloc {
    max_recid: u64,
    free: Vec<u64>,
}

pub struct StoreByteArray {
    shards: Box<[RwLock<HashMap<u64, Rec>>]>,
    alloc: Mutex<Alloc>,
    thread_safe: bool,
    closed: AtomicBool,
    #[allow(dead_code)] // read via StoreLease, used by the collection layer
    lease_table: Arc<LeaseTable>,
}

impl Default for StoreByteArray {
    fn default() -> Self {
        Self::new(true)
    }
}

impl StoreByteArray {
    pub fn new(thread_safe: bool) -> Self {
        let mut shards = Vec::with_capacity(SHARDS);
        for _ in 0..SHARDS {
            shards.push(RwLock::new(HashMap::new()));
        }
        StoreByteArray {
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
    fn shard(&self, recid: u64) -> &RwLock<HashMap<u64, Rec>> {
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

    /// Serialize `value` (or build a null record), reserving `headroom` trailing
    /// bytes. Runs outside any record lock.
    fn new_rec<R: Record>(
        value: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
        headroom: usize,
    ) -> Rec {
        match value {
            None => Rec {
                buf: vec![0u8; headroom],
                used: 0,
                state: RecState::Null,
            },
            Some(v) => {
                let mut out = DataOutput2::with_capacity(ser.size_hint());
                ser.serialize(&mut out, v);
                let used = out.pos();
                let mut buf = out.into_vec();
                buf.resize(used + headroom, 0);
                Rec {
                    buf,
                    used,
                    state: RecState::Live,
                }
            }
        }
    }
}

impl super::StoreLease for StoreByteArray {
    fn lease_table(&self) -> &Arc<LeaseTable> {
        &self.lease_table
    }
}

// The oracle store only offers the locked baseline read; the default
// `read_torn_safe` body delegates to `read`, which is correct here (D4).
impl super::StoreTornRead for StoreByteArray {}

#[inline]
fn nz(recid: u64) -> Recid {
    NonZeroU64::new(recid).expect("recid 0 is never allocated")
}

impl Store for StoreByteArray {
    fn preallocate(&self) -> Result<Recid> {
        assert_not_in_action("preallocate");
        self.check_closed()?;
        let recid = self.alloc_recid();
        self.shard(recid).write().insert(
            recid,
            Rec {
                buf: Vec::new(),
                used: 0,
                state: RecState::Prealloc,
            },
        );
        Ok(nz(recid))
    }

    fn put<R: Record>(&self, value: &R, ser: &(impl Serializer<R> + Sync)) -> Result<Recid> {
        assert_not_in_action("put");
        self.check_closed()?;
        let rec = Self::new_rec(Some(value), ser, 0);
        let recid = self.alloc_recid();
        self.shard(recid).write().insert(recid, rec);
        Ok(nz(recid))
    }

    fn get<R: Record>(&self, recid: Recid, ser: &(impl Serializer<R> + Sync)) -> Result<Option<R>> {
        assert_not_in_action("get");
        let guard = self.shard(recid.get()).read();
        self.check_closed()?;
        match guard.get(&recid.get()) {
            None => Err(DbError::GetVoid(recid.get())),
            Some(r) if r.state != RecState::Live => Ok(None),
            Some(r) => {
                let mut input = SliceInput::new(&r.buf[..r.used]);
                Ok(Some(ser.deserialize(&mut input, Some(r.used))?))
            }
        }
    }

    fn read(&self, recid: Recid, action: &mut dyn RecordRead) -> Result<i64> {
        assert_not_in_action("read");
        let guard = self.shard(recid.get()).read();
        self.check_closed()?;
        match guard.get(&recid.get()) {
            None => Err(DbError::GetVoid(recid.get())),
            Some(r) if r.state != RecState::Live => action.on_null(),
            Some(r) => {
                let mut input = SliceInput::new(&r.buf[..r.used]);
                action.on_bytes(&mut input, r.used)
            }
        }
    }

    fn update<R: Record>(
        &self,
        recid: Recid,
        value: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
    ) -> Result<()> {
        self.update_with_headroom_opt(recid, value, ser, 0)
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
        // Deserialize current under the write lock; build the new rec inside too
        // (small; correctness over lock-hold time for the oracle).
        let mut guard = self.shard(recid.get()).write();
        let current: Option<R> = match guard.get(&recid.get()) {
            None => return Err(DbError::GetVoid(recid.get())),
            Some(r) if r.state != RecState::Live => None,
            Some(r) => {
                let mut input = SliceInput::new(&r.buf[..r.used]);
                Some(ser.deserialize(&mut input, Some(r.used))?)
            }
        };
        let eq = match (&current, expect) {
            (None, None) => true,
            (Some(c), Some(e)) => ser.equals(c, e),
            _ => false,
        };
        if !eq {
            return Ok(false);
        }
        let rec = Self::new_rec(new, ser, 0);
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
        self.alloc.lock().free.clear();
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
        for s in self.shards.iter() {
            for (&recid, r) in s.read().iter() {
                if recid < 1 || recid > max {
                    return Err(DbError::VerifyFailed(format!(
                        "recid out of range: {recid}"
                    )));
                }
                if r.used > r.buf.len() {
                    return Err(DbError::VerifyFailed(format!(
                        "used beyond capacity, recid={recid}"
                    )));
                }
            }
        }
        for &free in &a.free {
            if self.shard(free).read().contains_key(&free) {
                return Err(DbError::VerifyFailed(format!("free recid is live: {free}")));
            }
        }
        Ok(())
    }

    fn get_all_recids(&self) -> Result<Vec<Recid>> {
        self.check_closed()?;
        let mut out = Vec::new();
        for s in self.shards.iter() {
            for (&recid, r) in s.read().iter() {
                if r.state != RecState::Prealloc {
                    out.push(nz(recid));
                }
            }
        }
        out.sort();
        Ok(out)
    }
}

impl StoreByteArray {
    fn update_with_headroom_opt<R: Record>(
        &self,
        recid: Recid,
        value: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
        headroom: usize,
    ) -> Result<()> {
        assert_not_in_action("update");
        self.check_closed()?;
        let rec = Self::new_rec(value, ser, headroom);
        let mut guard = self.shard(recid.get()).write();
        if !guard.contains_key(&recid.get()) {
            return Err(DbError::GetVoid(recid.get()));
        }
        guard.insert(recid.get(), rec);
        Ok(())
    }
}

impl StoreDelta for StoreByteArray {
    fn append(&self, recid: Recid, data: &[u8]) -> Result<AppendResult> {
        assert_not_in_action("append");
        self.check_closed()?;
        let mut guard = self.shard(recid.get()).write();
        let r = guard
            .get_mut(&recid.get())
            .ok_or(DbError::GetVoid(recid.get()))?;
        if r.used + data.len() > r.buf.len() {
            let never_provisioned = r.state != RecState::Live && r.buf.is_empty();
            if !never_provisioned {
                return Ok(AppendResult::Refused);
            }
            // first append establishes the record: capacity == len
            r.buf = vec![0u8; data.len()];
        }
        r.buf[r.used..r.used + data.len()].copy_from_slice(data);
        r.used += data.len();
        r.state = RecState::Live;
        Ok(AppendResult::NewSize(r.used))
    }

    fn capacity_remaining(&self, recid: Recid) -> Result<usize> {
        assert_not_in_action("capacity_remaining");
        let guard = self.shard(recid.get()).read();
        self.check_closed()?;
        let r = guard
            .get(&recid.get())
            .ok_or(DbError::GetVoid(recid.get()))?;
        Ok(r.buf.len() - r.used)
    }

    fn update_with_headroom<R: Record>(
        &self,
        recid: Recid,
        value: &R,
        ser: &(impl Serializer<R> + Sync),
        headroom: usize,
    ) -> Result<()> {
        self.update_with_headroom_opt(recid, Some(value), ser, headroom)
    }
}
