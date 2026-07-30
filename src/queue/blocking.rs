//! `PersistentBlockingQueue` — a store-backed FIFO / LIFO stack /
//! overwrite-on-full circular queue with blocking `take`/`put`, ported from
//! Java `org.mapdb.queue.PersistentBlockingQueue`.
//!
//! ## Modes
//!
//! The mode ([`Mode`]), the current head/tail/size pointers, and the circular
//! capacity all live in the **header record**, not in any
//! catalog key. `Mode::ordinal()` is wire-relevant (`FIFO=0`, `LIFO=1`,
//! `CIRCULAR=2`).
//!
//! ## Wire format (byte-for-byte with Java)
//!
//! Header record: `packInt(mode) ++ packLong(head) ++ packLong(tail) ++
//! packLong(size) ++ packLong(capacity)` (a non-circular queue stores
//! `capacity = Long.MAX_VALUE`). Node record: `packLong(next) ++
//! element_serializer(value)`. `head`/`tail`/`next` are node recids, `0` for
//! "none". Golden vectors in the test module pin this.
//!
//! ## Blocking coordination (and its limit)
//!
//! Blocking `take`/`put` use a per-handle `parking_lot::Mutex` + two
//! `Condvar`s (`not_empty` / `not_full`). Wakeups coordinate **only threads
//! that share the one live handle** (`Arc`-clone it to share) — the queue
//! *contents* are durable, but the condition signals are not cross-process.
//!
//! Java `take`/`put` are interruptible (`InterruptedException`). Rust threads
//! have no interruption; a blocked waiter is released only by data becoming
//! available, by a timeout ([`poll_timeout`](PersistentBlockingQueue::poll_timeout)
//! / [`offer_timeout`](PersistentBlockingQueue::offer_timeout)), or by
//! [`close_handle`](PersistentBlockingQueue::close_handle) (the shutdown flag),
//! after which blocked/subsequent operations return
//! [`DbError::StoreClosed`]. See `PORTING-GAPS.md`.
//!
//! Use one writable handle per header; direct callers must not open the same
//! header twice concurrently (locks/conditions are handle-local). The DB facade
//! The DB facade enforces this through its per-name handle cache.

use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use crate::ser::Serializer;
use crate::store::{Recid, Store};
use parking_lot::{Condvar, Mutex};
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Non-circular queues store this as their capacity (Java `Long.MAX_VALUE`).
/// Kept identical to Java for byte parity of the header record.
const UNBOUNDED: u64 = i64::MAX as u64;

#[inline]
fn nz(v: u64) -> Result<Recid> {
    NonZeroU64::new(v)
        .ok_or_else(|| DbError::corrupt("blocking queue: zero recid where a node was expected"))
}

/// Queue discipline. The ordinal is persisted, so variant order is wire-fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Fifo = 0,
    Lifo = 1,
    Circular = 2,
}

impl Mode {
    pub fn ordinal(self) -> i32 {
        self as i32
    }
    fn from_i32(v: i32) -> Result<Mode> {
        match v {
            0 => Ok(Mode::Fifo),
            1 => Ok(Mode::Lifo),
            2 => Ok(Mode::Circular),
            _ => Err(DbError::corrupt("invalid persistent queue mode")),
        }
    }
}

/// The mutable queue header (persisted in one record).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Header {
    mode: i32,
    head: u64,
    tail: u64,
    size: u64,
    capacity: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct HeaderSer;
static HEADER_SER: HeaderSer = HeaderSer;

impl Serializer<Header> for HeaderSer {
    fn serialize(&self, out: &mut DataOutput2, h: &Header) {
        out.pack_int(h.mode);
        out.pack_long(h.head);
        out.pack_long(h.tail);
        out.pack_long(h.size);
        out.pack_long(h.capacity);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Header> {
        Ok(Header {
            mode: input.unpack_int()?,
            head: input.unpack_long()?,
            tail: input.unpack_long()?,
            size: input.unpack_long()?,
            capacity: input.unpack_long()?,
        })
    }
    fn compare(&self, a: &Header, b: &Header) -> Ordering {
        (a.mode as i64, a.head, a.tail, a.size, a.capacity).cmp(&(
            b.mode as i64,
            b.head,
            b.tail,
            b.size,
            b.capacity,
        ))
    }
    fn equals(&self, a: &Header, b: &Header) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

/// A singly-linked node: `next` recid (`0` = end) plus the element.
#[derive(Debug, Clone)]
struct QNode<E> {
    next: u64,
    value: E,
}

/// Wire codec for [`QNode`], borrowing the element serializer for the duration
/// of a store op (`packLong(next) ++ element(value)`).
struct NodeSer<'a, E, Se: Serializer<E>> {
    elem: &'a Se,
    _p: PhantomData<fn() -> E>,
}

impl<'a, E, Se: Serializer<E>> Serializer<QNode<E>> for NodeSer<'a, E, Se> {
    fn serialize(&self, out: &mut DataOutput2, node: &QNode<E>) {
        out.pack_long(node.next);
        self.elem.serialize(out, &node.value);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<QNode<E>> {
        let next = input.unpack_long()?;
        let value = self.elem.deserialize(input, None)?;
        Ok(QNode { next, value })
    }
    fn compare(&self, a: &QNode<E>, b: &QNode<E>) -> Ordering {
        a.next
            .cmp(&b.next)
            .then_with(|| self.elem.compare(&a.value, &b.value))
    }
    fn equals(&self, a: &QNode<E>, b: &QNode<E>) -> bool {
        a.next == b.next && self.elem.equals(&a.value, &b.value)
    }
}

/// Store-backed blocking FIFO/LIFO/circular queue. See the module docs.
pub struct PersistentBlockingQueue<S: Store, E, Se: Serializer<E> + Sync> {
    store: Arc<S>,
    header_recid: Recid,
    serializer: Se,
    lock: Mutex<()>,
    not_empty: Condvar,
    not_full: Condvar,
    closed: AtomicBool,
    _p: PhantomData<fn() -> E>,
}

impl<S, E, Se> PersistentBlockingQueue<S, E, Se>
where
    S: Store,
    E: Clone + Send + Sync + 'static,
    Se: Serializer<E> + Sync,
{
    /// Create a fresh queue of the given `mode`. `capacity` is used only for
    /// [`Mode::Circular`]; FIFO/LIFO are unbounded (`Long.MAX_VALUE`). A
    /// circular `capacity` must be `> 0`.
    pub fn create(store: Arc<S>, serializer: Se, mode: Mode, capacity: u64) -> Result<Self> {
        let actual_capacity = if mode == Mode::Circular {
            capacity
        } else {
            UNBOUNDED
        };
        // Capacity is packed as an unsigned long here but read back by Java as a
        // signed `long` that must be `> 0`; reject anything Java could not
        // reopen (0, or a value in `(i64::MAX, u64::MAX]`).
        if actual_capacity == 0 || actual_capacity > UNBOUNDED {
            return Err(DbError::corrupt(
                "blocking queue: capacity must be in 1..=i64::MAX",
            ));
        }
        let header = Header {
            mode: mode.ordinal(),
            head: 0,
            tail: 0,
            size: 0,
            capacity: actual_capacity,
        };
        let header_recid = store.put(&header, &HEADER_SER)?;
        Self::build(store, header_recid, serializer)
    }

    /// Reopen an existing queue from its header recid.
    pub fn open(store: Arc<S>, header_recid: Recid, serializer: Se) -> Result<Self> {
        Self::build(store, header_recid, serializer)
    }

    fn build(store: Arc<S>, header_recid: Recid, serializer: Se) -> Result<Self> {
        let q = PersistentBlockingQueue {
            store,
            header_recid,
            serializer,
            lock: Mutex::new(()),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            closed: AtomicBool::new(false),
            _p: PhantomData,
        };
        let h = q.header()?;
        if h.mode < 0 || h.mode as usize >= 3 {
            return Err(DbError::corrupt("invalid persistent queue header: mode"));
        }
        // Capacity must be a value Java could reopen (a signed `long` in
        // `1..=i64::MAX`), and the stored size must not exceed it. A hostile header
        // with `size > capacity` would otherwise underflow `remaining_capacity`
        // and the `full()`/`size-1` paths (hardening, now reachable via the DB
        // facade — R7).
        if h.capacity == 0 || h.capacity > UNBOUNDED {
            return Err(DbError::corrupt(
                "invalid persistent queue header: capacity out of 1..=i64::MAX",
            ));
        }
        if h.size > h.capacity {
            return Err(DbError::corrupt(
                "invalid persistent queue header: size exceeds capacity",
            ));
        }
        Ok(q)
    }

    fn node_ser(&self) -> NodeSer<'_, E, Se> {
        NodeSer {
            elem: &self.serializer,
            _p: PhantomData,
        }
    }

    /// Read the header (caller must hold the handle lock). Errors once the
    /// handle is closed, matching Java `header()`.
    fn header(&self) -> Result<Header> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(DbError::StoreClosed);
        }
        self.store
            .get(self.header_recid, &HEADER_SER)?
            .ok_or_else(|| DbError::corrupt("blocking queue header missing"))
    }

    fn node(&self, recid: Recid) -> Result<QNode<E>> {
        self.store
            .get(recid, &self.node_ser())?
            .ok_or_else(|| DbError::corrupt("blocking queue node missing"))
    }

    fn full(&self, h: &Header) -> bool {
        h.size >= h.capacity
    }

    // ---- core enqueue / dequeue (caller holds the lock) --------------------

    fn enqueue(&self, h: &Header, value: &E) -> Result<()> {
        let ns = self.node_ser();
        let mode = Mode::from_i32(h.mode)?;
        let mut h = *h;
        if mode == Mode::Circular && self.full(&h) {
            h = self.dequeue(&h, None)?;
        }
        if mode == Mode::Lifo {
            let recid = self.store.put(
                &QNode {
                    next: h.head,
                    value: value.clone(),
                },
                &ns,
            )?;
            let tail = if h.size == 0 { recid.get() } else { h.tail };
            let nh = Header {
                mode: h.mode,
                head: recid.get(),
                tail,
                size: h.size + 1,
                capacity: h.capacity,
            };
            self.store
                .update(self.header_recid, Some(&nh), &HEADER_SER)?;
        } else {
            let recid = self.store.put(
                &QNode {
                    next: 0,
                    value: value.clone(),
                },
                &ns,
            )?;
            if h.tail != 0 {
                let tail_recid = nz(h.tail)?;
                let tail_node = self.node(tail_recid)?;
                self.store.update(
                    tail_recid,
                    Some(&QNode {
                        next: recid.get(),
                        value: tail_node.value,
                    }),
                    &ns,
                )?;
            }
            let head = if h.size == 0 { recid.get() } else { h.head };
            let nh = Header {
                mode: h.mode,
                head,
                tail: recid.get(),
                size: h.size + 1,
                capacity: h.capacity,
            };
            self.store
                .update(self.header_recid, Some(&nh), &HEADER_SER)?;
        }
        Ok(())
    }

    /// Remove the head node, optionally returning its value in `out`.
    fn dequeue(&self, h: &Header, out: Option<&mut Option<E>>) -> Result<Header> {
        if h.size == 0 {
            return Ok(*h);
        }
        let head_recid = nz(h.head)?;
        let n = self.node(head_recid)?;
        let next = Header {
            mode: h.mode,
            head: n.next,
            tail: if h.size == 1 { 0 } else { h.tail },
            // Guarded by the `size == 0` early-return above; `checked_sub` keeps a
            // hostile/inconsistent header from wrapping (R7 defense-in-depth).
            size: h
                .size
                .checked_sub(1)
                .ok_or_else(|| DbError::corrupt("blocking queue: size underflow"))?,
            capacity: h.capacity,
        };
        self.store
            .update(self.header_recid, Some(&next), &HEADER_SER)?;
        self.store.delete(head_recid)?;
        if let Some(slot) = out {
            *slot = Some(n.value);
        }
        Ok(next)
    }

    fn remove_head_locked(&self) -> Result<E> {
        let h = self.header()?;
        let mut slot = None;
        self.dequeue(&h, Some(&mut slot))?;
        self.not_full.notify_one();
        slot.ok_or_else(|| DbError::corrupt("blocking queue: empty dequeue slot"))
    }

    // ---- non-blocking API --------------------------------------------------

    /// Insert `value`; `Ok(false)` if a bounded (FIFO/LIFO) queue is full.
    pub fn offer(&self, value: E) -> Result<bool> {
        let _g = self.lock.lock();
        let h = self.header()?;
        if self.full(&h) && Mode::from_i32(h.mode)? != Mode::Circular {
            return Ok(false);
        }
        self.enqueue(&h, &value)?;
        self.not_empty.notify_one();
        Ok(true)
    }

    /// Insert `value`, erroring if a bounded queue is full (Java
    /// `AbstractQueue.add` / `IllegalStateException`, mapped to
    /// [`DbError::Unsupported`]).
    pub fn add(&self, value: E) -> Result<()> {
        if self.offer(value)? {
            Ok(())
        } else {
            Err(DbError::Unsupported("blocking queue is full"))
        }
    }

    /// [`add`](Self::add) every element of `values`.
    pub fn add_all<I: IntoIterator<Item = E>>(&self, values: I) -> Result<()> {
        for v in values {
            self.add(v)?;
        }
        Ok(())
    }

    /// Remove and return the head, or `None` if empty.
    pub fn poll(&self) -> Result<Option<E>> {
        let _g = self.lock.lock();
        let h = self.header()?;
        if h.size == 0 {
            return Ok(None);
        }
        let mut slot = None;
        self.dequeue(&h, Some(&mut slot))?;
        self.not_full.notify_one();
        Ok(slot)
    }

    /// The head element without removing it, or `None` if empty.
    pub fn peek(&self) -> Result<Option<E>> {
        let _g = self.lock.lock();
        let h = self.header()?;
        if h.size == 0 {
            Ok(None)
        } else {
            Ok(Some(self.node(nz(h.head)?)?.value))
        }
    }

    // ---- blocking API ------------------------------------------------------

    /// Block until an element is available, then remove and return it. Errors
    /// [`DbError::StoreClosed`] if the handle is closed while waiting.
    pub fn take(&self) -> Result<E> {
        let mut g = self.lock.lock();
        loop {
            let h = self.header()?;
            if h.size != 0 {
                return self.remove_head_locked();
            }
            self.not_empty.wait(&mut g);
        }
    }

    /// [`take`](Self::take) with a timeout; `Ok(None)` if it elapses.
    pub fn poll_timeout(&self, timeout: Duration) -> Result<Option<E>> {
        let deadline = Instant::now().checked_add(timeout);
        let mut g = self.lock.lock();
        loop {
            let h = self.header()?;
            if h.size != 0 {
                return self.remove_head_locked().map(Some);
            }
            match remaining(deadline) {
                Some(d) if !d.is_zero() => {
                    self.not_empty.wait_for(&mut g, d);
                }
                _ => return Ok(None),
            }
        }
    }

    /// Block until there is room, then insert `value`. Circular queues never
    /// block. Errors [`DbError::StoreClosed`] if closed while waiting.
    pub fn put(&self, value: E) -> Result<()> {
        let mut g = self.lock.lock();
        let mut h = self.header()?;
        while self.full(&h) && Mode::from_i32(h.mode)? != Mode::Circular {
            self.not_full.wait(&mut g);
            h = self.header()?;
        }
        self.enqueue(&h, &value)?;
        self.not_empty.notify_one();
        Ok(())
    }

    /// [`put`](Self::put) with a timeout; `Ok(false)` if it elapses.
    pub fn offer_timeout(&self, value: E, timeout: Duration) -> Result<bool> {
        let deadline = Instant::now().checked_add(timeout);
        let mut g = self.lock.lock();
        let mut h = self.header()?;
        while self.full(&h) && Mode::from_i32(h.mode)? != Mode::Circular {
            match remaining(deadline) {
                Some(d) if !d.is_zero() => {
                    self.not_full.wait_for(&mut g, d);
                    h = self.header()?;
                }
                _ => return Ok(false),
            }
        }
        self.enqueue(&h, &value)?;
        self.not_empty.notify_one();
        Ok(true)
    }

    // ---- collection helpers ------------------------------------------------

    /// Current element count.
    pub fn len(&self) -> Result<u64> {
        let _g = self.lock.lock();
        Ok(self.header()?.size)
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Remaining insertable capacity (`capacity - size`).
    pub fn remaining_capacity(&self) -> Result<u64> {
        let _g = self.lock.lock();
        let h = self.header()?;
        // Checked: a hostile header where `size > capacity` slipped past open must
        // not wrap the unsigned subtraction (R7).
        h.capacity
            .checked_sub(h.size)
            .ok_or_else(|| DbError::corrupt("blocking queue: size exceeds capacity"))
    }

    /// True if `value` is present (by serializer equality).
    pub fn contains(&self, value: &E) -> Result<bool> {
        let _g = self.lock.lock();
        let mut recid = self.header()?.head;
        while recid != 0 {
            let n = self.node(nz(recid)?)?;
            if self.serializer.equals(&n.value, value) {
                return Ok(true);
            }
            recid = n.next;
        }
        Ok(false)
    }

    /// Remove the first node equal to `value`; `Ok(true)` if one was removed.
    pub fn remove_value(&self, value: &E) -> Result<bool> {
        let _g = self.lock.lock();
        let h = self.header()?;
        let ns = self.node_ser();
        let mut previous_recid = 0u64;
        let mut recid = h.head;
        while recid != 0 {
            let cur = nz(recid)?;
            let n = self.node(cur)?;
            if self.serializer.equals(&n.value, value) {
                if previous_recid == 0 {
                    let nh = Header {
                        mode: h.mode,
                        head: n.next,
                        tail: if h.size == 1 { 0 } else { h.tail },
                        size: h.size - 1,
                        capacity: h.capacity,
                    };
                    self.store
                        .update(self.header_recid, Some(&nh), &HEADER_SER)?;
                } else {
                    let prev_recid = nz(previous_recid)?;
                    let previous = self.node(prev_recid)?;
                    self.store.update(
                        prev_recid,
                        Some(&QNode {
                            next: n.next,
                            value: previous.value,
                        }),
                        &ns,
                    )?;
                    let nh = Header {
                        mode: h.mode,
                        head: h.head,
                        tail: if h.tail == recid {
                            previous_recid
                        } else {
                            h.tail
                        },
                        size: h.size - 1,
                        capacity: h.capacity,
                    };
                    self.store
                        .update(self.header_recid, Some(&nh), &HEADER_SER)?;
                }
                self.store.delete(cur)?;
                self.not_full.notify_one();
                return Ok(true);
            }
            previous_recid = recid;
            recid = n.next;
        }
        Ok(false)
    }

    /// Move up to `max_elements` head elements into `target`; returns the count
    /// moved.
    pub fn drain_to(&self, target: &mut Vec<E>, max_elements: usize) -> Result<usize> {
        if max_elements == 0 {
            return Ok(0);
        }
        let _g = self.lock.lock();
        let mut count = 0;
        while count < max_elements && self.header()?.size != 0 {
            target.push(self.remove_head_locked()?);
            count += 1;
        }
        Ok(count)
    }

    /// Remove all elements.
    pub fn clear(&self) -> Result<()> {
        let _g = self.lock.lock();
        let mut h = self.header()?;
        while h.size != 0 {
            h = self.dequeue(&h, None)?;
        }
        self.not_full.notify_all();
        Ok(())
    }

    /// Free every node record AND the header record (DB `delete()` teardown).
    /// Reads the header directly so it works even after the handle is closed, and
    /// walks the head chain freeing nodes. Safe to free the header because every
    /// queue clone shares one handle whose `close_handle` globally closes it, so
    /// no live clone can write the freed recids (unlike a map's structural root).
    /// Best-effort: the caller runs this only AFTER the catalog is unlinked, so a
    /// failure leaks records rather than leaving a live catalog pointer.
    pub fn purge_records(&self) -> Result<()> {
        let _g = self.lock.lock();
        if let Some(h) = self.store.get(self.header_recid, &HEADER_SER)? {
            let mut recid = h.head;
            while recid != 0 {
                let cur = nz(recid)?;
                let next = self.node(cur)?.next;
                self.store.delete(cur)?;
                recid = next;
            }
        }
        self.store.delete(self.header_recid)?;
        Ok(())
    }

    /// Snapshot of the elements, head-first.
    pub fn to_vec(&self) -> Result<Vec<E>> {
        let _g = self.lock.lock();
        let mut out = Vec::new();
        let mut recid = self.header()?.head;
        while recid != 0 {
            let n = self.node(nz(recid)?)?;
            out.push(n.value);
            recid = n.next;
        }
        Ok(out)
    }

    /// Wake blocked operations and mark the handle closed (without touching the
    /// shared store). Subsequent operations return [`DbError::StoreClosed`].
    pub fn close_handle(&self) {
        let _g = self.lock.lock();
        self.closed.store(true, AtomicOrdering::Release);
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    pub fn header_recid(&self) -> Recid {
        self.header_recid
    }

    /// The queue's mode (reads the header).
    pub fn mode(&self) -> Result<Mode> {
        let _g = self.lock.lock();
        Mode::from_i32(self.header()?.mode)
    }

    pub fn serializer(&self) -> &Se {
        &self.serializer
    }

    /// Structural self-check; `Err(VerifyFailed)` on inconsistency.
    pub fn verify(&self) -> Result<()> {
        let _g = self.lock.lock();
        let h = self.header()?;
        let mut count = 0u64;
        let mut recid = h.head;
        let mut last = 0u64;
        while recid != 0 {
            count += 1;
            if count > h.size {
                return Err(DbError::VerifyFailed("queue cycle/size mismatch".into()));
            }
            last = recid;
            recid = self.node(nz(recid)?)?.next;
        }
        let tail_ok = if count == 0 {
            h.tail == 0
        } else {
            h.tail == last
        };
        if count != h.size || !tail_ok {
            return Err(DbError::VerifyFailed("queue header/link mismatch".into()));
        }
        Ok(())
    }
}

#[inline]
fn remaining(deadline: Option<Instant>) -> Option<Duration> {
    match deadline {
        Some(d) => Some(d.saturating_duration_since(Instant::now())),
        // `now + timeout` overflowed → effectively unbounded wait.
        None => Some(Duration::from_secs(3600)),
    }
}

#[cfg(test)]
mod golden {
    //! Golden wire-format vectors for the header/node records, hand-computed
    //! from the MapDB packed encoding to pin byte-for-byte Java parity.
    use super::*;
    use crate::ser::serializers::STRING;

    fn header_bytes(h: &Header) -> Vec<u8> {
        let mut out = DataOutput2::new();
        HEADER_SER.serialize(&mut out, h);
        out.into_vec()
    }

    #[test]
    fn header_fresh_fifo() {
        // create(FIFO): mode=0, head=tail=size=0, capacity=Long.MAX_VALUE.
        // packInt(0)=0x80; packLong(0)=0x80 (×3 for head/tail/size);
        // packLong(i64::MAX) = eight 0x7F groups then 0xFF terminator.
        let h = Header {
            mode: 0,
            head: 0,
            tail: 0,
            size: 0,
            capacity: UNBOUNDED,
        };
        assert_eq!(
            header_bytes(&h),
            vec![0x80, 0x80, 0x80, 0x80, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0xFF]
        );
    }

    #[test]
    fn header_fresh_circular_cap3() {
        // create(CIRCULAR, 3): mode=2 -> packInt(2)=0x82; three 0x80; capacity 3 -> 0x83.
        let h = Header {
            mode: 2,
            head: 0,
            tail: 0,
            size: 0,
            capacity: 3,
        };
        assert_eq!(header_bytes(&h), vec![0x82, 0x80, 0x80, 0x80, 0x83]);
    }

    #[test]
    fn header_populated() {
        // mode=1(LIFO), head=1, tail=2, size=2, capacity=MAX.
        let h = Header {
            mode: 1,
            head: 1,
            tail: 2,
            size: 2,
            capacity: UNBOUNDED,
        };
        assert_eq!(
            header_bytes(&h),
            vec![0x81, 0x81, 0x82, 0x82, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0xFF]
        );
        // round-trips
        let bytes = header_bytes(&h);
        let mut inp = crate::io::SliceInput::new(&bytes);
        assert_eq!(HEADER_SER.deserialize(&mut inp, None).unwrap(), h);
    }

    #[test]
    fn node_string_value() {
        // QNode{next:0, value:"a"}: packLong(0)=0x80; STRING("a")=packInt(1)0x81 + 'a'0x61.
        let ns: NodeSer<'_, String, _> = NodeSer {
            elem: &STRING,
            _p: PhantomData,
        };
        let mut out = DataOutput2::new();
        ns.serialize(
            &mut out,
            &QNode {
                next: 0,
                value: "a".to_string(),
            },
        );
        assert_eq!(out.into_vec(), vec![0x80, 0x81, 0x61]);

        // QNode{next:7, value:"bc"}: packLong(7)=0x87; STRING("bc")=0x82 'b'0x62 'c'0x63.
        let mut out2 = DataOutput2::new();
        ns.serialize(
            &mut out2,
            &QNode {
                next: 7,
                value: "bc".to_string(),
            },
        );
        assert_eq!(out2.into_vec(), vec![0x87, 0x82, 0x62, 0x63]);
    }
}

#[cfg(test)]
mod hostile_header {
    //! A persisted header is now reachable through the DB facade, so `open` must
    //! reject a hostile header rather than underflow later (R7).
    use super::*;
    use crate::ser::serializers::StringSer;
    use crate::store::{Store, StoreOnHeap};
    use std::sync::Arc;

    fn open_with(h: Header) -> Result<PersistentBlockingQueue<StoreOnHeap, String, StringSer>> {
        let store = Arc::new(StoreOnHeap::new(true));
        let recid = store.put(&h, &HEADER_SER).unwrap();
        PersistentBlockingQueue::open(store, recid, StringSer)
    }

    #[test]
    fn open_rejects_size_exceeding_capacity() {
        let h = Header {
            mode: 2,
            head: 0,
            tail: 0,
            size: 2,
            capacity: 1,
        };
        assert!(matches!(open_with(h), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn open_rejects_capacity_above_i64_max() {
        let h = Header {
            mode: 0,
            head: 0,
            tail: 0,
            size: 0,
            capacity: u64::MAX,
        };
        assert!(matches!(open_with(h), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn open_rejects_zero_capacity() {
        let h = Header {
            mode: 0,
            head: 0,
            tail: 0,
            size: 0,
            capacity: 0,
        };
        assert!(matches!(open_with(h), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn open_accepts_wellformed_header() {
        let h = Header {
            mode: 0,
            head: 0,
            tail: 0,
            size: 0,
            capacity: UNBOUNDED,
        };
        assert!(open_with(h).is_ok());
    }
}
