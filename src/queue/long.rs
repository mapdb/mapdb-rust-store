//! `QueueLong` — persistent FIFO of `(timestamp, value)` long pairs over a
//! Store4 store, ported from Java `org.mapdb.QueueLong` /
//! `QueueLongTakeUntil`.
//!
//! A queue is reopenable from just its three pointer recids
//! (`tail_recid`, `head_recid`, `head_prev_recid`) — there is no separate
//! header object. It is a **direct store primitive**, not a named DB catalog
//! object: the DB facade writes no QueueLong catalog entry
//! and constructs it directly from the three recids. See `PORTING-GAPS.md`.
//!
//! ## Wire format (byte-for-byte with Java)
//!
//! Each of the three pointer recids stores a single `LONG_PACKED` value (a node
//! recid, or `0` for "none"). A queue **node** record is four packed longs, in
//! order: `packLong(prev) ++ packLong(next) ++ packLong(timestamp) ++
//! packLong(value)`. Java writes `prev`/`next` via `Serializers.LONG_PACKED`
//! and `timestamp`/`value` via `DataOutput2.packLong`; for non-negative values
//! those two encoders emit identical bytes, so all four fields are plain packed
//! longs. Golden vectors in the test module pin this.
//!
//! ## Strictness (Java)
//!
//! Java's `QueueLong.Node` constructor rejects a negative `timestamp`/`value`
//! (or recid) with `IllegalArgumentException`, because the fields are packed as
//! **unsigned** longs — stricter than MapDB 3, which accepted negatives. This
//! port enforces that structurally: the public `put` API and every `Node`
//! field take `u64`, so a negative is unrepresentable at compile time
//! (decisions D8 / D9.2 — "API-misuse panics become compile-time"). See the
//! `PORTING-GAPS.md`.
//!
//! ## Concurrency
//!
//! Java's methods are `synchronized` on the handle. Here a per-handle
//! `parking_lot::Mutex<()>` serializes operations so each multi-record mutation
//! is atomic against other operations on the same handle. As in Java, do not
//! open two writable handles over the same pointer recids concurrently; clone /
//! share one handle instead.

use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use crate::ser::families::LONG_PACKED;
use crate::ser::Serializer;
use crate::store::{Recid, Store};
use parking_lot::Mutex;
use std::num::NonZeroU64;
use std::sync::Arc;

/// A queue node: two link recids plus the `(timestamp, value)` payload. `prev`
/// is `0` for the oldest (tail) node; `next` always points to the following
/// node or the head sentinel. All four fields are non-negative (see the module
/// strictness note), hence `u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node {
    pub prev: u64,
    pub next: u64,
    pub timestamp: u64,
    pub value: u64,
}

impl Node {
    pub fn new(prev: u64, next: u64, timestamp: u64, value: u64) -> Self {
        Node {
            prev,
            next,
            timestamp,
            value,
        }
    }

    fn with_prev(self, prev: u64) -> Self {
        Node { prev, ..self }
    }
    fn with_next(self, next: u64) -> Self {
        Node { next, ..self }
    }
    fn with_links_and_timestamp(self, prev: u64, next: u64, timestamp: u64) -> Self {
        Node {
            prev,
            next,
            timestamp,
            value: self.value,
        }
    }
}

/// Byte-for-byte codec for [`Node`] — four packed longs (`prev`, `next`,
/// `timestamp`, `value`). Matches Java `QueueLong.Node.SERIALIZER`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NodeSer;

/// Static instance so it can be handed to the store's generic API by reference.
pub static NODE_SER: NodeSer = NodeSer;

impl Serializer<Node> for NodeSer {
    fn serialize(&self, out: &mut DataOutput2, value: &Node) {
        out.pack_long(value.prev);
        out.pack_long(value.next);
        out.pack_long(value.timestamp);
        out.pack_long(value.value);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Node> {
        let prev = input.unpack_long()?;
        let next = input.unpack_long()?;
        let timestamp = input.unpack_long()?;
        let value = input.unpack_long()?;
        Ok(Node {
            prev,
            next,
            timestamp,
            value,
        })
    }
    fn compare(&self, a: &Node, b: &Node) -> std::cmp::Ordering {
        (a.prev, a.next, a.timestamp, a.value).cmp(&(b.prev, b.next, b.timestamp, b.value))
    }
    fn equals(&self, a: &Node, b: &Node) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

/// Callback for [`QueueLong::take_until`]. Java's `QueueLongTakeUntil`
/// functional interface; blanket-implemented for closures so callers can pass
/// `|recid, node| …`. Return `true` to consume the node and continue.
///
/// The callback runs while the handle lock is held and **must not** re-enter
/// the same queue handle — see [`QueueLong::take_until`] for details.
pub trait QueueLongTakeUntil {
    fn take(&mut self, node_recid: Recid, node: &Node) -> bool;
}

impl<F: FnMut(Recid, &Node) -> bool> QueueLongTakeUntil for F {
    fn take(&mut self, node_recid: Recid, node: &Node) -> bool {
        self(node_recid, node)
    }
}

#[inline]
fn nz(v: u64) -> Result<Recid> {
    NonZeroU64::new(v)
        .ok_or_else(|| DbError::corrupt("QueueLong: zero recid where a node was expected"))
}

/// Largest field value that round-trips through Java, whose packed longs decode
/// as **signed** `long`s. A `u64` in `(i64::MAX, u64::MAX]` would serialize a
/// record Java's `QueueLong.Node` rejects (negative field). See the module docs.
const MAX_FIELD: u64 = i64::MAX as u64;

#[inline]
fn check_field(v: u64) -> Result<()> {
    if v > MAX_FIELD {
        return Err(DbError::corrupt(
            "QueueLong: timestamp/value exceeds i64::MAX (would break Java interop)",
        ));
    }
    Ok(())
}

/// Resets the callback-owner marker on scope exit (including a panicking
/// callback), so a re-entry guard is never left stuck set.
struct OwnerGuard<'a>(&'a Mutex<Option<std::thread::ThreadId>>);
impl Drop for OwnerGuard<'_> {
    fn drop(&mut self) {
        *self.0.lock() = None;
    }
}

/// Persistent FIFO of `(timestamp, value)` long pairs with O(1) removal / bump
/// by node recid. See the module docs for the format and strictness contract.
pub struct QueueLong<S: Store> {
    store: Arc<S>,
    tail_recid: Recid,
    head_recid: Recid,
    head_prev_recid: Recid,
    lock: Mutex<()>,
    /// Thread currently running a `take_until`/`for_each` callback while holding
    /// [`lock`](Self::lock). Java's `synchronized` monitor is reentrant, so a
    /// callback re-entering the same handle works there; the non-reentrant
    /// `parking_lot::Mutex` would instead deadlock. [`enter`](Self::enter)
    /// consults this to fail such same-thread re-entry loudly (a distinct
    /// second lock, so an unrelated thread still blocks on `lock` normally).
    callback_owner: Mutex<Option<std::thread::ThreadId>>,
}

impl<S: Store> QueueLong<S> {
    /// Reopen a queue from its three pointer recids. Fails if
    /// `tail_recid == head_recid` (Java `IllegalArgumentException`).
    pub fn open(
        store: Arc<S>,
        tail_recid: Recid,
        head_recid: Recid,
        head_prev_recid: Recid,
    ) -> Result<Self> {
        if tail_recid == head_recid {
            return Err(DbError::corrupt("QueueLong: tailRecid == headRecid"));
        }
        Ok(QueueLong {
            store,
            tail_recid,
            head_recid,
            head_prev_recid,
            lock: Mutex::new(()),
            callback_owner: Mutex::new(None),
        })
    }

    /// Acquire the handle lock, first rejecting a same-thread re-entry from a
    /// running `take_until`/`for_each` callback (which would otherwise
    /// deadlock on the non-reentrant mutex).
    fn enter(&self) -> Result<parking_lot::MutexGuard<'_, ()>> {
        let owner = *self.callback_owner.lock();
        if owner == Some(std::thread::current().id()) {
            return Err(DbError::corrupt(
                "QueueLong: reentrant call into the same handle from a take_until/for_each callback",
            ));
        }
        Ok(self.lock.lock())
    }

    /// Allocate a fresh empty queue (sentinel + three pointer records) in
    /// `store`. Mirrors Java `QueueLong.make(store)`.
    pub fn make(store: Arc<S>) -> Result<Self> {
        let sentinel = store.preallocate()?;
        let s = sentinel.get() as i64;
        let tail = store.put(&s, &LONG_PACKED)?;
        let head = store.put(&s, &LONG_PACKED)?;
        let head_prev = store.put(&0i64, &LONG_PACKED)?;
        Self::open(store, tail, head, head_prev)
    }

    pub fn store(&self) -> &Arc<S> {
        &self.store
    }
    pub fn tail_recid(&self) -> Recid {
        self.tail_recid
    }
    pub fn head_recid(&self) -> Recid {
        self.head_recid
    }
    pub fn head_prev_recid(&self) -> Recid {
        self.head_prev_recid
    }

    // ---- pointer-record accessors ------------------------------------------

    fn read_ptr(&self, recid: Recid) -> Result<u64> {
        match self.store.get(recid, &LONG_PACKED)? {
            Some(v) => Ok(v as u64),
            None => Err(DbError::corrupt("QueueLong: missing queue pointer")),
        }
    }
    fn write_ptr(&self, recid: Recid, value: u64) -> Result<()> {
        self.store
            .update(recid, Some(&(value as i64)), &LONG_PACKED)
    }

    /// Recid of the oldest node (or the head sentinel when empty).
    pub fn tail(&self) -> Result<u64> {
        let _g = self.enter()?;
        self.read_ptr(self.tail_recid)
    }
    /// Recid of the head sentinel (the preallocated append slot).
    pub fn head(&self) -> Result<u64> {
        let _g = self.enter()?;
        self.read_ptr(self.head_recid)
    }
    /// Recid of the newest node (or `0` when empty).
    pub fn head_prev(&self) -> Result<u64> {
        let _g = self.enter()?;
        self.read_ptr(self.head_prev_recid)
    }

    fn tail_locked(&self) -> Result<u64> {
        self.read_ptr(self.tail_recid)
    }
    fn head_locked(&self) -> Result<u64> {
        self.read_ptr(self.head_recid)
    }
    fn head_prev_locked(&self) -> Result<u64> {
        self.read_ptr(self.head_prev_recid)
    }

    fn get_node(&self, recid: Recid) -> Result<Option<Node>> {
        self.store.get(recid, &NODE_SER)
    }

    // ---- mutations ---------------------------------------------------------

    /// Append `(timestamp, value)` at the head and return the new node's recid.
    ///
    /// `timestamp`/`value` must be `<= i64::MAX`; a larger `u64` would serialize
    /// a record Java rejects on read (see the module strictness note).
    pub fn put(&self, timestamp: u64, value: u64) -> Result<Recid> {
        check_field(timestamp)?;
        check_field(value)?;
        let _g = self.enter()?;
        let next = self.store.preallocate()?;
        let old_head = nz(self.head_locked()?)?;
        let old_prev = self.head_prev_locked()?;
        self.store.update(
            old_head,
            Some(&Node::new(old_prev, next.get(), timestamp, value)),
            &NODE_SER,
        )?;
        self.write_ptr(self.head_recid, next.get())?;
        self.write_ptr(self.head_prev_recid, old_head.get())?;
        Ok(old_head)
    }

    /// Insert a caller-preallocated node at the head. Mirrors Java
    /// `put(timestamp, value, nodeRecid)`.
    pub fn put_preallocated(&self, timestamp: u64, value: u64, node_recid: Recid) -> Result<()> {
        check_field(timestamp)?;
        check_field(value)?;
        let _g = self.enter()?;
        self.put_preallocated_locked(timestamp, value, node_recid)
    }

    /// Remove and return the oldest node, or `None` when empty.
    pub fn take(&self) -> Result<Option<Node>> {
        let _g = self.enter()?;
        self.take_locked()
    }

    fn take_locked(&self) -> Result<Option<Node>> {
        let old_tail = nz(self.tail_locked()?)?;
        let node = match self.get_node(old_tail)? {
            Some(n) => n,
            None => {
                self.write_ptr(self.head_prev_recid, 0)?;
                return Ok(None);
            }
        };
        self.store.delete(old_tail)?;
        self.write_ptr(self.tail_recid, node.next)?;
        // Reset headPrev to 0 iff it still points at the node we just removed
        // (the single-element case). Ignore the CAS result, as Java does.
        let _ = self.store.compare_and_swap(
            self.head_prev_recid,
            Some(&(old_tail.get() as i64)),
            Some(&0i64),
            &LONG_PACKED,
        )?;
        let next_recid = nz(node.next)?;
        if let Some(next) = self.get_node(next_recid)? {
            self.store
                .update(next_recid, Some(&next.with_prev(0)), &NODE_SER)?;
        }
        Ok(Some(node))
    }

    /// Consume oldest nodes while `callback` returns `true`.
    ///
    /// The callback runs while the handle lock is held. It **must not** call
    /// back into this same queue handle (`put`/`take`/`bump`/`remove`/…): unlike
    /// Java's reentrant `synchronized`, doing so returns
    /// [`DbError::DataCorruption`] (a re-entry guard) rather than deadlocking.
    pub fn take_until(&self, mut callback: impl QueueLongTakeUntil) -> Result<()> {
        let _g = self.enter()?;
        *self.callback_owner.lock() = Some(std::thread::current().id());
        let _owner = OwnerGuard(&self.callback_owner);
        loop {
            let recid = nz(self.tail_locked()?)?;
            let node = match self.get_node(recid)? {
                Some(n) => n,
                None => return Ok(()),
            };
            if !callback.take(recid, &node) {
                return Ok(());
            }
            self.take_locked()?;
        }
    }

    /// Unlink a node. When `remove_node` is false its record is left intact for
    /// the caller (used by [`bump`](Self::bump)).
    pub fn remove(&self, node_recid: Recid, remove_node: bool) -> Result<Node> {
        let _g = self.enter()?;
        self.remove_locked(node_recid, remove_node)
    }

    fn remove_locked(&self, node_recid: Recid, remove_node: bool) -> Result<Node> {
        let node = self
            .get_node(node_recid)?
            .ok_or_else(|| DbError::corrupt("QueueLong: node not found"))?;
        if remove_node {
            self.store.delete(node_recid)?;
        }
        let next_recid = nz(node.next)?;
        match self.get_node(next_recid)? {
            Some(next) => {
                if next.prev != node_recid.get() {
                    return Err(DbError::corrupt("QueueLong: next-node backlink mismatch"));
                }
                self.store
                    .update(next_recid, Some(&next.with_prev(node.prev)), &NODE_SER)?;
            }
            None => {
                if self.head_prev_locked()? != node_recid.get() {
                    return Err(DbError::corrupt("QueueLong: headPrev mismatch"));
                }
                self.write_ptr(self.head_prev_recid, node.prev)?;
            }
        }
        if node.prev != 0 {
            let prev_recid = nz(node.prev)?;
            let previous = self.get_node(prev_recid)?;
            match previous {
                Some(previous) if previous.next == node_recid.get() => {
                    self.store.update(
                        prev_recid,
                        Some(&previous.with_next(node.next)),
                        &NODE_SER,
                    )?;
                }
                _ => return Err(DbError::corrupt("QueueLong: previous-node link mismatch")),
            }
        } else {
            if self.tail_locked()? != node_recid.get() {
                return Err(DbError::corrupt("QueueLong: tail mismatch"));
            }
            self.write_ptr(self.tail_recid, node.next)?;
        }
        Ok(node)
    }

    /// Move a node to the newest position and replace its timestamp.
    pub fn bump(&self, node_recid: Recid, new_timestamp: u64) -> Result<()> {
        check_field(new_timestamp)?;
        let _g = self.enter()?;
        let newest = self.head_prev_locked()?;
        let node = self
            .get_node(node_recid)?
            .ok_or_else(|| DbError::corrupt("QueueLong: node not found"))?;
        if newest == node_recid.get() {
            self.store.update(
                node_recid,
                Some(&node.with_links_and_timestamp(node.prev, node.next, new_timestamp)),
                &NODE_SER,
            )?;
            return Ok(());
        }
        self.remove_locked(node_recid, false)?;
        self.put_preallocated_locked(new_timestamp, node.value, node_recid)
    }

    /// Locked body of [`put_preallocated`](Self::put_preallocated), reused by
    /// [`bump`](Self::bump) which already holds the handle lock.
    fn put_preallocated_locked(&self, timestamp: u64, value: u64, node_recid: Recid) -> Result<()> {
        let prev = self.head_prev_locked()?;
        let sentinel = self.head_locked()?;
        self.store.update(
            node_recid,
            Some(&Node::new(prev, sentinel, timestamp, value)),
            &NODE_SER,
        )?;
        self.write_ptr(self.head_prev_recid, node_recid.get())?;
        if prev != 0 {
            let prev_recid = nz(prev)?;
            let previous = self
                .get_node(prev_recid)?
                .ok_or_else(|| DbError::corrupt("QueueLong: previous node not found"))?;
            self.store.update(
                prev_recid,
                Some(&previous.with_next(node_recid.get())),
                &NODE_SER,
            )?;
        }
        if self.tail_locked()? == sentinel {
            self.write_ptr(self.tail_recid, node_recid.get())?;
        }
        Ok(())
    }

    /// Remove every node.
    pub fn clear(&self) -> Result<()> {
        self.take_until(|_recid: Recid, _node: &Node| true)
    }

    /// Number of nodes currently in the queue (O(n) walk, like Java).
    pub fn size(&self) -> Result<u64> {
        let _g = self.enter()?;
        let sentinel = self.head_locked()?;
        let mut recid = self.tail_locked()?;
        let mut count = 0u64;
        while recid != sentinel {
            let node = self
                .get_node(nz(recid)?)?
                .ok_or_else(|| DbError::corrupt("QueueLong: linked node not found"))?;
            recid = node.next;
            count += 1;
        }
        Ok(count)
    }

    /// Values from oldest to newest.
    pub fn values(&self) -> Result<Vec<u64>> {
        let _g = self.enter()?;
        let mut out = Vec::new();
        let mut recid = self.tail_locked()?;
        loop {
            match self.get_node(nz(recid)?)? {
                Some(node) => {
                    out.push(node.value);
                    recid = node.next;
                }
                None => return Ok(out),
            }
        }
    }

    /// Visit every node oldest-first: `f(node_recid, value, timestamp)`.
    ///
    /// `f` runs under the handle lock and **must not** call back into this same
    /// queue handle (see [`take_until`](Self::take_until) for the re-entry
    /// contract).
    pub fn for_each(&self, mut f: impl FnMut(Recid, u64, u64)) -> Result<()> {
        let _g = self.enter()?;
        *self.callback_owner.lock() = Some(std::thread::current().id());
        let _owner = OwnerGuard(&self.callback_owner);
        let mut recid = self.tail_locked()?;
        loop {
            let r = nz(recid)?;
            match self.get_node(r)? {
                Some(node) => {
                    f(r, node.value, node.timestamp);
                    recid = node.next;
                }
                None => return Ok(()),
            }
        }
    }

    /// Structural self-check; `Err(VerifyFailed)` on inconsistency.
    pub fn verify(&self) -> Result<()> {
        let _g = self.enter()?;
        let sentinel = self.head_locked()?;
        let first = self.tail_locked()?;
        let newest = self.head_prev_locked()?;
        if sentinel == first {
            if newest != 0 {
                return Err(DbError::VerifyFailed("empty QueueLong has headPrev".into()));
            }
            return Ok(());
        }
        let mut previous = 0u64;
        let mut recid = first;
        while recid != sentinel {
            let node = self
                .get_node(nz(recid)?)?
                .ok_or_else(|| DbError::VerifyFailed(format!("QueueLong node missing: {recid}")))?;
            if node.prev != previous {
                return Err(DbError::VerifyFailed("QueueLong backlink mismatch".into()));
            }
            previous = recid;
            recid = node.next;
        }
        if self.get_node(nz(sentinel)?)?.is_some() {
            return Err(DbError::VerifyFailed(
                "QueueLong sentinel is not preallocated".into(),
            ));
        }
        if previous != newest {
            return Err(DbError::VerifyFailed("QueueLong headPrev mismatch".into()));
        }
        Ok(())
    }
}
