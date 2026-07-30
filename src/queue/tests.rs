//! Ported behavioural suites for the queue layer (Java `QueueLongTest`,
//! `PersistentBlockingQueueTest`) plus a `QueueLong` node golden vector.

use std::sync::Arc;
use std::time::Duration;

use super::blocking::{Mode, PersistentBlockingQueue};
use super::long::{Node, QueueLong, NODE_SER};
use crate::error::Result;
use crate::io::{DataOutput2, SliceInput};
use crate::ser::serializers::STRING;
use crate::ser::Serializer;
use crate::store::{Store, StoreByteArray, StoreDirect, StoreOnHeap};

// ===================== QueueLong node golden vectors ======================

#[test]
fn queuelong_node_golden() {
    // Node(prev=0, next=5, ts=10, value=1):
    // packLong(0)=0x80, packLong(5)=0x85, packLong(10)=0x8A, packLong(1)=0x81.
    let mut out = DataOutput2::new();
    NODE_SER.serialize(&mut out, &Node::new(0, 5, 10, 1));
    assert_eq!(out.into_vec(), vec![0x80, 0x85, 0x8A, 0x81]);

    // Multi-byte node round-trip: prev=200 -> [0x01,0xC8], next=1 -> [0x81],
    // ts=128 -> [0x01,0x80], value=300 -> [0x02,0xAC].
    let n = Node::new(200, 1, 128, 300);
    let mut o2 = DataOutput2::new();
    NODE_SER.serialize(&mut o2, &n);
    let bytes = o2.into_vec();
    assert_eq!(bytes, vec![0x01, 0xC8, 0x81, 0x01, 0x80, 0x02, 0xAC]);
    let mut inp = SliceInput::new(&bytes);
    assert_eq!(NODE_SER.deserialize(&mut inp, None).unwrap(), n);
}

// ============================ QueueLong tests =============================

fn fifo_remove_bump_reopen<S: Store>(store: Arc<S>) -> Result<()> {
    let queue = QueueLong::make(store.clone())?;
    let a = queue.put(10, 1)?;
    let b = queue.put(20, 2)?;
    let c = queue.put(30, 3)?;
    assert_eq!(queue.values()?, vec![1, 2, 3]);

    assert_eq!(queue.remove(b, true)?.value, 2);
    assert_eq!(queue.values()?, vec![1, 3]);
    queue.bump(a, 40)?;
    assert_eq!(queue.values()?, vec![3, 1]);
    queue.verify()?;

    let reopened = QueueLong::open(
        store.clone(),
        queue.tail_recid(),
        queue.head_recid(),
        queue.head_prev_recid(),
    )?;
    assert_eq!(reopened.tail()?, c.get());
    assert_eq!(reopened.take()?.unwrap().value, 3);
    assert_eq!(reopened.take()?.unwrap().value, 1);
    assert!(reopened.take()?.is_none());
    reopened.verify()?;
    store.close()?;
    Ok(())
}

#[test]
fn queuelong_fifo_remove_bump_reopen_heap() {
    fifo_remove_bump_reopen(Arc::new(StoreOnHeap::new(true))).unwrap();
}

#[test]
fn queuelong_fifo_remove_bump_reopen_bytearray() {
    fifo_remove_bump_reopen(Arc::new(StoreByteArray::new(true))).unwrap();
}

#[test]
fn queuelong_fifo_remove_bump_reopen_direct() {
    fifo_remove_bump_reopen(Arc::new(StoreDirect::new_heap_ts(true).unwrap())).unwrap();
}

#[test]
fn queuelong_take_until_and_for_each() {
    let store = Arc::new(StoreOnHeap::new(true));
    let queue = QueueLong::make(store.clone()).unwrap();
    queue.put(10, 1).unwrap();
    queue.put(20, 2).unwrap();
    queue.put(30, 3).unwrap();
    queue
        .take_until(|_recid, node: &Node| node.timestamp <= 20)
        .unwrap();
    assert_eq!(queue.values().unwrap(), vec![3]);

    let mut seen: Vec<u64> = Vec::new();
    queue
        .for_each(|_recid, value, timestamp| {
            seen.push(value);
            seen.push(timestamp);
        })
        .unwrap();
    assert_eq!(seen, vec![3, 30]);

    queue.clear().unwrap();
    assert_eq!(queue.size().unwrap(), 0);
    queue.verify().unwrap();
    store.close().unwrap();
}

#[test]
fn queuelong_insert_preallocated_node() {
    let store = Arc::new(StoreOnHeap::new(true));
    let queue = QueueLong::make(store.clone()).unwrap();
    let recid = store.preallocate().unwrap();
    queue.put_preallocated(7, 9, recid).unwrap();
    assert_eq!(queue.tail().unwrap(), recid.get());
    assert_eq!(queue.values().unwrap(), vec![9]);
    queue.verify().unwrap();
    store.close().unwrap();
}

// ====================== PersistentBlockingQueue tests =====================

fn s(v: &str) -> String {
    v.to_string()
}

#[test]
fn blocking_fifo_stack_circular_and_reopen() {
    let store = Arc::new(StoreByteArray::new(true));

    let fifo =
        PersistentBlockingQueue::create(store.clone(), STRING, Mode::Fifo, u64::MAX).unwrap();
    fifo.add_all([s("a"), s("b"), s("c")]).unwrap();
    assert_eq!(fifo.poll().unwrap(), Some(s("a")));
    assert!(fifo.remove_value(&s("b")).unwrap());
    assert_eq!(fifo.peek().unwrap(), Some(s("c")));
    fifo.verify().unwrap();
    let reopened =
        PersistentBlockingQueue::open(store.clone(), fifo.header_recid(), STRING).unwrap();
    assert_eq!(reopened.take().unwrap(), s("c"));
    assert_eq!(reopened.poll().unwrap(), None);

    let stack =
        PersistentBlockingQueue::create(store.clone(), STRING, Mode::Lifo, u64::MAX).unwrap();
    stack.add_all([s("a"), s("b"), s("c")]).unwrap();
    assert_eq!(stack.poll().unwrap(), Some(s("c")));
    assert_eq!(stack.poll().unwrap(), Some(s("b")));
    stack.verify().unwrap();

    let circular =
        PersistentBlockingQueue::create(store.clone(), STRING, Mode::Circular, 3).unwrap();
    circular.add_all([s("a"), s("b"), s("c"), s("d")]).unwrap();
    assert_eq!(circular.len().unwrap(), 3);
    assert_eq!(circular.poll().unwrap(), Some(s("b")));
    assert_eq!(circular.poll().unwrap(), Some(s("c")));
    assert_eq!(circular.poll().unwrap(), Some(s("d")));
    circular.verify().unwrap();
    store.close().unwrap();
}

#[test]
fn blocking_take_wakes_on_put() {
    let store = Arc::new(StoreByteArray::new(true));
    let queue = Arc::new(
        PersistentBlockingQueue::create(store.clone(), STRING, Mode::Fifo, u64::MAX).unwrap(),
    );
    let q2 = queue.clone();
    let taker = std::thread::spawn(move || q2.take());
    // Give the taker time to block on the empty queue.
    std::thread::sleep(Duration::from_millis(100));
    queue.put(s("ready")).unwrap();
    let taken = taker.join().unwrap().unwrap();
    assert_eq!(taken, s("ready"));
    store.close().unwrap();
}

#[test]
fn blocking_close_handle_wakes_taker_with_store_closed() {
    let store = Arc::new(StoreByteArray::new(true));
    let queue = Arc::new(
        PersistentBlockingQueue::create(store.clone(), STRING, Mode::Fifo, u64::MAX).unwrap(),
    );
    let q2 = queue.clone();
    let taker = std::thread::spawn(move || q2.take());
    std::thread::sleep(Duration::from_millis(100));
    queue.close_handle();
    let result = taker.join().unwrap();
    assert!(matches!(result, Err(crate::error::DbError::StoreClosed)));
    store.close().unwrap();
}

#[test]
fn blocking_poll_timeout_returns_none() {
    let store = Arc::new(StoreByteArray::new(true));
    let queue =
        PersistentBlockingQueue::create(store.clone(), STRING, Mode::Fifo, u64::MAX).unwrap();
    assert_eq!(queue.poll_timeout(Duration::from_millis(20)).unwrap(), None);
    store.close().unwrap();
}

// ===================== review follow-up coverage =========================

#[test]
fn queuelong_bump_newest_in_place() {
    // Bumping the newest node hits the in-place timestamp-rewrite branch (not
    // remove+reinsert): order is unchanged, only the timestamp updates.
    let store = Arc::new(StoreOnHeap::new(true));
    let queue = QueueLong::make(store.clone()).unwrap();
    let _a = queue.put(10, 1).unwrap();
    let b = queue.put(20, 2).unwrap(); // newest == headPrev
    queue.bump(b, 99).unwrap();
    assert_eq!(queue.values().unwrap(), vec![1, 2]);
    let mut timestamps: Vec<u64> = Vec::new();
    queue
        .for_each(|_recid, _value, timestamp| timestamps.push(timestamp))
        .unwrap();
    assert_eq!(timestamps, vec![10, 99]);
    queue.verify().unwrap();
    store.close().unwrap();
}

#[test]
fn queuelong_rejects_fields_above_i64_max() {
    let store = Arc::new(StoreOnHeap::new(true));
    let queue = QueueLong::make(store.clone()).unwrap();
    assert!(queue.put(0, (i64::MAX as u64) + 1).is_err());
    assert!(queue.put((i64::MAX as u64) + 1, 0).is_err());
    // The boundary value is accepted.
    queue.put(i64::MAX as u64, i64::MAX as u64).unwrap();
    store.close().unwrap();
}

#[test]
fn queuelong_reentrant_callback_errors_not_deadlock() {
    // A callback that re-enters the same handle must return an error, not hang.
    let store = Arc::new(StoreOnHeap::new(true));
    let queue = QueueLong::make(store.clone()).unwrap();
    queue.put(10, 1).unwrap();
    let mut reentry_errored = false;
    queue
        .take_until(|_recid, _node: &Node| {
            reentry_errored = queue.size().is_err();
            false
        })
        .unwrap();
    assert!(reentry_errored);
    store.close().unwrap();
}

#[test]
fn blocking_remove_head_element_size_one() {
    // Head removal with size==1 exercises the `tail = 0` header-reset branch.
    let store = Arc::new(StoreByteArray::new(true));
    let queue =
        PersistentBlockingQueue::create(store.clone(), STRING, Mode::Fifo, u64::MAX).unwrap();
    queue.add(s("a")).unwrap();
    assert!(queue.remove_value(&s("a")).unwrap());
    assert_eq!(queue.len().unwrap(), 0);
    queue.verify().unwrap();
    store.close().unwrap();
}

#[test]
fn blocking_remove_tail_and_middle_elements() {
    let store = Arc::new(StoreByteArray::new(true));
    let queue =
        PersistentBlockingQueue::create(store.clone(), STRING, Mode::Fifo, u64::MAX).unwrap();
    // Remove the TAIL element: exercises the `h.tail == recid -> previousRecid`
    // header fix-up.
    queue.add_all([s("a"), s("b"), s("c")]).unwrap();
    assert!(queue.remove_value(&s("c")).unwrap());
    queue.verify().unwrap();
    assert_eq!(queue.to_vec().unwrap(), vec![s("a"), s("b")]);

    // Remove a MIDDLE element (previous != 0, tail unchanged).
    queue.clear().unwrap();
    queue.add_all([s("a"), s("b"), s("c")]).unwrap();
    assert!(queue.remove_value(&s("b")).unwrap());
    queue.verify().unwrap();
    assert_eq!(queue.to_vec().unwrap(), vec![s("a"), s("c")]);
    assert_eq!(queue.poll().unwrap(), Some(s("a")));
    assert_eq!(queue.poll().unwrap(), Some(s("c")));
    store.close().unwrap();
}

#[test]
fn blocking_offer_timeout_success_drain_and_contains() {
    let store = Arc::new(StoreByteArray::new(true));
    let queue =
        PersistentBlockingQueue::create(store.clone(), STRING, Mode::Fifo, u64::MAX).unwrap();
    // offer_timeout success path (queue not full, returns immediately).
    assert!(queue
        .offer_timeout(s("a"), Duration::from_millis(50))
        .unwrap());
    queue.add_all([s("b"), s("c")]).unwrap();

    assert!(queue.contains(&s("b")).unwrap());
    assert!(!queue.contains(&s("z")).unwrap());

    let mut out = Vec::new();
    assert_eq!(queue.drain_to(&mut out, 2).unwrap(), 2);
    assert_eq!(out, vec![s("a"), s("b")]);
    assert_eq!(queue.len().unwrap(), 1);
    assert_eq!(queue.poll().unwrap(), Some(s("c")));
    store.close().unwrap();
}

#[test]
fn blocking_create_rejects_capacity_above_i64_max() {
    let store = Arc::new(StoreByteArray::new(true));
    assert!(PersistentBlockingQueue::<_, String, _>::create(
        store.clone(),
        STRING,
        Mode::Circular,
        u64::MAX
    )
    .is_err());
    store.close().unwrap();
}
